use anyrender::{
    RegisterResourceErrorKind, RenderContext, ResourceId, WindowHandle, WindowRenderer,
};
use debug_timer::debug_timer;
use futures_channel::oneshot;
use peniko::{Color, ImageData};
use rustc_hash::FxHashMap;
use std::future::Future;
use std::sync::Arc;
use vello::{
    AaConfig, AaSupport, RenderParams, Renderer as VelloRenderer, RendererOptions,
    Scene as VelloScene,
};
use wgpu::{Features, Limits, PresentMode, SurfaceError, Texture, TextureFormat, TextureUsages};
use wgpu_context::{
    DeviceHandle, SurfaceRenderer, SurfaceRendererConfiguration, TextureConfiguration, WGPUContext,
};

use crate::{DEFAULT_THREADS, VelloScenePainter};

#[cfg(target_arch = "wasm32")]
fn spawn_init<F: Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_init<F: Future<Output = ()>>(f: F) {
    pollster::block_on(f);
}

struct ActiveRenderState {
    renderer: VelloRenderer,
    render_surface: SurfaceRenderer<'static>,
}

/// Result of a successful asynchronous resume; both the active state and the
/// `WGPUContext` are returned so the renderer can reclaim the context.
struct InitOutput {
    wgpu_context: WGPUContext,
    active: ActiveRenderState,
}

#[allow(clippy::large_enum_variant)]
enum RenderState {
    Suspended,
    Pending {
        receiver: oneshot::Receiver<InitOutput>,
    },
    Active(ActiveRenderState),
}

#[derive(Clone)]
pub struct VelloRendererOptions {
    pub features: Option<Features>,
    pub limits: Option<Limits>,
    pub base_color: Color,
    pub antialiasing_method: AaConfig,
}

impl Default for VelloRendererOptions {
    fn default() -> Self {
        Self {
            features: None,
            limits: None,
            base_color: Color::WHITE,
            antialiasing_method: AaConfig::Msaa16,
        }
    }
}

pub struct VelloWindowRenderer {
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,

    /// Optional so the spawned init future can take ownership of it. Restored from
    /// the future's result inside `complete_resume`.
    wgpu_context: Option<WGPUContext>,
    scene: VelloScene,
    config: VelloRendererOptions,

    // Resources
    texture_handles: FxHashMap<ResourceId, ImageData>,
}
impl VelloWindowRenderer {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_options(VelloRendererOptions::default())
    }

    pub fn with_options(config: VelloRendererOptions) -> Self {
        Self {
            wgpu_context: Some(build_wgpu_context(&config)),
            config,
            render_state: RenderState::Suspended,
            window_handle: None,
            scene: VelloScene::new(),
            texture_handles: FxHashMap::default(),
        }
    }

    pub fn current_device_handle(&self) -> Option<&DeviceHandle> {
        match &self.render_state {
            RenderState::Active(state) => Some(&state.render_surface.device_handle),
            _ => None,
        }
    }
}

impl RenderContext for VelloWindowRenderer {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn std::any::Any>,
    ) -> Result<ResourceId, anyrender::RegisterResourceError> {
        let RenderState::Active(state) = &mut self.render_state else {
            return Err(RegisterResourceErrorKind::NotActive.into());
        };

        if let Ok(texture) = resource.downcast::<Texture>() {
            let id = ResourceId::new();
            self.texture_handles
                .insert(id, state.renderer.register_texture(*texture));
            Ok(id)
        } else {
            Err(anyrender::RegisterResourceErrorKind::UnsupportedResourceKind.into())
        }
    }

    fn unregister_resource(&mut self, resource_id: ResourceId) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        if let Some(handle) = self.texture_handles.remove(&resource_id) {
            state.renderer.unregister_texture(handle);
        }
    }

    fn renderer_specific_context(&self) -> Option<Box<dyn std::any::Any>> {
        match &self.render_state {
            RenderState::Active(active_render_state) => Some(Box::new(
                active_render_state.render_surface.device_handle.clone(),
            )),
            RenderState::Pending { .. } => None,
            RenderState::Suspended => None,
        }
    }
}

fn build_wgpu_context(config: &VelloRendererOptions) -> WGPUContext {
    let features =
        config.features.unwrap_or_default() | Features::CLEAR_TEXTURE | Features::PIPELINE_CACHE;
    WGPUContext::with_features_and_limits(Some(features), config.limits.clone())
}

impl WindowRenderer for VelloWindowRenderer {
    type ScenePainter<'a>
        = VelloScenePainter<'a, 'a>
    where
        Self: 'a;

    fn is_active(&self) -> bool {
        matches!(self.render_state, RenderState::Active(_))
    }

    fn resume<F: FnOnce() + 'static>(
        &mut self,
        window_handle: Arc<dyn WindowHandle>,
        width: u32,
        height: u32,
        on_ready: F,
    ) {
        // Drain the wgpu_context. If we were `Pending`, the previous future still holds
        // its own context and will be silently abandoned (its `sender` becomes a no-op
        // when the new receiver replaces it). Build a fresh context on first use.
        let wgpu_context = self
            .wgpu_context
            .take()
            .unwrap_or_else(|| build_wgpu_context(&self.config));

        self.window_handle = Some(window_handle.clone());
        let (sender, receiver) = oneshot::channel();
        let num_init_threads = DEFAULT_THREADS;

        spawn_init(async move {
            let mut wgpu_context = wgpu_context;
            let render_surface = wgpu_context
                .create_surface(
                    window_handle,
                    SurfaceRendererConfiguration {
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                        formats: vec![TextureFormat::Rgba8Unorm, TextureFormat::Bgra8Unorm],
                        width,
                        height,
                        present_mode: PresentMode::AutoVsync,
                        desired_maximum_frame_latency: 2,
                        alpha_mode: wgpu::CompositeAlphaMode::Auto,
                        view_formats: vec![],
                    },
                    Some(TextureConfiguration {
                        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                    }),
                )
                .await
                .expect("Error creating surface");

            let renderer = VelloRenderer::new(
                render_surface.device(),
                RendererOptions {
                    antialiasing_support: AaSupport::all(),
                    use_cpu: false,
                    num_init_threads,
                    pipeline_cache: None,
                },
            )
            .unwrap();

            let _ = sender.send(InitOutput {
                wgpu_context,
                active: ActiveRenderState {
                    renderer,
                    render_surface,
                },
            });
            on_ready();
        });

        self.render_state = RenderState::Pending { receiver };
    }

    fn complete_resume(&mut self) -> bool {
        if let RenderState::Pending { receiver } = &mut self.render_state {
            let Ok(Some(InitOutput {
                wgpu_context,
                active,
            })) = receiver.try_recv()
            else {
                return false;
            };
            self.wgpu_context = Some(wgpu_context);

            self.render_state = RenderState::Active(active);
            return true;
        }
        matches!(self.render_state, RenderState::Active(_))
    }

    fn suspend(&mut self) {
        if let RenderState::Active(state) = &mut self.render_state {
            // Unregister all textures on suspend
            for (_id, handle) in self.texture_handles.drain() {
                state.renderer.unregister_texture(handle);
            }
        }

        self.render_state = RenderState::Suspended;
    }

    fn set_size(&mut self, width: u32, height: u32) {
        if let RenderState::Active(state) = &mut self.render_state {
            state.render_surface.resize(width, height);
        };
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        let render_surface = &mut state.render_surface;

        debug_timer!(timer, feature = "log_frame_times");

        // Regenerate the vello scene
        draw_fn(&mut VelloScenePainter {
            inner: &mut self.scene,
            renderer: Some(&mut state.renderer),
            device_handle: Some(&render_surface.device_handle),
            texture_handles: Some(&mut self.texture_handles),
        });
        timer.record_time("cmd");

        match render_surface.ensure_current_surface_texture() {
            Ok(_) => {}
            Err(SurfaceError::Timeout | SurfaceError::Lost | SurfaceError::Outdated) => {
                render_surface.clear_surface_texture();
                return;
            }
            Err(SurfaceError::OutOfMemory) => panic!("Out of memory"),
            Err(SurfaceError::Other) => panic!("Unknown error getting surface"),
        };

        let texture_view = render_surface
            .target_texture_view()
            .expect("handled errorss from ensure_current_surface_texture above");
        state
            .renderer
            .render_to_texture(
                render_surface.device(),
                render_surface.queue(),
                &self.scene,
                &texture_view,
                &RenderParams {
                    base_color: self.config.base_color,
                    width: render_surface.config.width,
                    height: render_surface.config.height,
                    antialiasing_method: self.config.antialiasing_method,
                },
            )
            .expect("failed to render to texture");
        timer.record_time("render");

        drop(texture_view);

        render_surface
            .maybe_blit_and_present()
            .expect("handled errorss from ensure_current_surface_texture above");
        timer.record_time("present");

        render_surface
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        timer.record_time("wait");
        timer.print_times("vello: ");

        // Empty the Vello scene (memory optimisation)
        self.scene.reset();
    }
}
