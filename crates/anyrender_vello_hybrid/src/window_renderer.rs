use anyrender::{
    RegisterResourceErrorKind, RenderContext, ResourceId, WindowHandle, WindowRenderer,
};
use debug_timer::debug_timer;
use futures_channel::oneshot;
use rustc_hash::FxHashMap;
use std::future::Future;
use std::sync::Arc;
use vello_common::{TextureId, paint::ImageId};
use vello_hybrid::{
    RenderSettings, RenderSize, RenderTargetConfig, Renderer as VelloHybridRenderer, Resources,
    Scene as VelloHybridScene, TextureBindings,
};
use wgpu::{
    CommandEncoderDescriptor, Features, Limits, PresentMode, Texture, TextureFormat, TextureView,
    TextureViewDescriptor,
};
use wgpu_context::{DeviceHandle, SurfaceRenderer, SurfaceRendererConfiguration, WGPUContext};

use crate::{VelloHybridScenePainter, scene::ImageManager};

/// Drive the wgpu init future. On wasm32 we spawn it onto the JS microtask
/// queue (blocking is not allowed). On native we drive it inline with
/// `pollster::block_on` — there's no ambient async runtime to spawn onto, and
/// `on_ready` then fires before `resume` returns.
#[cfg(target_arch = "wasm32")]
fn spawn_init<F: Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_init<F: Future<Output = ()>>(f: F) {
    pollster::block_on(f);
}

struct ActiveRenderState {
    renderer: VelloHybridRenderer,
    resources: Resources,
    texture_bindings: FxHashMap<ResourceId, TextureView>,
    render_surface: SurfaceRenderer<'static>,
}

/// Result of a successful asynchronous resume; both the active state and the
/// `WGPUContext` are returned so the renderer can reclaim the context.
struct InitOutput {
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

#[derive(Clone, Default)]
pub struct VelloHybridRendererOptions {
    pub features: Option<Features>,
    pub limits: Option<Limits>,
    pub render_settings: RenderSettings,
}

pub struct VelloHybridWindowRenderer {
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,

    wgpu_context: WGPUContext,
    scene: VelloHybridScene,
    config: VelloHybridRendererOptions,
    cached_images: FxHashMap<u64, ImageId>,
}
impl VelloHybridWindowRenderer {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_options(VelloHybridRendererOptions::default())
    }

    pub fn with_options(config: VelloHybridRendererOptions) -> Self {
        let render_settings = config.render_settings;
        let wgpu_context = build_wgpu_context(&config);
        Self {
            render_state: RenderState::Suspended,
            config,
            wgpu_context,
            window_handle: None,
            scene: VelloHybridScene::new_with(0, 0, render_settings),
            cached_images: FxHashMap::default(),
        }
    }

    pub fn current_device_handle(&self) -> Option<&DeviceHandle> {
        match &self.render_state {
            RenderState::Active(active) => Some(&active.render_surface.device_handle),
            _ => None,
        }
    }
}

fn build_wgpu_context(config: &VelloHybridRendererOptions) -> WGPUContext {
    let features =
        config.features.unwrap_or_default() | Features::CLEAR_TEXTURE | Features::PIPELINE_CACHE;
    WGPUContext::with_features_and_limits(Some(features), config.limits.clone())
}

// TODO: Make configurable?
#[cfg(target_os = "android")]
const DEFAULT_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
#[cfg(not(target_os = "android"))]
const DEFAULT_TEXTURE_FORMAT: TextureFormat = TextureFormat::Bgra8Unorm;

impl RenderContext for VelloHybridWindowRenderer {
    fn renderer_specific_context(&self) -> Option<Box<dyn std::any::Any>> {
        match &self.render_state {
            RenderState::Active(state) => {
                Some(Box::new(state.render_surface.device_handle.clone()) as _)
            }
            RenderState::Suspended => None,
            RenderState::Pending { .. } => None,
        }
    }

    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn std::any::Any>,
    ) -> Result<anyrender::ResourceId, anyrender::RegisterResourceError> {
        let RenderState::Active(state) = &mut self.render_state else {
            return Err(RegisterResourceErrorKind::Other.into());
        };

        // Try to downcast as Texture
        match resource.downcast::<Texture>() {
            Ok(texture) => {
                let id = ResourceId::new();
                let texture_view = texture.create_view(&TextureViewDescriptor::default());
                state.texture_bindings.insert(id, texture_view);
                Ok(id)
            }
            Err(resource) => {
                // Else try to downcast as TextureView
                if let Ok(texture_view) = resource.downcast::<TextureView>() {
                    let id = ResourceId::new();
                    state.texture_bindings.insert(id, *texture_view);
                    Ok(id)
                }
                // Else return error
                else {
                    Err(anyrender::RegisterResourceErrorKind::UnsupportedResourceKind.into())
                }
            }
        }
    }

    fn unregister_resource(&mut self, resource_id: ResourceId) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };
        state.texture_bindings.remove(&resource_id);
    }
}

impl WindowRenderer for VelloHybridWindowRenderer {
    type ScenePainter<'a>
        = VelloHybridScenePainter<'a>
    where
        Self: 'a;

    fn is_active(&self) -> bool {
        matches!(self.render_state, RenderState::Active { .. })
    }

    fn is_pending(&self) -> bool {
        matches!(self.render_state, RenderState::Pending { .. })
    }

    fn resume<F: FnOnce() + 'static>(
        &mut self,
        window_handle: Arc<dyn WindowHandle>,
        width: u32,
        height: u32,
        on_ready: F,
    ) {
        // Each `resume` must be preceded by `suspend` (or be the first call after
        // construction). Calling while `Pending` or `Active` is a state-machine bug
        // in the embedder: it would orphan the in-flight init's `WGPUContext` and
        // pay for a fresh adapter+device init on the fallback path below.
        if !matches!(self.render_state, RenderState::Suspended) {
            // #[cfg(feature = "tracing")]
            // tracing::warn!("WindowRenderer::resume called from non-Suspended state");
            return;
        }

        let (sender, receiver) = oneshot::channel();
        self.render_state = RenderState::Pending { receiver };
        self.window_handle = Some(window_handle.clone());

        // Reset the scene to the new dimensions before init kicks off, so callers that
        // query scene size (e.g. `set_size`) see consistent state.
        let render_settings = self.config.render_settings;
        self.scene = VelloHybridScene::new_with(width as u16, height as u16, render_settings);

        let surface = self
            .wgpu_context
            .create_surface(window_handle)
            .expect("Error creating surface");
        let instance = self.wgpu_context.instance.clone();
        let extra_features = self.wgpu_context.extra_features();
        let override_limits = self.wgpu_context.override_limits();
        let existing_device_handle = self
            .wgpu_context
            .find_compatible_device_handle(Some(&surface));

        spawn_init(async move {
            let device_handle = match existing_device_handle {
                Some(device_handle) => device_handle,
                None => DeviceHandle::new_from_compatible_surface(
                    instance,
                    Some(&surface),
                    extra_features,
                    override_limits,
                )
                .await
                .expect("Error creating DeviceHandle"),
            };

            let render_surface = SurfaceRenderer::new(
                surface,
                SurfaceRendererConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    formats: vec![DEFAULT_TEXTURE_FORMAT],
                    width,
                    height,
                    present_mode: PresentMode::AutoVsync,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: wgpu::CompositeAlphaMode::Auto,
                    view_formats: vec![],
                },
                None,
                device_handle,
            )
            .expect("Error creating SurfaceRenderer");

            let resources = Resources::new();
            let renderer = VelloHybridRenderer::new(
                render_surface.device(),
                &RenderTargetConfig {
                    format: DEFAULT_TEXTURE_FORMAT,
                    width,
                    height,
                },
            );

            let _ = sender.send(InitOutput {
                active: ActiveRenderState {
                    renderer,
                    resources,
                    render_surface,
                    texture_bindings: FxHashMap::default(),
                },
            });
            on_ready();
        });
    }

    fn complete_resume(&mut self) -> bool {
        match &mut self.render_state {
            RenderState::Active { .. } => true,
            RenderState::Suspended => false,
            RenderState::Pending { receiver } => match receiver.try_recv() {
                Ok(Some(InitOutput { active })) => {
                    let device_handle = active.render_surface.device_handle.clone();
                    self.wgpu_context.device_pool.push(device_handle);
                    self.render_state = RenderState::Active(active);
                    true
                }
                _ => false,
            },
        }
    }

    fn suspend(&mut self) {
        self.render_state = RenderState::Suspended;
    }

    fn set_size(&mut self, width: u32, height: u32) {
        if width as u16 != self.scene.width() || height as u16 != self.scene.height() {
            self.scene = VelloHybridScene::new_with(
                width as u16,
                height as u16,
                self.config.render_settings,
            );
            if let RenderState::Active(active) = &mut self.render_state {
                active.render_surface.resize(width, height);
            };
        }
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        let render_surface = &mut state.render_surface;

        debug_timer!(timer, feature = "log_frame_times");

        let mut encoder =
            render_surface
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Render scene"),
                });

        let image_manager = ImageManager {
            renderer: &mut state.renderer,
            resources: &mut state.resources,
            device: render_surface.device(),
            queue: render_surface.queue(),
            encoder: &mut encoder,
            cache: &mut self.cached_images,
        };

        // Regenerate the vello scene
        draw_fn(&mut VelloHybridScenePainter {
            scene: &mut self.scene,
            layer_stack: Vec::new(),
            image_manager,
            texture_bindings: &mut state.texture_bindings,
            device_handle: &render_surface.device_handle,
        });
        timer.record_time("cmd");

        let Ok(texture_view) = render_surface.target_texture_view() else {
            // Skip frame in case of error getting surface texture
            render_surface.clear_surface_texture();
            return;
        };

        // Construct Vello Hybrid TextureBindings
        let mut texture_bindings = TextureBindings::new();
        for (resource_id, texture_view) in state.texture_bindings.iter() {
            texture_bindings.insert(TextureId(resource_id.into_ffi()), texture_view.clone());
        }

        state
            .renderer
            .render(
                &self.scene,
                &mut state.resources,
                render_surface.device(),
                render_surface.queue(),
                &mut encoder,
                &RenderSize {
                    width: render_surface.config.width,
                    height: render_surface.config.height,
                },
                &texture_view,
                &texture_bindings,
            )
            .expect("failed to render to texture");
        render_surface.queue().submit([encoder.finish()]);
        timer.record_time("render");

        drop(texture_view);

        if render_surface.maybe_blit_and_present().is_err() {
            return;
        }
        timer.record_time("present");

        render_surface
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        timer.record_time("wait");
        timer.print_times("vello_hybrid: ");

        // Empty the Vello scene (memory optimisation)
        self.scene.reset();
    }
}
