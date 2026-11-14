use anyrender::{
    PaintScene, RegisterResourceErrorKind, RenderContext, ResourceId, WindowHandle, WindowRenderer,
    WindowRendererContext,
};
use debug_timer::debug_timer;
use futures_channel::oneshot;
use peniko::Color;
use rustc_hash::FxHashSet;
use std::future::Future;
use std::sync::{Arc, Mutex};
use vello_gpu::{
    RenderOptions, RenderSize, RenderTarget, RenderTargetConfig, Renderer as VelloHybridRenderer,
    Resources, Scene as VelloHybridScene,
};
use wgpu::{
    CommandEncoderDescriptor, CompositeAlphaMode, Features, PresentMode, Texture, TextureFormat,
    TextureView, TextureViewDescriptor,
};
use wgpu_context::{
    DeviceHandle, SharedWGPUContext, SurfaceRenderer, SurfaceRendererConfiguration, WGPUContext,
};

use super::scene::{ExternalTextures, ImageManager, VelloHybridScenePainter};
use crate::common::{ImageCache, VelloHybridRendererOptions};

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
    device_state: Arc<Mutex<DeviceState>>,
    /// Custom resources this window registered in the shared `texture_bindings`
    custom_resources: FxHashSet<ResourceId>,
    depth_texture_view: TextureView,
    render_surface: SurfaceRenderer<'static>,
}

impl Drop for ActiveRenderState {
    fn drop(&mut self) {
        if self.custom_resources.is_empty() {
            return;
        }
        if let Ok(mut device_state) = self.device_state.lock() {
            for id in self.custom_resources.drain() {
                device_state.texture_bindings.remove(id);
            }
        }
    }
}

/// Renderer state for one device, shared by every window that renders with that device.
struct DeviceState {
    renderer: VelloHybridRenderer,
    /// Glyph atlases
    resources: Resources,
    /// Uploaded images, keyed by image data
    cached_images: ImageCache,
    /// Textures for uploaded images and registered custom resources
    texture_bindings: ExternalTextures,
}

type DeviceStates = Vec<(wgpu::Device, Arc<Mutex<DeviceState>>)>;

/// Creates [`VelloHybridWindowRenderer`]s that share a wgpu instance, its device pool and,
/// per device, one renderer with its glyph atlases and image textures, instead of each window
/// creating its own.
///
/// Create one per application and call [`create_renderer`](WindowRendererContext::create_renderer)
/// for each window. Cloning is cheap: clones refer to the same shared state.
///
/// The renderer for a device is created with the `render_settings` of the first window that
/// uses that device.
///
/// On wasm, device creation is asynchronous, so windows resumed in the same tick may still
/// create a device each.
#[derive(Clone)]
pub struct VelloHybridContext {
    wgpu_context: SharedWGPUContext,
    devices: Arc<Mutex<DeviceStates>>,
    options: VelloHybridRendererOptions,
}

impl VelloHybridContext {
    /// Creates a context with a new wgpu instance, using the features and limits from `options`.
    pub fn new(options: impl Into<VelloHybridRendererOptions>) -> Self {
        let options = options.into();
        let wgpu_context = build_wgpu_context(&options);
        Self::with_wgpu_context(options, wgpu_context)
    }

    /// Creates a context that gets its devices from `wgpu_context`. Pass the same
    /// [`SharedWGPUContext`] to several contexts (e.g. a [`VelloHybridContext`] and a Vello
    /// context) to share devices between them. The `features` and `limits` in `options` are not
    /// used: devices are created with those of `wgpu_context`.
    // wgpu types are not `Send` on wasm, which is single-threaded
    #[cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
    pub fn with_wgpu_context(
        options: impl Into<VelloHybridRendererOptions>,
        wgpu_context: impl Into<SharedWGPUContext>,
    ) -> Self {
        Self {
            wgpu_context: wgpu_context.into(),
            devices: Arc::new(Mutex::new(Vec::new())),
            options: options.into(),
        }
    }

    /// Creates a context that renders with a device the application created: pass its
    /// `Instance`, `Adapter`, `Device` and `Queue` as a [`DeviceHandle`]. A window whose
    /// surface that device cannot present to gets a new device from the same instance.
    pub fn from_device_handle(
        options: impl Into<VelloHybridRendererOptions>,
        device_handle: DeviceHandle,
    ) -> Self {
        Self::with_wgpu_context(options, device_handle)
    }

    pub fn wgpu_context(&self) -> &SharedWGPUContext {
        &self.wgpu_context
    }

    pub fn options(&self) -> &VelloHybridRendererOptions {
        &self.options
    }

    /// Creates a renderer for a window with its own options (e.g. `base_color`), instead of the
    /// context's.
    pub fn create_renderer_with_options(
        &self,
        options: impl Into<VelloHybridRendererOptions>,
    ) -> VelloHybridWindowRenderer {
        let config = options.into();
        let level = config.render_settings.level();
        VelloHybridWindowRenderer {
            render_state: RenderState::Suspended,
            config,
            context: self.clone(),
            window_handle: None,
            scene: VelloHybridScene::new_with(0, 0, level),
        }
    }
}

impl WindowRendererContext for VelloHybridContext {
    type Renderer = VelloHybridWindowRenderer;

    fn create_renderer(&self) -> VelloHybridWindowRenderer {
        self.create_renderer_with_options(self.options.clone())
    }
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

pub struct VelloHybridWindowRenderer {
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,

    context: VelloHybridContext,
    scene: VelloHybridScene,
    config: VelloHybridRendererOptions,
}
impl VelloHybridWindowRenderer {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_options(VelloHybridRendererOptions::default())
    }

    /// Creates a renderer with its own [`VelloHybridContext`]. To share a device and renderer
    /// between windows, create them from one [`VelloHybridContext`] instead.
    pub fn with_options(config: impl Into<VelloHybridRendererOptions>) -> Self {
        VelloHybridContext::new(config).create_renderer()
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
                state.register_custom_resource(id, texture_view);
                Ok(id)
            }
            Err(resource) => {
                // Else try to downcast as TextureView
                if let Ok(texture_view) = resource.downcast::<TextureView>() {
                    let id = ResourceId::new();
                    state.register_custom_resource(id, *texture_view);
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
        if state.custom_resources.remove(&resource_id) {
            state
                .device_state
                .lock()
                .unwrap()
                .texture_bindings
                .remove(resource_id);
        }
    }
}

impl ActiveRenderState {
    fn register_custom_resource(&mut self, id: ResourceId, texture_view: TextureView) {
        self.custom_resources.insert(id);
        self.device_state
            .lock()
            .unwrap()
            .texture_bindings
            .insert(id, texture_view);
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

    // wgpu types are not `Send` on wasm, which is single-threaded
    #[cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
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
        let render_settings = self.config.render_settings.clone();
        self.scene =
            VelloHybridScene::new_with(width as u16, height as u16, render_settings.level());

        let context = self.context.clone();
        let mut wgpu_context = context.wgpu_context.lock();
        let surface = wgpu_context
            .create_surface(window_handle)
            .expect("Error creating surface");
        let instance = wgpu_context.instance.clone();
        let extra_features = wgpu_context.extra_features();
        let override_limits = wgpu_context.override_limits();
        let mut composite_alpha_mode = match self.config.composite_alpha_mode {
            anyrender::CompositeAlphaMode::Auto => CompositeAlphaMode::Auto,
            anyrender::CompositeAlphaMode::Opaque => CompositeAlphaMode::Opaque,
            anyrender::CompositeAlphaMode::Transparent => {
                #[cfg(target_vendor = "apple")]
                {
                    // wgpu is lying in apple's case it uses PreMultiplied in reality
                    // (do not modify shaders for PostMultiplied)
                    CompositeAlphaMode::PostMultiplied
                }
                #[cfg(not(target_vendor = "apple"))]
                {
                    CompositeAlphaMode::PreMultiplied
                }
            }
        };
        let desired_maximum_frame_latency = self.config.desired_maximum_frame_latency;
        let existing_device_handle = wgpu_context.find_compatible_device_handle(Some(&surface));
        drop(wgpu_context);

        spawn_init(async move {
            let device_handle = match existing_device_handle {
                Some(device_handle) => device_handle,
                None => {
                    let device_handle = DeviceHandle::new_from_compatible_surface(
                        instance,
                        Some(&surface),
                        extra_features,
                        override_limits,
                    )
                    .await
                    .expect("Error creating DeviceHandle");
                    context
                        .wgpu_context
                        .lock()
                        .add_device_handle(device_handle.clone());
                    device_handle
                }
            };

            let adapter = &device_handle.adapter;
            let caps = surface.get_capabilities(adapter);
            let mut alpha_modes = caps.alpha_modes;

            if !alpha_modes.contains(&composite_alpha_mode) {
                alpha_modes.sort_unstable_by(
                    |first: &CompositeAlphaMode, second: &CompositeAlphaMode| {
                        let first_num = match *first {
                            CompositeAlphaMode::PreMultiplied => 0,
                            CompositeAlphaMode::PostMultiplied => 1,
                            CompositeAlphaMode::Opaque
                            | CompositeAlphaMode::Inherit
                            | CompositeAlphaMode::Auto => 2,
                        };
                        let second_num = match *second {
                            CompositeAlphaMode::PreMultiplied => 0,
                            CompositeAlphaMode::PostMultiplied => 1,
                            CompositeAlphaMode::Opaque
                            | CompositeAlphaMode::Inherit
                            | CompositeAlphaMode::Auto => 2,
                        };
                        first_num.cmp(&second_num)
                    },
                );
                composite_alpha_mode = alpha_modes
                    .first()
                    .copied()
                    .expect("Surface didn't report any alpha modes");
            }

            // Vello Hybrid emits premultiplied alpha and renders directly into
            // its target. That matches a `PreMultiplied` surface (and Opaque
            // ignores alpha), so those render straight to the surface. Only
            // `PostMultiplied` (straight alpha) needs conversion, which routes
            // through an intermediate texture (using the renderer's target
            // format) that is un-premultiplied while blitting to the surface.
            #[cfg(not(target_vendor = "apple"))]
            let intermediate_texture = (composite_alpha_mode == CompositeAlphaMode::PostMultiplied)
                .then(|| {
                    use wgpu::TextureUsages;
                    use wgpu_context::{AlphaConversion, TextureConfiguration};
                    TextureConfiguration {
                        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                        format: DEFAULT_TEXTURE_FORMAT,
                        alpha_conversion: Some(AlphaConversion::Unpremultiply),
                    }
                });

            // Apple is almost guaranteed to be premultiplied
            // TODO: Remove below once gfx-rs/wgpu#9896 gets fixed
            #[cfg(target_vendor = "apple")]
            let intermediate_texture = None;

            let render_surface = SurfaceRenderer::new(
                surface,
                SurfaceRendererConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    formats: vec![DEFAULT_TEXTURE_FORMAT],
                    width,
                    height,
                    present_mode: PresentMode::AutoVsync,
                    desired_maximum_frame_latency,
                    alpha_mode: composite_alpha_mode,
                    view_formats: vec![],
                },
                intermediate_texture,
                device_handle,
            )
            .expect("Error creating SurfaceRenderer");

            let device = render_surface.device();
            let mut devices = context.devices.lock().unwrap();
            let existing_device_state = devices
                .iter()
                .find(|(d, _)| d == device)
                .map(|(_, state)| state.clone());
            let device_state = existing_device_state.unwrap_or_else(|| {
                let (renderer, resources) = VelloHybridRenderer::new_with(
                    device,
                    render_surface.queue(),
                    &RenderTargetConfig {
                        format: DEFAULT_TEXTURE_FORMAT,
                        width: width as u16,
                        height: height as u16,
                    },
                    render_settings,
                );
                let max = device
                    .limits()
                    .max_texture_dimension_2d
                    .min(u16::MAX as u32) as u16;
                let mut cached_images = ImageCache::default();
                cached_images.set_max_image_size((max, max));
                let state = Arc::new(Mutex::new(DeviceState {
                    renderer,
                    resources,
                    cached_images,
                    texture_bindings: ExternalTextures::default(),
                }));
                devices.push((device.clone(), state.clone()));
                state
            });
            drop(devices);
            let depth_texture_view = device_state
                .lock()
                .unwrap()
                .renderer
                .create_depth_texture_view(&RenderSize {
                    width: width as u16,
                    height: height as u16,
                });

            let _ = sender.send(InitOutput {
                active: ActiveRenderState {
                    device_state,
                    custom_resources: FxHashSet::default(),
                    depth_texture_view,
                    render_surface,
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
                self.config.render_settings.level(),
            );
            if let RenderState::Active(active) = &mut self.render_state {
                active.render_surface.resize(width, height);
                active.depth_texture_view = active
                    .device_state
                    .lock()
                    .unwrap()
                    .renderer
                    .create_depth_texture_view(&RenderSize {
                        width: width as u16,
                        height: height as u16,
                    });
            };
        }
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        let render_surface = &mut state.render_surface;
        let mut device_state = state.device_state.lock().unwrap();
        let DeviceState {
            renderer,
            resources,
            cached_images,
            texture_bindings,
        } = &mut *device_state;

        debug_timer!(timer, feature = "log_frame_times");

        let mut encoder =
            render_surface
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Render scene"),
                });

        let image_manager = ImageManager {
            resources,
            device: render_surface.device(),
            queue: render_surface.queue(),
            cache: cached_images,
            texture_bindings,
        };

        let mut scene_painter = VelloHybridScenePainter {
            scene: &mut self.scene,
            layer_stack: Vec::new(),
            image_manager,
            device_handle: &render_surface.device_handle,
            glyph_caching: self.config.glyph_caching,
        };
        if self.config.base_color != Color::TRANSPARENT {
            scene_painter.fill(
                peniko::Fill::NonZero,
                kurbo::Affine::IDENTITY,
                self.config.base_color,
                None,
                &kurbo::Rect::new(
                    0.,
                    0.,
                    render_surface.config.width as f64,
                    render_surface.config.height as f64,
                ),
            );
        }
        // Regenerate the vello scene
        draw_fn(&mut scene_painter);
        timer.record_time("cmd");

        let Ok(texture_view) = render_surface.target_texture_view() else {
            // Skip frame in case of error getting surface texture
            render_surface.clear_surface_texture();
            self.scene.reset();
            ImageManager::new(
                resources,
                render_surface.device(),
                render_surface.queue(),
                cached_images,
                texture_bindings,
            )
            .finish_frame();
            render_surface.queue().submit([encoder.finish()]);
            return;
        };

        renderer
            .render(
                &self.scene,
                resources,
                &mut encoder,
                RenderTarget {
                    view: &texture_view,
                    depth_view: Some(&state.depth_texture_view),
                },
                texture_bindings.bindings(),
                RenderOptions::default(),
            )
            .expect("failed to render to texture");
        ImageManager::new(
            resources,
            render_surface.device(),
            render_surface.queue(),
            cached_images,
            texture_bindings,
        )
        .finish_frame();
        render_surface.queue().submit([encoder.finish()]);
        timer.record_time("render");

        drop(texture_view);

        if render_surface.maybe_blit_and_present().is_err() {
            self.scene.reset();
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

#[cfg(not(target_arch = "wasm32"))]
const _: fn() = || {
    fn assert_send<T: Send>() {}
    assert_send::<VelloHybridWindowRenderer>();
    assert_send::<VelloHybridContext>();
};
