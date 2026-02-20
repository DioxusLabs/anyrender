use anyrender::{ImageResource, RenderContext, ResourceId, WindowHandle, WindowRenderer};
use debug_timer::debug_timer;
use peniko::{Color, ImageData};
use rustc_hash::FxHashMap;
use std::sync::{
    Arc,
    atomic::{self, AtomicU64},
};
use vello::{
    AaConfig, AaSupport, RenderParams, Renderer as VelloRenderer, RendererOptions,
    Scene as VelloScene,
};
use wgpu::{Features, Limits, PresentMode, TextureFormat, TextureUsages};
use wgpu_context::{
    DeviceHandle, SurfaceRenderer, SurfaceRendererConfiguration, TextureConfiguration, WGPUContext,
};

use crate::{CustomPaintSource, DEFAULT_THREADS, VelloRenderContext, VelloScenePainter};

static PAINT_SOURCE_ID: AtomicU64 = AtomicU64::new(0);

// Simple struct to hold the state of the renderer
struct ActiveRenderState {
    renderer: VelloRenderer,
    render_surface: SurfaceRenderer<'static>,
}

#[allow(clippy::large_enum_variant)]
enum RenderState {
    Active(ActiveRenderState),
    Suspended,
}

impl RenderState {
    fn current_device_handle(&self) -> Option<&DeviceHandle> {
        let RenderState::Active(state) = self else {
            return None;
        };
        Some(&state.render_surface.device_handle)
    }
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
    ctx: VelloRenderContext,
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,

    // Vello
    wgpu_context: WGPUContext,
    scene: VelloScene,
    config: VelloRendererOptions,

    custom_paint_sources: FxHashMap<u64, Box<dyn CustomPaintSource>>,
}
impl VelloWindowRenderer {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_options(VelloRendererOptions::default())
    }

    pub fn with_options(config: VelloRendererOptions) -> Self {
        let features = config.features.unwrap_or_default()
            | Features::CLEAR_TEXTURE
            | Features::PIPELINE_CACHE;
        Self {
            ctx: VelloRenderContext::new(),
            wgpu_context: WGPUContext::with_features_and_limits(
                Some(features),
                config.limits.clone(),
            ),
            config,
            render_state: RenderState::Suspended,
            window_handle: None,
            scene: VelloScene::new(),
            custom_paint_sources: FxHashMap::default(),
        }
    }

    pub fn current_device_handle(&self) -> Option<&DeviceHandle> {
        self.render_state.current_device_handle()
    }

    pub fn register_custom_paint_source(&mut self, mut source: Box<dyn CustomPaintSource>) -> u64 {
        if let Some(device_handle) = self.render_state.current_device_handle() {
            source.resume(device_handle);
        }
        let id = PAINT_SOURCE_ID.fetch_add(1, atomic::Ordering::SeqCst);
        self.custom_paint_sources.insert(id, source);

        id
    }

    pub fn unregister_custom_paint_source(&mut self, id: u64) {
        if let Some(mut source) = self.custom_paint_sources.remove(&id) {
            source.suspend();
            drop(source);
        }
    }
}

impl RenderContext for VelloWindowRenderer {
    fn register_image(&mut self, image: ImageData) -> ImageResource {
        self.ctx.register_image(image)
    }

    fn unregister_resource(&mut self, id: ResourceId) {
        self.ctx.unregister_resource(id)
    }
}

impl WindowRenderer for VelloWindowRenderer {
    type ScenePainter<'a>
        = VelloScenePainter<'a, 'a>
    where
        Self: 'a;

    fn is_active(&self) -> bool {
        matches!(self.render_state, RenderState::Active(_))
    }

    fn resume(&mut self, window_handle: Arc<dyn WindowHandle>, width: u32, height: u32) {
        // Create wgpu_context::SurfaceRenderer
        let render_surface = pollster::block_on(self.wgpu_context.create_surface(
            window_handle.clone(),
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
        ))
        .expect("Error creating surface");

        // Create vello::Renderer
        let renderer = VelloRenderer::new(
            render_surface.device(),
            RendererOptions {
                antialiasing_support: AaSupport::all(),
                use_cpu: false,
                num_init_threads: DEFAULT_THREADS,
                // TODO: add pipeline cache
                pipeline_cache: None,
            },
        )
        .unwrap();

        // Resume custom paint sources
        let device_handle = &render_surface.device_handle;
        for source in self.custom_paint_sources.values_mut() {
            source.resume(device_handle)
        }

        // Set state to Active
        self.window_handle = Some(window_handle);
        self.render_state = RenderState::Active(ActiveRenderState {
            renderer,
            render_surface,
        });
    }

    fn suspend(&mut self) {
        // Suspend custom paint sources
        for source in self.custom_paint_sources.values_mut() {
            source.suspend()
        }

        // Set state to Suspended
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

        debug_timer!(timer, feature = "log_frame_times");

        // Regenerate the vello scene
        draw_fn(&mut VelloScenePainter {
            ctx: &self.ctx,
            inner: &mut self.scene,
            renderer: Some(&mut state.renderer),
            custom_paint_sources: Some(&mut self.custom_paint_sources),
        });
        timer.record_time("cmd");

        let render_surface = &mut state.render_surface;
        let texture_view = render_surface.target_texture_view();
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

        render_surface.maybe_blit_and_present();
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
