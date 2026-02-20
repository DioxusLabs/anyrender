use anyrender::{ImageResource, RenderContext, ResourceId, WindowHandle, WindowRenderer};
use debug_timer::debug_timer;
use peniko::ImageData;
use std::sync::Arc;
use vello_hybrid::{
    RenderSettings, RenderSize, RenderTargetConfig, Renderer as VelloHybridRenderer,
    Scene as VelloHybridScene,
};
use wgpu::{CommandEncoderDescriptor, Features, Limits, PresentMode, TextureFormat};
use wgpu_context::{DeviceHandle, SurfaceRenderer, SurfaceRendererConfiguration, WGPUContext};

use crate::{VelloHybridScenePainter, scene::VelloHybridRenderContext};

// Simple struct to hold the state of the renderer
struct ActiveRenderState {
    renderer: VelloHybridRenderer,
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

#[derive(Clone, Default)]
pub struct VelloHybridRendererOptions {
    pub features: Option<Features>,
    pub limits: Option<Limits>,
    pub render_settings: RenderSettings,
}

pub struct VelloHybridWindowRenderer {
    ctx: VelloHybridRenderContext,
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,

    // Vello
    wgpu_context: WGPUContext,
    scene: VelloHybridScene,
    config: VelloHybridRendererOptions,
}
impl VelloHybridWindowRenderer {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_options(VelloHybridRendererOptions::default())
    }

    pub fn with_options(config: VelloHybridRendererOptions) -> Self {
        let features = config.features.unwrap_or_default()
            | Features::CLEAR_TEXTURE
            | Features::PIPELINE_CACHE;
        let render_settings = config.render_settings;
        Self {
            ctx: VelloHybridRenderContext::new(),
            wgpu_context: WGPUContext::with_features_and_limits(
                Some(features),
                config.limits.clone(),
            ),
            config,
            render_state: RenderState::Suspended,
            window_handle: None,
            scene: VelloHybridScene::new_with(0, 0, render_settings),
        }
    }

    pub fn current_device_handle(&self) -> Option<&DeviceHandle> {
        self.render_state.current_device_handle()
    }
}

// TODO: Make configurable?
#[cfg(target_os = "android")]
const DEFAULT_TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
#[cfg(not(target_os = "android"))]
const DEFAULT_TEXTURE_FORMAT: TextureFormat = TextureFormat::Bgra8Unorm;

impl RenderContext for VelloHybridWindowRenderer {
    fn register_image(&mut self, image: ImageData) -> ImageResource {
        self.ctx.register_image(image)
    }

    fn unregister_resource(&mut self, id: ResourceId) {
        self.ctx.unregister_resource(id)
    }
}

impl WindowRenderer for VelloHybridWindowRenderer {
    type ScenePainter<'a>
        = VelloHybridScenePainter<'a>
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
                formats: vec![DEFAULT_TEXTURE_FORMAT],
                width,
                height,
                present_mode: PresentMode::AutoVsync,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
            },
            None,
        ))
        .expect("Error creating surface");

        // Create vello::Renderer
        let renderer = VelloHybridRenderer::new(
            render_surface.device(),
            &RenderTargetConfig {
                format: DEFAULT_TEXTURE_FORMAT,
                width,
                height,
            },
        );

        // Create a Scene with the correct dimensions
        self.scene =
            VelloHybridScene::new_with(width as u16, height as u16, self.config.render_settings);

        // Set state to Active
        self.window_handle = Some(window_handle);
        self.render_state = RenderState::Active(ActiveRenderState {
            renderer,
            render_surface,
        });
    }

    fn suspend(&mut self) {
        // Set state to Suspended
        self.render_state = RenderState::Suspended;
    }

    fn set_size(&mut self, width: u32, height: u32) {
        if width as u16 != self.scene.width() || height as u16 != self.scene.height() {
            self.scene = VelloHybridScene::new_with(
                width as u16,
                height as u16,
                self.config.render_settings,
            );
            if let RenderState::Active(state) = &mut self.render_state {
                state.render_surface.resize(width, height);
            };
        }
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        debug_timer!(timer, feature = "log_frame_times");

        // Regenerate the vello scene.
        {
            let device = state.render_surface.device();
            let queue = state.render_surface.queue();
            draw_fn(&mut VelloHybridScenePainter {
                ctx: &mut self.ctx,
                renderer: &mut state.renderer,
                device,
                queue,
                scene: &mut self.scene,
                layer_stack: Vec::new(),
            });
        }
        timer.record_time("cmd");

        let render_surface = &mut state.render_surface;

        let mut encoder =
            render_surface
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Render scene"),
                });

        let texture_view = render_surface.target_texture_view();

        state
            .renderer
            .render(
                &self.scene,
                render_surface.device(),
                render_surface.queue(),
                &mut encoder,
                &RenderSize {
                    width: render_surface.config.width,
                    height: render_surface.config.height,
                },
                &texture_view,
            )
            .expect("failed to render to texture");
        render_surface.queue().submit([encoder.finish()]);
        timer.record_time("render");

        drop(texture_view);

        render_surface.maybe_blit_and_present();
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
