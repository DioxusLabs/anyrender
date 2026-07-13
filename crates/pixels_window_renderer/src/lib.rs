//! An AnyRender WindowRenderer for rendering pixel buffers using the pixels crate

#![cfg_attr(docsrs, feature(doc_cfg))]

use anyrender::{ImageRenderer, RenderContext, WindowHandle, WindowRenderer};
use debug_timer::debug_timer;
use pixels::{
    Pixels, PixelsBuilder, SurfaceTexture,
    wgpu::{Color, CompositeAlphaMode},
};
use std::sync::Arc;

// Simple struct to hold the state of the renderer
pub struct ActiveRenderState {
    // surface: SurfaceTexture<Arc<dyn WindowHandle>>,
    pixels: Pixels<'static>,
}

#[allow(clippy::large_enum_variant)]
pub enum RenderState {
    Active(ActiveRenderState),
    Suspended,
}

/// Configuration options for the Pixels renderer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PixelsRendererOptions {
    /// Background color used to clear the frame.
    pub base_color: Color,
    /// Alpha mode used when compositing the window surface.
    pub composite_alpha_mode: CompositeAlphaMode,
}

impl Default for PixelsRendererOptions {
    fn default() -> Self {
        Self {
            base_color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            composite_alpha_mode: CompositeAlphaMode::Auto,
        }
    }
}

pub struct PixelsWindowRenderer<Renderer: ImageRenderer> {
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,
    renderer: Renderer,
    config: PixelsRendererOptions,
}

impl<Renderer: ImageRenderer> PixelsWindowRenderer<Renderer> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_renderer(Renderer::new(0, 0))
    }

    pub fn with_renderer<R: ImageRenderer>(renderer: R) -> PixelsWindowRenderer<R> {
        PixelsWindowRenderer {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer,
            config: PixelsRendererOptions::default(),
        }
    }

    pub fn with_options(config: PixelsRendererOptions) -> Self {
        Self {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer: Renderer::new(0, 0),
            config,
        }
    }

    pub fn with_options_and_renderer<R: ImageRenderer>(
        renderer: R,
        config: PixelsRendererOptions,
    ) -> PixelsWindowRenderer<R> {
        PixelsWindowRenderer {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer,
            config,
        }
    }
}

impl<Renderer: ImageRenderer> RenderContext for PixelsWindowRenderer<Renderer> {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn std::any::Any>,
    ) -> Result<anyrender::ResourceId, anyrender::RegisterResourceError> {
        self.renderer.try_register_custom_resource(resource)
    }

    fn unregister_resource(&mut self, resource_id: anyrender::ResourceId) {
        self.renderer.unregister_resource(resource_id);
    }
}
impl<Renderer: ImageRenderer> WindowRenderer for PixelsWindowRenderer<Renderer> {
    type ScenePainter<'a>
        = <Renderer as ImageRenderer>::ScenePainter<'a>
    where
        Renderer: 'a;

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
        let surface = SurfaceTexture::new(width, height, window_handle.clone());
        let pixels = PixelsBuilder::new(width, height, surface)
            .enable_vsync(true)
            .alpha_mode(self.config.composite_alpha_mode)
            .clear_color(self.config.base_color)
            .build()
            .unwrap();
        self.render_state = RenderState::Active(ActiveRenderState { pixels });
        self.window_handle = Some(window_handle);

        self.set_size(width, height);
        on_ready();
    }

    fn complete_resume(&mut self) -> bool {
        true
    }

    fn suspend(&mut self) {
        self.render_state = RenderState::Suspended;
    }

    fn set_size(&mut self, physical_width: u32, physical_height: u32) {
        if let RenderState::Active(state) = &mut self.render_state {
            state
                .pixels
                .resize_buffer(physical_width, physical_height)
                .unwrap();
            state
                .pixels
                .resize_surface(physical_width, physical_height)
                .unwrap();
            self.renderer.resize(physical_width, physical_height);
        };
    }

    fn render<F: FnOnce(&mut Renderer::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        debug_timer!(timer, feature = "log_frame_times");

        // Paint
        self.renderer.render(draw_fn, state.pixels.frame_mut());
        timer.record_time("render");
        state.pixels.render().unwrap();
        timer.record_time("present");
        timer.print_times("pixels: ");

        // Reset the renderer ready for the next render
        self.renderer.reset();
    }
}
