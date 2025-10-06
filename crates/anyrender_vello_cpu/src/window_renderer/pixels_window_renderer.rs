use anyrender::{ImageRenderer, WindowHandle, WindowRenderer};
use debug_timer::debug_timer;
use pixels::{Pixels, SurfaceTexture, wgpu::Color};
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

pub struct PixelsWindowRenderer<Renderer: ImageRenderer> {
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,
    renderer: Renderer,
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
        }
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

    fn resume(&mut self, window_handle: Arc<dyn WindowHandle>, width: u32, height: u32) {
        let surface = SurfaceTexture::new(width, height, window_handle.clone());
        let mut pixels = Pixels::new(width, height, surface).unwrap();
        pixels.enable_vsync(true);
        pixels.clear_color(Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        });
        self.render_state = RenderState::Active(ActiveRenderState { pixels });
        self.window_handle = Some(window_handle);

        self.set_size(width, height);
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
        timer.print_times("Frame time: ");

        // Reset the renderer ready for the next render
        self.renderer.reset();
    }
}
