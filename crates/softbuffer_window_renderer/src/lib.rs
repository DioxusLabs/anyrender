//! An AnyRender WindowRenderer for rendering pixel buffers using the softbuffer crate

#![cfg_attr(docsrs, feature(doc_cfg))]

use anyrender::{ImageRenderer, WindowHandle, WindowRenderer};
use debug_timer::debug_timer;
use softbuffer::{Context, Surface};
use std::{num::NonZero, sync::Arc};

// Simple struct to hold the state of the renderer
pub struct ActiveRenderState {
    _context: Context<Arc<dyn WindowHandle>>,
    surface: Surface<Arc<dyn WindowHandle>, Arc<dyn WindowHandle>>,
}

#[allow(clippy::large_enum_variant)]
pub enum RenderState {
    Active(ActiveRenderState),
    Suspended,
}

pub struct SoftbufferWindowRenderer<Renderer: ImageRenderer> {
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,
    renderer: Renderer,
}

impl<Renderer: ImageRenderer> SoftbufferWindowRenderer<Renderer> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_renderer(Renderer::new(0, 0))
    }

    pub fn with_renderer<R: ImageRenderer>(renderer: R) -> SoftbufferWindowRenderer<R> {
        SoftbufferWindowRenderer {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer,
        }
    }
}

impl<Renderer: ImageRenderer> WindowRenderer for SoftbufferWindowRenderer<Renderer> {
    type ScenePainter<'a>
        = Renderer::ScenePainter<'a>
    where
        Self: 'a;

    fn is_active(&self) -> bool {
        matches!(self.render_state, RenderState::Active(_))
    }

    fn resume(&mut self, window_handle: Arc<dyn WindowHandle>, width: u32, height: u32) {
        let context = Context::new(window_handle.clone()).unwrap();
        let surface = Surface::new(&context, window_handle.clone()).unwrap();
        self.render_state = RenderState::Active(ActiveRenderState {
            _context: context,
            surface,
        });
        self.window_handle = Some(window_handle);

        self.set_size(width, height);
    }

    fn suspend(&mut self) {
        self.render_state = RenderState::Suspended;
    }

    fn set_size(&mut self, physical_width: u32, physical_height: u32) {
        if let RenderState::Active(state) = &mut self.render_state {
            state
                .surface
                .resize(
                    NonZero::new(physical_width.max(1)).unwrap(),
                    NonZero::new(physical_height.max(1)).unwrap(),
                )
                .unwrap();
            self.renderer.resize(physical_width, physical_height);
        };
    }

    fn render<F: FnOnce(&mut Renderer::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        debug_timer!(timer, feature = "log_frame_times");

        let Ok(mut surface_buffer) = state.surface.buffer_mut() else {
            return;
        };
        let out = surface_buffer.as_mut();
        timer.record_time("buffer_mut");

        // Paint
        let (prefix, out_u8, suffix) = unsafe { out.align_to_mut::<u8>() };
        assert_eq!(prefix.len(), 0);
        assert_eq!(suffix.len(), 0);

        self.renderer.render(draw_fn, out_u8);
        timer.record_time("render");

        // Swizel
        for pixel in out.iter_mut() {
            if *pixel >> 24 == 0 {
                *pixel = u32::MAX >> 8;
            } else {
                *pixel = pixel.swap_bytes() >> 8;
            }
        }
        timer.record_time("swizel");

        surface_buffer.present().unwrap();
        timer.record_time("present");
        timer.print_times("softbuffer: ");

        // Reset the renderer ready for the next render
        self.renderer.reset();
    }
}
