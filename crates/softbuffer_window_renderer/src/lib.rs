//! An AnyRender WindowRenderer for rendering pixel buffers using the softbuffer crate

#![cfg_attr(docsrs, feature(doc_cfg))]

use anyrender::{ImageRenderer, PaintScene, RenderContext, WindowHandle, WindowRenderer};
use debug_timer::debug_timer;
use kurbo::{Affine, Rect};
use softbuffer::{Context, Surface};
use std::{num::NonZero, sync::Arc};

/// Configuration options for the Softbuffer renderer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SoftbufferRendererOptions {
    /// Background color used to clear the frame.
    pub base_color: peniko::Color,
}

impl Default for SoftbufferRendererOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftbufferRendererOptions {
    pub const fn new() -> Self {
        Self {
            base_color: peniko::Color::WHITE,
        }
    }

    pub const fn base_color(self, base_color: peniko::Color) -> Self {
        Self { base_color, ..self }
    }
}

impl From<anyrender::RendererConfig> for SoftbufferRendererOptions {
    fn from(config: anyrender::RendererConfig) -> Self {
        let mut options = Self::default();
        if let Some(color) = config.base_color {
            options.base_color = color;
        }
        // `composite_alpha_mode` is intentionally ignored: softbuffer does not
        // support transparency, so any requested alpha mode degrades to opaque.
        options
    }
}

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
    buffer: Vec<u8>,
    config: SoftbufferRendererOptions,
    width: u32,
    height: u32,
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
            buffer: Vec::new(),
            config: SoftbufferRendererOptions::default(),
            width: 0,
            height: 0,
        }
    }

    pub fn with_options(config: impl Into<SoftbufferRendererOptions>) -> Self {
        Self {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer: Renderer::new(0, 0),
            buffer: Vec::new(),
            config: config.into(),
            width: 0,
            height: 0,
        }
    }

    pub fn try_with_options<E: std::error::Error>(
        config: impl TryInto<SoftbufferRendererOptions, Error = E>,
    ) -> Result<Self, E> {
        Ok(Self {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer: Renderer::new(0, 0),
            buffer: Vec::new(),
            config: config.try_into()?,
            width: 0,
            height: 0,
        })
    }

    pub fn with_options_and_renderer<R: ImageRenderer>(
        renderer: R,
        config: impl TryInto<SoftbufferRendererOptions, Error = impl std::error::Error>,
    ) -> SoftbufferWindowRenderer<R> {
        SoftbufferWindowRenderer {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer,
            buffer: Vec::new(),
            config: config
                .try_into()
                .expect("Invalid Softbuffer renderer configuration"),
            width: 0,
            height: 0,
        }
    }
}

impl<Renderer: ImageRenderer> RenderContext for SoftbufferWindowRenderer<Renderer> {
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
impl<Renderer: ImageRenderer> WindowRenderer for SoftbufferWindowRenderer<Renderer> {
    type ScenePainter<'a>
        = Renderer::ScenePainter<'a>
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
        let context = Context::new(window_handle.clone()).unwrap();
        let surface = Surface::new(&context, window_handle.clone()).unwrap();
        self.render_state = RenderState::Active(ActiveRenderState {
            _context: context,
            surface,
        });
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
        self.width = physical_width;
        self.height = physical_height;
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
        timer.record_time("buffer_mut");

        // Paint
        let base_color = self.config.base_color;
        let width = self.width as f64;
        let height = self.height as f64;

        let wrapped_draw_fn = |painter: &mut Renderer::ScenePainter<'_>| {
            painter.fill(
                peniko::Fill::NonZero,
                Affine::IDENTITY,
                base_color,
                None,
                &Rect::new(0.0, 0.0, width, height),
            );
            draw_fn(painter);
        };

        self.renderer
            .render_to_vec(wrapped_draw_fn, &mut self.buffer);
        timer.record_time("render");

        let out = surface_buffer.as_mut();

        let (chunks, remainder) = self.buffer.as_chunks::<4>();
        assert_eq!(chunks.len(), out.len());
        assert_eq!(remainder.len(), 0);

        for (&src, dest) in chunks.iter().zip(out.iter_mut()) {
            let [r, g, b, _a] = src;
            let r = r as u32;
            let g = g as u32;
            let b = b as u32;

            *dest = (r << 16) | (g << 8) | b;
        }
        timer.record_time("swizel");

        surface_buffer.present().unwrap();
        timer.record_time("present");
        timer.print_times("softbuffer: ");

        // Reset the renderer ready for the next render
        self.renderer.reset();
    }
}
