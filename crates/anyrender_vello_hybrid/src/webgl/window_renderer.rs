//! A [`WindowRenderer`] that renders with [`vello_gpu::WebGlRenderer`] into a WebGL2 canvas.

use std::sync::Arc;

use anyrender::{PaintScene, RenderContext, WindowHandle, WindowRenderer};
use debug_timer::debug_timer;
use peniko::Color;
use raw_window_handle::RawWindowHandle;
use vello_gpu::{
    RenderSize, Resources, Scene, WebGlRenderOptions, WebGlRenderTarget, WebGlRenderer,
    WebGlRendererInit, WebGlRendererInitStatus, WebGlTextureBindings,
};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

use super::scene::{WebGlImageManager, WebGlScenePainter};
use crate::common::{ImageCache, VelloHybridRendererOptions};

struct ActiveRenderState {
    renderer: WebGlRenderer,
    resources: Resources,
    canvas: HtmlCanvasElement,
}

#[allow(clippy::large_enum_variant)]
enum RenderState {
    Suspended,
    Pending {
        init: WebGlRendererInit,
        resources: Resources,
        canvas: HtmlCanvasElement,
    },
    Active(ActiveRenderState),
}

/// A [`WindowRenderer`] that draws to the window's `<canvas>` using Vello GPU's WebGL2 backend.
///
/// Unlike [`VelloHybridWindowRenderer`](crate::VelloHybridWindowRenderer), this does not need
/// WebGPU, so it works in any browser that supports WebGL2.
///
/// `resume` only starts shader compilation and never calls `on_ready`. The renderer stays
/// [`is_pending`](WindowRenderer::is_pending) until the caller polls
/// [`complete_resume`](WindowRenderer::complete_resume) (e.g. once per redraw) and it returns
/// `true`, so the browser's main thread isn't blocked where `KHR_parallel_shader_compile` is
/// available.
///
/// The wgpu-specific fields of [`VelloHybridRendererOptions`] (`features`, `limits`,
/// `composite_alpha_mode`, `desired_maximum_frame_latency` and `device_creator`) are ignored.
///
/// Custom resources and WebGL context loss are not yet supported.
pub struct VelloHybridWebGlWindowRenderer {
    render_state: RenderState,
    scene: Scene,
    config: VelloHybridRendererOptions,
    cached_images: ImageCache,
    texture_bindings: WebGlTextureBindings,
}

impl VelloHybridWebGlWindowRenderer {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_options(VelloHybridRendererOptions::default())
    }

    pub fn with_options(config: impl Into<VelloHybridRendererOptions>) -> Self {
        let config = config.into();
        Self {
            render_state: RenderState::Suspended,
            scene: Scene::new_with(0, 0, config.render_settings.level()),
            config,
            cached_images: ImageCache::default(),
            texture_bindings: WebGlTextureBindings::new(),
        }
    }

    /// The WebGL2 context, while the renderer is active.
    pub fn gl_context(&self) -> Option<&web_sys::WebGl2RenderingContext> {
        match &self.render_state {
            RenderState::Active(state) => Some(state.renderer.gl_context()),
            _ => None,
        }
    }
}

fn canvas_from_window_handle(window: &dyn WindowHandle) -> HtmlCanvasElement {
    let handle = window.window_handle().expect("Error getting window handle");
    match handle.as_raw() {
        RawWindowHandle::WebCanvas(handle) => {
            // SAFETY: `obj` points to a `JsValue` holding the canvas, which outlives `handle`.
            let value: &JsValue = unsafe { handle.obj.cast().as_ref() };
            value.clone().unchecked_into()
        }
        RawWindowHandle::Web(handle) => {
            let selector = format!("canvas[data-raw-handle=\"{}\"]", handle.id);
            web_sys::window()
                .and_then(|window| window.document())
                .and_then(|document| document.query_selector(&selector).ok().flatten())
                .expect("No canvas matches the window handle")
                .unchecked_into()
        }
        _ => panic!("VelloHybridWebGlWindowRenderer requires a web canvas window handle"),
    }
}

fn set_canvas_size(canvas: &HtmlCanvasElement, width: u32, height: u32) {
    // Setting the size clears the canvas, even if it is unchanged.
    if canvas.width() != width {
        canvas.set_width(width);
    }
    if canvas.height() != height {
        canvas.set_height(height);
    }
}

impl RenderContext for VelloHybridWebGlWindowRenderer {}

impl WindowRenderer for VelloHybridWebGlWindowRenderer {
    type ScenePainter<'a>
        = WebGlScenePainter<'a>
    where
        Self: 'a;

    fn is_active(&self) -> bool {
        matches!(self.render_state, RenderState::Active(_))
    }

    fn is_pending(&self) -> bool {
        matches!(self.render_state, RenderState::Pending { .. })
    }

    fn resume<F: FnOnce() + 'static>(
        &mut self,
        window_handle: Arc<dyn WindowHandle>,
        width: u32,
        height: u32,
        _on_ready: F,
    ) {
        if !matches!(self.render_state, RenderState::Suspended) {
            return;
        }

        let canvas = canvas_from_window_handle(&*window_handle);
        set_canvas_size(&canvas, width, height);
        self.scene.reset_and_resize(width as u16, height as u16);

        let (init, resources) =
            WebGlRenderer::begin_with(&canvas, self.config.render_settings.clone(), true)
                .expect("Error creating WebGL renderer");

        self.render_state = RenderState::Pending {
            init,
            resources,
            canvas,
        };
    }

    fn complete_resume(&mut self) -> bool {
        match std::mem::replace(&mut self.render_state, RenderState::Suspended) {
            RenderState::Active(active) => {
                self.render_state = RenderState::Active(active);
                true
            }
            RenderState::Suspended => false,
            RenderState::Pending {
                init,
                resources,
                canvas,
            } => match init.try_finish().expect("Error compiling WebGL shaders") {
                WebGlRendererInitStatus::Pending(init) => {
                    self.render_state = RenderState::Pending {
                        init,
                        resources,
                        canvas,
                    };
                    false
                }
                WebGlRendererInitStatus::Complete(renderer) => {
                    let gl = renderer.gl_context();
                    let max = gl
                        .get_parameter(WebGl2RenderingContext::MAX_TEXTURE_SIZE)
                        .ok()
                        .and_then(|v| v.as_f64())
                        .map(|v| v as u16)
                        .unwrap_or(2048);
                    self.cached_images.clear();
                    self.cached_images.set_max_image_size((max, max));
                    self.render_state = RenderState::Active(ActiveRenderState {
                        renderer,
                        resources,
                        canvas,
                    });
                    true
                }
            },
        }
    }

    fn suspend(&mut self) {
        self.render_state = RenderState::Suspended;
        self.cached_images.clear();
    }

    fn set_size(&mut self, width: u32, height: u32) {
        if width as u16 != self.scene.width() || height as u16 != self.scene.height() {
            self.scene.reset_and_resize(width as u16, height as u16);
        }
        if let RenderState::Active(state) = &self.render_state {
            set_canvas_size(&state.canvas, width, height);
        }
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        debug_timer!(timer, feature = "log_frame_times");

        // The browser may clamp the drawing buffer below the canvas size, and the renderer must
        // render at the drawing buffer size.
        let gl = state.renderer.gl_context();
        let render_size = RenderSize {
            width: gl.drawing_buffer_width() as u16,
            height: gl.drawing_buffer_height() as u16,
        };
        if render_size.width != self.scene.width() || render_size.height != self.scene.height() {
            self.scene
                .reset_and_resize(render_size.width, render_size.height);
        }

        let image_manager = WebGlImageManager::new(
            &mut state.renderer,
            &mut state.resources,
            &mut self.cached_images,
            &mut self.texture_bindings,
        );
        let mut scene_painter = WebGlScenePainter::new(&mut self.scene, image_manager)
            .with_glyph_caching(self.config.glyph_caching);
        if self.config.base_color != Color::TRANSPARENT {
            scene_painter.fill(
                peniko::Fill::NonZero,
                kurbo::Affine::IDENTITY,
                self.config.base_color,
                None,
                &kurbo::Rect::new(0., 0., render_size.width as f64, render_size.height as f64),
            );
        }
        draw_fn(&mut scene_painter);
        timer.record_time("cmd");

        let mut options = WebGlRenderOptions::default();
        options.clear_color = Color::TRANSPARENT;
        state
            .renderer
            .render(
                &self.scene,
                &mut state.resources,
                WebGlRenderTarget::DrawingBuffer,
                &self.texture_bindings,
                options,
            )
            .expect("failed to render to canvas");
        WebGlImageManager::new(
            &mut state.renderer,
            &mut state.resources,
            &mut self.cached_images,
            &mut self.texture_bindings,
        )
        .finish_frame()
        .expect("failed to evict images");
        timer.record_time("render");
        timer.print_times("vello_hybrid (webgl): ");

        self.scene.reset();
    }
}
