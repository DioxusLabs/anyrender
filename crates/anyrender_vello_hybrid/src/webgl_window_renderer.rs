//! A [`WindowRenderer`] that renders with [`vello_gpu::WebGlRenderer`] into a WebGL2 canvas.

use std::sync::Arc;

use anyrender::{PaintScene, RenderContext, WindowHandle, WindowRenderer};
use debug_timer::debug_timer;
use futures_channel::oneshot;
use peniko::Color;
use raw_window_handle::RawWindowHandle;
use vello_gpu::{
    RenderSize, Resources, Scene, WebGlRenderer, WebGlRendererInitStatus, WebGlTextureBindings,
};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::HtmlCanvasElement;

use crate::{ImageCache, VelloHybridRendererOptions, WebGlImageManager, WebGlScenePainter};

struct ActiveRenderState {
    renderer: WebGlRenderer,
    resources: Resources,
    canvas: HtmlCanvasElement,
}

#[allow(clippy::large_enum_variant)]
enum RenderState {
    Suspended,
    Pending {
        receiver: oneshot::Receiver<ActiveRenderState>,
    },
    Active(ActiveRenderState),
}

/// A [`WindowRenderer`] that draws to the window's `<canvas>` using Vello GPU's WebGL2 backend.
///
/// Unlike [`VelloHybridWindowRenderer`](crate::VelloHybridWindowRenderer), this does not need
/// WebGPU, so it works in any browser that supports WebGL2. Shader compilation is polled once per
/// animation frame, so `resume` does not block the browser's main thread where
/// `KHR_parallel_shader_compile` is available.
///
/// The wgpu-specific fields of [`VelloHybridRendererOptions`] (`features`, `limits`,
/// `composite_alpha_mode` and `desired_maximum_frame_latency`) are ignored.
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
            scene: Scene::new_with(0, 0, config.render_settings.level),
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

async fn next_animation_frame() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .expect("No global window")
            .request_animation_frame(&resolve)
            .expect("Error requesting animation frame");
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
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
        on_ready: F,
    ) {
        if !matches!(self.render_state, RenderState::Suspended) {
            return;
        }

        let canvas = canvas_from_window_handle(&*window_handle);
        set_canvas_size(&canvas, width, height);
        self.scene.reset_and_resize(width as u16, height as u16);

        let (init, resources) =
            WebGlRenderer::begin_with(&canvas, self.config.render_settings, true)
                .expect("Error creating WebGL renderer");

        let (sender, receiver) = oneshot::channel();
        self.render_state = RenderState::Pending { receiver };

        wasm_bindgen_futures::spawn_local(async move {
            let mut init = init;
            let renderer = loop {
                match init.try_finish().expect("Error compiling WebGL shaders") {
                    WebGlRendererInitStatus::Complete(renderer) => break renderer,
                    WebGlRendererInitStatus::Pending(pending) => {
                        init = pending;
                        next_animation_frame().await;
                    }
                }
            };
            let _ = sender.send(ActiveRenderState {
                renderer,
                resources,
                canvas,
            });
            on_ready();
        });
    }

    fn complete_resume(&mut self) -> bool {
        match &mut self.render_state {
            RenderState::Active(_) => true,
            RenderState::Suspended => false,
            RenderState::Pending { receiver } => match receiver.try_recv() {
                Ok(Some(active)) => {
                    let atlas = active.renderer.atlas_info();
                    self.cached_images.clear();
                    self.cached_images
                        .set_max_image_size((atlas.width, atlas.height));
                    self.render_state = RenderState::Active(active);
                    true
                }
                _ => false,
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

        state
            .renderer
            .render(
                &self.scene,
                &mut state.resources,
                &render_size,
                &self.texture_bindings,
                Color::TRANSPARENT,
            )
            .expect("failed to render to canvas");
        WebGlImageManager::new(
            &mut state.renderer,
            &mut state.resources,
            &mut self.cached_images,
        )
        .finish_frame()
        .expect("failed to evict images");
        timer.record_time("render");
        timer.print_times("vello_hybrid (webgl): ");

        self.scene.reset();
    }
}
