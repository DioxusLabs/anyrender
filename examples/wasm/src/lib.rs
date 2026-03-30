//! Minimal WASM example: paint a simple scene onto an `HtmlCanvasElement`.
#![cfg(target_arch = "wasm32")]

use anyrender::{PaintScene, WindowHandle, WindowRenderer};
use anyrender_vello::VelloWindowRenderer;
use kurbo::{Affine, Circle, Point, Rect, Stroke};
use peniko::{Color, Fill};
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawWindowHandle,
    WebCanvasWindowHandle, WindowHandle as RwhWindowHandle,
};
use std::cell::RefCell;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::HtmlCanvasElement;

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

struct RendererState {
    renderer: VelloWindowRenderer,
    canvas: Option<Arc<HtmlCanvasElement>>,
    width: u32,
    height: u32,
}

impl RendererState {
    fn new() -> Self {
        Self {
            renderer: VelloWindowRenderer::new(),
            canvas: None,
            width: 0,
            height: 0,
        }
    }
}

thread_local! {
    static RENDERER_STATE: RefCell<Option<RendererState>> = RefCell::new(Some(RendererState::new()));
}

struct CanvasPresentationTarget {
    canvas: Arc<HtmlCanvasElement>,
}

impl HasDisplayHandle for CanvasPresentationTarget {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(DisplayHandle::web())
    }
}

impl HasWindowHandle for CanvasPresentationTarget {
    fn window_handle(&self) -> Result<RwhWindowHandle<'_>, HandleError> {
        let js: &wasm_bindgen::JsValue = self.canvas.as_ref().unchecked_ref();
        let wh = WebCanvasWindowHandle::from_wasm_bindgen_0_2(js);
        let raw = RawWindowHandle::WebCanvas(wh);
        Ok(unsafe { RwhWindowHandle::borrow_raw(raw) })
    }
}

/// Paint `html` into `canvas`.
#[wasm_bindgen(js_name = paintHtml)]
pub async fn paint_html(
    canvas: &HtmlCanvasElement,
    html: &str,
    css_width: f64,
    css_height: f64,
    device_pixel_ratio: f64,
) -> Result<(), JsValue> {
    let dpr = device_pixel_ratio.max(1.0);
    let css_width = css_width.max(1.0);
    let css_height = css_height.max(1.0);

    let phys_w = (css_width * dpr).round().max(1.0) as u32;
    let phys_h = (css_height * dpr).round().max(1.0) as u32;

    if canvas.width() != phys_w {
        canvas.set_width(phys_w);
    }
    if canvas.height() != phys_h {
        canvas.set_height(phys_h);
    }

    let _ = html;
    let canvas_arc = Arc::new(canvas.clone());

    let mut renderer_state = RENDERER_STATE
        .with(|slot| slot.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("Renderer state already in use."))?;

    let result = async {
        let canvas_changed = renderer_state
            .canvas
            .as_ref()
            .map_or(true, |existing| !existing.is_same_node(Some(canvas.as_ref())));

        if canvas_changed {
            renderer_state.renderer.suspend();
            renderer_state.canvas = Some(canvas_arc.clone());
        }

        if !renderer_state.renderer.is_active() {
            let window: Arc<dyn WindowHandle> = Arc::new(CanvasPresentationTarget {
                canvas: canvas_arc.clone(),
            });
            renderer_state.renderer.resume(window, phys_w, phys_h).await;
        }

        if !renderer_state.renderer.is_active() {
            return Err(JsValue::from_str(
                "Failed to create WebGPU surface for this canvas.",
            ));
        }

        if renderer_state.width != phys_w || renderer_state.height != phys_h {
            renderer_state.renderer.set_size(phys_w, phys_h);
            renderer_state.width = phys_w;
            renderer_state.height = phys_h;
        }

        renderer_state
            .renderer
            .render(|scene| paint_simple_scene(scene, css_width, css_height, dpr));

        Ok(())
    }
    .await;

    RENDERER_STATE.with(|slot| {
        *slot.borrow_mut() = Some(renderer_state);
    });

    result
}

fn paint_simple_scene(
    scene: &mut impl PaintScene,
    css_width: f64,
    css_height: f64,
    dpr: f64,
) {
    let width = css_width.max(1.0);
    let height = css_height.max(1.0);
    let transform = Affine::scale(dpr);

    scene.fill(
        Fill::NonZero,
        transform,
        Color::WHITE,
        None,
        &Rect::new(0.0, 0.0, width, height),
    );

    let inset = 8.0;
    if width > inset && height > inset {
        scene.stroke(
            &Stroke::new(2.0),
            transform,
            Color::from_rgb8(30, 41, 59),
            None,
            &Rect::new(inset, inset, width - inset, height - inset),
        );
    }

    let radius = (width.min(height) * 0.18).max(6.0);
    scene.fill(
        Fill::NonZero,
        transform,
        Color::from_rgb8(59, 130, 246),
        None,
        &Circle::new(Point::new(width * 0.5, height * 0.5), radius),
    );
}
