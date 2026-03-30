//! Paint HTML (via Blitz) onto a `<canvas>` with WebGPU + Vello.
//!
//! Flow each `paintHtml` call: sync canvas pixel size → (re)create GPU surface if needed →
//! parse HTML into a Blitz document → layout (`resolve`) → raster (`paint_scene`).
#![cfg(target_arch = "wasm32")]

mod net;

use anyrender::{WindowHandle, WindowRenderer};
use anyrender_vello::VelloWindowRenderer;
use blitz_dom::{DEFAULT_CSS, DocumentConfig, FontContext};
use blitz_html::HtmlDocument;
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use linebender_resource_handle::Blob;
use net::WasmNetProvider;
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawWindowHandle,
    WebCanvasWindowHandle, WindowHandle as RwhWindowHandle,
};
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

/// Bundled fonts: no system font DB on wasm with `blitz-dom` default features off.
fn wasm_font_context() -> FontContext {
    let mut ctx = FontContext::new();
    ctx.collection.register_fonts(
        Blob::new(Arc::new(
            include_bytes!("../../../assets/fonts/roboto/Roboto.ttf").to_vec(),
        )),
        None,
    );
    ctx.collection
        .register_fonts(Blob::new(Arc::new(blitz_dom::BULLET_FONT.to_vec())), None);
    ctx
}

struct CanvasTarget {
    canvas: Arc<HtmlCanvasElement>,
}

impl HasDisplayHandle for CanvasTarget {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(DisplayHandle::web())
    }
}

impl HasWindowHandle for CanvasTarget {
    fn window_handle(&self) -> Result<RwhWindowHandle<'_>, HandleError> {
        let js: &wasm_bindgen::JsValue = self.canvas.as_ref().unchecked_ref();
        let raw = RawWindowHandle::WebCanvas(WebCanvasWindowHandle::from_wasm_bindgen_0_2(js));
        Ok(unsafe { RwhWindowHandle::borrow_raw(raw) })
    }
}

/// One WebGPU session: reuse `VelloWindowRenderer`, only suspend when the canvas element changes.
struct Session {
    net: Arc<WasmNetProvider>,
    gpu: VelloWindowRenderer,
    canvas_key: usize,
    phys_w: u32,
    phys_h: u32,
    html: String,
    doc: Option<HtmlDocument>,
}

impl Session {
    fn new() -> Self {
        Self {
            net: WasmNetProvider::new(),
            gpu: VelloWindowRenderer::new(),
            canvas_key: 0,
            phys_w: 0,
            phys_h: 0,
            html: String::new(),
            doc: None,
        }
    }
}

thread_local! {
    static SESSION: RefCell<Option<Session>> = RefCell::new(None);
    static BUSY: Cell<bool> = Cell::new(false);
}

struct BusyGuard;
impl Drop for BusyGuard {
    fn drop(&mut self) {
        BUSY.set(false);
    }
}

fn try_enter_paint() -> Result<BusyGuard, JsValue> {
    if BUSY.get() {
        return Err(JsValue::from_str("busy"));
    }
    BUSY.set(true);
    Ok(BusyGuard)
}

#[wasm_bindgen(js_name = paintHtml)]
pub async fn paint_html(
    canvas: &HtmlCanvasElement,
    html: &str,
    css_width: f64,
    css_height: f64,
    device_pixel_ratio: f64,
) -> Result<bool, JsValue> {
    let _busy = try_enter_paint()?;

    let dpr = device_pixel_ratio.max(1.0);
    let css_w = css_width.max(1.0);
    let css_h = css_height.max(1.0);
    let phys_w = (css_w * dpr).round().max(1.0) as u32;
    let phys_h = (css_h * dpr).round().max(1.0) as u32;

    if canvas.width() != phys_w {
        canvas.set_width(phys_w);
    }
    if canvas.height() != phys_h {
        canvas.set_height(phys_h);
    }

    let canvas_key = canvas as *const HtmlCanvasElement as usize;
    let canvas_arc = Arc::new(canvas.clone());
    let window: Arc<dyn WindowHandle> = Arc::new(CanvasTarget {
        canvas: canvas_arc.clone(),
    });

    let mut session = SESSION
        .with(|s| s.borrow_mut().take())
        .unwrap_or_else(Session::new);

    if session.canvas_key != canvas_key {
        session.gpu.suspend();
        session.canvas_key = canvas_key;
    }

    if !session.gpu.is_active() {
        session.gpu.resume(window, phys_w, phys_h).await;
    } else if session.phys_w != phys_w || session.phys_h != phys_h {
        session.gpu.set_size(phys_w, phys_h);
    }

    if !session.gpu.is_active() {
        SESSION.with(|s| *s.borrow_mut() = Some(session));
        return Err(JsValue::from_str("Failed to create WebGPU surface."));
    }

    session.phys_w = phys_w;
    session.phys_h = phys_h;

    if session.html != html {
        let mut config = DocumentConfig::default();
        config.net_provider = Some(session.net.clone());
        config.font_ctx = Some(wasm_font_context());
        config.ua_stylesheets = Some(vec![DEFAULT_CSS.to_string()]);
        let mut doc = HtmlDocument::from_html(html, config);
        doc.set_viewport(Viewport::new(
            phys_w,
            phys_h,
            dpr as f32,
            ColorScheme::Light,
        ));
        session.doc = Some(doc);
        session.html = html.to_string();
    }

    let doc = session
        .doc
        .as_mut()
        .ok_or_else(|| JsValue::from_str("No document"))?;

    doc.handle_messages();
    doc.set_viewport(Viewport::new(
        phys_w,
        phys_h,
        dpr as f32,
        ColorScheme::Light,
    ));
    doc.resolve(0.0);
    doc.handle_messages();

    let (w, h) = doc.viewport().window_size;
    let scale = doc.viewport().scale_f64();
    session
        .gpu
        .render(|scene| paint_scene(scene, &*doc, scale, w, h, 0, 0));

    let pending = doc.has_pending_critical_resources() || session.net.has_in_flight();
    SESSION.with(|s| *s.borrow_mut() = Some(session));
    Ok(!pending)
}
