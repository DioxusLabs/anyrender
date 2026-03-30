//! Minimal WASM example: paint HTML onto an `HtmlCanvasElement`.
#![cfg(target_arch = "wasm32")]

use anyrender::{WindowHandle, WindowRenderer};
use blitz_dom::{DocumentConfig, FontContext, DEFAULT_CSS};
use anyrender_vello::VelloWindowRenderer;
use blitz_html::HtmlDocument;
use blitz_paint::paint_scene;
use blitz_traits::net::{Bytes, NetHandler, NetProvider, Request};
use blitz_traits::shell::{ColorScheme, Viewport};
use js_sys::Uint8Array;
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawWindowHandle,
    WebCanvasWindowHandle, WindowHandle as RwhWindowHandle,
};
use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::HtmlCanvasElement;

/// Blitz is built with `default-features = false`, so there is no system font backend on wasm.
/// Ship a small font and point UA styles at it so body text can shape and paint.
fn wasm_font_context() -> FontContext {
    use linebender_resource_handle::Blob;

    let mut font_ctx = FontContext::new();
    font_ctx.collection.register_fonts(
        Blob::new(Arc::new(
            include_bytes!("../../../assets/fonts/roboto/Roboto.ttf").to_vec(),
        )),
        None,
    );
    font_ctx.collection.register_fonts(
        Blob::new(Arc::new(blitz_dom::BULLET_FONT.to_vec())),
        None,
    );
    font_ctx
}

#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

struct RendererState {
    renderer: VelloWindowRenderer,
    canvas: Option<Arc<HtmlCanvasElement>>,
    width: u32,
    height: u32,
    doc: Option<HtmlDocument>,
    doc_html: String,
    net_provider: Arc<WasmNetProvider>,
}

impl RendererState {
    fn new() -> Self {
        Self {
            renderer: VelloWindowRenderer::new(),
            canvas: None,
            width: 0,
            height: 0,
            doc: None,
            doc_html: String::new(),
            net_provider: Arc::new(WasmNetProvider::new()),
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

struct WasmNetProvider {
    in_flight: Arc<AtomicUsize>,
}

impl WasmNetProvider {
    fn new() -> Self {
        Self {
            in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn has_in_flight(&self) -> bool {
        self.in_flight.load(Ordering::SeqCst) > 0
    }
}

struct InFlightGuard {
    counter: Arc<AtomicUsize>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::SeqCst);
    }
}

impl NetProvider for WasmNetProvider {
    fn fetch(&self, _doc_id: usize, request: Request, handler: Box<dyn NetHandler>) {
        let url = request.url.to_string();
        let counter = self.in_flight.clone();
        counter.fetch_add(1, Ordering::SeqCst);
        wasm_bindgen_futures::spawn_local(async move {
            let _guard = InFlightGuard { counter };
            let window = web_sys::window();
            let bytes = match window {
                Some(window) => {
                    let resp_value = match JsFuture::from(window.fetch_with_str(&url)).await {
                        Ok(value) => value,
                        Err(_) => {
                            handler.bytes(url, Bytes::new());
                            return;
                        }
                    };
                    let resp: web_sys::Response = match resp_value.dyn_into() {
                        Ok(resp) => resp,
                        Err(_) => {
                            handler.bytes(url, Bytes::new());
                            return;
                        }
                    };
                    let resp_url = resp.url();
                    let buffer = match JsFuture::from(resp.array_buffer().unwrap()).await {
                        Ok(buffer) => buffer,
                        Err(_) => {
                            handler.bytes(resp_url, Bytes::new());
                            return;
                        }
                    };
                    let array = Uint8Array::new(&buffer);
                    let mut data = vec![0u8; array.length() as usize];
                    array.copy_to(&mut data);
                    handler.bytes(resp_url, Bytes::from(data));
                    return;
                }
                None => Bytes::new(),
            };
            handler.bytes(url, bytes);
        });
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
) -> Result<bool, JsValue> {
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

        let html_changed = renderer_state.doc_html != html;
        if html_changed || renderer_state.doc.is_none() {
            let mut config = DocumentConfig::default();
            config.net_provider = Some(renderer_state.net_provider.clone());
            config.font_ctx = Some(wasm_font_context());
            config.ua_stylesheets = Some(vec![
                DEFAULT_CSS.to_string(),
                "html, body, p { font-family: Roboto, ui-sans-serif, system-ui, sans-serif !important; }\n\
                 code, kbd, pre, samp { font-family: Roboto, ui-monospace, monospace !important; }\n"
                    .to_string(),
            ]);
            let mut doc = HtmlDocument::from_html(html, config);
            doc.set_viewport(Viewport::new(phys_w, phys_h, dpr as f32, ColorScheme::Light));
            renderer_state.doc = Some(doc);
            renderer_state.doc_html = html.to_string();
        }

        let doc = renderer_state
            .doc
            .as_mut()
            .ok_or_else(|| JsValue::from_str("Document not initialized."))?;

        doc.handle_messages();
        doc.set_viewport(Viewport::new(phys_w, phys_h, dpr as f32, ColorScheme::Light));
        doc.resolve(0.0);
        doc.handle_messages();

        let (width, height) = doc.viewport().window_size;
        let scale = doc.viewport().scale_f64();
        let pending_resources =
            doc.has_pending_critical_resources() || renderer_state.net_provider.has_in_flight();
        renderer_state
            .renderer
            .render(|scene| paint_scene(scene, &*doc, scale, width, height, 0, 0));

        Ok(!pending_resources)
    }
    .await;

    RENDERER_STATE.with(|slot| {
        *slot.borrow_mut() = Some(renderer_state);
    });

    result
}
