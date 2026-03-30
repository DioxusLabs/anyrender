//! Fetch subresources for Blitz (`@font-face`, stylesheets) using the browser `fetch` API.

use blitz_traits::net::{Bytes, NetHandler, NetProvider, Request};
use js_sys::Uint8Array;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::window;

pub struct WasmNetProvider {
    in_flight: Arc<AtomicUsize>,
}

impl WasmNetProvider {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            in_flight: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn has_in_flight(&self) -> bool {
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
            let Some(win) = window() else {
                handler.bytes(url, Bytes::new());
                return;
            };
            let Ok(resp_val) = JsFuture::from(win.fetch_with_str(&url)).await else {
                handler.bytes(url, Bytes::new());
                return;
            };
            let Ok(resp) = resp_val.dyn_into::<web_sys::Response>() else {
                handler.bytes(url, Bytes::new());
                return;
            };
            let resp_url = resp.url();
            let Ok(buf) = JsFuture::from(resp.array_buffer().unwrap()).await else {
                handler.bytes(resp_url, Bytes::new());
                return;
            };
            let arr = Uint8Array::new(&buf);
            let mut data = vec![0u8; arr.length() as usize];
            arr.copy_to(&mut data);
            handler.bytes(resp_url, Bytes::from(data));
        });
    }
}
