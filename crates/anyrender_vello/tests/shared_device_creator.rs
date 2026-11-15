//! Windows of a Vello and a vello_hybrid context that share one `WGPUContext` created with a
//! `DeviceCreator` all render with the one device the hook created.
//!
//! Runs without the test harness because winit needs the main thread. Needs a display and a
//! wgpu adapter (lavapipe is enough); skips itself when either is missing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyrender::{WindowHandle, WindowRenderer, WindowRendererContext};
use anyrender_vello::wgpu::{self, Features};
use anyrender_vello::{
    DeviceCreator, SharedWGPUContext, VelloContext, VelloRendererOptions, WGPUContext,
};
use anyrender_vello_hybrid::{VelloHybridContext, VelloHybridRendererOptions};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const WINDOWS_PER_BACKEND: usize = 2;

#[derive(Default)]
struct App {
    /// The device the hook created, and how often it was called
    created: Arc<Mutex<Option<wgpu::Device>>>,
    calls: Arc<AtomicUsize>,
    /// The device each window renders with
    vello_devices: Vec<wgpu::Device>,
    hybrid_devices: Vec<wgpu::Device>,
    pool_len: usize,
}

impl App {
    fn creator(&self) -> DeviceCreator {
        let created = self.created.clone();
        let calls = self.calls.clone();
        DeviceCreator::new(move |adapter, descriptor| {
            calls.fetch_add(1, Ordering::SeqCst);
            let (device, queue) = pollster::block_on(adapter.request_device(descriptor))?;
            *created.lock().unwrap() = Some(device.clone());
            Ok((device, queue))
        })
    }
}

fn resume<R: WindowRenderer>(renderer: &mut R, event_loop: &ActiveEventLoop) {
    let size = PhysicalSize::new(64, 64);
    let window = event_loop
        .create_window(Window::default_attributes().with_inner_size(size))
        .unwrap();
    let window: Arc<dyn WindowHandle> = Arc::new(window);
    renderer.resume(window, size.width, size.height, || {});
    assert!(renderer.complete_resume());
    renderer.render(|_| {});
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let features = Features::CLEAR_TEXTURE | Features::PIPELINE_CACHE;
        let wgpu_context = SharedWGPUContext::new(WGPUContext::with_device_creator(
            Some(features),
            None,
            self.creator(),
        ));
        let vello =
            VelloContext::with_wgpu_context(VelloRendererOptions::new(), wgpu_context.clone());
        let hybrid = VelloHybridContext::with_wgpu_context(
            VelloHybridRendererOptions::new(),
            wgpu_context.clone(),
        );

        for _ in 0..WINDOWS_PER_BACKEND {
            let mut renderer = vello.create_renderer();
            resume(&mut renderer, event_loop);
            let device = &renderer.current_device_handle().unwrap().device;
            self.vello_devices.push(device.clone());

            let mut renderer = hybrid.create_renderer();
            resume(&mut renderer, event_loop);
            let device = &renderer.current_device_handle().unwrap().device;
            self.hybrid_devices.push(device.clone());
        }
        self.pool_len = wgpu_context.lock().device_pool.len();
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

fn main() {
    let adapter = pollster::block_on(
        WGPUContext::new()
            .instance
            .request_adapter(&Default::default()),
    );
    if adapter.is_err() {
        eprintln!("no wgpu adapter, skipping");
        return;
    }
    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(err) => {
            eprintln!("no display ({err}), skipping");
            return;
        }
    };

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();

    assert_eq!(app.calls.load(Ordering::SeqCst), 1, "hook called once");
    let created = app.created.lock().unwrap().clone().unwrap();
    assert_eq!(app.vello_devices.len(), WINDOWS_PER_BACKEND);
    assert_eq!(app.hybrid_devices.len(), WINDOWS_PER_BACKEND);
    for device in app.vello_devices.iter().chain(&app.hybrid_devices) {
        assert!(*device == created, "window renders with the hook's device");
    }
    assert_eq!(app.pool_len, 1);
    println!(
        "shared_device_creator: {} Vello and {} vello_hybrid windows rendered with the device the DeviceCreator made",
        WINDOWS_PER_BACKEND, WINDOWS_PER_BACKEND
    );
}
