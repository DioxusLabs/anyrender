use std::sync::Arc;

use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_app_kit::NSView;
use objc2_core_foundation::CGSize;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};
use objc2_quartz_core::{CAAutoresizingMask, CAMetalDrawable, CAMetalLayer};
#[cfg(target_os = "ios")]
use objc2_ui_kit::UIView;
use skia_safe::{
    ColorType, Surface,
    gpu::{self, DirectContext, SurfaceOrigin, backend_render_targets, mtl},
    scalar,
};

use crate::window_renderer::SkiaBackend;

pub struct MetalBackend {
    pub metal_layer: Retained<CAMetalLayer>,
    pub command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub skia: DirectContext,
    prepared_drawable: Option<Retained<ProtocolObject<dyn objc2_metal::MTLDrawable>>>,
}

impl MetalBackend {
    pub fn new(window: Arc<dyn anyrender::WindowHandle>, width: u32, height: u32) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("no device found");

        let metal_layer = {
            let layer = CAMetalLayer::new();
            layer.setDevice(Some(&device));
            layer.setPixelFormat(objc2_metal::MTLPixelFormat::BGRA8Unorm);
            layer.setPresentsWithTransaction(false);
            // Match wgpu's `desired_maximum_frame_latency` of 1. This could be configurable but this is typical for gui usage.
            layer.setMaximumDrawableCount(2);
            // Avoid timing out while AppKit is in a live-resize loop and drawable delivery is momentarily delayed.
            layer.setAllowsNextDrawableTimeout(false);
            // Disabling this option allows Skia's Blend Mode to work.
            // More about: https://developer.apple.com/documentation/quartzcore/cametallayer/1478168-framebufferonly
            layer.setFramebufferOnly(false);
            layer.setDrawableSize(CGSize::new(width as f64, height as f64));

            let view_ptr = match window.window_handle().unwrap().as_raw() {
                #[cfg(target_os = "macos")]
                raw_window_handle::RawWindowHandle::AppKit(appkit) => {
                    appkit.ns_view.as_ptr() as *mut NSView
                }
                #[cfg(target_os = "ios")]
                raw_window_handle::RawWindowHandle::UiKit(uikit) => {
                    uikit.ui_view.as_ptr() as *mut UIView
                }
                _ => panic!("Wrong window handle type"),
            };
            let view = unsafe { view_ptr.as_ref().unwrap() };

            #[cfg(target_os = "macos")]
            {
                // AppKit needs the NSView to stay layer-backed so live resize stays on its normal path.
                view.setWantsLayer(true);
                // Use AppKit's root backing layer instead of replacing it with a CAMetalLayer.
                let root_layer = view.layer().expect("NSView should be layer-backed");
                // Match the metal sublayer to the current view-backed layer bounds immediately.
                layer.setFrame(root_layer.bounds());
                // Match the current backing scale so Retina resize does not blur or jump.
                layer.setContentsScale(root_layer.contentsScale());
                // Let Core Animation keep the metal sublayer sized to the view during live resize.
                layer.setAutoresizingMask(
                    CAAutoresizingMask::LayerWidthSizable | CAAutoresizingMask::LayerHeightSizable,
                );
                // Attach CAMetalLayer as a child so AppKit still owns the root layer transaction model.
                root_layer.addSublayer(&layer.clone().into_super());
            }

            #[cfg(target_os = "ios")]
            {
                // TODO: consider using raw-window-metal crate. It synchronises some properties
                // from the parent UIView layer to the child metal layer when they change
                layer.setFrame(view.layer().frame());
                view.layer().addSublayer(&layer)
            }

            layer
        };

        let command_queue = device
            .newCommandQueue()
            .expect("unable to get command queue");

        let backend = unsafe {
            mtl::BackendContext::new(
                Retained::as_ptr(&device) as mtl::Handle,
                Retained::as_ptr(&command_queue) as mtl::Handle,
            )
        };

        let skia_context = gpu::direct_contexts::make_metal(&backend, None).unwrap();

        Self {
            metal_layer,
            command_queue,
            skia: skia_context,
            prepared_drawable: None,
        }
    }
}

impl SkiaBackend for MetalBackend {
    fn set_size(&mut self, width: u32, height: u32) {
        // Re-read the parent layer geometry because AppKit may have updated it during live resize.
        if let Some(superlayer) = self.metal_layer.superlayer() {
            // Keep the metal sublayer aligned with the AppKit-managed backing layer bounds.
            self.metal_layer.setFrame(superlayer.bounds());
            // Keep drawable scale in sync with the current backing scale.
            self.metal_layer
                .setContentsScale(superlayer.contentsScale());
        }
        // Update Metal's pixel-backed drawable size to match Floem's logical resize dimensions.
        self.metal_layer
            .setDrawableSize(CGSize::new(width as f64, height as f64));
    }

    fn prepare(&mut self) -> Option<Surface> {
        let drawable = self.metal_layer.nextDrawable()?;

        let (drawable_width, drawable_height) = {
            let size = self.metal_layer.drawableSize();
            (size.width as scalar, size.height as scalar)
        };

        let surface = {
            let texture_info = unsafe {
                mtl::TextureInfo::new(Retained::as_ptr(&drawable.texture()) as mtl::Handle)
            };

            let backend_render_target = backend_render_targets::make_mtl(
                (drawable_width as i32, drawable_height as i32),
                &texture_info,
            );

            gpu::surfaces::wrap_backend_render_target(
                &mut self.skia,
                &backend_render_target,
                SurfaceOrigin::TopLeft,
                ColorType::BGRA8888,
                None,
                None,
            )
            .unwrap()
        };

        self.prepared_drawable = Some((&drawable).into());

        Some(surface)
    }

    fn flush(&mut self, surface: Surface) {
        self.skia.flush_and_submit();
        drop(surface);
        let command_buffer = self
            .command_queue
            .commandBuffer()
            .expect("unable to get command buffer");

        // TODO: save drawable
        let drawable = self.prepared_drawable.take().unwrap();
        command_buffer.presentDrawable(&drawable);
        command_buffer.commit();
    }
}
