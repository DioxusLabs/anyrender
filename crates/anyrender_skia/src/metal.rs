use std::sync::Arc;

use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_app_kit::NSView;
use objc2_core_foundation::CGSize;
use objc2_metal::MTLDevice;
use objc2_quartz_core::CAMetalLayer;
#[cfg(target_os = "ios")]
use objc2_ui_kit::UIView;

#[cfg(feature = "ganesh")]
mod ganesh;
#[cfg(feature = "graphite")]
mod graphite;

#[cfg(feature = "ganesh")]
pub(crate) use ganesh::MetalBackend;
#[cfg(feature = "graphite")]
pub(crate) use graphite::MetalGraphiteBackend;

/// Creates a `CAMetalLayer` for `device` and attaches it to the window's view.
fn create_metal_layer(
    device: &ProtocolObject<dyn MTLDevice>,
    window: Arc<dyn anyrender::WindowHandle>,
    width: u32,
    height: u32,
    composite_alpha_mode: anyrender::CompositeAlphaMode,
) -> Retained<CAMetalLayer> {
    let layer = CAMetalLayer::new();
    layer.setDevice(Some(device));
    layer.setPixelFormat(objc2_metal::MTLPixelFormat::BGRA8Unorm);
    layer.setOpaque(matches!(
        composite_alpha_mode,
        anyrender::CompositeAlphaMode::Opaque | anyrender::CompositeAlphaMode::Auto
    ));
    layer.setPresentsWithTransaction(false);
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
        raw_window_handle::RawWindowHandle::UiKit(uikit) => uikit.ui_view.as_ptr() as *mut UIView,
        _ => panic!("Wrong window handle type"),
    };
    let view = unsafe { view_ptr.as_ref().unwrap() };

    #[cfg(target_os = "macos")]
    {
        view.setWantsLayer(true);
        view.setLayer(Some(&layer.clone().into_super()));
    }

    #[cfg(target_os = "ios")]
    {
        // TODO: consider using raw-window-metal crate. It synchronises some properties
        // from the parent UIView layer to the child metal layer when they change
        layer.setFrame(view.layer().frame());
        view.layer().addSublayer(&layer)
    }

    layer
}
