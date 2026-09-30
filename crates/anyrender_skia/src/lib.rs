mod image_renderer;
mod scene;
mod window_renderer;

// Backends
mod cache;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod metal;
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    not(any(feature = "ganesh", feature = "graphite"))
))]
compile_error!("anyrender_skia requires the `ganesh` or `graphite` feature on macOS/iOS");
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod opengl;
#[cfg(feature = "vulkan")]
mod vulkan;

pub use image_renderer::SkiaImageRenderer;
pub use scene::{SkiaSceneCache, SkiaScenePainter};
pub use window_renderer::*;
