mod image_renderer;
mod scene;
mod window_renderer;

// Backends
mod cache;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod metal;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod opengl;
#[cfg(feature = "vulkan")]
mod vulkan;

pub use image_renderer::SkiaImageRenderer;
pub use scene::{SkiaSceneCache, SkiaScenePainter};
pub use window_renderer::*;

/// An error initializing a Skia GPU backend.
#[derive(Debug)]
pub(crate) struct BackendError(String);

impl BackendError {
    pub(crate) fn new(message: impl std::fmt::Display) -> Self {
        Self(message.to_string())
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BackendError {}
