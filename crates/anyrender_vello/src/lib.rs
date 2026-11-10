//! A [`vello`] backend for the [`anyrender`] 2D drawing abstraction

#[cfg(not(target_arch = "wasm32"))]
mod image_renderer;
mod scene;
mod window_renderer;

#[cfg(not(target_arch = "wasm32"))]
pub use image_renderer::VelloImageRenderer;
pub use scene::VelloScenePainter;
pub use wgpu_context::{DeviceHandle, SharedWGPUContext, WGPUContext};
pub use window_renderer::{VelloContext, VelloRendererOptions, VelloWindowRenderer};

pub use wgpu;

use std::num::NonZeroUsize;

#[cfg(target_os = "macos")]
const DEFAULT_THREADS: Option<NonZeroUsize> = NonZeroUsize::new(1);
#[cfg(not(target_os = "macos"))]
const DEFAULT_THREADS: Option<NonZeroUsize> = None;
