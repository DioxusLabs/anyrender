//! A [`vello_cpu`] backend for the [`anyrender`] 2D drawing abstraction
#![cfg_attr(docsrs, feature(doc_cfg))]

mod image_renderer;
mod scene;
mod window_renderer;

pub use image_renderer::VelloCpuImageRenderer;
pub use scene::VelloCpuScenePainter;

#[cfg(any(
    feature = "pixels_window_renderer",
    feature = "softbuffer_window_renderer"
))]
pub use window_renderer::*;
