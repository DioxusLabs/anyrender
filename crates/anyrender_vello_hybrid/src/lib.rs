//! A [`vello_gpu`] backend for the [`anyrender`] 2D drawing abstraction
#![cfg_attr(docsrs, feature(doc_cfg))]

mod filters;
mod image_cache;
mod scene;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
mod webgl_scene;
mod window_renderer;

pub use image_cache::ImageCache;
pub use scene::ImageManager;
pub use scene::VelloHybridScenePainter;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub use webgl_scene::*;
pub use window_renderer::*;

pub use wgpu;
