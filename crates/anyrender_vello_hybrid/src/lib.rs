//! A [`vello_gpu`] backend for the [`anyrender`] 2D drawing abstraction
#![cfg_attr(docsrs, feature(doc_cfg))]

mod filters;
mod image_cache;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
mod webgl;
// Not named `wgpu`, which would clash with the `wgpu` crate re-export below.
#[path = "wgpu/mod.rs"]
mod wgpu_backend;

pub use image_cache::ImageCache;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub use webgl::*;
pub use wgpu_backend::*;

pub use wgpu;
