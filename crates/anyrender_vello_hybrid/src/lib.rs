//! A [`vello_gpu`] backend for the [`anyrender`] 2D drawing abstraction
#![cfg_attr(docsrs, feature(doc_cfg))]

mod common;
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
mod webgl;
// Not named `wgpu`, which would clash with the `wgpu` crate re-export below.
#[path = "wgpu/mod.rs"]
mod wgpu_backend;

pub use common::{ImageCache, VelloHybridRendererOptions};
#[cfg(all(target_arch = "wasm32", feature = "webgl"))]
pub use webgl::*;
pub use wgpu_backend::*;
pub use wgpu_context::{DeviceCreator, DeviceCreatorError};

pub use wgpu;
pub use wgpu_context::{DeviceHandle, WGPUContext};
