//! A [`vello_hybrid`] backend for the [`anyrender`] 2D drawing abstraction
#![cfg_attr(docsrs, feature(doc_cfg))]

mod scene;
mod window_renderer;

pub use scene::VelloHybridScenePainter;
pub use window_renderer::*;
