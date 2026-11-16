//! Code shared by the wgpu and WebGL backends.

mod filters;
mod image_cache;
mod options;
mod scene;

pub(crate) use filters::convert_filter;
pub use image_cache::ImageCache;
pub use options::VelloHybridRendererOptions;
pub(crate) use scene::{DEFAULT_TOLERANCE, LayerKind, supports_glyph_caching};
