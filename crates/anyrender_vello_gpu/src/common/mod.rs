//! Code shared by the wgpu and WebGL backends.

mod filters;
mod image_cache;
mod options;
mod scene;

pub(crate) use filters::convert_filter;
pub use image_cache::ImageCache;
pub use options::VelloGpuRendererOptions;
pub(crate) use scene::{
    DEFAULT_TOLERANCE, LayerKind, draw_glyphs_bitmap_safe, font_has_bitmaps,
    supports_bitmap_atlas_caching, supports_glyph_caching,
};
