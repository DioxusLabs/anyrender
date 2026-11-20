//! Minimal RGBA8 pixel buffers: the owned [`Pixmap`] and the borrowed [`PixmapRef`] and
//! [`PixmapMut`] views.
//!
//! [`Pixmap`] is the pixel container shared by AnyRender's glyph and atlas crates, so that
//! neither has to depend on the other or on a renderer. It has no dependencies, and its buffer
//! is a plain `Vec<u8>` so that converting to a renderer's own image type can move the pixels
//! instead of copying them.
//!
//! Owned [`Pixmap`] buffers are always tightly packed. Borrowed [`PixmapRef`] and [`PixmapMut`]
//! views carry an explicit byte stride between rows, allowing zero-copy subregions and
//! interoperability with row-padded external or GPU buffers.
//!
//! ## Cargo features
//!
//! All features are disabled by default so the core pixel buffer, blitting, and alpha-conversion
//! helpers stay minimal and dependency-free:
//!
//! - `blend`: Porter-Duff compositing and W3C blend modes (`BlendMode`, `Mix`, `Compose`,
//!   `composite`, `composite_color`, `composite_rgba8`, `composite_color_rgba8`).
//! - `resample`: Separable image resampling and scaling (`ResampleFilter`, `resize`,
//!   `resize_into`).
//! - `transform`: Axis-aligned structural transforms and padding (`flip_*`, `rotate_*`,
//!   `transposed`, `padded`, `padded_clamped`).
//! - `png`: PNG encoding and decoding (`from_png`, `decode_png`, `load_png`, `into_png`,
//!   `encode_png`, `write_png`, `save_png`).

#![no_std]

extern crate alloc;
#[cfg(feature = "png")]
extern crate std;

mod alpha;
#[cfg(feature = "blend")]
mod blend;
mod blit;
mod pixmap;
mod pixmap_mut;
mod pixmap_ref;
#[cfg(feature = "png")]
mod png;
#[cfg(feature = "resample")]
mod resample;
#[cfg(feature = "transform")]
mod transform;
mod util;

#[cfg(feature = "png")]
pub use ::png::{DecodingError as PngDecodingError, EncodingError as PngEncodingError};
pub use alpha::{multiply_alpha_rgba8, premultiply_rgba8, swap_rb_rgba8, unpremultiply_rgba8};
#[cfg(feature = "blend")]
pub use blend::{BlendMode, Compose, Mix, composite_color_rgba8, composite_rgba8};
pub use pixmap::Pixmap;
pub use pixmap_mut::{PixmapMut, RowsMut};
pub use pixmap_ref::{PixmapRef, Rows};
#[cfg(feature = "resample")]
pub use resample::ResampleFilter;

/// The number of bytes one pixel occupies.
pub const BYTES_PER_PIXEL: usize = 4;
