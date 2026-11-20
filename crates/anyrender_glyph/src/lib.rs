// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Glyph preparation and drawing for AnyRender backends.
//!
//! This crate is a fork of [`glifo`](https://docs.rs/glifo) 0.5 that does not depend on a
//! renderer. It turns a font and a sequence of positioned glyphs into draw commands (outlines
//! with hinting, variations and synthetic emboldening, COLR v0/v1 glyphs, and decoded bitmap
//! glyphs) for a renderer that implements [`GlyphRenderer`].
//!
//! It does not rasterize and it does not store rasterized glyphs. Instead it decides which
//! glyphs can be served from a raster cache and under which [`GlyphCacheKey`], asks a
//! [`GlyphCache`] to look them up and store them, and asks the renderer to draw them.
//! [`GlyphRasterCache`] is a ready-made store for caches that keep one image per glyph, and
//! [`NoGlyphCache`] draws every glyph uncached.
//!
//! # Features
//!
//! - `std` (enabled by default): Get floating point functions from the standard library
//!   (likely using your target's libc).
//! - `libm`: Use floating point implementations from `libm`.
//! - `png`: Enables PNG support for drawing bitmap glyphs.
//!
//! At least one of `std` and `libm` is required.

// LINEBENDER LINT SET - lib.rs - v3
// See https://linebender.org/wiki/canonical-lints/
// These lints shouldn't apply to examples or tests.
#![cfg_attr(not(test), warn(unused_crate_dependencies))]
// These lints shouldn't apply to examples.
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![allow(
    clippy::too_many_arguments,
    reason = "inherited from glifo, whose lint configuration allows it"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![no_std]

extern crate alloc;
#[cfg(all(feature = "std", feature = "libm"))]
use core_maths as _;

// Currently used for debugging in `cache.rs`, but only in debug build.
use log as _;
#[cfg(feature = "png")]
use png as _;
#[cfg(feature = "std")]
extern crate std;

use peniko::{self, color, kurbo};

mod cache;
mod colr;
mod glyph;
mod interface;
mod key;
mod raster;
mod renderer;
mod util;

pub use cache::{GlyphRasterCache, GlyphRasterCacheStats};
pub use glyph::{
    FontEmbolden, Glyph, GlyphColr, GlyphPrepCache, GlyphPrepCacheMut, GlyphRenderError, GlyphRun,
    GlyphRunBackend, GlyphRunBuilder, GlyphRunRenderer, GlyphSkipReason, HintCache, HintKey,
    NormalizedCoord, OutlineCache, SkippedGlyph,
};
pub use interface::{DrawSink, GlyphCache, GlyphPaint, GlyphRenderer, NoGlyphCache};
pub use key::GlyphCacheKey;
pub use raster::{GlyphCacheConfig, GlyphCommands, GlyphRaster, RasterMetrics};
