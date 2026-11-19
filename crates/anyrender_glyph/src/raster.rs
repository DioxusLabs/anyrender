// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a renderer needs to rasterize a glyph into its raster cache.

use crate::colr::ColrPainter;
use crate::glyph::{GlyphColr, OutlineCacheSession};
use crate::interface::{DrawSink, GlyphPaint};
use crate::kurbo::{Affine, BezPath, Rect, Shape, Vec2};
use crate::peniko::BlendMode;
use crate::peniko::color::palette::css::BLACK;
use crate::peniko::color::{AlphaColor, Srgb};
use core::fmt::Debug;
use pixmap::Pixmap;

/// Configuration of a renderer's glyph raster cache.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphCacheConfig {
    /// Maximum font size (in ppem) that will be cached. Glyphs rendered at
    /// sizes above this threshold are drawn directly each frame, since very
    /// large glyphs consume disproportionate cache space.
    pub max_cached_font_size: f32,
}

impl Default for GlyphCacheConfig {
    fn default() -> Self {
        Self {
            max_cached_font_size: 128.0,
        }
    }
}

/// Size and placement of a rasterized glyph (no pixel data).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RasterMetrics {
    /// Width of the rasterized glyph (pixels).
    pub width: u16,
    /// Height of the rasterized glyph (pixels).
    pub height: u16,
    /// Horizontal bearing (offset from glyph origin to left edge of bitmap).
    pub bearing_x: i16,
    /// Vertical bearing (offset from glyph origin to top edge of bitmap).
    pub bearing_y: i16,
}

/// The content of a glyph that is about to be stored in a raster cache.
#[derive(Debug)]
pub enum GlyphRaster<'a> {
    /// A bitmap glyph: the pixels already exist.
    Pixmap(Pixmap),
    /// An outline or COLR glyph, to be rasterized by the renderer.
    Commands(GlyphCommands<'a>),
}

/// The draw commands of one glyph, positioned for a surface the size of its [`RasterMetrics`].
pub struct GlyphCommands<'a>(pub(crate) GlyphCommandsInner<'a>);

pub(crate) enum GlyphCommandsInner<'a> {
    /// Fill `path` under `transform` in opaque black, giving an alpha mask.
    Outline {
        path: &'a BezPath,
        transform: Affine,
    },
    Colr {
        glyph: &'a GlyphColr<'a>,
        /// Where the glyph's area starts in the image, less than a pixel in each direction.
        offset: Vec2,
        context_color: AlphaColor<Srgb>,
        outline_cache: &'a mut dyn ColrOutlines,
    },
}

/// Moves everything drawn into a sink by `offset`.
struct OffsetSink<'a> {
    sink: &'a mut dyn DrawSink,
    offset: Vec2,
}

impl DrawSink for OffsetSink<'_> {
    fn set_transform(&mut self, t: Affine) {
        self.sink.set_transform(t.then_translate(self.offset));
    }

    fn set_paint(&mut self, paint: GlyphPaint) {
        self.sink.set_paint(paint);
    }

    // Paint transforms are relative to the transform, so they move with it.
    fn set_paint_transform(&mut self, t: Affine) {
        self.sink.set_paint_transform(t);
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.sink.fill_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.sink.fill_rect(rect);
    }

    fn push_clip_layer(&mut self, clip: &BezPath) {
        self.sink.push_clip_layer(clip);
    }

    fn push_clip_path(&mut self, clip: &BezPath) {
        self.sink.push_clip_path(clip);
    }

    fn push_clip_rect(&mut self, clip: &Rect) {
        self.sink.push_clip_rect(clip);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.sink.push_blend_layer(blend_mode);
    }

    fn pop_layer(&mut self) {
        self.sink.pop_layer();
    }

    fn pop_clip_path(&mut self) {
        self.sink.pop_clip_path();
    }

    fn width(&self) -> u16 {
        self.sink.width()
    }

    fn height(&self) -> u16 {
        self.sink.height()
    }
}

/// Type-erases the lifetime of an [`OutlineCacheSession`].
pub(crate) trait ColrOutlines {
    fn paint_colr(
        &mut self,
        glyph: &GlyphColr<'_>,
        context_color: AlphaColor<Srgb>,
        sink: &mut dyn DrawSink,
    );
}

impl ColrOutlines for OutlineCacheSession<'_> {
    fn paint_colr(
        &mut self,
        glyph: &GlyphColr<'_>,
        context_color: AlphaColor<Srgb>,
        sink: &mut dyn DrawSink,
    ) {
        // TODO: Maybe ColrPainter can be reused across glyphs?
        let mut colr_painter = ColrPainter::new(glyph, context_color, sink, self);
        colr_painter.paint();
    }
}

impl Debug for GlyphCommands<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GlyphCommands").finish_non_exhaustive()
    }
}

impl GlyphCommands<'_> {
    /// Draw the glyph into `sink`, with the top-left corner of its image at the origin.
    ///
    /// This can be called more than once, so a renderer can rasterize the glyph and still hand
    /// the commands back if it then has nowhere to store the result.
    pub fn draw(&mut self, sink: &mut dyn DrawSink) {
        match &mut self.0 {
            GlyphCommandsInner::Outline { path, transform } => {
                sink.set_transform(*transform);
                sink.set_paint(BLACK.into());
                sink.fill_path(path);
            }
            GlyphCommandsInner::Colr {
                glyph,
                offset,
                context_color,
                outline_cache,
            } => {
                let sink = &mut OffsetSink {
                    sink,
                    offset: *offset,
                };
                sink.set_transform(Affine::IDENTITY);
                // See the comment in `render_uncached_colr_glyph` for why we wrap COLR glyphs
                // in a clip layer.
                if glyph.has_non_default_blend {
                    sink.push_clip_layer(&glyph.area.to_path(0.1));
                } else {
                    sink.push_clip_rect(&glyph.area);
                }

                outline_cache.paint_colr(glyph, *context_color, sink);

                if glyph.has_non_default_blend {
                    sink.pop_layer();
                } else {
                    sink.pop_clip_path();
                }
            }
        }
    }
}
