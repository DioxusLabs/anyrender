// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! What a renderer needs to rasterize a glyph into its raster cache.

use crate::colr::ColrPainter;
use crate::glyph::{GlyphColr, OutlineCacheSession};
use crate::interface::{DrawSink, GlyphPaint};
use crate::key::GlyphCacheKey;
use crate::kurbo::{Affine, BezPath, Rect, Shape};
use crate::peniko::BlendMode;
use crate::peniko::color::palette::css::BLACK;
use crate::peniko::color::{AlphaColor, Srgb};
use alloc::vec::Vec;
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
    /// Draw recorded commands.
    Recorded(&'a [DrawCommand]),
    /// Fill `path` under `transform` in opaque black, giving an alpha mask.
    Outline {
        path: &'a BezPath,
        transform: Affine,
    },
    Colr {
        glyph: &'a GlyphColr<'a>,
        context_color: AlphaColor<Srgb>,
        outline_cache: &'a mut dyn ColrOutlines,
    },
}

/// A glyph to rasterize into the raster cache, as part of a batch.
///
/// Unlike [`GlyphRaster`], this is plain data that can be sent to another thread, so a renderer
/// can rasterize the glyphs of a batch in parallel.
#[derive(Clone, Debug)]
pub struct GlyphRasterJob<'a> {
    /// The key to store the glyph under.
    pub key: &'a GlyphCacheKey,
    /// The size of the glyph's image.
    pub metrics: RasterMetrics,
    pub(crate) content: JobContent<'a>,
}

#[derive(Clone, Debug)]
pub(crate) enum JobContent<'a> {
    /// Fill `path` under `transform` in opaque black, giving an alpha mask.
    Outline {
        path: &'a BezPath,
        transform: Affine,
    },
    /// The recorded drawing of a COLR glyph.
    Recorded(&'a [DrawCommand]),
}

impl<'a> GlyphRasterJob<'a> {
    /// Draw the glyph into `sink`, with the top-left corner of its image at the origin.
    pub fn draw(&self, sink: &mut dyn DrawSink) {
        match self.content {
            JobContent::Outline { path, transform } => {
                sink.set_transform(transform);
                sink.set_paint(BLACK.into());
                sink.fill_path(path);
            }
            JobContent::Recorded(commands) => replay(commands, Affine::IDENTITY, sink),
        }
    }

    /// The glyph as the argument of [`GlyphCache::store`](crate::GlyphCache::store).
    pub fn to_raster(&self) -> GlyphRaster<'a> {
        GlyphRaster::Commands(GlyphCommands(match self.content {
            JobContent::Outline { path, transform } => {
                GlyphCommandsInner::Outline { path, transform }
            }
            JobContent::Recorded(commands) => GlyphCommandsInner::Recorded(commands),
        }))
    }
}

/// One call on a [`DrawSink`].
#[derive(Clone, Debug)]
pub(crate) enum DrawCommand {
    SetTransform(Affine),
    SetPaint(GlyphPaint),
    SetPaintTransform(Affine),
    FillPath(BezPath),
    FillRect(Rect),
    PushClipLayer(BezPath),
    PushClipPath(BezPath),
    PushClipRect(Rect),
    PushBlendLayer(BlendMode),
    PopLayer,
    PopClipPath,
}

/// Draw recorded commands into `sink`, with `base` applied before each of their transforms.
pub(crate) fn replay(commands: &[DrawCommand], base: Affine, sink: &mut dyn DrawSink) {
    for command in commands {
        match command {
            DrawCommand::SetTransform(t) => sink.set_transform(base * *t),
            DrawCommand::SetPaint(paint) => sink.set_paint(paint.clone()),
            DrawCommand::SetPaintTransform(t) => sink.set_paint_transform(*t),
            DrawCommand::FillPath(path) => sink.fill_path(path),
            DrawCommand::FillRect(rect) => sink.fill_rect(rect),
            DrawCommand::PushClipLayer(clip) => sink.push_clip_layer(clip),
            DrawCommand::PushClipPath(clip) => sink.push_clip_path(clip),
            DrawCommand::PushClipRect(clip) => sink.push_clip_rect(clip),
            DrawCommand::PushBlendLayer(blend_mode) => sink.push_blend_layer(*blend_mode),
            DrawCommand::PopLayer => sink.pop_layer(),
            DrawCommand::PopClipPath => sink.pop_clip_path(),
        }
    }
}

/// A [`DrawSink`] that records what is drawn into it, for a surface of the given size.
pub(crate) struct Recorder {
    pub(crate) commands: Vec<DrawCommand>,
    pub(crate) width: u16,
    pub(crate) height: u16,
}

impl DrawSink for Recorder {
    fn set_transform(&mut self, t: Affine) {
        self.commands.push(DrawCommand::SetTransform(t));
    }

    fn set_paint(&mut self, paint: GlyphPaint) {
        self.commands.push(DrawCommand::SetPaint(paint));
    }

    fn set_paint_transform(&mut self, t: Affine) {
        self.commands.push(DrawCommand::SetPaintTransform(t));
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.commands.push(DrawCommand::FillPath(path.clone()));
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.commands.push(DrawCommand::FillRect(*rect));
    }

    fn push_clip_layer(&mut self, clip: &BezPath) {
        self.commands.push(DrawCommand::PushClipLayer(clip.clone()));
    }

    fn push_clip_path(&mut self, clip: &BezPath) {
        self.commands.push(DrawCommand::PushClipPath(clip.clone()));
    }

    fn push_clip_rect(&mut self, clip: &Rect) {
        self.commands.push(DrawCommand::PushClipRect(*clip));
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.commands.push(DrawCommand::PushBlendLayer(blend_mode));
    }

    fn pop_layer(&mut self) {
        self.commands.push(DrawCommand::PopLayer);
    }

    fn pop_clip_path(&mut self) {
        self.commands.push(DrawCommand::PopClipPath);
    }

    fn width(&self) -> u16 {
        self.width
    }

    fn height(&self) -> u16 {
        self.height
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
            GlyphCommandsInner::Recorded(commands) => replay(commands, Affine::IDENTITY, sink),
            GlyphCommandsInner::Outline { path, transform } => {
                sink.set_transform(*transform);
                sink.set_paint(BLACK.into());
                sink.fill_path(path);
            }
            GlyphCommandsInner::Colr {
                glyph,
                context_color,
                outline_cache,
            } => {
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
