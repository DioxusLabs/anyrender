// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Traits for rendering glyphs and replaying them.

use crate::color::{AlphaColor, Srgb};
use crate::key::GlyphCacheKey;
use crate::kurbo::{Affine, BezPath, Rect, Shape};
use crate::peniko::{BlendMode, Gradient, ImageQuality};
use crate::raster::{GlyphCacheConfig, GlyphRaster, RasterMetrics};
use pixmap::Pixmap;

/// The paint of a glyph draw command: a solid colour or a gradient.
#[derive(Clone, Debug)]
pub enum GlyphPaint {
    /// A solid colour (used for outlines and COLR solid fills).
    Solid(AlphaColor<Srgb>),
    /// A gradient (used for COLR gradient fills).
    Gradient(Gradient),
}

impl From<AlphaColor<Srgb>> for GlyphPaint {
    fn from(c: AlphaColor<Srgb>) -> Self {
        Self::Solid(c)
    }
}

impl From<Gradient> for GlyphPaint {
    fn from(g: Gradient) -> Self {
        Self::Gradient(g)
    }
}

// TODO: This trait is only temporary and will hopefully be replaced once we have a better
// unifying imaging API.
/// A sink for low-level glyph drawing commands.
pub trait DrawSink {
    /// Set the current transform.
    fn set_transform(&mut self, t: Affine);
    /// Set the current paint.
    fn set_paint(&mut self, paint: GlyphPaint);
    /// Set the paint transform.
    fn set_paint_transform(&mut self, t: Affine);
    /// Fill a path with the current paint and transform.
    fn fill_path(&mut self, path: &BezPath);
    /// Fill a rectangle with the current paint and transform.
    fn fill_rect(&mut self, rect: &Rect);
    // TODO: `push/pop_clip_layer` are only needed temporarily for Vello renderers so the impact of
    // destructive blend modes can be limited to a smaller area. Once the limitation is lifted,
    // we can remove this method and just use `push_clip_path` and `push_blend_layer`.
    /// Push a clip layer defined by a path.
    fn push_clip_layer(&mut self, clip: &BezPath);
    /// Push a clip path.
    ///
    /// This method _can_ be implemented in terms of `push_clip_layer` without losing correctness.
    /// However, clients are allowed to choose a more efficient clipping implementation that
    /// doesn't require pushing an isolated layer.
    ///
    /// When providing a custom implementation of this method, the `pop_clip_path` method also
    /// needs to be overridden correspondingly.
    fn push_clip_path(&mut self, clip: &BezPath) {
        self.push_clip_layer(clip);
    }
    /// Push a rectangular clip path.
    fn push_clip_rect(&mut self, clip: &Rect) {
        self.push_clip_path(&clip.to_path(0.1));
    }
    /// Push a blend/compositing layer.
    fn push_blend_layer(&mut self, blend_mode: BlendMode);
    /// Pop the most recent clip or blend layer.
    fn pop_layer(&mut self);
    /// Pop the most recent clip path.
    ///
    /// See the documentation for [`DrawSink::push_clip_path`].
    fn pop_clip_path(&mut self) {
        self.pop_layer();
    }
    /// Width of the surface.
    fn width(&self) -> u16;
    /// Height of the surface.
    fn height(&self) -> u16;
}

/// A stateful renderer that can draw sequences of glyphs.
///
/// It draws glyphs; it does not store them. Rasterized glyphs are kept by a [`GlyphCache`],
/// which hands out [`CachedGlyph`](Self::CachedGlyph)s for the renderer to draw.
pub trait GlyphRenderer: DrawSink {
    /// The type of state used by the renderer.
    type SavedState;

    /// What the renderer needs to draw a glyph that was rasterized earlier, such as a handle
    /// to its image or its place in an atlas. `()` for a renderer that can't draw cached
    /// glyphs.
    type CachedGlyph;

    /// Save the current state.
    fn save_state(&mut self) -> Self::SavedState;

    /// Restore the current state.
    fn restore_state(&mut self, state: Self::SavedState);

    /// Stroke a path with the current paint and stroke settings.
    fn stroke_path(&mut self, path: &BezPath);

    /// Get the context color from the renderer's current paint, used for resolving the
    /// context-dependent colors of COLR glyphs.
    fn get_context_color(&self) -> AlphaColor<Srgb>;

    /// Whether the currently active paint is a solid colour.
    ///
    /// Cached outline glyphs are coloured by tinting, which only works for solid colours.
    fn has_solid_paint(&self) -> bool;

    /// Draw a bitmap glyph that is not in a [`GlyphCache`].
    ///
    /// The pixmap's top-left corner is at the origin of the current transform, with one pixel
    /// per unit, and only `area` is filled.
    fn draw_pixmap(&mut self, pixmap: Pixmap, area: &Rect);

    /// Draw a glyph that a [`GlyphCache`] holds.
    ///
    /// Only called with what [`GlyphCache::get`] or [`GlyphCache::store`] just returned. The
    /// cached image's top-left corner is at the origin of the current transform, with one
    /// pixel per unit, and only `area` is filled. With a `tint`, the image is an alpha mask to
    /// be drawn in that colour.
    fn draw_cached_glyph(
        &mut self,
        glyph: &Self::CachedGlyph,
        area: &Rect,
        quality: ImageQuality,
        tint: Option<AlphaColor<Srgb>>,
    );
}

/// A store of rasterized glyphs.
///
/// The store owns the images and rasterizes the glyphs it is given; this crate decides which
/// glyphs are cacheable and under which [`GlyphCacheKey`]. `G` is what a renderer needs to
/// draw a stored glyph: [`GlyphRenderer::CachedGlyph`].
///
/// The cache is asked for a glyph once per draw: looking a glyph up or storing it returns a
/// `G`, which is all the renderer gets to draw it.
pub trait GlyphCache<G> {
    /// The raster cache settings, or `None` if glyphs are not to be cached.
    fn config(&self) -> Option<GlyphCacheConfig>;

    /// Look up a glyph, marking it as used.
    fn get(&mut self, key: &GlyphCacheKey) -> Option<(RasterMetrics, G)>;

    /// Rasterize a glyph and store it under `key`.
    ///
    /// `metrics` gives the size of the image. Returns the raster back if the glyph can't be
    /// drawn from an image (for example because the cache is full and the renderer can only
    /// draw images that are in it), in which case it is drawn uncached.
    fn store<'a>(
        &mut self,
        key: &GlyphCacheKey,
        metrics: RasterMetrics,
        raster: GlyphRaster<'a>,
    ) -> Result<G, GlyphRaster<'a>>;
}

/// A [`GlyphCache`] that caches nothing: every glyph is drawn uncached.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoGlyphCache;

impl<G> GlyphCache<G> for NoGlyphCache {
    fn config(&self) -> Option<GlyphCacheConfig> {
        None
    }

    fn get(&mut self, _: &GlyphCacheKey) -> Option<(RasterMetrics, G)> {
        None
    }

    fn store<'a>(
        &mut self,
        _: &GlyphCacheKey,
        _: RasterMetrics,
        raster: GlyphRaster<'a>,
    ) -> Result<G, GlyphRaster<'a>> {
        Err(raster)
    }
}
