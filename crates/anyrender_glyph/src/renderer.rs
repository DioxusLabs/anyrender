// Copyright 2026 the Vello Authors and the Parley Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared glyph rendering logic for rendering backends.

use crate::colr::ColrPainter;
use crate::glyph::{
    CachedGlyphType, GlyphBitmap, GlyphColr, GlyphOutline, GlyphType, OutlineCacheSession,
    PreparedGlyph,
};
use crate::interface::GlyphRenderer;
use crate::key::{GlyphCacheKey, subpixel_offset};
use crate::raster::{GlyphCommands, GlyphCommandsInner, GlyphRaster, RasterMetrics};
use crate::util::AffineExt;
use crate::{kurbo, peniko};
#[cfg(not(feature = "std"))]
use core_maths::CoreFloat as _;
use kurbo::{Affine, BezPath, Rect, Shape};
use peniko::ImageQuality;
use peniko::color::{AlphaColor, Srgb};

/// Fill a prepared glyph, using the renderer's raster cache when possible and falling
/// back to direct rendering otherwise.
pub(crate) fn fill_glyph(
    renderer: &mut impl GlyphRenderer,
    prepared_glyph: PreparedGlyph<'_>,
    outline_cache: &mut OutlineCacheSession<'_>,
) {
    let cache_key = prepared_glyph.cache_key;
    let transform = prepared_glyph.outline_transform;
    let paint_transform = prepared_glyph.relative_paint_transform;

    match prepared_glyph.glyph_type {
        GlyphType::Outline(glyph) => {
            if let Some(key) = cache_key
                && cache_and_render_outline(renderer, &glyph, transform, &key)
            {
                return;
            }

            fill_uncached_outline_glyph(
                renderer,
                &glyph.path,
                glyph.scale,
                transform,
                paint_transform,
            );
        }
        GlyphType::Bitmap(glyph) => {
            let glyph = match cache_key {
                Some(key) => match cache_and_render_bitmap(renderer, glyph, transform, &key) {
                    Ok(()) => return,
                    Err(glyph) => glyph,
                },
                None => glyph,
            };

            render_uncached_bitmap_glyph(renderer, glyph, transform);
        }
        GlyphType::Colr(glyph) => {
            if let Some(key) = cache_key
                && cache_and_render_colr(renderer, &glyph, transform, &key, outline_cache)
            {
                return;
            }

            let context_color = renderer.get_context_color();
            render_uncached_colr_glyph(renderer, &glyph, transform, context_color, outline_cache);
        }
    }
}

/// Stroke a prepared glyph. Stroked outlines are never cached; bitmap and COLR glyphs are
/// always filled.
pub(crate) fn stroke_glyph(
    renderer: &mut impl GlyphRenderer,
    prepared_glyph: PreparedGlyph<'_>,
    outline_cache: &mut OutlineCacheSession<'_>,
) {
    match prepared_glyph.glyph_type {
        GlyphType::Outline(glyph) => {
            stroke_uncached_outline_glyph(
                renderer,
                &glyph.path,
                glyph.scale,
                prepared_glyph.outline_transform,
                prepared_glyph.relative_paint_transform,
            );
        }
        GlyphType::Bitmap(_) | GlyphType::Colr(_) => {
            fill_glyph(renderer, prepared_glyph, outline_cache);
        }
    }
}

fn fill_uncached_outline_glyph(
    renderer: &mut impl GlyphRenderer,
    path: &BezPath,
    scale: f64,
    outline_transform: Affine,
    paint_transform: Affine,
) {
    let state = renderer.save_state();
    renderer.set_transform(outline_transform.pre_scale(scale));
    renderer.set_paint_transform(paint_transform);
    renderer.fill_path(path);
    renderer.restore_state(state);
}

fn stroke_uncached_outline_glyph(
    renderer: &mut impl GlyphRenderer,
    path: &BezPath,
    scale: f64,
    outline_transform: Affine,
    paint_transform: Affine,
) {
    let state = renderer.save_state();
    renderer.set_transform(outline_transform.pre_scale(scale));
    renderer.set_paint_transform(paint_transform);
    renderer.stroke_path(path);
    renderer.restore_state(state);
}

fn render_uncached_bitmap_glyph(
    renderer: &mut impl GlyphRenderer,
    glyph: GlyphBitmap,
    outline_transform: Affine,
) {
    let state = renderer.save_state();
    renderer.set_transform(outline_transform);
    renderer.draw_pixmap(&glyph.key, glyph.pixmap, &glyph.area);
    renderer.restore_state(state);
}

fn render_uncached_colr_glyph(
    renderer: &mut impl GlyphRenderer,
    glyph: &GlyphColr<'_>,
    outline_transform: Affine,
    context_color: AlphaColor<Srgb>,
    outline_cache: &mut OutlineCacheSession<'_>,
) {
    let state = renderer.save_state();
    renderer.set_transform(outline_transform);
    // Two reasons why we wrap COLR glyphs in a clip layer:
    // 1) We need a layer to make sure they are isolated and don't blend into the main surface (unless
    // the glyph is guaranteed to only use default blending, in which case we don't need this).
    // Otherwise, blend modes that are part of the glyph could affect already drawn contents.
    // 2) We do the clipping as a temporary measure to allow the Vello renderers to get a bounding box
    // of the glyph, necessary to keep the cost of blending operations with
    // destructive blend modes to a minimum.
    if glyph.has_non_default_blend {
        renderer.push_clip_layer(&glyph.area.to_path(0.1));
    } else {
        renderer.push_clip_rect(&glyph.area);
    }

    // TODO: Maybe ColrPainter can be reused across glyphs?
    let mut colr_painter = ColrPainter::new(glyph, context_color, renderer, outline_cache);
    colr_painter.paint();
    if glyph.has_non_default_blend {
        renderer.pop_layer();
    } else {
        renderer.pop_clip_path();
    }

    renderer.restore_state(state);
}

/// Render a glyph that is present in the renderer's raster cache.
pub(crate) fn render_cached_glyph(
    renderer: &mut impl GlyphRenderer,
    key: &GlyphCacheKey,
    metrics: RasterMetrics,
    transform: Affine,
    glyph_type: CachedGlyphType,
) {
    match glyph_type {
        CachedGlyphType::Outline => {
            let tint = renderer.get_context_color();
            let [_, _, _, _, tx, ty] = transform.as_coeffs();
            let rect_transform = Affine::translate((
                tx.floor() + metrics.bearing_x as f64,
                ty.floor() + metrics.bearing_y as f64,
            ));
            let area = Rect::new(0.0, 0.0, metrics.width as f64, metrics.height as f64);
            render_from_cache(
                renderer,
                key,
                rect_transform,
                area,
                ImageQuality::Low,
                Some(tint),
            );
        }
        CachedGlyphType::Bitmap => {
            let area = Rect::new(0.0, 0.0, metrics.width as f64, metrics.height as f64);
            render_from_cache(renderer, key, transform, area, ImageQuality::Medium, None);
        }
        // The fractional area preserves sub-pixel accuracy, avoiding scaling artifacts.
        CachedGlyphType::Colr(area) => {
            let quality = quality_for_skew(&transform);
            render_from_cache(renderer, key, transform, area, quality, None);
        }
    }
}

fn render_from_cache(
    renderer: &mut impl GlyphRenderer,
    key: &GlyphCacheKey,
    rect_transform: Affine,
    area: Rect,
    quality: ImageQuality,
    tint: Option<AlphaColor<Srgb>>,
) {
    let state = renderer.save_state();
    renderer.set_transform(rect_transform);
    renderer.draw_cached_glyph(key, &area, quality, tint);
    renderer.restore_state(state);
}

/// Store an outline glyph in the raster cache and render it from there.
///
/// The caller is responsible for checking the cache first and only calling this on a miss.
/// Returns `false` if the glyph has to be drawn uncached.
fn cache_and_render_outline(
    renderer: &mut impl GlyphRenderer,
    glyph: &GlyphOutline,
    outline_transform: Affine,
    key: &GlyphCacheKey,
) -> bool {
    if !supports_raster_caching(&outline_transform, CachedGlyphType::Outline) {
        return false;
    }

    let bounds = glyph.bbox.scale_from_origin(glyph.scale);
    let metrics = calculate_raster_metrics(&bounds);
    let subpixel_offset = subpixel_offset(key.subpixel_x);

    let transform =
        Affine::scale_non_uniform(glyph.scale, -glyph.scale).then_translate(kurbo::Vec2::new(
            subpixel_offset as f64 - metrics.bearing_x as f64,
            -(metrics.bearing_y as f64),
        ));
    let raster = GlyphRaster::Commands(GlyphCommands(GlyphCommandsInner::Outline {
        path: &glyph.path,
        transform,
    }));
    if renderer.cache_glyph(key, metrics, raster).is_err() {
        return false;
    }

    render_cached_glyph(
        renderer,
        key,
        metrics,
        outline_transform,
        CachedGlyphType::Outline,
    );
    true
}

/// Store a bitmap glyph in the raster cache and render it from there, or hand the glyph back
/// if it has to be drawn uncached.
#[allow(
    clippy::result_large_err,
    reason = "the glyph is handed back by value so its pixmap is not copied; not large on 32-bit targets"
)]
fn cache_and_render_bitmap(
    renderer: &mut impl GlyphRenderer,
    glyph: GlyphBitmap,
    transform: Affine,
    key: &GlyphCacheKey,
) -> Result<(), GlyphBitmap> {
    if !supports_raster_caching(&transform, CachedGlyphType::Bitmap) {
        return Err(glyph);
    }

    let metrics = RasterMetrics {
        width: glyph.pixmap.width(),
        height: glyph.pixmap.height(),
        bearing_x: 0,
        bearing_y: 0,
    };
    let GlyphBitmap {
        pixmap,
        area,
        key: bitmap_key,
    } = glyph;

    match renderer.cache_glyph(key, metrics, GlyphRaster::Pixmap(pixmap)) {
        Ok(()) => {
            render_from_cache(renderer, key, transform, area, ImageQuality::Medium, None);
            Ok(())
        }
        Err(GlyphRaster::Pixmap(pixmap)) => Err(GlyphBitmap {
            pixmap,
            area,
            key: bitmap_key,
        }),
        Err(GlyphRaster::Commands(_)) => {
            unreachable!("a renderer must hand back the raster it was given")
        }
    }
}

/// Store a COLR glyph in the raster cache and render it from there.
///
/// Returns `false` if the glyph has to be drawn uncached.
fn cache_and_render_colr(
    renderer: &mut impl GlyphRenderer,
    glyph: &GlyphColr<'_>,
    outline_transform: Affine,
    key: &GlyphCacheKey,
    outline_cache: &mut OutlineCacheSession<'_>,
) -> bool {
    if !supports_raster_caching(&outline_transform, CachedGlyphType::Colr(Rect::ZERO)) {
        return false;
    }

    let metrics = RasterMetrics {
        width: glyph.pix_width,
        height: glyph.pix_height,
        bearing_x: 0,
        bearing_y: 0,
    };
    let raster = GlyphRaster::Commands(GlyphCommands(GlyphCommandsInner::Colr {
        glyph,
        context_color: key.context_color,
        outline_cache,
    }));
    if renderer.cache_glyph(key, metrics, raster).is_err() {
        return false;
    }

    render_cached_glyph(
        renderer,
        key,
        metrics,
        outline_transform,
        CachedGlyphType::Colr(glyph.area),
    );
    true
}

/// Calculate raster metrics (pixel bounds, bearings) from a glyph's bounding box.
#[expect(
    clippy::cast_possible_truncation,
    reason = "glyph bounds fit in i32/u16/i16 at reasonable ppem values"
)]
#[inline]
pub(crate) fn calculate_raster_metrics(bounds: &Rect) -> RasterMetrics {
    // Floor/ceil round outward from the fractional bounding box. Width gets an
    // extra pixel to accommodate the horizontal subpixel offset (up to 0.75 px)
    // applied when rasterising into the cache; the Y axis has no subpixel shift
    // so floor/ceil alone is sufficient. Padding around the cached image
    // provides the guard band needed by the GPU renderer's Extend::Pad sampling.
    let min_x = bounds.x0.floor() as i32;
    let max_x = bounds.x1.ceil() as i32 + 1;

    // For Y, we flip the coordinate system: font Y up -> screen Y down
    // After flipping Y, min_y becomes -max_y and max_y becomes -min_y
    let flipped_min_y = (-bounds.y1).floor() as i32;
    let flipped_max_y = (-bounds.y0).ceil() as i32;

    let width = (max_x - min_x) as u16;
    let height = (flipped_max_y - flipped_min_y) as u16;

    RasterMetrics {
        width,
        height,
        bearing_x: min_x as i16,
        bearing_y: flipped_min_y as i16,
    }
}

/// Choose image sampling quality based on skew presence.
///
/// Skewed transforms need `Medium` quality to avoid aliasing; axis-aligned
/// transforms use `Low` (nearest-neighbour) since the content was already
/// rasterized at pixel boundaries.
#[inline]
pub(crate) fn quality_for_skew(transform: &Affine) -> ImageQuality {
    if transform.has_skew() {
        ImageQuality::Medium
    } else {
        ImageQuality::Low
    }
}

/// Returns `true` if the transform is safe for raster-cached glyph rendering.
#[inline]
pub(crate) fn supports_raster_caching(transform: &Affine, glyph_type: CachedGlyphType) -> bool {
    // TODO: Investigate whether we can support arbitrary mirroring. From some
    // initial experiments, allowing x-mirroring leads to slightly shifted glyphs, so
    // we don't support this now. Y-mirroring also needs more consideration.

    let [a, _, _, d, _, _] = transform.as_coeffs();

    match glyph_type {
        // For those glyphs, we expect any scaling factor to have been completely absorbed. Due to the fact
        // that we had to apply a flip transform for outlines, the y-scaling factor is expected to be negative.
        CachedGlyphType::Outline | CachedGlyphType::Colr(_) => {
            !transform.has_non_unit_skew_or_scale() && a.is_sign_positive() && d.is_sign_negative()
        }
        // For bitmap glyphs, we need to relax the condition a bit, since bitmap glyphs already have a fixed
        // size and thus might not correspond 100% to the font size. Therefore, they likely don't have a unit
        // transform.
        CachedGlyphType::Bitmap => {
            !transform.has_skew() && a.is_sign_positive() && d.is_sign_positive()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::supports_raster_caching;
    use crate::glyph::CachedGlyphType;
    use peniko::kurbo::Affine;
    use peniko::kurbo::Rect;

    #[test]
    fn supports_bitmap_caching_for_identity_and_translation() {
        assert!(supports_raster_caching(
            &Affine::IDENTITY,
            CachedGlyphType::Bitmap
        ));
        assert!(supports_raster_caching(
            &Affine::translate((12.0, -3.5)),
            CachedGlyphType::Bitmap
        ));
    }

    #[test]
    fn rejects_skewed_transforms() {
        assert!(!supports_raster_caching(
            &Affine::new([1.0, 0.1, 0.0, 1.0, 0.0, 0.0]),
            CachedGlyphType::Bitmap
        ));
        assert!(!supports_raster_caching(
            &Affine::skew(0.2, 0.0),
            CachedGlyphType::Outline
        ));
        assert!(!supports_raster_caching(
            &Affine::skew(0.2, 0.0),
            CachedGlyphType::Colr(Rect::ZERO)
        ));
    }

    #[test]
    fn outline_and_colr_reject_non_unit_scales() {
        assert!(!supports_raster_caching(
            &Affine::scale(2.0),
            CachedGlyphType::Outline
        ));
        assert!(!supports_raster_caching(
            &Affine::scale_non_uniform(1.0, -0.5),
            CachedGlyphType::Outline
        ));
        assert!(!supports_raster_caching(
            &Affine::scale(2.0),
            CachedGlyphType::Colr(Rect::ZERO)
        ));
    }

    #[test]
    fn outline_and_colr_requires_negative_y_and_positive_x() {
        assert!(supports_raster_caching(
            &Affine::scale_non_uniform(1.0, -1.0),
            CachedGlyphType::Outline
        ));
        assert!(supports_raster_caching(
            &Affine::scale_non_uniform(1.0, -1.0),
            CachedGlyphType::Colr(Rect::ZERO)
        ));
        assert!(!supports_raster_caching(
            &Affine::scale_non_uniform(-1.0, -1.0),
            CachedGlyphType::Outline
        ));
        assert!(!supports_raster_caching(
            &Affine::scale_non_uniform(1.0, 1.0),
            CachedGlyphType::Outline
        ));
    }

    #[test]
    fn bitmap_allows_positive_scales_only() {
        assert!(supports_raster_caching(
            &Affine::scale_non_uniform(1.0, 1.0),
            CachedGlyphType::Bitmap
        ));
        assert!(supports_raster_caching(
            &Affine::scale_non_uniform(2.0, 3.0),
            CachedGlyphType::Bitmap
        ));
        assert!(!supports_raster_caching(
            &Affine::scale_non_uniform(1.0, -1.0),
            CachedGlyphType::Bitmap
        ));
        assert!(!supports_raster_caching(
            &Affine::scale_non_uniform(-1.0, 1.0),
            CachedGlyphType::Bitmap
        ));
    }
}
