//! Pattern paints: a recorded [`Scene`] repeated as a tile.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use kurbo::{Affine, Rect, Shape, Stroke, StrokeOpts, Vec2};
use peniko::{BlendMode, Color, Compose, Fill, FontData, Mix, StyleRef};

use crate::{Glyph, NormalizedCoord, PaintScene, Scene};

static PATTERN_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Patterns covering more tiles than this are not drawn by [`fill_pattern_tiles`] and
/// [`stroke_pattern_tiles`].
/// The fallback draws nothing when it would draw more than this many commands (tiles times the
/// commands in the pattern's scene).
const MAX_PATTERN_COMMANDS: f64 = 4_000_000.0;

/// Rasterized tiles are downscaled to at most this many pixels, and this many on each side.
const MAX_TILE_PIXELS: f64 = 4096.0 * 4096.0;
const MAX_TILE_SIDE: f64 = 16384.0;

/// The pixel size at which backends rasterize one tile of a pattern drawn with `transform`
/// (mapping pattern space to pixels), or `None` if the tile is empty.
///
/// This is intended for use by backends.
#[doc(hidden)]
pub fn pattern_raster_size(tile: Rect, transform: Affine) -> Option<(u32, u32)> {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let mut width = (tile.width() * a.hypot(b)).round().max(1.0);
    let mut height = (tile.height() * c.hypot(d)).round().max(1.0);
    if !(width.is_finite() && height.is_finite() && tile.width() > 0.0 && tile.height() > 0.0) {
        return None;
    }
    let scale =
        (MAX_TILE_SIDE / width.max(height)).min((MAX_TILE_PIXELS / (width * height)).sqrt());
    if scale < 1.0 {
        width = (width * scale).floor().max(1.0);
        height = (height * scale).floor().max(1.0);
    }
    Some((width as u32, height as u32))
}

/// A paint that repeats the content of a [`Scene`] in a grid of tiles.
///
/// The scene is drawn in pattern space, where one tile covers `tile`, and the tile is repeated
/// with a period of `tile.width()` by `tile.height()` in both directions. Content outside of
/// `tile` is clipped. Pattern space is mapped to the space of the shape being painted by the
/// brush transform of the draw call.
///
/// A pattern is immutable and has an [`id`](Self::id) that is unique to it and kept by its
/// clones, so backends can cache the rasterized tile across draws and frames.
#[derive(Clone, Debug)]
pub struct Pattern {
    id: u64,
    scene: Arc<Scene>,
    tile: Rect,
}

impl Pattern {
    /// Creates a pattern that repeats the area `tile` of `scene`.
    pub fn new(scene: impl Into<Arc<Scene>>, tile: Rect) -> Self {
        Self {
            id: PATTERN_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            scene: scene.into(),
            tile,
        }
    }

    /// An id unique to this pattern (and its clones).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The content of the pattern, in pattern space.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// The tile, in pattern space.
    pub fn tile(&self) -> Rect {
        self.tile
    }
}

impl PartialEq for Pattern {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

/// Fills `shape` with `pattern` by drawing a clipped copy of the pattern's scene for every tile
/// that intersects the shape. This is a fallback for backends that can't paint with a
/// pattern directly. Takes the same arguments as [`PaintScene::fill`].
///
/// This is intended for use by backends.
#[doc(hidden)]
pub fn fill_pattern_tiles<S: PaintScene>(
    painter: &mut S,
    pattern: &Pattern,
    fill: Fill,
    transform: Affine,
    brush_transform: Option<Affine>,
    shape: &impl Shape,
) {
    fill_pattern_tiles_with(
        painter,
        pattern,
        fill,
        transform,
        brush_transform,
        shape,
        |painter, tile_transform| pattern.scene().replay(painter, tile_transform),
    );
}

/// Strokes `shape` with `pattern`, as [`fill_pattern_tiles`] fills it. Takes the same arguments
/// as [`PaintScene::stroke`].
///
/// This is intended for use by backends.
#[doc(hidden)]
pub fn stroke_pattern_tiles<S: PaintScene>(
    painter: &mut S,
    pattern: &Pattern,
    style: &Stroke,
    transform: Affine,
    brush_transform: Option<Affine>,
    shape: &impl Shape,
) {
    stroke_pattern_tiles_with(
        painter,
        pattern,
        style,
        transform,
        brush_transform,
        shape,
        |painter, tile_transform| pattern.scene().replay(painter, tile_transform),
    );
}

/// Like [`fill_pattern_tiles`], but calls `draw_tile` to draw the content of each tile, with
/// the transform mapping pattern space to the painter's space for that tile. Backends use this
/// to draw a tile they converted to their own format once.
///
/// This is intended for use by backends.
#[doc(hidden)]
pub fn fill_pattern_tiles_with<S: PaintScene>(
    painter: &mut S,
    pattern: &Pattern,
    fill: Fill,
    transform: Affine,
    brush_transform: Option<Affine>,
    shape: &impl Shape,
    draw_tile: impl FnMut(&mut S, Affine),
) {
    let brush_transform = brush_transform.unwrap_or(Affine::IDENTITY);
    if brush_transform.determinant() == 0.0 {
        return;
    }
    let bounds = brush_transform
        .inverse()
        .transform_rect_bbox(shape.bounding_box());
    painter.push_clip_layer(fill, transform, shape);
    draw_tiles(
        painter,
        pattern,
        transform * brush_transform,
        bounds,
        draw_tile,
    );
    painter.pop_layer();
}

/// Like [`stroke_pattern_tiles`], but calls `draw_tile` to draw the content of each tile, as
/// [`fill_pattern_tiles_with`] does.
///
/// This is intended for use by backends.
#[doc(hidden)]
pub fn stroke_pattern_tiles_with<S: PaintScene>(
    painter: &mut S,
    pattern: &Pattern,
    style: &Stroke,
    transform: Affine,
    brush_transform: Option<Affine>,
    shape: &impl Shape,
    draw_tile: impl FnMut(&mut S, Affine),
) {
    if style.width <= 0.0 {
        return;
    }
    // Flatten to 0.1px in the painter's space.
    let [a, b, c, d, _, _] = (transform * brush_transform.unwrap_or(Affine::IDENTITY)).as_coeffs();
    let scale = a.hypot(b).max(c.hypot(d));
    let tolerance = if scale > 0.0 { 0.1 / scale } else { 0.1 };
    let outline = kurbo::stroke(
        shape.path_elements(tolerance),
        style,
        &StrokeOpts::default(),
        tolerance,
    );
    fill_pattern_tiles_with(
        painter,
        pattern,
        Fill::NonZero,
        transform,
        brush_transform,
        &outline,
        draw_tile,
    );
}

/// Draws a glyph run filled or stroked with `pattern`, by drawing the pattern's tiles into a
/// layer composited onto the glyphs with [`Compose::SrcIn`]. This is a fallback for backends
/// that can't paint glyphs with a pattern directly. Takes the same arguments as
/// [`PaintScene::draw_glyphs`], with pattern space being the glyph run's space.
///
/// This is intended for use by backends.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn draw_glyphs_pattern_tiles<'a, S: PaintScene>(
    painter: &mut S,
    pattern: &Pattern,
    font: &'a FontData,
    font_size: f32,
    hint: bool,
    normalized_coords: &'a [NormalizedCoord],
    embolden: Vec2,
    style: impl Into<StyleRef<'a>>,
    brush_alpha: f32,
    transform: Affine,
    glyph_transform: Option<Affine>,
    glyphs: impl Iterator<Item = Glyph> + Clone,
) {
    draw_glyphs_pattern_tiles_with(
        painter,
        pattern,
        font,
        font_size,
        hint,
        normalized_coords,
        embolden,
        style,
        brush_alpha,
        transform,
        glyph_transform,
        glyphs,
        |painter, tile_transform| pattern.scene().replay(painter, tile_transform),
    );
}

/// Like [`draw_glyphs_pattern_tiles`], but calls `draw_tile` to draw the content of each tile,
/// as [`fill_pattern_tiles_with`] does.
///
/// This is intended for use by backends.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn draw_glyphs_pattern_tiles_with<'a, S: PaintScene>(
    painter: &mut S,
    pattern: &Pattern,
    font: &'a FontData,
    font_size: f32,
    hint: bool,
    normalized_coords: &'a [NormalizedCoord],
    embolden: Vec2,
    style: impl Into<StyleRef<'a>>,
    brush_alpha: f32,
    transform: Affine,
    glyph_transform: Option<Affine>,
    glyphs: impl Iterator<Item = Glyph> + Clone,
    draw_tile: impl FnMut(&mut S, Affine),
) {
    let style = style.into();
    let Some(bounds) = glyph_run_bounds(
        font,
        font_size,
        embolden,
        &style,
        glyph_transform,
        glyphs.clone(),
    ) else {
        return;
    };
    painter.push_layer(
        Fill::NonZero,
        Mix::Normal,
        brush_alpha,
        transform,
        &bounds,
        None,
        None,
    );
    painter.draw_glyphs(
        font,
        font_size,
        hint,
        normalized_coords,
        embolden,
        style,
        Color::BLACK,
        1.0,
        transform,
        glyph_transform,
        glyphs,
    );
    painter.push_layer(
        Fill::NonZero,
        BlendMode::new(Mix::Normal, Compose::SrcIn),
        1.0,
        transform,
        &bounds,
        None,
        None,
    );
    draw_tiles(painter, pattern, transform, bounds, draw_tile);
    painter.pop_layer();
    painter.pop_layer();
}

/// A conservative bounding box of a glyph run, in the run's space: each glyph is assumed to fit
/// within the font's bounding box (or two ems of its origin, if the font has none).
fn glyph_run_bounds(
    font: &FontData,
    font_size: f32,
    embolden: Vec2,
    style: &StyleRef<'_>,
    glyph_transform: Option<Affine>,
    glyphs: impl Iterator<Item = Glyph>,
) -> Option<Rect> {
    let em_box = font_bbox(font)
        .map(|bbox| bbox.inflate(0.1, 0.1))
        .unwrap_or(Rect::new(-2.0, -2.0, 2.0, 2.0));
    let em = f64::from(font_size);
    let glyph_box = Rect::new(
        em_box.x0 * em,
        em_box.y0 * em,
        em_box.x1 * em,
        em_box.y1 * em,
    );
    let mut glyph_box = glyph_transform
        .unwrap_or(Affine::IDENTITY)
        .transform_rect_bbox(glyph_box)
        .inflate(embolden.x.abs(), embolden.y.abs());
    if let StyleRef::Stroke(stroke) = style {
        glyph_box = glyph_box.inflate(stroke.width, stroke.width);
    }
    glyphs
        .map(|glyph| glyph_box + Vec2::new(f64::from(glyph.x), f64::from(glyph.y)))
        .reduce(|a, b| a.union(b))
}

/// Draws every tile of `pattern` that intersects `bounds` (in pattern space), with
/// `transform` mapping pattern space to the painter's space.
/// The bounding box of all glyphs in `font`, in ems with y pointing down, from its `head` table.
fn font_bbox(font: &FontData) -> Option<Rect> {
    let data = font.data.data();
    let u16_at = |at: usize| Some(u16::from_be_bytes(data.get(at..at + 2)?.try_into().ok()?));
    let i16_at = |at: usize| Some(f64::from(u16_at(at)? as i16));
    let u32_at = |at: usize| Some(u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?));
    let mut font_offset = 0;
    if data.get(0..4)? == b"ttcf" {
        font_offset = u32_at(12 + 4 * font.index as usize)? as usize;
    }
    let num_tables = u16_at(font_offset + 4)? as usize;
    let head = (0..num_tables)
        .map(|i| font_offset + 12 + 16 * i)
        .find(|&record| data.get(record..record + 4) == Some(b"head"))?;
    let head = u32_at(head + 8)? as usize;
    let units_per_em = f64::from(u16_at(head + 18)?);
    if units_per_em == 0.0 {
        return None;
    }
    let [x0, y0, x1, y1] = [36, 38, 40, 42].map(|at| i16_at(head + at));
    let scale = 1.0 / units_per_em;
    Some(Rect::new(
        x0? * scale,
        -y1? * scale,
        x1? * scale,
        -y0? * scale,
    ))
}

fn draw_tiles<S: PaintScene>(
    painter: &mut S,
    pattern: &Pattern,
    transform: Affine,
    bounds: Rect,
    mut draw_tile: impl FnMut(&mut S, Affine),
) {
    let tile = pattern.tile();
    let (width, height) = (tile.width(), tile.height());
    if !(width > 0.0 && height > 0.0 && bounds.is_finite()) {
        return;
    }
    let first_x = ((bounds.x0 - tile.x0) / width).floor();
    let last_x = ((bounds.x1 - tile.x0) / width).ceil();
    let first_y = ((bounds.y0 - tile.y0) / height).floor();
    let last_y = ((bounds.y1 - tile.y0) / height).ceil();
    let count = (last_x - first_x) * (last_y - first_y);
    let commands = pattern.scene().commands.len().max(1) as f64;
    if !(count > 0.0 && count * commands <= MAX_PATTERN_COMMANDS) {
        return;
    }
    // When pattern space is axis-aligned with the painter's space, snap tile edges to whole
    // pixels, so adjacent tiles' antialiased clips don't leave seams between them.
    let [a, b, c, d, e, f] = transform.as_coeffs();
    let snap = b == 0.0 && c == 0.0 && (a * width).abs() >= 1.0 && (d * height).abs() >= 1.0;
    for y in first_y as i64..last_y as i64 {
        for x in first_x as i64..last_x as i64 {
            let tile_transform = if snap {
                let left = (a * (tile.x0 + x as f64 * width) + e).round();
                let right = (a * (tile.x0 + (x + 1) as f64 * width) + e).round();
                let top = (d * (tile.y0 + y as f64 * height) + f).round();
                let bottom = (d * (tile.y0 + (y + 1) as f64 * height) + f).round();
                let (sx, sy) = ((right - left) / width, (bottom - top) / height);
                Affine::new([sx, 0.0, 0.0, sy, left - sx * tile.x0, top - sy * tile.y0])
            } else {
                transform * Affine::translate((x as f64 * width, y as f64 * height))
            };
            painter.push_clip_layer(Fill::NonZero, tile_transform, &tile);
            draw_tile(painter, tile_transform);
            painter.pop_layer();
        }
    }
}
