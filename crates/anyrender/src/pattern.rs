//! Pattern paints: a recorded [`Scene`] repeated as a tile.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use kurbo::{Affine, Rect, Shape, Stroke, StrokeOpts};
use peniko::Fill;

use crate::{PaintScene, Scene};

static PATTERN_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Patterns covering more tiles than this are not drawn by [`fill_pattern_tiles`] and
/// [`stroke_pattern_tiles`].
const MAX_PATTERN_TILES: u64 = 10_000;

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
    let outline = kurbo::stroke(shape.path_elements(0.1), style, &StrokeOpts::default(), 0.1);
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

/// Draws every tile of `pattern` that intersects `bounds` (in pattern space), with
/// `transform` mapping pattern space to the painter's space.
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
    if !(count > 0.0 && count <= MAX_PATTERN_TILES as f64) {
        return;
    }
    for y in first_y as i64..last_y as i64 {
        for x in first_x as i64..last_x as i64 {
            let tile_transform =
                transform * Affine::translate((x as f64 * width, y as f64 * height));
            painter.push_clip_layer(Fill::NonZero, tile_transform, &tile);
            draw_tile(painter, tile_transform);
            painter.pop_layer();
        }
    }
}
