use core::ops::RangeInclusive;

use anyrender::NormalizedCoord;
use glifo::{
    AtlasCacher, AtlasSlot, DrawSink, FontEmbolden, Glyph, GlyphCacheConfig, GlyphPrepCache,
    GlyphRenderError, GlyphRenderer, GlyphRun, GlyphRunBackend, GlyphRunBuilder,
};
use kurbo::{Affine, BezPath, Diagonal2, Rect, Vec2};
use peniko::{BlendMode, Color, FontData, StyleRef};
use skrifa::{FontRef, MetadataProvider as _};
use vello_common::{
    TextureId,
    paint::{Image, ImageSource, PaintType, TextureRegion, Tint},
    pixmap::Pixmap,
};

pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

pub(crate) enum LayerKind {
    Layer,
    Clip,
    /// A filter layer whose input and output are clipped to its filter region.
    FilterRegionLayer,
}

/// Whether a glyph run drawn with the given transforms can use Vello's glyph atlas cache.
///
/// Vello only supports caching glyphs that are drawn upright with a uniform scale. It checks
/// this itself when inserting a glyph into the atlas, but not when looking one up, so a skewed
/// (e.g. synthetic italic) or rotated run would otherwise be drawn using upright cached glyphs.
pub(crate) fn supports_glyph_caching(transform: Affine, glyph_transform: Option<Affine>) -> bool {
    let [a, b, c, d, _, _] = (transform * glyph_transform.unwrap_or_default()).as_coeffs();
    b == 0.0 && c == 0.0 && a == d && a > 0.0
}

/// Whether `font` contains bitmap glyph strikes (e.g. CBDT/CBLC, sbix, EBDT/EBLC).
pub(crate) fn font_has_bitmaps(font: &FontData) -> bool {
    FontRef::from_index(font.data.as_ref(), font.index)
        .ok()
        .is_some_and(|f| !f.bitmap_strikes().is_empty())
}

/// Whether `vello_gpu`'s built-in glyph atlas will accept bitmap glyphs for this run without
/// falling back to `ImageSource::Pixmap` (which `vello_gpu` panics on).
pub(crate) fn supports_bitmap_atlas_caching(
    font_size: f32,
    transform: Affine,
    glyph_transform: Option<Affine>,
) -> bool {
    if !font_size.is_finite() || font_size <= 0.0 {
        return false;
    }
    let [a, b, c, d, _, _] = (transform * glyph_transform.unwrap_or_default()).as_coeffs();
    b == 0.0
        && c == 0.0
        && a == d
        && a > 0.0
        && (font_size * d as f32) <= GlyphCacheConfig::default().max_cached_font_size
}

/// Wrapper around `vello_gpu::Scene` that intercepts `ImageSource::Pixmap` paints emitted by
/// `glifo`'s uncached bitmap glyph path and uploads them as external GPU textures.
struct BitmapSafeGlyphRenderer<'a, F> {
    scene: &'a mut vello_gpu::Scene,
    upload_pixmap: F,
}

impl<F> DrawSink for BitmapSafeGlyphRenderer<'_, F> {
    #[inline]
    fn set_transform(&mut self, t: Affine) {
        self.scene.set_transform(t);
    }

    #[inline]
    fn set_paint(&mut self, paint: glifo::AtlasPaint) {
        self.scene.set_paint(paint);
    }

    #[inline]
    fn set_paint_transform(&mut self, t: Affine) {
        self.scene.set_paint_transform(t);
    }

    #[inline]
    fn fill_path(&mut self, path: &BezPath) {
        self.scene.fill_path(path);
    }

    #[inline]
    fn fill_rect(&mut self, rect: &Rect) {
        self.scene.fill_rect(rect);
    }

    #[inline]
    fn push_clip_layer(&mut self, clip: &BezPath) {
        self.scene.push_clip_layer(clip);
    }

    #[inline]
    fn push_clip_path(&mut self, clip: &BezPath) {
        self.scene.push_clip_path(clip);
    }

    #[inline]
    fn push_clip_rect(&mut self, clip: &Rect) {
        self.scene.push_clip_rect(clip);
    }

    #[inline]
    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.scene.push_blend_layer(blend_mode);
    }

    #[inline]
    fn pop_layer(&mut self) {
        self.scene.pop_layer();
    }

    #[inline]
    fn pop_clip_path(&mut self) {
        self.scene.pop_clip();
    }

    #[inline]
    fn width(&self) -> u16 {
        self.scene.width()
    }

    #[inline]
    fn height(&self) -> u16 {
        self.scene.height()
    }
}

impl<F> GlyphRenderer for BitmapSafeGlyphRenderer<'_, F>
where
    F: FnMut(&Pixmap) -> Option<(TextureId, bool)>,
{
    type SavedState = <vello_gpu::Scene as GlyphRenderer>::SavedState;

    #[inline]
    fn save_state(&mut self) -> Self::SavedState {
        GlyphRenderer::save_state(self.scene)
    }

    #[inline]
    fn restore_state(&mut self, state: Self::SavedState) {
        GlyphRenderer::restore_state(self.scene, state);
    }

    #[inline]
    fn stroke_path(&mut self, path: &BezPath) {
        self.scene.stroke_path(path);
    }

    #[inline]
    fn set_paint_image(&mut self, mut image: Image) {
        if let ImageSource::Pixmap(pixmap) = image.image {
            if let Some((texture_id, may_have_transparency)) = (self.upload_pixmap)(&pixmap) {
                image.image = ImageSource::external_texture(
                    texture_id,
                    TextureRegion::Full,
                    may_have_transparency,
                );
                self.scene.set_paint(image);
            } else {
                self.scene.set_paint(PaintType::Solid(Color::TRANSPARENT));
            }
        } else {
            self.scene.set_paint(image);
        }
    }

    #[inline]
    fn set_tint(&mut self, tint: Option<Tint>) {
        self.scene.set_tint(tint);
    }

    #[inline]
    fn get_context_color(&self) -> peniko::color::AlphaColor<peniko::color::Srgb> {
        GlyphRenderer::get_context_color(self.scene)
    }

    #[inline]
    fn current_paint(&self) -> &PaintType {
        GlyphRenderer::current_paint(self.scene)
    }

    #[inline]
    fn atlas_image_source(&self, atlas_slot: &AtlasSlot) -> ImageSource {
        GlyphRenderer::atlas_image_source(self.scene, atlas_slot)
    }

    #[inline]
    fn atlas_paint_transform(&self, atlas_slot: &AtlasSlot) -> Affine {
        GlyphRenderer::atlas_paint_transform(self.scene, atlas_slot)
    }
}

struct BitmapSafeGlyphRunBackend<'a, F> {
    renderer: BitmapSafeGlyphRenderer<'a, F>,
}

impl<'a, F> GlyphRunBackend<'a> for BitmapSafeGlyphRunBackend<'a, F>
where
    F: FnMut(&Pixmap) -> Option<(TextureId, bool)>,
{
    fn atlas_cache(self, _enabled: bool) -> Self {
        self
    }

    fn fill_glyphs<Glyphs>(
        mut self,
        run: GlyphRun<'a>,
        glyphs: Glyphs,
    ) -> Result<(), GlyphRenderError>
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let mut prep_cache = GlyphPrepCache::default();
        let mut glyph_run = run.build(glyphs, prep_cache.as_mut(), AtlasCacher::Disabled);
        glyph_run.fill_glyphs(&mut self.renderer)
    }

    fn stroke_glyphs<Glyphs>(
        mut self,
        run: GlyphRun<'a>,
        glyphs: Glyphs,
    ) -> Result<(), GlyphRenderError>
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let mut prep_cache = GlyphPrepCache::default();
        let mut glyph_run = run.build(glyphs, prep_cache.as_mut(), AtlasCacher::Disabled);
        let stroke_adjustment = glyph_run.stroke_adjustment();
        let original_stroke = self.renderer.scene.stroke().clone();
        let mut adjusted_stroke = original_stroke.clone();
        adjusted_stroke.width *= stroke_adjustment;
        self.renderer.scene.set_stroke(adjusted_stroke);
        let result = glyph_run.stroke_glyphs(&mut self.renderer);
        self.renderer.scene.set_stroke(original_stroke);
        result
    }

    fn render_decoration<Glyphs>(
        mut self,
        run: GlyphRun<'a>,
        glyphs: Glyphs,
        x_range: RangeInclusive<f32>,
        baseline_y: f32,
        offset: f32,
        size: f32,
        buffer: f32,
    ) where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let mut prep_cache = GlyphPrepCache::default();
        let mut glyph_run = run.build(glyphs, prep_cache.as_mut(), AtlasCacher::Disabled);
        glyph_run.render_decoration(
            x_range,
            baseline_y,
            offset,
            size,
            buffer,
            &mut self.renderer,
        );
    }
}

/// Draw a glyph run through `glifo` while converting any uncached bitmap glyph pixmaps
/// into external GPU textures via `upload_pixmap`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_glyphs_bitmap_safe<'a>(
    scene: &'a mut vello_gpu::Scene,
    font: &'a FontData,
    font_size: f32,
    hint: bool,
    normalized_coords: &'a [NormalizedCoord],
    embolden: Vec2,
    style: StyleRef<'a>,
    transform: Affine,
    glyph_transform: Option<Affine>,
    glyphs: impl Iterator<Item = Glyph> + Clone,
    upload_pixmap: impl FnMut(&Pixmap) -> Option<(TextureId, bool)>,
) {
    let backend = BitmapSafeGlyphRunBackend {
        renderer: BitmapSafeGlyphRenderer {
            scene,
            upload_pixmap,
        },
    };
    match style {
        StyleRef::Fill(fill) => {
            backend.renderer.scene.set_fill_rule(fill);
            let _ = GlyphRunBuilder::new(font.clone(), transform, Affine::IDENTITY, backend)
                .font_size(font_size)
                .hint(hint)
                .normalized_coords(normalized_coords)
                .font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)))
                .glyph_transform(glyph_transform.unwrap_or_default())
                .fill_glyphs(glyphs);
        }
        StyleRef::Stroke(stroke) => {
            backend.renderer.scene.set_stroke(stroke.clone());
            let _ = GlyphRunBuilder::new(font.clone(), transform, Affine::IDENTITY, backend)
                .font_size(font_size)
                .hint(hint)
                .normalized_coords(normalized_coords)
                .glyph_transform(glyph_transform.unwrap_or_default())
                .stroke_glyphs(glyphs);
        }
    }
}
