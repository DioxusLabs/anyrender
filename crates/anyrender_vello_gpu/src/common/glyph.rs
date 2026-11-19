//! Glyph drawing with [`anyrender_glyph`], with rasterized glyphs packed into one atlas texture.
//!
//! This joins the glyph crate and the atlas crate: glyphs that miss the atlas are rasterized on
//! the CPU with `vello_cpu` and queued for upload, and are drawn as a region of the atlas
//! texture. Glyphs that can't be cached are drawn as paths by Vello.

use std::ops::RangeInclusive;

use anyrender::{AntiAliasing, NormalizedCoord, ResourceId};
use anyrender_atlas::{Atlas, AtlasConfig, AtlasStorage, PixelRect};
use anyrender_glyph::{
    DrawSink, FontEmbolden, Glyph, GlyphCacheConfig, GlyphCacheKey, GlyphPaint, GlyphPrepCache,
    GlyphRaster, GlyphRenderError, GlyphRenderer, GlyphRun, GlyphRunBackend, GlyphRunBuilder,
    RasterMetrics,
};
use kurbo::{Affine, BezPath, Diagonal2, Rect};
use peniko::color::palette::css::BLACK;
use peniko::color::{AlphaColor, Srgb};
use peniko::{BlendMode, FontData, ImageBrush, ImageQuality, ImageSampler, StyleRef};
use pixmap::Pixmap;
use vello_common::TextureId;
use vello_common::geometry::RectU16;
use vello_common::paint::{ImageSource, PaintType, TextureRegion, Tint, TintMode};
use vello_common::render_state::RenderState;
use vello_gpu::Scene;

/// Number of frames a rasterized glyph is kept without being drawn.
const MAX_IDLE_FRAMES: u64 = 120;

/// Pixels to write into the atlas texture.
pub(crate) struct AtlasUpload {
    pub(crate) rect: PixelRect,
    /// Premultiplied RGBA8, `rect.width * rect.height * 4` bytes.
    pub(crate) data: Vec<u8>,
}

#[derive(Default)]
struct Uploads(Vec<AtlasUpload>);

impl AtlasStorage for Uploads {
    fn write_pixels(&mut self, rect: PixelRect, data: &[u8]) {
        self.0.push(AtlasUpload {
            rect,
            data: data.to_vec(),
        });
    }
}

/// The glyph caches of a renderer: outlines, and rasterized glyphs in an atlas texture.
///
/// `T` is the backend's texture type. The backend creates the texture when there is none, binds
/// it under [`GlyphAtlas::texture_id`] and writes [`GlyphAtlas::take_uploads`] to it before
/// rendering. [`GlyphAtlas::clear`] has to be called when the device or context the texture
/// belongs to is lost or replaced.
pub struct GlyphAtlas<T> {
    prep: GlyphPrepCache,
    rasters: Rasters,
    texture: Option<T>,
}

struct Rasters {
    atlas: Atlas<GlyphCacheKey, RasterMetrics>,
    uploads: Uploads,
    texture_id: TextureId,
    /// The context glyphs are rasterized in, kept to reuse its allocations.
    scratch: Option<(vello_cpu::RenderContext, vello_cpu::Resources)>,
}

impl<T> Default for GlyphAtlas<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> GlyphAtlas<T> {
    /// Creates empty caches. The atlas texture is created when a glyph is first cached.
    pub fn new() -> Self {
        Self {
            prep: GlyphPrepCache::default(),
            rasters: Rasters {
                atlas: Atlas::new(AtlasConfig::default()),
                uploads: Uploads::default(),
                texture_id: TextureId(ResourceId::new().into_ffi()),
                scratch: None,
            },
            texture: None,
        }
    }

    /// The id the atlas texture is bound under.
    pub(crate) fn texture_id(&self) -> TextureId {
        self.rasters.texture_id
    }

    /// The size of the atlas texture, in pixels.
    pub(crate) fn texture_size(&self) -> (u16, u16) {
        let config = self.rasters.atlas.config();
        (config.width, config.height)
    }

    pub(crate) fn texture(&self) -> Option<&T> {
        self.texture.as_ref()
    }

    pub(crate) fn set_texture(&mut self, texture: T) {
        self.texture = Some(texture);
    }

    pub(crate) fn has_uploads(&self) -> bool {
        !self.rasters.uploads.0.is_empty()
    }

    /// The pixels that were added since the last call, to be written to the atlas texture.
    pub(crate) fn take_uploads(&mut self) -> Vec<AtlasUpload> {
        std::mem::take(&mut self.rasters.uploads.0)
    }

    /// Evict what has not been used recently. Called once per frame, after rendering.
    pub(crate) fn finish_frame(&mut self) {
        self.prep.maintain();
        self.rasters.atlas.evict_idle(MAX_IDLE_FRAMES);
        self.rasters.atlas.begin_frame();
    }

    /// Forget every rasterized glyph and the atlas texture.
    pub fn clear(&mut self) {
        self.rasters.atlas.clear();
        self.rasters.uploads.0.clear();
        self.texture = None;
    }
}

impl Rasters {
    /// Rasterize outline or COLR glyph commands on the CPU.
    fn rasterize(
        &mut self,
        metrics: RasterMetrics,
        commands: &mut anyrender_glyph::GlyphCommands<'_>,
    ) -> Pixmap {
        let (ctx, resources) = self.scratch.get_or_insert_with(|| {
            // Single-threaded: a glyph is too small to be worth handing to worker threads,
            // which costs far more than rasterizing it.
            let settings = vello_cpu::RenderSettings {
                num_threads: 0,
                ..vello_cpu::RenderSettings::default()
            };
            (
                vello_cpu::RenderContext::new_with(1, 1, settings),
                vello_cpu::Resources::new(),
            )
        });
        ctx.reset_and_resize(metrics.width, metrics.height);
        commands.draw(&mut CpuSink(ctx));
        let mut pixmap = vello_cpu::Pixmap::new(metrics.width, metrics.height);
        ctx.flush();
        ctx.render(pixmap.as_mut(), resources);
        Pixmap::from_parts(
            pixmap.data_as_u8_slice().to_vec(),
            metrics.width,
            metrics.height,
        )
    }

    fn insert(
        &mut self,
        key: &GlyphCacheKey,
        metrics: RasterMetrics,
        image: &Pixmap,
    ) -> Option<PixelRect> {
        self.atlas
            .insert(&mut self.uploads, key.clone(), image, metrics)
    }

    /// Draw the atlas region `rect` into `area` of `scene`.
    fn draw(
        &self,
        scene: &mut Scene,
        rect: PixelRect,
        area: &Rect,
        quality: ImageQuality,
        tint: Option<AlphaColor<Srgb>>,
    ) {
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        let region = RectU16::new(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
        scene.set_paint(PaintType::Image(ImageBrush {
            image: ImageSource::external_texture(
                self.texture_id,
                TextureRegion::Rect(region),
                true,
            ),
            sampler: ImageSampler::new().with_quality(quality),
        }));
        scene.set_paint_transform(Affine::IDENTITY);
        scene.set_tint(tint.map(|color| Tint {
            color,
            mode: TintMode::AlphaMask,
        }));
        scene.fill_rect(area);
    }
}

/// Draws glyph commands into a `vello_cpu` context.
struct CpuSink<'a>(&'a mut vello_cpu::RenderContext);

impl DrawSink for CpuSink<'_> {
    fn set_transform(&mut self, t: Affine) {
        self.0.set_transform(t);
    }

    fn set_paint(&mut self, paint: GlyphPaint) {
        match paint {
            GlyphPaint::Solid(color) => self.0.set_paint(color),
            GlyphPaint::Gradient(gradient) => self.0.set_paint(gradient),
        }
    }

    fn set_paint_transform(&mut self, t: Affine) {
        self.0.set_paint_transform(t);
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.0.fill_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.0.fill_rect(rect);
    }

    fn push_clip_layer(&mut self, clip: &BezPath) {
        self.0.push_clip_layer(clip);
    }

    fn push_clip_path(&mut self, clip: &BezPath) {
        self.0.push_clip_path(clip);
    }

    fn push_clip_rect(&mut self, clip: &Rect) {
        self.0.push_clip_rect(clip);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.0.push_blend_layer(blend_mode);
    }

    fn pop_layer(&mut self) {
        self.0.pop_layer();
    }

    fn pop_clip_path(&mut self) {
        self.0.pop_clip();
    }

    fn width(&self) -> u16 {
        self.0.width()
    }

    fn height(&self) -> u16 {
        self.0.height()
    }
}

impl<T> GlyphAtlas<T> {
    /// Draw a glyph run into `scene` with the scene's current paint.
    ///
    /// With `caching`, upright and uniformly scaled glyphs with a solid paint are drawn from
    /// the atlas. Bitmap glyphs always are, because Vello can only draw them from a texture.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw_glyphs<'a>(
        &'a mut self,
        scene: &'a mut Scene,
        caching: bool,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: StyleRef<'a>,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
        anti_aliasing: AntiAliasing,
    ) {
        scene.reset_paint_transform();
        scene.set_transform(transform);
        scene.set_aliasing_threshold((!anti_aliasing.is_enabled()).then_some(127));
        let font_embolden = match style {
            StyleRef::Fill(fill) => {
                scene.set_fill_rule(fill);
                FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y))
            }
            StyleRef::Stroke(stroke) => {
                scene.set_stroke(stroke.clone());
                FontEmbolden::default()
            }
        };
        let backend = GlyphBackend {
            scene: &mut *scene,
            prep: &mut self.prep,
            rasters: &mut self.rasters,
            // Cached glyphs are anti-aliased.
            caching: caching && anti_aliasing.is_enabled(),
        };
        let mut run = GlyphRunBuilder::new(font.clone(), transform, Affine::IDENTITY, backend)
            .font_size(font_size)
            .font_embolden(font_embolden)
            .hint(hint)
            .normalized_coords(bytemuck::cast_slice(normalized_coords));
        if let Some(glyph_transform) = glyph_transform {
            run = run.glyph_transform(glyph_transform);
        }
        // A glyph that has no drawable representation is skipped; the rest of the run is drawn.
        let _ = match style {
            StyleRef::Fill(_) => run.fill_glyphs(glyphs),
            StyleRef::Stroke(_) => run.stroke_glyphs(glyphs),
        };
        scene.set_aliasing_threshold(None);
    }
}

/// Draws glyph runs into a scene.
struct GlyphBackend<'a> {
    scene: &'a mut Scene,
    prep: &'a mut GlyphPrepCache,
    rasters: &'a mut Rasters,
    /// Whether cacheable outline and COLR glyphs are served from the atlas. Bitmap glyphs
    /// always are, because Vello can only draw them from a texture.
    caching: bool,
}

impl GlyphBackend<'_> {
    fn sink<'b>(scene: &'b mut Scene, rasters: &'b mut Rasters, caching: bool) -> GlyphSink<'b> {
        GlyphSink {
            scene,
            rasters,
            caching,
        }
    }
}

impl<'a> GlyphRunBackend<'a> for GlyphBackend<'a> {
    fn fill_glyphs<Glyphs>(self, run: GlyphRun<'a>, glyphs: Glyphs) -> Result<(), GlyphRenderError>
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let mut run = run.build(glyphs, self.prep.as_mut());
        run.fill_glyphs(&mut Self::sink(self.scene, self.rasters, self.caching))
    }

    fn stroke_glyphs<Glyphs>(
        self,
        run: GlyphRun<'a>,
        glyphs: Glyphs,
    ) -> Result<(), GlyphRenderError>
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let mut run = run.build(glyphs, self.prep.as_mut());
        let width = self.scene.stroke().width;
        let mut stroke = self.scene.stroke().clone();
        stroke.width = width * run.stroke_adjustment();
        self.scene.set_stroke(stroke.clone());
        let result = run.stroke_glyphs(&mut Self::sink(self.scene, self.rasters, self.caching));
        stroke.width = width;
        self.scene.set_stroke(stroke);
        result
    }

    fn render_decoration<Glyphs>(
        self,
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
        let mut run = run.build(glyphs, self.prep.as_mut());
        let mut sink = Self::sink(self.scene, self.rasters, self.caching);
        run.render_decoration(x_range, baseline_y, offset, size, buffer, &mut sink);
    }
}

/// Draws glyphs into a scene, serving cacheable ones from the atlas.
struct GlyphSink<'a> {
    scene: &'a mut Scene,
    rasters: &'a mut Rasters,
    caching: bool,
}

impl DrawSink for GlyphSink<'_> {
    fn set_transform(&mut self, t: Affine) {
        self.scene.set_transform(t);
    }

    fn set_paint(&mut self, paint: GlyphPaint) {
        match paint {
            GlyphPaint::Solid(color) => self.scene.set_paint(color),
            GlyphPaint::Gradient(gradient) => self.scene.set_paint(gradient),
        }
    }

    fn set_paint_transform(&mut self, t: Affine) {
        self.scene.set_paint_transform(t);
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.scene.fill_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.scene.fill_rect(rect);
    }

    fn push_clip_layer(&mut self, clip: &BezPath) {
        self.scene.push_clip_layer(clip);
    }

    fn push_clip_path(&mut self, clip: &BezPath) {
        self.scene.push_clip_path(clip);
    }

    fn push_clip_rect(&mut self, clip: &Rect) {
        self.scene.push_clip_rect(clip);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.scene.push_blend_layer(blend_mode);
    }

    fn pop_layer(&mut self) {
        self.scene.pop_layer();
    }

    fn pop_clip_path(&mut self) {
        self.scene.pop_clip();
    }

    fn width(&self) -> u16 {
        self.scene.width()
    }

    fn height(&self) -> u16 {
        self.scene.height()
    }
}

impl GlyphRenderer for GlyphSink<'_> {
    type SavedState = RenderState;
    /// Where the glyph is in the atlas texture.
    type CachedGlyph = PixelRect;

    fn save_state(&mut self) -> Self::SavedState {
        self.scene.save_current_state()
    }

    fn restore_state(&mut self, state: Self::SavedState) {
        self.scene.restore_state(state);
    }

    fn stroke_path(&mut self, path: &BezPath) {
        self.scene.stroke_path(path);
    }

    fn get_context_color(&self) -> AlphaColor<Srgb> {
        match self.scene.paint() {
            PaintType::Solid(color) => *color,
            _ => BLACK,
        }
    }

    fn has_solid_paint(&self) -> bool {
        matches!(self.scene.paint(), PaintType::Solid(_))
    }

    /// Vello can't draw a pixmap, so an uncached bitmap glyph goes through the atlas too. If
    /// the atlas has no room for it, it is not drawn.
    fn draw_pixmap(&mut self, key: &GlyphCacheKey, pixmap: Pixmap, area: &Rect) {
        let rect = match self.rasters.atlas.get(key) {
            Some(entry) => entry.rect,
            None => {
                let metrics = RasterMetrics {
                    width: pixmap.width(),
                    height: pixmap.height(),
                    bearing_x: 0,
                    bearing_y: 0,
                };
                let Some(rect) = self.rasters.insert(key, metrics, &pixmap) else {
                    return;
                };
                rect
            }
        };
        self.rasters
            .draw(self.scene, rect, area, ImageQuality::Medium, None);
    }

    fn glyph_cache_config(&self) -> Option<GlyphCacheConfig> {
        self.caching.then(GlyphCacheConfig::default)
    }

    fn cached_glyph(&mut self, key: &GlyphCacheKey) -> Option<(RasterMetrics, PixelRect)> {
        let entry = self.rasters.atlas.get(key)?;
        Some((*entry.data, entry.rect))
    }

    fn cache_glyph<'a>(
        &mut self,
        key: &GlyphCacheKey,
        metrics: RasterMetrics,
        raster: GlyphRaster<'a>,
    ) -> Result<PixelRect, GlyphRaster<'a>> {
        match raster {
            GlyphRaster::Pixmap(pixmap) => match self.rasters.insert(key, metrics, &pixmap) {
                Some(rect) => Ok(rect),
                None => Err(GlyphRaster::Pixmap(pixmap)),
            },
            GlyphRaster::Commands(mut commands) => {
                let pixmap = self.rasters.rasterize(metrics, &mut commands);
                match self.rasters.insert(key, metrics, &pixmap) {
                    Some(rect) => Ok(rect),
                    None => Err(GlyphRaster::Commands(commands)),
                }
            }
        }
    }

    fn draw_cached_glyph(
        &mut self,
        rect: &PixelRect,
        area: &Rect,
        quality: ImageQuality,
        tint: Option<AlphaColor<Srgb>>,
    ) {
        self.rasters.draw(self.scene, *rect, area, quality, tint);
    }
}
