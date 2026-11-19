//! Glyph drawing with [`anyrender_glyph`], with rasterized glyphs cached one pixmap per glyph.

use std::sync::Arc;

use std::ops::RangeInclusive;

use anyrender_glyph::{
    DrawSink, Glyph, GlyphCacheConfig, GlyphCacheKey, GlyphPaint, GlyphPrepCache, GlyphRaster,
    GlyphRasterCache, GlyphRenderError, GlyphRenderer, GlyphRun, GlyphRunBackend, RasterMetrics,
};
use kurbo::{Affine, BezPath, Rect};
use peniko::color::palette::css::BLACK;
use peniko::color::{AlphaColor, Srgb};
use peniko::{BlendMode, ImageBrush, ImageQuality, ImageSampler};
use vello_common::paint::{ImageSource, Tint, TintMode};
use vello_common::pixmap::PixelMetadata;
use vello_common::render_state::RenderState;
use vello_cpu::{PaintType, Pixmap, RenderContext, Resources};

/// Number of frames a rasterized glyph is kept without being drawn.
const MAX_IDLE_FRAMES: u64 = 120;

/// The glyph caches of a scene painter.
pub(crate) struct GlyphCaches {
    /// Outlines and hinting instances.
    prep: GlyphPrepCache,
    rasters: RasterCache,
}

/// Rasterized glyphs, and what is needed to rasterize more.
pub(crate) struct RasterCache {
    images: GlyphRasterCache<Arc<Pixmap>>,
    /// The context glyphs are rasterized in, kept to reuse its allocations.
    scratch: Option<(RenderContext, Resources)>,
}

impl GlyphCaches {
    pub(crate) fn new() -> Self {
        Self {
            prep: GlyphPrepCache::default(),
            rasters: RasterCache {
                images: GlyphRasterCache::default(),
                scratch: None,
            },
        }
    }

    /// Evict what has not been used recently. Called once per frame.
    pub(crate) fn maintain(&mut self) {
        self.prep.maintain();
        self.rasters.images.evict_idle(MAX_IDLE_FRAMES);
        self.rasters.images.begin_frame();
    }
}

impl RasterCache {
    fn rasterize(&mut self, metrics: RasterMetrics, raster: GlyphRaster<'_>) -> Pixmap {
        match raster {
            GlyphRaster::Pixmap(pixmap) => convert_pixmap(pixmap),
            GlyphRaster::Commands(mut commands) => {
                let (ctx, resources) = self
                    .scratch
                    .get_or_insert_with(|| (RenderContext::new(1, 1), Resources::new()));
                ctx.reset_and_resize(metrics.width, metrics.height);
                commands.draw(&mut ContextSink(ctx));
                let mut pixmap = Pixmap::new(metrics.width, metrics.height);
                ctx.flush();
                ctx.render(pixmap.as_mut(), resources);
                pixmap
            }
        }
    }
}

fn convert_pixmap(pixmap: pixmap::Pixmap) -> Pixmap {
    let (width, height) = (pixmap.width(), pixmap.height());
    Pixmap::from_parts(pixmap.into_data(), width, height, PixelMetadata::default())
}

fn draw_image(ctx: &mut RenderContext, image: Arc<Pixmap>, quality: ImageQuality, area: &Rect) {
    ctx.set_paint(PaintType::Image(ImageBrush {
        image: ImageSource::Pixmap(image),
        sampler: ImageSampler::new().with_quality(quality),
    }));
    ctx.set_paint_transform(Affine::IDENTITY);
    ctx.fill_rect(area);
}

/// Draws glyph commands into a `vello_cpu` context.
struct ContextSink<'a>(&'a mut RenderContext);

impl DrawSink for ContextSink<'_> {
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

/// Draws glyph runs into a scene's context.
pub(crate) struct GlyphBackend<'a> {
    pub(crate) ctx: &'a mut RenderContext,
    pub(crate) caches: &'a mut GlyphCaches,
    /// Whether cacheable glyphs are served from the raster cache.
    pub(crate) caching: bool,
}

impl<'a> GlyphRunBackend<'a> for GlyphBackend<'a> {
    fn fill_glyphs<Glyphs>(self, run: GlyphRun<'a>, glyphs: Glyphs) -> Result<(), GlyphRenderError>
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let GlyphCaches { prep, rasters } = self.caches;
        let mut run = run.build(glyphs, prep.as_mut());
        run.fill_glyphs(&mut GlyphSink::new(
            self.ctx,
            self.caching.then_some(rasters),
        ))
    }

    fn stroke_glyphs<Glyphs>(
        self,
        run: GlyphRun<'a>,
        glyphs: Glyphs,
    ) -> Result<(), GlyphRenderError>
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let GlyphCaches { prep, rasters } = self.caches;
        let mut run = run.build(glyphs, prep.as_mut());
        let width = self.ctx.stroke().width;
        let mut stroke = self.ctx.stroke().clone();
        stroke.width = width * run.stroke_adjustment();
        self.ctx.set_stroke(stroke.clone());
        let result = run.stroke_glyphs(&mut GlyphSink::new(
            self.ctx,
            self.caching.then_some(rasters),
        ));
        stroke.width = width;
        self.ctx.set_stroke(stroke);
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
        let GlyphCaches { prep, rasters } = self.caches;
        let mut run = run.build(glyphs, prep.as_mut());
        let mut sink = GlyphSink::new(self.ctx, self.caching.then_some(rasters));
        run.render_decoration(x_range, baseline_y, offset, size, buffer, &mut sink);
    }
}

/// Draws glyphs into a scene's context, serving cacheable ones from its raster cache.
struct GlyphSink<'a> {
    ctx: ContextSink<'a>,
    /// `None` disables caching.
    rasters: Option<&'a mut RasterCache>,
}

impl<'a> GlyphSink<'a> {
    fn new(ctx: &'a mut RenderContext, rasters: Option<&'a mut RasterCache>) -> Self {
        Self {
            ctx: ContextSink(ctx),
            rasters,
        }
    }
}

impl DrawSink for GlyphSink<'_> {
    fn set_transform(&mut self, t: Affine) {
        self.ctx.set_transform(t);
    }

    fn set_paint(&mut self, paint: GlyphPaint) {
        self.ctx.set_paint(paint);
    }

    fn set_paint_transform(&mut self, t: Affine) {
        self.ctx.set_paint_transform(t);
    }

    fn fill_path(&mut self, path: &BezPath) {
        self.ctx.fill_path(path);
    }

    fn fill_rect(&mut self, rect: &Rect) {
        self.ctx.fill_rect(rect);
    }

    fn push_clip_layer(&mut self, clip: &BezPath) {
        self.ctx.push_clip_layer(clip);
    }

    fn push_clip_path(&mut self, clip: &BezPath) {
        self.ctx.push_clip_path(clip);
    }

    fn push_clip_rect(&mut self, clip: &Rect) {
        self.ctx.push_clip_rect(clip);
    }

    fn push_blend_layer(&mut self, blend_mode: BlendMode) {
        self.ctx.push_blend_layer(blend_mode);
    }

    fn pop_layer(&mut self) {
        self.ctx.pop_layer();
    }

    fn pop_clip_path(&mut self) {
        self.ctx.pop_clip_path();
    }

    fn width(&self) -> u16 {
        self.ctx.width()
    }

    fn height(&self) -> u16 {
        self.ctx.height()
    }
}

impl GlyphRenderer for GlyphSink<'_> {
    type SavedState = RenderState;
    type CachedGlyph = Arc<Pixmap>;

    fn save_state(&mut self) -> Self::SavedState {
        self.ctx.0.save_current_state()
    }

    fn restore_state(&mut self, state: Self::SavedState) {
        self.ctx.0.restore_state(state);
    }

    fn stroke_path(&mut self, path: &BezPath) {
        self.ctx.0.stroke_path(path);
    }

    fn get_context_color(&self) -> AlphaColor<Srgb> {
        match self.ctx.0.paint() {
            PaintType::Solid(color) => *color,
            _ => BLACK,
        }
    }

    fn has_solid_paint(&self) -> bool {
        matches!(self.ctx.0.paint(), PaintType::Solid(_))
    }

    fn draw_pixmap(&mut self, _key: &GlyphCacheKey, pixmap: pixmap::Pixmap, area: &Rect) {
        let image = Arc::new(convert_pixmap(pixmap));
        draw_image(self.ctx.0, image, ImageQuality::Medium, area);
    }

    fn glyph_cache_config(&self) -> Option<GlyphCacheConfig> {
        self.rasters.as_ref().map(|_| GlyphCacheConfig::default())
    }

    fn cached_glyph(&mut self, key: &GlyphCacheKey) -> Option<(RasterMetrics, Arc<Pixmap>)> {
        let (image, metrics) = self.rasters.as_mut()?.images.get(key)?;
        Some((metrics, image.clone()))
    }

    fn cache_glyph<'a>(
        &mut self,
        key: &GlyphCacheKey,
        metrics: RasterMetrics,
        raster: GlyphRaster<'a>,
    ) -> Result<Arc<Pixmap>, GlyphRaster<'a>> {
        let Some(rasters) = self.rasters.as_mut() else {
            return Err(raster);
        };
        let bytes = usize::from(metrics.width) * usize::from(metrics.height) * 4;
        if bytes > rasters.images.max_bytes() {
            return Err(raster);
        }
        let image = Arc::new(rasters.rasterize(metrics, raster));
        // The image fits the budget, so storing it only fails if everything else in the cache
        // was used this frame. It is drawn either way.
        let _ = rasters
            .images
            .insert(key.clone(), image.clone(), metrics, bytes);
        Ok(image)
    }

    fn draw_cached_glyph(
        &mut self,
        image: &Arc<Pixmap>,
        area: &Rect,
        quality: ImageQuality,
        tint: Option<AlphaColor<Srgb>>,
    ) {
        self.ctx.0.set_tint(tint.map(|color| Tint {
            color,
            mode: TintMode::AlphaMask,
        }));
        draw_image(self.ctx.0, image.clone(), quality, area);
    }
}
