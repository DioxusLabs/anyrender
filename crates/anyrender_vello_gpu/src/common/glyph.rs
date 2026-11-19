//! Glyph drawing with [`anyrender_glyph`], with rasterized glyphs packed into one atlas texture.
//!
//! This joins the glyph crate and the atlas crate: glyphs that miss the atlas are rasterized on
//! the CPU with `vello_cpu` and queued for upload, and are drawn as a region of the atlas
//! texture. Glyphs that can't be cached are drawn by Vello: outline and COLR glyphs as paths,
//! bitmap glyphs as images in a texture of their own.

use std::ops::RangeInclusive;
use std::sync::Arc;

use anyrender::{AntiAliasing, NormalizedCoord, ResourceId};
use anyrender_atlas::{Atlas, AtlasConfig, AtlasStorage, PixelRect};
use anyrender_glyph::{
    DrawSink, FontEmbolden, Glyph, GlyphCache, GlyphCacheConfig, GlyphCacheKey, GlyphPaint,
    GlyphPrepCache, GlyphRaster, GlyphRenderError, GlyphRenderer, GlyphRun, GlyphRunBackend,
    GlyphRunBuilder, NoGlyphCache, RasterMetrics,
};
use kurbo::{Affine, BezPath, Diagonal2, Rect};
use peniko::color::palette::css::BLACK;
use peniko::color::{AlphaColor, Srgb};
use peniko::{
    BlendMode, Blob, FontData, ImageAlphaType, ImageBrush, ImageData, ImageFormat, ImageQuality,
    ImageSampler, StyleRef,
};
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

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn texture_mut(&mut self) -> Option<&mut T> {
        self.texture.as_mut()
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
    /// the atlas. A bitmap glyph that is not drawn from the atlas is drawn from a texture of
    /// its own, which `upload_image` creates.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw_glyphs<'a>(
        &'a mut self,
        scene: &'a mut Scene,
        upload_image: &'a mut UploadImage<'a>,
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
            upload_image,
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

/// Uploads an image to a texture of its own and binds it, returning the id it is bound under
/// and whether it may have transparency. `None` if the image can't be uploaded.
pub(crate) type UploadImage<'a> = dyn FnMut(&ImageData) -> Option<(TextureId, bool)> + 'a;

/// Draws glyph runs into a scene.
struct GlyphBackend<'a> {
    scene: &'a mut Scene,
    upload_image: &'a mut UploadImage<'a>,
    prep: &'a mut GlyphPrepCache,
    rasters: &'a mut Rasters,
    /// Whether cacheable glyphs are served from the atlas.
    caching: bool,
}

impl<'a> GlyphRunBackend<'a> for GlyphBackend<'a> {
    fn fill_glyphs<Glyphs>(self, run: GlyphRun<'a>, glyphs: Glyphs) -> Result<(), GlyphRenderError>
    where
        Glyphs: Iterator<Item = Glyph> + Clone,
    {
        let mut run = run.build(glyphs, self.prep.as_mut());
        let mut sink = GlyphSink {
            scene: self.scene,
            upload_image: self.upload_image,
            atlas_texture: self.rasters.texture_id,
        };
        if self.caching {
            run.fill_glyphs(&mut sink, self.rasters)
        } else {
            run.fill_glyphs(&mut sink, &mut NoGlyphCache)
        }
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
        let mut sink = GlyphSink {
            scene: self.scene,
            upload_image: self.upload_image,
            atlas_texture: self.rasters.texture_id,
        };
        let result = if self.caching {
            run.stroke_glyphs(&mut sink, self.rasters)
        } else {
            run.stroke_glyphs(&mut sink, &mut NoGlyphCache)
        };
        stroke.width = width;
        sink.scene.set_stroke(stroke);
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
        let mut sink = GlyphSink {
            scene: self.scene,
            upload_image: self.upload_image,
            atlas_texture: self.rasters.texture_id,
        };
        run.render_decoration(x_range, baseline_y, offset, size, buffer, &mut sink);
    }
}

/// Draws glyphs into a scene.
struct GlyphSink<'a> {
    scene: &'a mut Scene,
    upload_image: &'a mut UploadImage<'a>,
    /// The id the atlas texture is bound under.
    atlas_texture: TextureId,
}

impl GlyphSink<'_> {
    /// Fill `area` with `image`.
    fn draw_image(
        &mut self,
        image: ImageSource,
        area: &Rect,
        quality: ImageQuality,
        tint: Option<AlphaColor<Srgb>>,
    ) {
        self.scene.set_paint(PaintType::Image(ImageBrush {
            image,
            sampler: ImageSampler::new().with_quality(quality),
        }));
        self.scene.set_paint_transform(Affine::IDENTITY);
        self.scene.set_tint(tint.map(|color| Tint {
            color,
            mode: TintMode::AlphaMask,
        }));
        self.scene.fill_rect(area);
    }
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

    /// Vello can't draw a pixmap, so the bitmap is uploaded to a texture of its own, like any
    /// other image. If that fails, it is not drawn.
    fn draw_pixmap(&mut self, pixmap: Pixmap, area: &Rect) {
        let (width, height) = (pixmap.width(), pixmap.height());
        let image = ImageData {
            data: Blob::new(Arc::new(pixmap.into_data())),
            format: ImageFormat::Rgba8,
            alpha_type: ImageAlphaType::AlphaPremultiplied,
            width: u32::from(width),
            height: u32::from(height),
        };
        let Some((texture, may_have_transparency)) = (self.upload_image)(&image) else {
            return;
        };
        self.draw_image(
            ImageSource::external_texture(texture, TextureRegion::Full, may_have_transparency),
            area,
            ImageQuality::Medium,
            None,
        );
    }

    fn draw_cached_glyph(
        &mut self,
        rect: &PixelRect,
        area: &Rect,
        quality: ImageQuality,
        tint: Option<AlphaColor<Srgb>>,
    ) {
        if rect.width == 0 || rect.height == 0 {
            return;
        }
        let region = RectU16::new(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
        self.draw_image(
            ImageSource::external_texture(self.atlas_texture, TextureRegion::Rect(region), true),
            area,
            quality,
            tint,
        );
    }
}

impl GlyphCache<PixelRect> for Rasters {
    fn config(&self) -> Option<GlyphCacheConfig> {
        Some(GlyphCacheConfig::default())
    }

    fn get(&mut self, key: &GlyphCacheKey) -> Option<(RasterMetrics, PixelRect)> {
        let entry = self.atlas.get(key)?;
        Some((*entry.data, entry.rect))
    }

    fn store<'a>(
        &mut self,
        key: &GlyphCacheKey,
        metrics: RasterMetrics,
        raster: GlyphRaster<'a>,
    ) -> Result<PixelRect, GlyphRaster<'a>> {
        match raster {
            GlyphRaster::Pixmap(pixmap) => match self.insert(key, metrics, &pixmap) {
                Some(rect) => Ok(rect),
                None => Err(GlyphRaster::Pixmap(pixmap)),
            },
            GlyphRaster::Commands(mut commands) => {
                let pixmap = self.rasterize(metrics, &mut commands);
                match self.insert(key, metrics, &pixmap) {
                    Some(rect) => Ok(rect),
                    None => Err(GlyphRaster::Commands(commands)),
                }
            }
        }
    }
}
