//! WebGL-compatible [`PaintScene`] implementation for [`vello_gpu::Scene`].

use anyrender::{Filter, Glyph, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, StyleRef};
use vello_common::paint::PaintType;

use peniko::ImageBrush;
use rustc_hash::FxHashMap;
use vello_common::paint::{ImageId, ImageSource};

use std::sync::Arc;

const DEFAULT_TOLERANCE: f64 = 0.1;

pub struct WebGlImageManager<'a> {
    pub(crate) renderer: &'a mut vello_gpu::WebGlRenderer,
    pub(crate) resources: &'a mut vello_gpu::Resources,
    pub(crate) cache: &'a mut FxHashMap<u64, ImageId>,
}

impl<'a> WebGlImageManager<'a> {
    pub fn new(
        renderer: &'a mut vello_gpu::WebGlRenderer,
        resources: &'a mut vello_gpu::Resources,
        cache: &'a mut FxHashMap<u64, ImageId>,
    ) -> Self {
        Self {
            renderer,
            resources,
            cache,
        }
    }

    pub(crate) fn upload_image(
        &mut self,
        image: &peniko::ImageData,
    ) -> Result<ImageId, vello_gpu::WebGlError> {
        let peniko_id = image.data.id();

        if let Some(atlas_id) = self.cache.get(&peniko_id) {
            return Ok(*atlas_id);
        }

        let ImageSource::Pixmap(pixmap) = ImageSource::from_peniko_image_data(image) else {
            unreachable!();
        };

        let atlas_id = self.renderer.upload_image(self.resources, &pixmap)?;
        self.cache.insert(peniko_id, atlas_id);
        Ok(atlas_id)
    }
}

enum LayerKind {
    Layer,
    Clip,
}

pub struct WebGlScenePainter<'s> {
    scene: &'s mut vello_gpu::Scene,
    layer_stack: Vec<LayerKind>,
    image_manager: WebGlImageManager<'s>,
    glyph_caching: bool,
}

impl<'s> WebGlScenePainter<'s> {
    pub fn new(scene: &'s mut vello_gpu::Scene, image_manager: WebGlImageManager<'s>) -> Self {
        Self {
            scene,
            layer_stack: Vec::with_capacity(16),
            image_manager,
            glyph_caching: false,
        }
    }

    /// Enable or disable caching of rasterized glyphs in Vello's glyph atlas.
    ///
    /// Defaults to `false`.
    ///
    /// Note: Vello considers atlas-backed glyph caching experimental.
    pub fn with_glyph_caching(mut self, enabled: bool) -> Self {
        self.glyph_caching = enabled;
        self
    }
}

impl WebGlScenePainter<'_> {
    fn convert_paint(&mut self, paint: PaintRef<'_>) -> PaintType {
        match paint {
            Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
            Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
            Paint::Image(image_brush) => self.convert_image_paint(image_brush),

            // TODO: custom paint
            Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
            Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        }
    }

    fn convert_image_paint(&mut self, image_brush: peniko::ImageBrushRef<'_>) -> PaintType {
        let Ok(image_id) = self.image_manager.upload_image(image_brush.image) else {
            return PaintType::Solid(Color::TRANSPARENT);
        };
        PaintType::Image(ImageBrush {
            image: ImageSource::OpaqueId {
                id: image_id,
                // TODO: optimize opaque case
                may_have_transparency: true,
            },
            sampler: image_brush.sampler,
        })
    }
}

impl RenderContext for WebGlScenePainter<'_> {}
impl PaintScene for WebGlScenePainter<'_> {
    fn reset(&mut self) {
        self.scene.reset();
    }

    fn push_layer(
        &mut self,
        fill: Fill,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        filter: Option<Arc<Filter>>,
        _backdrop_filter: Option<Arc<Filter>>,
    ) {
        let filter = filter.and_then(crate::filters::convert_filter);
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(fill);
        self.layer_stack.push(LayerKind::Layer);
        if let Some(rect) = clip.as_rect() {
            self.scene.push_clip_rect(&rect);
        } else {
            self.scene
                .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
        }
        self.scene
            .push_layer(None, Some(blend.into()), Some(alpha), None, filter);
    }

    fn push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &impl Shape) {
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(fill);
        self.layer_stack.push(LayerKind::Clip);
        if let Some(rect) = clip.as_rect() {
            self.scene.push_clip_rect(&rect);
        } else {
            self.scene
                .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
        }
    }

    fn pop_layer(&mut self) {
        if let Some(kind) = self.layer_stack.pop() {
            match kind {
                LayerKind::Layer => {
                    self.scene.pop_layer();
                    self.scene.pop_clip();
                }
                LayerKind::Clip => self.scene.pop_clip(),
            }
        }
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.scene.set_transform(transform);
        self.scene.set_stroke(style.clone());
        let paint = self.convert_paint(paint.into());
        self.scene.set_paint(paint);
        self.scene
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.scene.stroke_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(style);
        let paint = self.convert_paint(paint.into());
        self.scene.set_paint(paint);
        self.scene
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        if let Some(rect) = shape.as_rect() {
            self.scene.fill_rect(&rect);
        } else {
            self.scene.fill_path(&shape.into_path(DEFAULT_TOLERANCE));
        }
    }

    fn draw_glyphs<'a, 's2: 'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        _brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
    ) {
        let paint = self.convert_paint(paint.into());
        self.scene.set_paint(paint);
        self.scene.set_transform(transform);

        let glyph_caching =
            self.glyph_caching && crate::scene::supports_glyph_caching(transform, glyph_transform);
        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.scene.set_fill_rule(fill);
                let _ = self
                    .scene
                    .glyph_run(self.image_manager.resources, font)
                    .atlas_cache(glyph_caching)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)))
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(|g| glifo::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
            StyleRef::Stroke(stroke) => {
                self.scene.set_stroke(stroke.clone());
                let _ = self
                    .scene
                    .glyph_run(self.image_manager.resources, font)
                    .atlas_cache(glyph_caching)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(|g| glifo::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
        }
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        box_shape: &anyrender::NonUniformRoundedRect,
        offset: kurbo::Vec2,
        spread: f64,
        std_dev: f64,
        color: Color,
        kind: anyrender::BoxShadowKind,
    ) {
        let geometry = anyrender::BoxShadowGeometry::new(box_shape, offset, spread, std_dev, kind);
        self.scene.set_transform(transform);
        self.scene.set_paint(PaintType::Solid(color));
        self.scene.reset_paint_transform();
        self.scene.set_fill_rule(Fill::NonZero);
        if geometry.std_dev == 0.0 {
            // A single fill, so a non-isolated clip anti-aliases the clip edge only once.
            let clip = geometry.needs_clip();
            if clip {
                self.scene.push_clip_path(&geometry.area);
            }
            self.scene.fill_path(&geometry.unblurred_path());
            if clip {
                self.scene.pop_clip();
            }
            return;
        }
        if geometry.is_inset() {
            geometry.draw_inset_with_layers(self, transform, color);
            return;
        }
        // A single fill, so a non-isolated clip anti-aliases the clip edge only once.
        let clip = geometry.needs_clip();
        if clip {
            self.scene.push_clip_path(&geometry.area);
        }
        // TODO: draw shadows with matching individual radii instead of averaging them
        self.scene.fill_blurred_rounded_rect(
            &geometry.shadow.rect,
            geometry.shadow.average_radius() as f32,
            geometry.std_dev as f32,
            false,
        );
        if clip {
            self.scene.pop_clip();
        }
    }
}
