//! WebGL-compatible [`PaintScene`] implementation for [`vello_hybrid::Scene`].

use anyrender::{Glyph, NormalizedCoord, Paint, PaintRef, PaintScene};
use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, StyleRef};
use vello_common::paint::PaintType;

use peniko::ImageBrush;
use rustc_hash::FxHashMap;
use vello_common::paint::{ImageId, ImageSource};

const DEFAULT_TOLERANCE: f64 = 0.1;

pub struct WebGlImageManager<'a> {
    pub(crate) renderer: &'a mut vello_hybrid::WebGlRenderer,
    pub(crate) cache: &'a mut FxHashMap<u64, ImageId>,
}

impl<'a> WebGlImageManager<'a> {
    pub fn new(
        renderer: &'a mut vello_hybrid::WebGlRenderer,
        cache: &'a mut FxHashMap<u64, ImageId>,
    ) -> Self {
        Self { renderer, cache }
    }

    pub(crate) fn upload_image(&mut self, image: &peniko::ImageData) -> ImageId {
        let peniko_id = image.data.id();

        if let Some(atlas_id) = self.cache.get(&peniko_id) {
            return *atlas_id;
        }

        let ImageSource::Pixmap(pixmap) = ImageSource::from_peniko_image_data(image) else {
            unreachable!();
        };

        let atlas_id = self.renderer.upload_image(&pixmap);
        self.cache.insert(peniko_id, atlas_id);
        atlas_id
    }
}

enum LayerKind {
    Layer,
    Clip,
}

pub struct WebGlScenePainter<'s> {
    scene: &'s mut vello_hybrid::Scene,
    layer_stack: Vec<LayerKind>,
    image_manager: WebGlImageManager<'s>,
}

impl<'s> WebGlScenePainter<'s> {
    pub fn new(scene: &'s mut vello_hybrid::Scene, image_manager: WebGlImageManager<'s>) -> Self {
        Self {
            scene,
            layer_stack: Vec::with_capacity(16),
            image_manager,
        }
    }
}

impl WebGlScenePainter<'_> {
    fn convert_paint(&mut self, paint: PaintRef<'_>) -> PaintType {
        match paint {
            Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
            Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
            Paint::Image(image_brush) => self.convert_image_paint(image_brush),
            Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        }
    }

    fn convert_image_paint(&mut self, image_brush: peniko::ImageBrushRef<'_>) -> PaintType {
        let image_id = self.image_manager.upload_image(image_brush.image);
        PaintType::Image(ImageBrush {
            image: ImageSource::OpaqueId(image_id),
            sampler: image_brush.sampler,
        })
    }
}

impl PaintScene for WebGlScenePainter<'_> {
    fn reset(&mut self) {
        self.scene.reset();
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
    ) {
        self.scene.set_transform(transform);
        self.layer_stack.push(LayerKind::Layer);
        self.scene.push_layer(
            Some(&clip.into_path(DEFAULT_TOLERANCE)),
            Some(blend.into()),
            Some(alpha),
            None,
            None,
        );
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        self.scene.set_transform(transform);
        self.layer_stack.push(LayerKind::Clip);
        self.scene
            .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
    }

    fn pop_layer(&mut self) {
        if let Some(kind) = self.layer_stack.pop() {
            match kind {
                LayerKind::Layer => self.scene.pop_layer(),
                LayerKind::Clip => self.scene.pop_clip_path(),
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
        self.scene.fill_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn draw_glyphs<'a, 's2: 'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        _brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph>,
    ) {
        let paint = self.convert_paint(paint.into());
        self.scene.set_paint(paint);
        self.scene.set_transform(transform);

        fn into_vello_glyph(g: Glyph) -> vello_common::glyph::Glyph {
            vello_common::glyph::Glyph {
                id: g.id,
                x: g.x,
                y: g.y,
            }
        }

        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.scene.set_fill_rule(fill);
                self.scene
                    .glyph_run(font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(into_vello_glyph));
            }
            StyleRef::Stroke(stroke) => {
                self.scene.set_stroke(stroke.clone());
                self.scene
                    .glyph_run(font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(into_vello_glyph));
            }
        }
    }

    fn draw_box_shadow(
        &mut self,
        _transform: Affine,
        _rect: Rect,
        _color: Color,
        _radius: f64,
        _std_dev: f64,
    ) {
        // Not yet supported in vello_hybrid WebGL.
    }
}
