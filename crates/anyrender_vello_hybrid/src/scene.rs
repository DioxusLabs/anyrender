use anyrender::{NormalizedCoord, Paint, PaintRef, PaintScene};
use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, ImageData, StyleRef};
use rustc_hash::FxHashMap;
use vello_common::paint::{ImageId, ImageSource, PaintType};
use vello_hybrid::Renderer;
use wgpu::{CommandEncoder, Device, Queue};

const DEFAULT_TOLERANCE: f64 = 0.1;

fn anyrender_paint_to_vello_hybrid_paint<'a>(
    paint: PaintRef<'a>,
    image_manager: &mut ImageManager<'_>,
) -> PaintType {
    match paint {
        Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
        Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),

        Paint::Image(image_brush) => {
            let image_id = image_manager.upload_image(image_brush.image);
            PaintType::Image(ImageBrush {
                image: ImageSource::OpaqueId(image_id),
                sampler: image_brush.sampler,
            })
        }

        // TODO: custom paint
        Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
    }
}

pub struct ImageManager<'a> {
    pub(crate) renderer: &'a mut Renderer,
    pub(crate) device: &'a Device,
    pub(crate) queue: &'a Queue,
    pub(crate) encoder: &'a mut CommandEncoder,
    pub(crate) cache: &'a mut FxHashMap<u64, ImageId>,
}

impl<'a> ImageManager<'a> {
    pub fn new(
        renderer: &'a mut Renderer,
        device: &'a Device,
        queue: &'a Queue,
        encoder: &'a mut CommandEncoder,
        cache: &'a mut FxHashMap<u64, ImageId>,
    ) -> Self {
        Self {
            renderer,
            device,
            queue,
            encoder,
            cache,
        }
    }

    pub(crate) fn upload_image(&mut self, image: &ImageData) -> ImageId {
        let peniko_id = image.data.id();

        // Try to get ImageId from cache first
        if let Some(atlas_id) = self.cache.get(&peniko_id) {
            return *atlas_id;
        };

        // Convert ImageData to Pixmap
        let ImageSource::Pixmap(pixmap) = ImageSource::from_peniko_image_data(image) else {
            unreachable!(); // ImageSource::from_peniko_image_data always return a Pixmap
        };

        // Upload Pixamp
        let atlas_id = self
            .renderer
            .upload_image(self.device, self.queue, self.encoder, &pixmap);

        // Store ImageId in cache
        self.cache.insert(peniko_id, atlas_id);

        // Return ImageId
        atlas_id
    }
}

pub(crate) enum LayerKind {
    Layer,
    Clip,
}

pub struct VelloHybridScenePainter<'s> {
    pub(crate) scene: &'s mut vello_hybrid::Scene,
    pub(crate) layer_stack: Vec<LayerKind>,
    pub(crate) image_manager: ImageManager<'s>,
}

impl VelloHybridScenePainter<'_> {
    pub fn new<'s>(
        scene: &'s mut vello_hybrid::Scene,
        image_manager: ImageManager<'s>,
    ) -> VelloHybridScenePainter<'s> {
        VelloHybridScenePainter {
            scene,
            layer_stack: Vec::with_capacity(16),
            image_manager,
        }
    }
}

impl PaintScene for VelloHybridScenePainter<'_> {
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
        let paint = anyrender_paint_to_vello_hybrid_paint(paint.into(), &mut self.image_manager);
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
        let paint = anyrender_paint_to_vello_hybrid_paint(paint.into(), &mut self.image_manager);
        self.scene.set_paint(paint);
        self.scene
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.scene.fill_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn draw_glyphs<'a, 's: 'a>(
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
        glyphs: impl Iterator<Item = anyrender::Glyph>,
    ) {
        let paint = anyrender_paint_to_vello_hybrid_paint(paint.into(), &mut self.image_manager);
        self.scene.set_paint(paint);
        self.scene.set_transform(transform);

        fn into_vello_hybrid_glyph(g: anyrender::Glyph) -> vello_common::glyph::Glyph {
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
                    .fill_glyphs(glyphs.map(into_vello_hybrid_glyph));
            }
            StyleRef::Stroke(stroke) => {
                self.scene.set_stroke(stroke.clone());
                self.scene
                    .glyph_run(font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(into_vello_hybrid_glyph));
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
        // FIXME: implement once supported in vello_hybrid
        //
        // self.scene.set_transform(transform);
        // self.scene.set_paint(PaintType::Solid(color));
        // self.scene
        //     .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32);
    }
}
