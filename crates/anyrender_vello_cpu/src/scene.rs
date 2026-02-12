use anyrender::{
    ImageResource, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext, ResourceId,
};
use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, ImageData, StyleRef};
use std::collections::HashMap;
use vello_cpu::{ImageSource, PaintType, Pixmap};

const DEFAULT_TOLERANCE: f64 = 0.1;

pub struct VelloCpuRenderContext {
    pub(crate) resource_map: HashMap<ResourceId, ImageSource>,
    next_resource_id: u64,
}

impl VelloCpuRenderContext {
    pub fn new() -> Self {
        Self {
            resource_map: HashMap::new(),
            next_resource_id: 0,
        }
    }
}

impl Default for VelloCpuRenderContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderContext for VelloCpuRenderContext {
    fn register_image(&mut self, image: ImageData) -> ImageResource {
        let resource_id = ResourceId(self.next_resource_id);
        self.next_resource_id += 1;

        let image_source = ImageSource::from_peniko_image_data(&image);
        self.resource_map.insert(resource_id, image_source);

        ImageResource {
            id: resource_id,
            width: image.width,
            height: image.height,
        }
    }

    fn unregister_resource(&mut self, id: ResourceId) {
        self.resource_map.remove(&id);
    }
}

fn anyrender_paint_to_vello_cpu_paint(
    paint: PaintRef<'_>,
    ctx: &VelloCpuRenderContext,
) -> PaintType {
    match paint {
        Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
        Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
        Paint::Image(image) => PaintType::Image(ImageBrush {
            image: ctx.resource_map[&image.image.id].clone(),
            sampler: image.sampler,
        }),
        // TODO: custom paint
        Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
    }
}

pub struct VelloCpuScenePainter<'a> {
    pub(crate) ctx: &'a VelloCpuRenderContext,
    pub render_ctx: &'a mut vello_cpu::RenderContext,
}

impl<'a> VelloCpuScenePainter<'a> {
    pub fn new(
        ctx: &'a VelloCpuRenderContext,
        render_ctx: &'a mut vello_cpu::RenderContext,
    ) -> Self {
        Self { ctx, render_ctx }
    }

    pub fn finish(self) -> Pixmap {
        let mut pixmap = Pixmap::new(self.render_ctx.width(), self.render_ctx.height());
        self.render_ctx.render_to_pixmap(&mut pixmap);
        pixmap
    }
}

impl PaintScene for VelloCpuScenePainter<'_> {
    fn reset(&mut self) {
        self.render_ctx.reset();
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.push_layer(
            Some(&clip.into_path(DEFAULT_TOLERANCE)),
            Some(blend.into()),
            Some(alpha),
            None,
            None,
        );
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        self.render_ctx.set_transform(transform);
        self.render_ctx
            .push_clip_layer(&clip.into_path(DEFAULT_TOLERANCE));
    }

    fn pop_layer(&mut self) {
        self.render_ctx.pop_layer();
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_stroke(style.clone());
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into(), self.ctx));
        self.render_ctx
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.render_ctx
            .stroke_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(style);
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into(), self.ctx));
        self.render_ctx
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.render_ctx
            .fill_path(&shape.into_path(DEFAULT_TOLERANCE));
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
        self.render_ctx.set_transform(transform);
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into(), self.ctx));

        fn into_vello_cpu_glyph(g: anyrender::Glyph) -> vello_cpu::Glyph {
            vello_cpu::Glyph {
                id: g.id,
                x: g.x,
                y: g.y,
            }
        }

        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.render_ctx.set_fill_rule(fill);
                self.render_ctx
                    .glyph_run(font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(into_vello_cpu_glyph));
            }
            StyleRef::Stroke(stroke) => {
                self.render_ctx.set_stroke(stroke.clone());
                self.render_ctx
                    .glyph_run(font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(into_vello_cpu_glyph));
            }
        }
    }
    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        color: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_paint(PaintType::Solid(color));
        self.render_ctx
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32);
    }
}
