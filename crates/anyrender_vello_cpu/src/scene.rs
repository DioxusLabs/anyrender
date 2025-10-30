use anyrender::{NormalizedCoord, Paint, PaintRef, PaintScene};
use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, ImageData, StyleRef};
use vello_cpu::{ImageSource, PaintType, Pixmap};

const DEFAULT_TOLERANCE: f64 = 0.1;

fn anyrender_paint_to_vello_cpu_paint<'a>(paint: PaintRef<'a>) -> PaintType {
    match paint {
        Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
        Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
        Paint::Image(image) => PaintType::Image(ImageBrush {
            image: ImageSource::from_peniko_image_data(image.image),
            sampler: image.sampler,
        }),
        // TODO: custom paint
        Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
    }
}

#[allow(unused)]
fn convert_image_cached(image: &ImageData) -> ImageSource {
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};
    static CACHE: LazyLock<Mutex<HashMap<u64, ImageSource>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    let mut map = CACHE.lock().unwrap();
    let id = image.data.id();
    map.entry(id)
        .or_insert_with(|| ImageSource::from_peniko_image_data(image))
        .clone()
}

pub struct VelloCpuScenePainter(pub vello_cpu::RenderContext);

impl VelloCpuScenePainter {
    pub fn finish(self) -> Pixmap {
        let mut pixmap = Pixmap::new(self.0.width(), self.0.height());
        self.0.render_to_pixmap(&mut pixmap);
        pixmap
    }
}

impl PaintScene for VelloCpuScenePainter {
    fn reset(&mut self) {
        self.0.reset();
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
    ) {
        self.0.set_transform(transform);
        self.0.push_layer(
            Some(&clip.into_path(DEFAULT_TOLERANCE)),
            Some(blend.into()),
            Some(alpha),
            None,
        );
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        self.0.set_transform(transform);
        self.0.push_clip_layer(&clip.into_path(DEFAULT_TOLERANCE));
    }

    fn pop_layer(&mut self) {
        self.0.pop_layer();
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.0.set_transform(transform);
        self.0.set_stroke(style.clone());
        self.0
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));
        self.0
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.0.stroke_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.0.set_transform(transform);
        self.0.set_fill_rule(style);
        self.0
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));
        self.0
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.0.fill_path(&shape.into_path(DEFAULT_TOLERANCE));
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
        self.0.set_transform(transform);
        self.0
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));

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
                self.0.set_fill_rule(fill);
                self.0
                    .glyph_run(font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(into_vello_cpu_glyph));
            }
            StyleRef::Stroke(stroke) => {
                self.0.set_stroke(stroke.clone());
                self.0
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
        self.0.set_transform(transform);
        self.0.set_paint(PaintType::Solid(color));
        self.0
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32);
    }
}
