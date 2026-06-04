use std::sync::Arc;

use anyrender::{
    Filter, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext,
    filters::{EdgeMode, FilterEffect},
};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, StyleRef};
use vello_common::filter_effects::FilterPrimitive;
use vello_cpu::{ImageSource, PaintType, Pixmap};

const DEFAULT_TOLERANCE: f64 = 0.1;

fn anyrender_paint_to_vello_cpu_paint<'a>(paint: PaintRef<'a>) -> PaintType {
    match paint {
        Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
        Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
        Paint::Image(image) => PaintType::Image(ImageBrush {
            #[cfg(not(feature = "experimental_image_cache"))]
            image: ImageSource::from_peniko_image_data(image.image),
            #[cfg(feature = "experimental_image_cache")]
            image: convert_image_cached(image.image),
            sampler: image.sampler,
        }),
        // TODO: custom paint
        Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
    }
}

#[cfg(feature = "experimental_image_cache")]
fn convert_image_cached(image: &peniko::ImageData) -> ImageSource {
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

pub struct VelloCpuScenePainter {
    pub render_ctx: vello_cpu::RenderContext,
    pub resources: vello_cpu::Resources,
}

impl VelloCpuScenePainter {
    pub fn finish(mut self) -> Pixmap {
        let mut pixmap = Pixmap::new(self.render_ctx.width(), self.render_ctx.height());
        self.render_ctx
            .render_to_pixmap(&mut self.resources, &mut pixmap);
        pixmap
    }
}

impl RenderContext for VelloCpuScenePainter {}
impl PaintScene for VelloCpuScenePainter {
    fn reset(&mut self) {
        self.render_ctx.reset();
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        filter: Option<Arc<Filter>>,
        _backdrop_filter: Option<Arc<Filter>>,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.push_layer(
            Some(&clip.into_path(DEFAULT_TOLERANCE)),
            Some(blend.into()),
            Some(alpha),
            None,
            filter.and_then(convert_filter),
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
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));
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
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));
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
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        _brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = anyrender::Glyph> + Clone,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));

        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.render_ctx.set_fill_rule(fill);
                self.render_ctx
                    .glyph_run(&mut self.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)))
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(|g| vello_cpu::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
            StyleRef::Stroke(stroke) => {
                self.render_ctx.set_stroke(stroke.clone());
                self.render_ctx
                    .glyph_run(&mut self.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(|g| vello_cpu::Glyph {
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

fn convert_filter(filter: Arc<Filter>) -> Option<vello_common::filter_effects::Filter> {
    let nodes = filter.nodes();
    if nodes.is_empty() {
        return None;
    }

    // Vello CPU only supports single-node filters at the moment
    let node = &filter.nodes()[0];
    let primitive = convert_filter_effect(&node.effect)?;
    Some(vello_common::filter_effects::Filter::from_primitive(
        primitive,
    ))
}

fn convert_filter_effect(effect: &FilterEffect) -> Option<FilterPrimitive> {
    Some(match effect {
        FilterEffect::Flood(color) => FilterPrimitive::Flood { color: *color },
        FilterEffect::GaussianBlur(blur) => FilterPrimitive::GaussianBlur {
            std_deviation: blur.std_deviation,
            edge_mode: convert_edge_mode(blur.edge_mode),
        },
        FilterEffect::DropShadow(shadow) => FilterPrimitive::DropShadow {
            dx: shadow.dx,
            dy: shadow.dy,
            std_deviation: shadow.std_deviation,
            color: shadow.color,
            edge_mode: convert_edge_mode(shadow.edge_mode),
        },
        FilterEffect::Offset(offset) => FilterPrimitive::Offset {
            dx: offset.x as f32,
            dy: offset.y as f32,
        },
        FilterEffect::ColorMatrix(matrix) => FilterPrimitive::ColorMatrix { matrix: matrix.0 },
        FilterEffect::ComponentTransfer(_component_transfer_filter) => return None,
        FilterEffect::Blend(mode) => FilterPrimitive::Blend { mode: *mode },
        FilterEffect::Composite(_composite_operator) => return None,
        FilterEffect::Morphology(_morphology_filter) => return None,
        FilterEffect::ConvolveMatrix(_convolution_kernel) => return None,
        FilterEffect::Turbulence(_turbulence_filter) => return None,
        FilterEffect::DisplacementMap(_displacement_map_filter) => return None,
        FilterEffect::Image(_external_image_source) => return None,
        FilterEffect::Tile => return None,
        FilterEffect::DiffuseLighting(_diffuse_lighting_filter) => return None,
        FilterEffect::SpecularLighting(_specular_lighting_filter) => return None,
    })
}

fn convert_edge_mode(edge_mode: EdgeMode) -> vello_common::filter_effects::EdgeMode {
    match edge_mode {
        EdgeMode::Duplicate => vello_common::filter_effects::EdgeMode::Duplicate,
        EdgeMode::Wrap => vello_common::filter_effects::EdgeMode::Wrap,
        EdgeMode::Mirror => vello_common::filter_effects::EdgeMode::Mirror,
        EdgeMode::None => vello_common::filter_effects::EdgeMode::None,
    }
}
