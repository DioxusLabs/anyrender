use anyrender::{
    CustomPaint, ImageResource, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext,
    ResourceId,
};
use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, ImageData, StyleRef};
use rustc_hash::FxHashMap;
use vello::Renderer as VelloRenderer;

use crate::{CustomPaintSource, custom_paint_source::CustomPaintCtx};

pub struct VelloRenderContext {
    pub(crate) resource_map: FxHashMap<ResourceId, ImageData>,
    next_resource_id: u64,
}

impl VelloRenderContext {
    pub fn new() -> Self {
        Self {
            resource_map: FxHashMap::default(),
            next_resource_id: 0,
        }
    }
}

impl Default for VelloRenderContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderContext for VelloRenderContext {
    fn register_image(&mut self, image: ImageData) -> ImageResource {
        let resource_id = ResourceId(self.next_resource_id);
        self.next_resource_id += 1;
        let width = image.width;
        let height = image.height;
        self.resource_map.insert(resource_id, image);
        ImageResource {
            id: resource_id,
            width,
            height,
        }
    }

    fn unregister_resource(&mut self, id: ResourceId) {
        self.resource_map.remove(&id);
    }
}

pub struct VelloScenePainter<'r, 's> {
    pub(crate) ctx: &'r VelloRenderContext,
    pub(crate) renderer: Option<&'r mut VelloRenderer>,
    pub(crate) custom_paint_sources: Option<&'r mut FxHashMap<u64, Box<dyn CustomPaintSource>>>,
    pub(crate) inner: &'s mut vello::Scene,
}

impl VelloScenePainter<'_, '_> {
    pub fn new<'r, 's>(
        ctx: &'r VelloRenderContext,
        scene: &'s mut vello::Scene,
    ) -> VelloScenePainter<'r, 's> {
        VelloScenePainter {
            ctx,
            renderer: None,
            custom_paint_sources: None,
            inner: scene,
        }
    }

    fn render_custom_source(&mut self, custom_paint: CustomPaint) -> Option<peniko::ImageBrush> {
        let (Some(renderer), Some(custom_paint_sources)) =
            (&mut self.renderer, &mut self.custom_paint_sources)
        else {
            return None;
        };

        let CustomPaint {
            source_id,
            width,
            height,
            scale,
        } = custom_paint;

        // Render custom paint source
        let source = custom_paint_sources.get_mut(&source_id)?;
        let ctx = CustomPaintCtx::new(renderer);
        let texture_handle = source.render(ctx, width, height, scale)?;

        // Return dummy image
        Some(ImageBrush::new(texture_handle.0))
    }

    /// Convert a PaintRef to an owned peniko::Brush, looking up image resources from the context.
    fn paint_to_brush(&self, paint: PaintRef<'_>) -> Option<peniko::Brush> {
        Some(match paint {
            Paint::Solid(color) => peniko::Brush::Solid(color),
            Paint::Gradient(gradient) => peniko::Brush::Gradient(gradient.clone()),
            Paint::Image(image_brush) => {
                let image_data = self.ctx.resource_map[&image_brush.image.id].clone();
                peniko::Brush::Image(ImageBrush {
                    image: image_data,
                    sampler: image_brush.sampler,
                })
            }
            Paint::Custom(_) => return None,
        })
    }
}

impl PaintScene for VelloScenePainter<'_, '_> {
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
    ) {
        self.inner
            .push_layer(Fill::NonZero, blend, alpha, transform, clip);
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        self.inner.push_clip_layer(Fill::NonZero, transform, clip);
    }

    fn pop_layer(&mut self) {
        self.inner.pop_layer();
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint_ref: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let Some(brush) = self.paint_to_brush(paint_ref.into()) else {
            return;
        };
        self.inner
            .stroke(style, transform, &brush, brush_transform, shape);
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let paint: PaintRef<'_> = paint.into();

        let brush: peniko::Brush = match paint {
            Paint::Solid(color) => peniko::Brush::Solid(color),
            Paint::Gradient(gradient) => peniko::Brush::Gradient(gradient.clone()),
            Paint::Image(image_brush) => {
                let image_data = self.ctx.resource_map[&image_brush.image.id].clone();
                peniko::Brush::Image(ImageBrush {
                    image: image_data,
                    sampler: image_brush.sampler,
                })
            }
            Paint::Custom(custom_paint) => {
                let Some(custom_paint) = custom_paint.downcast_ref::<CustomPaint>() else {
                    return;
                };
                let Some(image) = self.render_custom_source(*custom_paint) else {
                    return;
                };
                peniko::Brush::Image(image)
            }
        };

        self.inner
            .fill(style, transform, &brush, brush_transform, shape);
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = anyrender::Glyph>,
    ) {
        let paint: PaintRef<'_> = paint.into();
        let resource_map = &self.ctx.resource_map;

        let glyph_iter = glyphs.map(|g: anyrender::Glyph| vello::Glyph {
            id: g.id,
            x: g.x,
            y: g.y,
        });

        let mut glyph_renderer = self
            .inner
            .draw_glyphs(font)
            .font_size(font_size)
            .hint(hint)
            .normalized_coords(normalized_coords)
            .brush_alpha(brush_alpha)
            .transform(transform)
            .glyph_transform(glyph_transform);

        match paint {
            Paint::Solid(color) => {
                glyph_renderer = glyph_renderer.brush(peniko::Brush::Solid(color))
            }
            Paint::Gradient(gradient) => {
                glyph_renderer = glyph_renderer.brush(peniko::Brush::Gradient(gradient))
            }
            Paint::Image(image_brush) => {
                let image_data = &resource_map[&image_brush.image.id];
                let brush = ImageBrush {
                    image: image_data,
                    sampler: image_brush.sampler,
                };
                glyph_renderer = glyph_renderer.brush(brush);
            }
            Paint::Custom(_) => {}
        }

        glyph_renderer.draw(style, glyph_iter);
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        brush: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.inner
            .draw_blurred_rounded_rect(transform, rect, brush, radius, std_dev);
    }
}
