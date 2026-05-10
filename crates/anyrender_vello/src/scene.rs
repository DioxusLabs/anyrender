use anyrender::{NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext, ResourceId};
use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, BrushRef, Color, Fill, FontData, ImageBrush, ImageData, StyleRef};
use rustc_hash::FxHashMap;
use vello::Renderer as VelloRenderer;
use wgpu::Texture;
use wgpu_context::DeviceHandle;

pub struct VelloScenePainter<'r, 's> {
    pub(crate) renderer: Option<&'r mut VelloRenderer>,
    pub(crate) device_handle: Option<&'r DeviceHandle>,
    pub(crate) texture_handles: Option<&'r mut FxHashMap<ResourceId, ImageData>>,
    pub(crate) inner: &'s mut vello::Scene,
}

impl RenderContext for VelloScenePainter<'_, '_> {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn std::any::Any>,
    ) -> Result<ResourceId, anyrender::RegisterResourceError> {
        if let Some(renderer) = &mut self.renderer
            && let Some(texture_handles) = &mut self.texture_handles
        {
            if let Ok(texture) = resource.downcast::<Texture>() {
                let id = ResourceId::new();
                texture_handles.insert(id, renderer.register_texture(*texture));
                Ok(id)
            } else {
                Err(anyrender::RegisterResourceErrorKind::UnsupportedResourceKind.into())
            }
        } else {
            Err(anyrender::RegisterResourceErrorKind::Unimplemented.into())
        }
    }

    fn unregister_resource(&mut self, resource_id: ResourceId) {
        if let Some(renderer) = &mut self.renderer
            && let Some(texture_handles) = &mut self.texture_handles
            && let Some(handle) = texture_handles.remove(&resource_id)
        {
            renderer.unregister_texture(handle);
        }
    }

    fn renderer_specific_context(&self) -> Option<Box<dyn std::any::Any>> {
        self.device_handle
            .map(|device_handle| Box::new(device_handle.clone()) as _)
    }
}

impl VelloScenePainter<'_, '_> {
    pub fn new<'s>(scene: &'s mut vello::Scene) -> VelloScenePainter<'static, 's> {
        VelloScenePainter {
            renderer: None,
            device_handle: None,
            texture_handles: None,
            inner: scene,
        }
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
        let paint_ref: PaintRef<'_> = paint_ref.into();
        let brush_ref: BrushRef<'_> = paint_ref.into();
        self.inner
            .stroke(style, transform, brush_ref, brush_transform, shape);
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
        let brush_ref: BrushRef<'_> = match paint {
            Paint::Solid(color) => BrushRef::Solid(color),
            Paint::Gradient(gradient) => BrushRef::Gradient(gradient),
            Paint::Image(image) => BrushRef::Image(image),
            Paint::Resource(brush) => {
                let resource_id = brush.image;
                if let Some(texture_handle) = self
                    .texture_handles
                    .as_ref()
                    .and_then(|texture_handles| texture_handles.get(&resource_id))
                {
                    peniko::Brush::Image(ImageBrush {
                        image: texture_handle,
                        sampler: brush.sampler,
                    })
                } else {
                    BrushRef::Solid(Color::TRANSPARENT)
                }
            }
            Paint::Custom(_) => BrushRef::Solid(Color::TRANSPARENT),
        };

        self.inner
            .fill(style, transform, brush_ref, brush_transform, shape);
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
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = anyrender::Glyph>,
    ) {
        self.inner
            .draw_glyphs(font)
            .font_size(font_size)
            .hint(hint)
            .normalized_coords(normalized_coords)
            .font_embolden(vello::FontEmbolden::new(kurbo::Diagonal2::new(
                embolden.x, embolden.y,
            )))
            .brush(paint.into())
            .brush_alpha(brush_alpha)
            .transform(transform)
            .glyph_transform(glyph_transform)
            .draw(
                style,
                glyphs.map(|g: anyrender::Glyph| vello::Glyph {
                    id: g.id,
                    x: g.x,
                    y: g.y,
                }),
            );
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
