use std::sync::Arc;

use anyrender::{Filter, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext, ResourceId};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, ImageData, StyleRef};
use rustc_hash::FxHashMap;
use vello_common::{
    TextureId,
    geometry::RectU16,
    paint::{ImageId, ImageSource, PaintType},
};
use vello_hybrid::{Renderer, Resources, SampleRect};
use wgpu::{CommandEncoder, Device, Queue, Texture, TextureView, TextureViewDescriptor};
use wgpu_context::DeviceHandle;

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
                image: ImageSource::OpaqueId {
                    id: image_id,
                    // TODO: optimize opaque case
                    may_have_transparency: true,
                },
                sampler: image_brush.sampler,
            })
        }

        // TODO: custom paint
        Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
    }
}

pub struct ImageManager<'a> {
    pub(crate) renderer: &'a mut Renderer,
    pub(crate) resources: &'a mut Resources,
    pub(crate) device: &'a Device,
    pub(crate) queue: &'a Queue,
    pub(crate) encoder: &'a mut CommandEncoder,
    pub(crate) cache: &'a mut FxHashMap<u64, ImageId>,
}

impl<'a> ImageManager<'a> {
    pub fn new(
        renderer: &'a mut Renderer,
        resources: &'a mut Resources,
        device: &'a Device,
        queue: &'a Queue,
        encoder: &'a mut CommandEncoder,
        cache: &'a mut FxHashMap<u64, ImageId>,
    ) -> Self {
        Self {
            renderer,
            resources,
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
        let atlas_id = self.renderer.upload_image(
            self.resources,
            self.device,
            self.queue,
            self.encoder,
            &pixmap,
        );

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
    pub(crate) texture_bindings: &'s mut FxHashMap<ResourceId, TextureView>,
    pub(crate) device_handle: &'s DeviceHandle,
}

impl VelloHybridScenePainter<'_> {
    pub fn new<'s>(
        scene: &'s mut vello_hybrid::Scene,
        image_manager: ImageManager<'s>,
        texture_bindings: &'s mut FxHashMap<ResourceId, TextureView>,
        device_handle: &'s DeviceHandle,
    ) -> VelloHybridScenePainter<'s> {
        VelloHybridScenePainter {
            scene,
            layer_stack: Vec::with_capacity(16),
            image_manager,
            texture_bindings,
            device_handle,
        }
    }
}

impl RenderContext for VelloHybridScenePainter<'_> {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn std::any::Any>,
    ) -> Result<ResourceId, anyrender::RegisterResourceError> {
        // Try to downcast as Texture
        match resource.downcast::<Texture>() {
            Ok(texture) => {
                let id = ResourceId::new();
                let texture_view = texture.create_view(&TextureViewDescriptor::default());
                self.texture_bindings.insert(id, texture_view);
                Ok(id)
            }
            Err(resource) => {
                // Else try to downcast as TextureView
                if let Ok(texture_view) = resource.downcast::<TextureView>() {
                    let id = ResourceId::new();
                    self.texture_bindings.insert(id, *texture_view);
                    Ok(id)
                }
                // Else return error
                else {
                    Err(anyrender::RegisterResourceErrorKind::UnsupportedResourceKind.into())
                }
            }
        }
    }

    fn unregister_resource(&mut self, resource_id: ResourceId) {
        self.texture_bindings.remove(&resource_id);
    }

    fn renderer_specific_context(&self) -> Option<Box<dyn std::any::Any>> {
        Some(Box::new(self.device_handle.clone()) as _)
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
        filter: Option<Arc<Filter>>,
        _backdrop_filter: Option<Arc<Filter>>,
    ) {
        let filter = filter.and_then(crate::filters::convert_filter);
        self.scene.set_transform(transform);
        self.layer_stack.push(LayerKind::Layer);
        self.scene
            .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
        self.scene
            .push_layer(None, Some(blend.into()), Some(alpha), None, filter);
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
                LayerKind::Layer => {
                    self.scene.pop_layer();
                    self.scene.pop_clip_path();
                }
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
        let paint = paint.into();

        match paint {
            Paint::Resource(brush) => {
                if let Some(texture_view) = self.texture_bindings.get(&brush.image) {
                    let texture_id = TextureId(brush.image.into_ffi());

                    let src_width = texture_view.texture().width();
                    let src_height = texture_view.texture().height();

                    let rect = shape.bounding_box();

                    self.scene.draw_texture_rects(
                        texture_id,
                        brush.sampler.quality,
                        [SampleRect {
                            source_region: RectU16 {
                                x0: 0,
                                y0: 0,
                                x1: src_width as u16,
                                y1: src_height as u16,
                            },
                            transform: Affine::translate(rect.origin().to_vec2()),
                        }],
                    );
                }
            }
            _ => {
                let paint = anyrender_paint_to_vello_hybrid_paint(paint, &mut self.image_manager);
                self.scene.set_paint(paint);
                self.scene
                    .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
                self.scene.fill_path(&shape.into_path(DEFAULT_TOLERANCE));
            }
        }
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
        let paint = anyrender_paint_to_vello_hybrid_paint(paint.into(), &mut self.image_manager);
        self.scene.set_paint(paint);
        self.scene.set_transform(transform);

        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.scene.set_fill_rule(fill);
                self.scene
                    .glyph_run(self.image_manager.resources, font)
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
                self.scene
                    .glyph_run(self.image_manager.resources, font)
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
        rect: Rect,
        color: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.scene.set_transform(transform);
        self.scene.set_paint(PaintType::Solid(color));
        self.scene
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32);
    }
}
