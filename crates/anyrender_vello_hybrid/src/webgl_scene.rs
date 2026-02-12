//! WebGL-compatible [`PaintScene`] implementation for [`vello_hybrid::Scene`].

use anyrender::{
    Glyph, ImageResource, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext, ResourceId,
};
use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, ImageData, StyleRef};
use rustc_hash::FxHashMap;
use vello_common::paint::{ImageId, ImageSource, PaintType};

const DEFAULT_TOLERANCE: f64 = 0.1;

pub struct WebGlRenderContext {
    resource_map: FxHashMap<ResourceId, ImageId>,
    next_id: u64,
    pending_uploads: Vec<(ResourceId, ImageData)>,
}

impl WebGlRenderContext {
    pub fn new() -> Self {
        Self {
            resource_map: FxHashMap::default(),
            next_id: 0,
            pending_uploads: Vec::new(),
        }
    }
}

impl Default for WebGlRenderContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderContext for WebGlRenderContext {
    fn register_image(&mut self, image: ImageData) -> ImageResource {
        let resource_id = ResourceId(self.next_id);
        self.next_id += 1;
        let width = image.width;
        let height = image.height;
        self.pending_uploads.push((resource_id, image));
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

enum LayerKind {
    Layer,
    Clip,
}

pub struct WebGlScenePainter<'s> {
    ctx: &'s mut WebGlRenderContext,
    renderer: &'s mut vello_hybrid::WebGlRenderer,
    scene: &'s mut vello_hybrid::Scene,
    layer_stack: Vec<LayerKind>,
}

impl<'s> WebGlScenePainter<'s> {
    pub fn new(
        ctx: &'s mut WebGlRenderContext,
        renderer: &'s mut vello_hybrid::WebGlRenderer,
        scene: &'s mut vello_hybrid::Scene,
    ) -> Self {
        Self {
            ctx,
            renderer,
            scene,
            layer_stack: Vec::with_capacity(16),
        }
    }
}

impl WebGlScenePainter<'_> {
    /// Upload any images that have been registered with the context but not yet
    /// sent to the WebGL renderer. Called automatically before resolving image paints.
    fn flush_pending_uploads(&mut self) {
        if self.ctx.pending_uploads.is_empty() {
            return;
        }

        for (resource_id, image_data) in self.ctx.pending_uploads.drain(..) {
            let ImageSource::Pixmap(pixmap) = ImageSource::from_peniko_image_data(&image_data)
            else {
                unreachable!();
            };

            let image_id = self.renderer.upload_image(&pixmap);
            self.ctx.resource_map.insert(resource_id, image_id);
        }
    }

    fn convert_paint(&mut self, paint: PaintRef<'_>) -> PaintType {
        match paint {
            Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
            Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
            Paint::Image(image_brush) => {
                self.flush_pending_uploads();
                let image_id = self.ctx.resource_map[&image_brush.image.id];
                PaintType::Image(ImageBrush {
                    image: ImageSource::OpaqueId(image_id),
                    sampler: image_brush.sampler,
                })
            }
            Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        }
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
