use std::sync::Arc;

use crate::common::{DEFAULT_TOLERANCE, ImageCache, LayerKind, supports_glyph_caching};
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
use vello_gpu::{Renderer, Resources};
use wgpu::{CommandEncoder, Device, Queue, Texture, TextureView, TextureViewDescriptor};
use wgpu_context::DeviceHandle;

/// Snaps glyph origins to the horizontal subpixel grid of the platform's native text renderer.
///
/// CoreGraphics (and so Safari and other native Apple text) floors each glyph origin's device x
/// to a grid that depends on the device font size: 1/4px at 9-11ppem, 1/3px at 12-16ppem, 1/2px
/// at 17-33ppem and whole pixels from 34ppem. Vello positions glyphs continuously, so without
/// this, stems land up to 1/2px away from native ones. Elsewhere, x is rounded to the nearest
/// 1/4px, as Skia does for horizontal text (and so Chrome off Apple). On Apple, CoreGraphics
/// also floors the device y of the baseline to a whole pixel (y up, so the baseline moves down
/// in Vello's y-down device space); off Apple, y is not snapped.
///
/// CoreGraphics font smoothing also dilates glyphs upwards only: stems grow by the same amount
/// on both sides in x, but the bottom edge stays on the baseline and the top grows by twice
/// that. Vello's embolden grows both sides in y, so emboldened glyphs are raised by the
/// vertical embolden amount, which keeps baselines and other bottom edges as crisp as native.
#[derive(Clone, Copy)]
pub(crate) struct GlyphOriginSnap {
    scale_x: f64,
    offset_x: f64,
    positions_per_px: f64,
    /// Skia rounds to the nearest grid position; CoreGraphics floors.
    round_x: f64,
    /// The transform's y scale and offset when device y is snapped, as on Apple.
    snap_y: Option<(f64, f64)>,
    lift_y: f64,
}

impl GlyphOriginSnap {
    pub(crate) fn new(transform: Affine, font_size: f32, embolden: kurbo::Vec2) -> Option<Self> {
        if cfg!(any(target_os = "macos", target_os = "ios")) {
            Self::core_graphics(transform, font_size, embolden)
        } else {
            Self::skia(transform)
        }
    }

    fn skia(transform: Affine) -> Option<Self> {
        let [a, b, c, _, e, _] = transform.as_coeffs();
        if b != 0.0 || c != 0.0 || a <= 0.0 {
            return None;
        }
        Some(Self {
            scale_x: a,
            offset_x: e,
            positions_per_px: 4.0,
            round_x: 0.5,
            snap_y: None,
            lift_y: 0.0,
        })
    }

    fn core_graphics(transform: Affine, font_size: f32, embolden: kurbo::Vec2) -> Option<Self> {
        let [a, b, c, d, e, f] = transform.as_coeffs();
        let ppem = font_size as f64 * d.abs();
        if b != 0.0 || c != 0.0 || a <= 0.0 || !ppem.is_finite() || ppem <= 0.0 {
            return None;
        }
        Some(Self {
            scale_x: a,
            offset_x: e,
            // CoreGraphics' grid gets coarser as glyphs grow: n = ceil(100 / (3 * ppem)), at most 5.
            positions_per_px: (100.0 / (3.0 * ppem)).ceil().min(5.0),
            round_x: 0.0,
            snap_y: Some((d, f)),
            // glifo scales the embolden amount with the transform, so it is in user units.
            lift_y: embolden.y * d.signum(),
        })
    }

    pub(crate) fn x(self, x: f32) -> f32 {
        let device_x = self.scale_x * x as f64 + self.offset_x;
        let snapped =
            (device_x * self.positions_per_px + self.round_x).floor() / self.positions_per_px;
        ((snapped - self.offset_x) / self.scale_x) as f32
    }

    pub(crate) fn y(self, y: f32) -> f32 {
        let mut y = y as f64;
        if let Some((scale_y, offset_y)) = self.snap_y {
            // CoreGraphics tolerates about 1/1000 device px before moving to the next pixel.
            let snapped = (scale_y * y + offset_y - 1.0 / 1024.0).ceil();
            y = (snapped - offset_y) / scale_y;
        }
        (y - self.lift_y) as f32
    }
}

fn anyrender_paint_to_vello_hybrid_paint<'a>(
    paint: PaintRef<'a>,
    image_manager: &mut ImageManager<'_>,
    texture_bindings: &FxHashMap<ResourceId, TextureView>,
) -> PaintType {
    match paint {
        Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
        Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),

        Paint::Image(image_brush) => {
            let Some(image_id) = image_manager.upload_image(image_brush.image) else {
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

        Paint::Resource(brush) => match texture_bindings.get(&brush.image) {
            Some(texture_view) => {
                let texture = texture_view.texture();
                PaintType::Image(ImageBrush {
                    image: ImageSource::external_texture(
                        TextureId(brush.image.into_ffi()),
                        RectU16 {
                            x0: 0,
                            y0: 0,
                            x1: texture.width().min(u16::MAX as u32) as u16,
                            y1: texture.height().min(u16::MAX as u32) as u16,
                        },
                        true,
                    ),
                    sampler: brush.sampler,
                })
            }
            None => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        },

        // TODO: custom paint
        Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
    }
}

pub struct ImageManager<'a> {
    pub(crate) renderer: &'a mut Renderer,
    pub(crate) resources: &'a mut Resources,
    pub(crate) device: &'a Device,
    pub(crate) queue: &'a Queue,
    pub(crate) encoder: &'a mut CommandEncoder,
    pub(crate) cache: &'a mut ImageCache,
}

impl<'a> ImageManager<'a> {
    pub fn new(
        renderer: &'a mut Renderer,
        resources: &'a mut Resources,
        device: &'a Device,
        queue: &'a Queue,
        encoder: &'a mut CommandEncoder,
        cache: &'a mut ImageCache,
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

    pub(crate) fn upload_image(&mut self, image: &ImageData) -> Option<ImageId> {
        if !self.cache.fits(image.width, image.height) {
            return None;
        }
        let peniko_id = image.data.id();

        // Try to get ImageId from cache first
        if let Some(atlas_id) = self.cache.get(peniko_id) {
            return Some(atlas_id);
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
        self.cache
            .insert(&image.data, atlas_id, image.width, image.height);

        // Return ImageId
        Some(atlas_id)
    }

    /// Evict unused images after encoding the frame's rendering, before submitting the encoder.
    pub fn finish_frame(&mut self) {
        for id in self.cache.finish_frame() {
            self.renderer
                .destroy_image(self.resources, self.encoder, id);
        }
    }
}

pub struct VelloHybridScenePainter<'s> {
    pub(crate) scene: &'s mut vello_gpu::Scene,
    pub(crate) layer_stack: Vec<LayerKind>,
    pub(crate) image_manager: ImageManager<'s>,
    pub(crate) texture_bindings: &'s mut FxHashMap<ResourceId, TextureView>,
    pub(crate) device_handle: &'s DeviceHandle,
    pub(crate) glyph_caching: bool,
}

impl VelloHybridScenePainter<'_> {
    pub fn new<'s>(
        scene: &'s mut vello_gpu::Scene,
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

    fn convert_paint(&mut self, paint: PaintRef<'_>) -> PaintType {
        anyrender_paint_to_vello_hybrid_paint(paint, &mut self.image_manager, self.texture_bindings)
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
        fill: Fill,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        filter: Option<Arc<Filter>>,
        _backdrop_filter: Option<Arc<Filter>>,
    ) {
        let filter = filter.and_then(crate::common::convert_filter);
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
        let paint = self.convert_paint(paint.into());
        self.scene.set_paint(paint);
        self.scene.reset_paint_transform();
        self.scene.set_transform(transform);

        let glyph_caching =
            self.glyph_caching && supports_glyph_caching(transform, glyph_transform);
        let snap = GlyphOriginSnap::new(transform, font_size, embolden);
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
                        x: snap.map_or(g.x, |snap| snap.x(g.x)),
                        y: snap.map_or(g.y, |snap| snap.y(g.y)),
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
                        x: snap.map_or(g.x, |snap| snap.x(g.x)),
                        y: snap.map_or(g.y, |snap| snap.y(g.y)),
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
        self.scene.reset_paint_transform();
        self.scene
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32, false);
    }
}

#[cfg(test)]
mod glyph_origin_snap_tests {
    use super::GlyphOriginSnap;
    use kurbo::Affine;

    fn snapped_device_x(transform: Affine, font_size: f32, x: f32) -> f64 {
        let snap = GlyphOriginSnap::core_graphics(transform, font_size, kurbo::Vec2::ZERO).unwrap();
        transform.as_coeffs()[0] * snap.x(x) as f64 + transform.as_coeffs()[4]
    }

    #[test]
    fn grid_depends_on_device_ppem() {
        for (ppem, step) in [
            (8.0, 0.2),
            (11.0, 0.25),
            (12.0, 1.0 / 3.0),
            (16.0, 1.0 / 3.0),
            (17.0, 0.5),
            (33.0, 0.5),
            (34.0, 1.0),
            (72.0, 1.0),
        ] {
            let snap =
                GlyphOriginSnap::core_graphics(Affine::IDENTITY, ppem, kurbo::Vec2::ZERO).unwrap();
            assert_eq!(snap.positions_per_px, (1.0f64 / step).round(), "{ppem}ppem");
        }
    }

    #[test]
    fn floors_device_x() {
        assert_eq!(snapped_device_x(Affine::IDENTITY, 11.0, 10.7), 10.5);
        assert_eq!(snapped_device_x(Affine::IDENTITY, 16.0, 10.3), 10.0);
        assert_eq!(snapped_device_x(Affine::IDENTITY, 40.0, 10.9), 10.0);
        // 8px at 2x is 16 device ppem, so a 1/3 device px grid.
        let transform = Affine::translate((3.0, 0.0)) * Affine::scale(2.0);
        assert!((snapped_device_x(transform, 8.0, 5.2) - 13.333333).abs() < 1e-4);
    }

    #[test]
    fn rounds_device_x_to_quarter_px_for_skia() {
        let device_x = |transform: Affine, x: f32| {
            let snap = GlyphOriginSnap::skia(transform).unwrap();
            transform.as_coeffs()[0] * snap.x(x) as f64 + transform.as_coeffs()[4]
        };
        assert_eq!(device_x(Affine::IDENTITY, 10.1), 10.0);
        assert_eq!(device_x(Affine::IDENTITY, 10.2), 10.25);
        assert_eq!(device_x(Affine::IDENTITY, 10.9), 11.0);
        let transform = Affine::translate((3.0, 0.0)) * Affine::scale(2.0);
        assert_eq!(device_x(transform, 5.2), 13.5);
        assert!(GlyphOriginSnap::skia(Affine::rotate(0.5)).is_none());
    }

    #[test]
    fn snaps_device_y_down_to_whole_px() {
        let device_y = |transform: Affine, y: f32| {
            let snap = GlyphOriginSnap::core_graphics(transform, 16.0, kurbo::Vec2::ZERO).unwrap();
            transform.as_coeffs()[3] * snap.y(y) as f64 + transform.as_coeffs()[5]
        };
        assert_eq!(device_y(Affine::IDENTITY, 10.0), 10.0);
        assert_eq!(device_y(Affine::IDENTITY, 10.0005), 10.0);
        assert_eq!(device_y(Affine::IDENTITY, 10.1), 11.0);
        assert_eq!(device_y(Affine::IDENTITY, 10.9), 11.0);
        // At scale(1.5), a baseline at y = 7 is 10.5 device px down.
        assert!((device_y(Affine::scale(1.5), 7.0) - 11.0).abs() < 1e-6);
        let transform = Affine::translate((0.0, 0.25)) * Affine::scale(2.0);
        assert!((device_y(transform, 5.0) - 11.0).abs() < 1e-6);
        let snap = GlyphOriginSnap::skia(Affine::IDENTITY).unwrap();
        assert_eq!(snap.y(10.3), 10.3);
    }

    #[test]
    fn raises_emboldened_glyphs_by_vertical_embolden() {
        let embolden = kurbo::Vec2::new(0.1, 0.3);
        let snap = GlyphOriginSnap::core_graphics(Affine::IDENTITY, 16.0, embolden).unwrap();
        assert!((snap.y(10.0) - 9.7).abs() < 1e-6);
        // The lift is in user units, so at 2x it is 0.6 device px.
        let snap = GlyphOriginSnap::core_graphics(Affine::scale(2.0), 8.0, embolden).unwrap();
        assert!((snap.y(10.0) - 9.7).abs() < 1e-6);
        let snap = GlyphOriginSnap::core_graphics(Affine::IDENTITY, 16.0, kurbo::Vec2::ZERO);
        assert_eq!(snap.unwrap().y(10.0), 10.0);
    }

    #[test]
    fn skips_rotated_runs() {
        assert!(
            GlyphOriginSnap::core_graphics(Affine::rotate(0.3), 16.0, kurbo::Vec2::ZERO).is_none()
        );
    }
}
