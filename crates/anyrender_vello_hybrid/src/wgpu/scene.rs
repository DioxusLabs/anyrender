use std::sync::Arc;

use crate::common::{DEFAULT_TOLERANCE, ImageCache, LayerKind, supports_glyph_caching};
use anyrender::{
    Filter, MaskMode, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext, ResourceId,
};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Compose, Fill, FontData, ImageBrush, ImageData, Mix, StyleRef};
use rustc_hash::FxHashSet;
use vello_common::{
    TextureId,
    paint::{ImageSource, PaintType, TextureRegion},
};
use vello_gpu::{Resources, TextureBindings};
use wgpu::{
    Device, Extent3d, Origin3d, Queue, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};
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
) -> PaintType {
    let texture_bindings = &*image_manager.texture_bindings;
    match paint {
        Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
        Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),

        Paint::Image(image_brush) => {
            let Some((texture_id, may_have_transparency)) =
                image_manager.upload_image(image_brush.image)
            else {
                return PaintType::Solid(Color::TRANSPARENT);
            };
            PaintType::Image(ImageBrush {
                image: ImageSource::external_texture(
                    texture_id,
                    TextureRegion::Full,
                    may_have_transparency,
                ),
                sampler: image_brush.sampler,
            })
        }

        Paint::Resource(brush) if texture_bindings.contains(brush.image) => {
            PaintType::Image(ImageBrush {
                image: ImageSource::external_texture(
                    TextureId(brush.image.into_ffi()),
                    TextureRegion::Full,
                    true,
                ),
                sampler: brush.sampler,
            })
        }
        Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),

        // TODO: custom paint
        Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
    }
}

/// Textures bound for rendering (uploaded images and registered custom
/// resources), kept in a persistent [`TextureBindings`] keyed by
/// `TextureId(resource_id.into_ffi())`.
#[derive(Debug, Default)]
pub struct ExternalTextures {
    bindings: TextureBindings,
    ids: FxHashSet<ResourceId>,
}

impl ExternalTextures {
    pub fn insert(&mut self, id: ResourceId, view: TextureView) {
        self.ids.insert(id);
        self.bindings.insert(TextureId(id.into_ffi()), view);
    }

    pub fn remove(&mut self, id: ResourceId) {
        if self.ids.remove(&id) {
            self.bindings.remove(TextureId(id.into_ffi()));
        }
    }

    pub fn contains(&self, id: ResourceId) -> bool {
        self.ids.contains(&id)
    }

    pub fn bindings(&self) -> &TextureBindings {
        &self.bindings
    }
}

pub struct ImageManager<'a> {
    pub(crate) resources: &'a mut Resources,
    pub(crate) device: &'a Device,
    pub(crate) queue: &'a Queue,
    pub(crate) cache: &'a mut ImageCache,
    pub(crate) texture_bindings: &'a mut ExternalTextures,
}

impl<'a> ImageManager<'a> {
    pub fn new(
        resources: &'a mut Resources,
        device: &'a Device,
        queue: &'a Queue,
        cache: &'a mut ImageCache,
        texture_bindings: &'a mut ExternalTextures,
    ) -> Self {
        Self {
            resources,
            device,
            queue,
            cache,
            texture_bindings,
        }
    }

    /// Upload `image` to its own texture and bind it for rendering, returning
    /// its [`TextureId`] and transparency hint.
    pub(crate) fn upload_image(&mut self, image: &ImageData) -> Option<(TextureId, bool)> {
        if !self.cache.fits(image.width, image.height) {
            return None;
        }
        let peniko_id = image.data.id();

        // Try to get TextureId from cache first
        if let Some(cached) = self.cache.get(peniko_id) {
            return Some(cached);
        };

        // Convert ImageData to Pixmap
        let ImageSource::Pixmap(pixmap) = ImageSource::from_peniko_image_data(image) else {
            unreachable!(); // ImageSource::from_peniko_image_data always returns a Pixmap
        };
        let may_have_transparency = pixmap.may_have_transparency();

        // Upload the pixmap to its own texture.
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("AnyRender uploaded image"),
            size: Extent3d {
                width: u32::from(pixmap.width()),
                height: u32::from(pixmap.height()),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            pixmap.data_as_u8_slice(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * u32::from(pixmap.width())),
                rows_per_image: Some(u32::from(pixmap.height())),
            },
            Extent3d {
                width: u32::from(pixmap.width()),
                height: u32::from(pixmap.height()),
                depth_or_array_layers: 1,
            },
        );

        // Bind the texture and remember the TextureId in the cache.
        let resource_id = ResourceId::new();
        self.texture_bindings.insert(
            resource_id,
            texture.create_view(&TextureViewDescriptor::default()),
        );
        let texture_id = TextureId(resource_id.into_ffi());
        self.cache.insert(
            &image.data,
            texture_id,
            may_have_transparency,
            image.width,
            image.height,
        );

        Some((texture_id, may_have_transparency))
    }

    /// Evict unused images after encoding the frame's rendering.
    pub fn finish_frame(&mut self) {
        for id in self.cache.finish_frame() {
            // Dropping the binding drops the last handle to the texture.
            self.texture_bindings.remove(ResourceId::from_ffi(id.0));
        }
    }
}

pub struct VelloHybridScenePainter<'s> {
    pub(crate) scene: &'s mut vello_gpu::Scene,
    pub(crate) layer_stack: Vec<LayerKind>,
    pub(crate) image_manager: ImageManager<'s>,
    pub(crate) device_handle: &'s DeviceHandle,
    pub(crate) glyph_caching: bool,
}

impl VelloHybridScenePainter<'_> {
    pub fn new<'s>(
        scene: &'s mut vello_gpu::Scene,
        image_manager: ImageManager<'s>,
        device_handle: &'s DeviceHandle,
    ) -> VelloHybridScenePainter<'s> {
        VelloHybridScenePainter {
            scene,
            layer_stack: Vec::with_capacity(16),
            image_manager,
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
        anyrender_paint_to_vello_hybrid_paint(paint, &mut self.image_manager)
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
                self.image_manager.texture_bindings.insert(id, texture_view);
                Ok(id)
            }
            Err(resource) => {
                // Else try to downcast as TextureView
                if let Ok(texture_view) = resource.downcast::<TextureView>() {
                    let id = ResourceId::new();
                    self.image_manager
                        .texture_bindings
                        .insert(id, *texture_view);
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
        self.image_manager.texture_bindings.remove(resource_id);
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
        let (filter, region) = match filter.as_deref().and_then(crate::common::convert_filter) {
            Some(converted) => (Some(converted), filter.and_then(|f| f.region())),
            None => (None, None),
        };
        self.scene.set_transform(transform);
        if let Some(region) = &region {
            self.scene.push_clip_rect(region);
        }
        self.scene.set_fill_rule(fill);
        if let Some(rect) = clip.as_rect() {
            self.scene.push_clip_rect(&rect);
        } else {
            self.scene
                .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
        }
        self.scene
            .push_layer(None, Some(blend.into()), Some(alpha), None, filter);
        match &region {
            Some(region) => {
                self.scene.push_clip_rect(region);
                self.layer_stack.push(LayerKind::FilterRegionLayer);
            }
            None => self.layer_stack.push(LayerKind::Layer),
        }
    }

    fn push_mask_layer(
        &mut self,
        fill: Fill,
        _mode: MaskMode,
        transform: Affine,
        clip: &impl Shape,
    ) {
        let blend = BlendMode::new(Mix::Normal, Compose::DestIn);

        // Everything outside of the clip is masked out
        let everything = Rect::new(-1e6, -1e6, 1e6, 1e6);
        self.push_layer(
            Fill::NonZero,
            blend,
            1.0,
            Affine::IDENTITY,
            &everything,
            None,
            None,
        );
        self.fill(fill, transform, Color::BLACK, None, clip);
        self.pop_layer();

        // Luminance masks aren't supported yet, so they are treated as alpha masks
        self.push_layer(fill, blend, 1.0, transform, clip, None, None);
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
                LayerKind::FilterRegionLayer => {
                    self.scene.pop_clip();
                    self.scene.pop_layer();
                    self.scene.pop_clip();
                    self.scene.pop_clip();
                }
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
        box_shape: &anyrender::NonUniformRoundedRect,
        offset: kurbo::Vec2,
        spread: f64,
        std_dev: f64,
        color: Color,
        kind: anyrender::BoxShadowKind,
    ) {
        let geometry = anyrender::BoxShadowGeometry::new(box_shape, offset, spread, std_dev, kind);
        self.scene.set_transform(transform);
        self.scene.set_paint(PaintType::Solid(color));
        self.scene.reset_paint_transform();
        self.scene.set_fill_rule(Fill::NonZero);
        if geometry.std_dev == 0.0 {
            // A single fill, so a non-isolated clip anti-aliases the clip edge only once.
            let clip = geometry.needs_clip();
            if clip {
                self.scene.push_clip_path(&geometry.area);
            }
            self.scene.fill_path(&geometry.unblurred_path());
            if clip {
                self.scene.pop_clip();
            }
            return;
        }
        // TODO: draw shadows with matching individual radii instead of averaging them
        self.scene.fill_blurred_rounded_rect_in(
            &geometry.area,
            &geometry.shadow.rect,
            geometry.shadow.average_radius() as f32,
            geometry.std_dev as f32,
            geometry.is_inset(),
        );
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
