use std::sync::Arc;

use anyrender::{Filter, MaskMode, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Compose, Fill, FontData, ImageBrush, Mix, StyleRef};
use vello_cpu::{PaintType, Pixmap};

use crate::image_cache::{ImageCache, ImageCacheConfig};

const DEFAULT_TOLERANCE: f64 = 0.1;

/// Whether a glyph run drawn with the given transforms can use Vello's glyph atlas cache.
///
/// Vello only supports caching glyphs that are drawn upright with a uniform scale. It checks
/// this itself when inserting a glyph into the atlas, but not when looking one up, so a skewed
/// (e.g. synthetic italic) or rotated run would otherwise be drawn using upright cached glyphs.
fn supports_glyph_caching(transform: Affine, glyph_transform: Option<Affine>) -> bool {
    let [a, b, c, d, _, _] = (transform * glyph_transform.unwrap_or_default()).as_coeffs();
    b == 0.0 && c == 0.0 && a == d && a > 0.0
}

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

enum LayerKind {
    Layer,
    Clip,
    /// A filter layer whose input and output are clipped to its filter region.
    FilterRegionLayer,
}

pub struct VelloCpuScenePainter {
    pub(crate) render_ctx: vello_cpu::RenderContext,
    pub(crate) resources: vello_cpu::Resources,
    pub(crate) image_cache: ImageCache,
    layer_stack: Vec<LayerKind>,
    glyph_caching: bool,
}

impl VelloCpuScenePainter {
    pub fn new(width: u16, height: u16) -> Self {
        Self::with_image_cache_config(width, height, ImageCacheConfig::default())
    }

    pub fn with_image_cache_config(width: u16, height: u16, config: ImageCacheConfig) -> Self {
        Self {
            render_ctx: vello_cpu::RenderContext::new(width, height),
            resources: vello_cpu::Resources::new(),
            image_cache: ImageCache::new(config),
            layer_stack: Vec::new(),
            glyph_caching: false,
        }
    }

    /// Enable or disable caching of rasterized glyphs in Vello's glyph atlas.
    ///
    /// Defaults to `false`.
    ///
    /// Note: Vello considers atlas-backed glyph caching experimental.
    pub fn set_glyph_caching(&mut self, enabled: bool) {
        self.glyph_caching = enabled;
    }

    fn convert_paint(&mut self, paint: PaintRef<'_>) -> PaintType {
        match paint {
            Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
            Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
            Paint::Image(image) => PaintType::Image(ImageBrush {
                image: self.image_cache.get_or_register(image.image),
                sampler: image.sampler,
            }),
            // TODO: custom paint
            Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
            Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
            // Patterns are drawn with `fill_pattern_tiles`
            Paint::Pattern(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        }
    }

    /// Advance the image cache's frame counter and evict stale entries.
    ///
    /// Should be called once per frame, after rendering.
    pub fn maintain(&mut self) {
        self.image_cache.maintain();
    }

    /// Drop all cached image conversions.
    pub fn clear_image_cache(&mut self) {
        self.image_cache.clear();
    }

    pub fn finish(mut self) -> Pixmap {
        let mut pixmap = Pixmap::new(self.render_ctx.width(), self.render_ctx.height());
        self.render_ctx.flush();
        self.render_ctx.render(pixmap.as_mut(), &mut self.resources);
        pixmap
    }
}

impl RenderContext for VelloCpuScenePainter {}
impl PaintScene for VelloCpuScenePainter {
    fn reset(&mut self) {
        self.render_ctx.reset();
        self.layer_stack.clear();
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
        #[cfg(feature = "filters")]
        let (filter, region) = match filter
            .as_deref()
            .and_then(crate::filters::convert_filter)
            .filter(|_| cfg!(not(feature = "multithreading")))
        {
            Some(converted) => (Some(converted), filter.and_then(|f| f.region())),
            None => (None, None),
        };

        #[cfg(not(feature = "filters"))]
        let (filter, region): (_, Option<Rect>) = {
            let _ = filter;
            (None, None)
        };

        self.render_ctx.set_transform(transform);
        let region = region.map(|region| region.to_path(DEFAULT_TOLERANCE));
        if let Some(region) = &region {
            self.render_ctx.set_fill_rule(Fill::NonZero);
            self.render_ctx.push_clip_path(region);
        }
        self.render_ctx.set_fill_rule(fill);
        self.render_ctx.push_layer(
            Some(&clip.into_path(DEFAULT_TOLERANCE)),
            Some(blend.into()),
            Some(alpha),
            None,
            filter,
        );
        match &region {
            Some(region) => {
                self.render_ctx.set_fill_rule(Fill::NonZero);
                self.render_ctx.push_clip_path(region);
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
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(fill);
        if let Some(rect) = clip.as_rect() {
            self.layer_stack.push(LayerKind::Clip);
            self.render_ctx.push_clip_rect(&rect);
        } else {
            // An isolated layer (rather than `push_clip_path`) so that overlapping draws are
            // anti-aliased against the clip edge once, instead of once per draw.
            self.layer_stack.push(LayerKind::Layer);
            self.render_ctx
                .push_clip_layer(&clip.into_path(DEFAULT_TOLERANCE));
        }
    }

    fn pop_layer(&mut self) {
        match self.layer_stack.pop() {
            Some(LayerKind::Layer) => self.render_ctx.pop_layer(),
            Some(LayerKind::Clip) => self.render_ctx.pop_clip(),
            Some(LayerKind::FilterRegionLayer) => {
                self.render_ctx.pop_clip();
                self.render_ctx.pop_layer();
                self.render_ctx.pop_clip();
            }
            None => {}
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
        let paint: anyrender::PaintRef<'a> = paint.into();
        if let anyrender::Paint::Pattern(pattern) = paint {
            anyrender::stroke_pattern_tiles(
                self,
                pattern,
                style,
                transform,
                brush_transform,
                shape,
            );
            return;
        }
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_stroke(style.clone());
        let paint = self.convert_paint(paint);
        self.render_ctx.set_paint(paint);
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
        let paint: anyrender::PaintRef<'a> = paint.into();
        if let anyrender::Paint::Pattern(pattern) = paint {
            anyrender::fill_pattern_tiles(self, pattern, style, transform, brush_transform, shape);
            return;
        }
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(style);
        let paint = self.convert_paint(paint);
        self.render_ctx.set_paint(paint);
        self.render_ctx
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        if let Some(rect) = shape.as_rect() {
            self.render_ctx.fill_rect(&rect);
        } else {
            self.render_ctx
                .fill_path(&shape.into_path(DEFAULT_TOLERANCE));
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
        self.render_ctx.set_transform(transform);
        let paint = self.convert_paint(paint.into());
        self.render_ctx.set_paint(paint);

        let glyph_caching =
            self.glyph_caching && supports_glyph_caching(transform, glyph_transform);
        let snap = GlyphOriginSnap::new(transform, font_size, embolden);
        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.render_ctx.set_fill_rule(fill);
                let _ = self
                    .render_ctx
                    .glyph_run(&mut self.resources, font)
                    .atlas_cache(glyph_caching)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)))
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(|g| vello_cpu::Glyph {
                        id: g.id,
                        x: snap.map_or(g.x, |snap| snap.x(g.x)),
                        y: snap.map_or(g.y, |snap| snap.y(g.y)),
                    }));
            }
            StyleRef::Stroke(stroke) => {
                self.render_ctx.set_stroke(stroke.clone());
                let _ = self
                    .render_ctx
                    .glyph_run(&mut self.resources, font)
                    .atlas_cache(glyph_caching)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(|g| vello_cpu::Glyph {
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
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_paint(PaintType::Solid(color));
        self.render_ctx.reset_paint_transform();
        self.render_ctx.set_fill_rule(Fill::NonZero);
        if geometry.std_dev == 0.0 {
            // A single fill, so a non-isolated clip anti-aliases the clip edge only once.
            let clip = geometry.needs_clip();
            if clip {
                self.render_ctx.push_clip_path(&geometry.area);
            }
            self.render_ctx.fill_path(&geometry.unblurred_path());
            if clip {
                self.render_ctx.pop_clip();
            }
            return;
        }
        // TODO: draw shadows with matching individual radii instead of averaging them
        self.render_ctx.fill_blurred_rounded_rect_in(
            &geometry.area,
            &geometry.shadow.rect,
            geometry.shadow.average_radius() as f32,
            geometry.std_dev as f32,
            geometry.is_inset(),
        );
    }
}

#[cfg(test)]
mod clip_rule_tests {
    use anyrender::{PaintScene, recording::Scene, render_to_buffer};
    use kurbo::{Affine, BezPath, Rect};
    use peniko::{Fill, Mix, color::palette::css::RED};

    use crate::VelloCpuImageRenderer;

    fn render_clip(fill: Fill, compositing_layer: bool, replay: bool) -> Vec<u8> {
        let path = BezPath::from_svg("M0 0H100V100H0Z M25 25H75V75H25Z").unwrap();
        let draw = |scene: &mut crate::VelloCpuScenePainter| {
            if replay {
                let mut recording = Scene::new();
                draw_clipped(&mut recording, fill, compositing_layer, &path);
                scene.append_scene(recording, Affine::IDENTITY);
            } else {
                draw_clipped(scene, fill, compositing_layer, &path);
            }
        };
        render_to_buffer::<VelloCpuImageRenderer, _>(draw, 100, 100)
    }

    fn draw_clipped(
        scene: &mut impl PaintScene,
        fill: Fill,
        compositing_layer: bool,
        path: &BezPath,
    ) {
        if compositing_layer {
            scene.push_layer(fill, Mix::Normal, 1.0, Affine::IDENTITY, path, None, None);
        } else {
            scene.push_clip_layer(fill, Affine::IDENTITY, path);
        }
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            RED,
            None,
            &Rect::new(0.0, 0.0, 100.0, 100.0),
        );
        scene.pop_layer();
    }

    fn assert_pixels(compositing_layer: bool, replay: bool) {
        for fill in [Fill::NonZero, Fill::EvenOdd] {
            let buffer = render_clip(fill, compositing_layer, replay);
            let pixel = |x: usize, y: usize| &buffer[(y * 100 + x) * 4..(y * 100 + x) * 4 + 4];
            assert_eq!(pixel(10, 10), &[255, 0, 0, 255]);
            assert_eq!(
                pixel(50, 50),
                if fill == Fill::EvenOdd {
                    &[0, 0, 0, 0]
                } else {
                    &[255, 0, 0, 255]
                }
            );
        }
    }

    #[test]
    fn clip_layers_respect_fill_rule() {
        assert_pixels(false, false);
    }

    #[test]
    fn compositing_layers_respect_fill_rule() {
        assert_pixels(true, false);
    }

    #[test]
    fn overlapping_fills_in_path_clip_are_antialiased_once() {
        let render = |fills: usize| {
            render_to_buffer::<VelloCpuImageRenderer, _>(
                |scene| {
                    scene.push_clip_layer(
                        Fill::NonZero,
                        Affine::IDENTITY,
                        &kurbo::Circle::new((50.0, 50.0), 40.3),
                    );
                    for _ in 0..fills {
                        scene.fill(
                            Fill::NonZero,
                            Affine::IDENTITY,
                            RED,
                            None,
                            &Rect::new(0.0, 0.0, 100.0, 100.0),
                        );
                    }
                    scene.pop_layer();
                },
                100,
                100,
            )
        };
        let once = render(1);
        let twice = render(3);
        let edge = (50 * 100 + 90) * 4;
        assert!(once[edge + 3] > 0 && once[edge + 3] < 255);
        assert_eq!(once, twice);
    }

    #[test]
    fn rect_clips_and_fills() {
        let buffer = render_to_buffer::<VelloCpuImageRenderer, _>(
            |scene| {
                scene.push_layer(
                    Fill::NonZero,
                    Mix::Normal,
                    1.0,
                    Affine::IDENTITY,
                    &Rect::new(0.0, 0.0, 80.0, 100.0),
                    None,
                    None,
                );
                scene.push_clip_layer(
                    Fill::NonZero,
                    Affine::translate((20.0, 20.0)),
                    &Rect::new(0.0, 0.0, 80.0, 60.0),
                );
                scene.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    RED,
                    None,
                    &Rect::new(0.0, 0.0, 100.0, 100.0),
                );
                scene.pop_layer();
                scene.pop_layer();
                scene.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    RED,
                    None,
                    &Rect::new(90.0, 90.0, 100.0, 100.0),
                );
            },
            100,
            100,
        );
        let pixel = |x: usize, y: usize| &buffer[(y * 100 + x) * 4..(y * 100 + x) * 4 + 4];
        assert_eq!(pixel(50, 50), &[255, 0, 0, 255]);
        assert_eq!(pixel(10, 50), &[0, 0, 0, 0]);
        assert_eq!(pixel(50, 10), &[0, 0, 0, 0]);
        assert_eq!(pixel(50, 90), &[0, 0, 0, 0]);
        assert_eq!(pixel(85, 50), &[0, 0, 0, 0]);
        assert_eq!(pixel(95, 95), &[255, 0, 0, 255]);
    }

    #[test]
    fn scene_replay_preserves_clip_rules() {
        assert_pixels(false, true);
        assert_pixels(true, true);
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

#[cfg(test)]
mod box_shadow_tests {
    use anyrender::{BoxShadowKind, NonUniformRoundedRect, PaintScene, render_to_buffer};
    use kurbo::{Affine, Rect, Vec2};
    use peniko::color::palette::css::RED;

    use crate::VelloCpuImageRenderer;

    fn render_unblurred(box_rect: Rect, offset: Vec2, spread: f64, kind: BoxShadowKind) -> Vec<u8> {
        render_to_buffer::<VelloCpuImageRenderer, _>(
            |scene| {
                scene.draw_box_shadow(
                    Affine::IDENTITY,
                    &NonUniformRoundedRect::from(box_rect),
                    offset,
                    spread,
                    0.0,
                    RED,
                    kind,
                );
            },
            100,
            100,
        )
    }

    fn alpha(buffer: &[u8], x: usize, y: usize) -> u8 {
        buffer[(y * 100 + x) * 4 + 3]
    }

    #[test]
    fn unblurred_outset_shadow_is_clipped_to_outside_of_box() {
        let kind = BoxShadowKind::Outset;
        let buffer = render_unblurred(
            Rect::new(20.0, 20.0, 60.0, 60.0),
            Vec2::new(10.0, 10.0),
            2.0,
            kind,
        );
        assert_eq!(alpha(&buffer, 65, 65), 255);
        assert_eq!(alpha(&buffer, 40, 40), 0);
        assert_eq!(alpha(&buffer, 15, 15), 0);
        assert_eq!(alpha(&buffer, 75, 75), 0);

        let kind = BoxShadowKind::OutsetUnclipped;
        let buffer = render_unblurred(Rect::new(20.0, 20.0, 60.0, 60.0), Vec2::ZERO, 0.0, kind);
        assert_eq!(alpha(&buffer, 40, 40), 255);
    }

    #[test]
    fn blurred_shadows_are_clipped_to_box() {
        let render = |kind| {
            render_to_buffer::<VelloCpuImageRenderer, _>(
                |scene| {
                    scene.draw_box_shadow(
                        Affine::IDENTITY,
                        &NonUniformRoundedRect::from(Rect::new(20.0, 20.0, 80.0, 80.0)),
                        Vec2::ZERO,
                        10.0,
                        2.0,
                        RED,
                        kind,
                    );
                },
                100,
                100,
            )
        };
        let inset = render(BoxShadowKind::Inset);
        assert!(alpha(&inset, 22, 50) > 250);
        assert_eq!(alpha(&inset, 50, 50), 0);
        assert_eq!(alpha(&inset, 15, 50), 0);

        let outset = render(BoxShadowKind::Outset);
        assert!(alpha(&outset, 15, 50) > 250);
        assert_eq!(alpha(&outset, 50, 50), 0);
        assert_eq!(alpha(&outset, 2, 50), 0);
    }

    #[test]
    fn blurred_inset_shadow_keeps_background() {
        for box_shape in [
            NonUniformRoundedRect::from(Rect::new(20.0, 20.0, 80.0, 80.0)),
            NonUniformRoundedRect::from(kurbo::RoundedRect::new(20.0, 20.0, 80.0, 80.0, 8.0)),
        ] {
            let buffer = render_to_buffer::<VelloCpuImageRenderer, _>(
                |scene| {
                    scene.fill(
                        peniko::Fill::NonZero,
                        Affine::IDENTITY,
                        peniko::color::palette::css::WHITE,
                        None,
                        &Rect::new(0.0, 0.0, 100.0, 100.0),
                    );
                    scene.draw_box_shadow(
                        Affine::IDENTITY,
                        &box_shape,
                        Vec2::ZERO,
                        5.0,
                        2.0,
                        RED,
                        BoxShadowKind::Inset,
                    );
                },
                100,
                100,
            );
            let center = (50 * 100 + 50) * 4;
            assert_eq!(&buffer[center..center + 4], &[255, 255, 255, 255]);
            let edge = (50 * 100 + 21) * 4;
            assert!(buffer[edge + 1] < 8 && buffer[edge + 2] < 8 && buffer[edge + 3] == 255);
        }
    }

    #[test]
    fn unblurred_inset_shadow_surrounds_hole() {
        let buffer = render_unblurred(
            Rect::new(20.0, 20.0, 80.0, 80.0),
            Vec2::ZERO,
            10.0,
            BoxShadowKind::Inset,
        );
        assert_eq!(alpha(&buffer, 25, 50), 255);
        assert_eq!(alpha(&buffer, 50, 50), 0);
        assert_eq!(alpha(&buffer, 10, 10), 0);
    }
}
