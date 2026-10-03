use std::sync::Arc;

use anyrender::{Filter, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, StyleRef};
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

enum LayerKind {
    Layer,
    Clip,
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
                image: self
                    .image_cache
                    .get_or_register(&mut self.resources, image.image),
                sampler: image.sampler,
            }),
            // TODO: custom paint
            Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
            Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        }
    }

    /// Advance the image cache's frame counter and evict stale entries.
    ///
    /// Should be called once per frame, after rendering.
    pub fn maintain(&mut self) {
        self.image_cache.maintain(&mut self.resources);
    }

    /// Drop all cached image conversions.
    pub fn clear_image_cache(&mut self) {
        self.image_cache.clear(&mut self.resources);
    }

    pub fn finish(mut self) -> Pixmap {
        let mut pixmap = Pixmap::new(self.render_ctx.width(), self.render_ctx.height());
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
        let filter = filter
            .and_then(crate::filters::convert_filter)
            .filter(|_| cfg!(not(feature = "multithreading")));

        #[cfg(not(feature = "filters"))]
        let filter = {
            let _ = filter;
            None
        };

        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(fill);
        self.layer_stack.push(LayerKind::Layer);
        self.render_ctx.push_layer(
            Some(&clip.into_path(DEFAULT_TOLERANCE)),
            Some(blend.into()),
            Some(alpha),
            None,
            filter,
        );
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
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_stroke(style.clone());
        let paint = self.convert_paint(paint.into());
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
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(style);
        let paint = self.convert_paint(paint.into());
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
                        x: g.x,
                        y: g.y,
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
                        x: g.x,
                        y: g.y,
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
        if geometry.is_inset() {
            geometry.draw_inset_with_layers(self, transform, color);
            return;
        }
        // A single fill, so a non-isolated clip anti-aliases the clip edge only once.
        let clip = geometry.needs_clip();
        if clip {
            self.render_ctx.push_clip_path(&geometry.area);
        }
        // TODO: draw shadows with matching individual radii instead of averaging them
        self.render_ctx.fill_blurred_rounded_rect(
            &geometry.shadow.rect,
            geometry.shadow.average_radius() as f32,
            geometry.std_dev as f32,
            false,
        );
        if clip {
            self.render_ctx.pop_clip();
        }
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
        let kind = BoxShadowKind::Outset { clip_to_box: true };
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

        let kind = BoxShadowKind::Outset { clip_to_box: false };
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

        let outset = render(BoxShadowKind::Outset { clip_to_box: true });
        assert!(alpha(&outset, 15, 50) > 250);
        assert_eq!(alpha(&outset, 50, 50), 0);
        assert_eq!(alpha(&outset, 2, 50), 0);
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
