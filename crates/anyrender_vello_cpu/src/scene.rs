use std::sync::Arc;

use anyrender::{Filter, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Rect, Shape, Stroke};
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
            glyph_caching: cfg!(feature = "glyph_caching"),
        }
    }

    /// Enable or disable caching of rasterized glyphs in Vello's glyph atlas.
    ///
    /// Defaults to `false` unless the `glyph_caching` cargo feature is enabled.
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
        } else if self.render_ctx.is_multi_threaded() {
            // The multi-threaded dispatcher flushes its pending work and rasterizes the clip on
            // the main thread for every `push_clip_path`, which is slower than an isolated layer.
            self.layer_stack.push(LayerKind::Layer);
            self.render_ctx
                .push_clip_layer(&clip.into_path(DEFAULT_TOLERANCE));
        } else {
            self.layer_stack.push(LayerKind::Clip);
            self.render_ctx
                .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
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
        rect: Rect,
        color: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_paint(PaintType::Solid(color));
        self.render_ctx.reset_paint_transform();
        self.render_ctx
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32, false);
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
