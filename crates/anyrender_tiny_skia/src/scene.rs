//! A [`tiny-skia`] backend for the [`anyrender`] 2D drawing abstraction

#![allow(clippy::too_many_arguments)]

use anyhow::{Result, anyhow};
use anyrender::{NormalizedCoord, Paint as AnyRenderPaint, PaintRef, PaintScene};
use kurbo::{Affine, PathEl, Point, Rect, Shape};
use peniko::{
    BlendMode, BrushRef, Color, Compose, Fill, FontData, GradientKind, ImageBrushRef, Mix,
    StyleRef, color::palette,
};
use resvg::tiny_skia::StrokeDash;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use swash::{
    FontRef, GlyphId,
    scale::{Render, ScaleContext, Source, StrikeWith, image::Content as SwashContent},
    zeno::Format,
};
use tiny_skia::{
    self, FillRule, FilterQuality, GradientStop, LineCap, LineJoin, LinearGradient, Mask, MaskType,
    Paint, Path, PathBuilder, Pattern, Pixmap, PixmapPaint, RadialGradient, Shader, SpreadMode,
    Stroke, Transform,
};

thread_local! {
    #[allow(clippy::type_complexity)]
    static IMAGE_CACHE: RefCell<HashMap<Vec<u8>, (CacheColor, Rc<Pixmap>)>> = RefCell::new(HashMap::new());
    #[allow(clippy::type_complexity)]
    // The cache key includes: (glyph_id, font_size_bits, coords_hash, hint, fill_rule, transform_hash, offset_hash), color
    static GLYPH_CACHE: RefCell<HashMap<((GlyphId, u32, u64, bool, u8, u64 ), u32), (CacheColor, Option<Rc<Glyph>>)>> = RefCell::new(HashMap::new());
    static SWASH_SCALER: RefCell<ScaleContext> = RefCell::new(ScaleContext::new());
}

fn cache_image(cache_color: CacheColor, image: &ImageBrushRef) -> Option<Rc<Pixmap>> {
    let data_key = image.image.data.data().to_vec();

    if let Some(cached_pixmap) = IMAGE_CACHE.with_borrow_mut(|ic| {
        if let Some((color, pixmap)) = ic.get_mut(&data_key) {
            *color = cache_color;
            Some(pixmap.clone())
        } else {
            None
        }
    }) {
        return Some(cached_pixmap);
    }

    // Convert peniko ImageData to tiny-skia Pixmap
    let pixmap = match image.image.format {
        peniko::ImageFormat::Rgba8 => {
            let mut pixmap = Pixmap::new(image.image.width, image.image.height)?;
            let data = image.image.data.data();

            for (i, chunk) in data.chunks_exact(4).enumerate() {
                if let [r, g, b, a] = chunk {
                    let color = tiny_skia::ColorU8::from_rgba(*r, *g, *b, *a);
                    let x = (i as u32) % image.image.width;
                    let y = (i as u32) / image.image.width;
                    if x < image.image.width && y < image.image.height {
                        pixmap.pixels_mut()[i] = color.premultiply();
                    }
                }
            }

            Some(Rc::new(pixmap))
        }
        _ => None, // Other formats not supported yet
    };

    if let Some(pixmap) = pixmap.clone() {
        IMAGE_CACHE.with_borrow_mut(|ic| {
            ic.insert(data_key, (cache_color, pixmap));
        });
    }

    pixmap
}

fn cache_glyph(
    cache_color: CacheColor,
    glyph_id: GlyphId,
    font_size: f32,
    color: Color,
    font: &FontData,
    normalized_coords: &[NormalizedCoord],
    hint: bool,
    fill: peniko::Fill,
    glyph_transform: Option<Affine>,
) -> Option<Rc<Glyph>> {
    let c = color.to_rgba8();
    // Create a more comprehensive cache key including normalized coords, hinting, fill rule, transform, and offset
    let coords_hash = normalized_coords
        .iter()
        .fold(0u64, |acc, x| acc.wrapping_mul(31).wrapping_add(*x as u64));
    let transform_hash = glyph_transform
        .map(|t| {
            let coeffs = t.as_coeffs();
            coeffs.iter().fold(0u64, |acc, &x| {
                acc.wrapping_mul(31).wrapping_add(x.to_bits())
            })
        })
        .unwrap_or(0);
    let cache_key = (
        glyph_id,
        font_size.to_bits(),
        coords_hash,
        hint,
        fill as u8,
        transform_hash,
    );

    if let Some(opt_glyph) = GLYPH_CACHE.with_borrow_mut(|gc| {
        if let Some((color, glyph)) = gc.get_mut(&(cache_key, c.to_u32())) {
            *color = cache_color;
            Some(glyph.clone())
        } else {
            None
        }
    }) {
        return opt_glyph;
    };

    let image = SWASH_SCALER.with_borrow_mut(|context| {
        let font_ref = FontRef::from_index(font.data.as_ref(), font.index as usize)?;
        let mut scaler = context
            .builder(font_ref)
            .size(font_size)
            .hint(hint)
            .normalized_coords(normalized_coords)
            .build();

        let zeno_transform = if let Some(transform) = glyph_transform {
            let coeffs = transform.as_coeffs();
            let swash_transform = swash::zeno::Transform::new(
                coeffs[0] as f32,
                coeffs[1] as f32,
                coeffs[2] as f32,
                coeffs[3] as f32,
                coeffs[4] as f32,
                coeffs[5] as f32,
            );
            Some(swash_transform)
        } else {
            None
        };
        Render::new(&[
            Source::ColorOutline(0),
            Source::ColorBitmap(StrikeWith::BestFit),
            Source::Outline,
        ])
        .format(Format::Alpha)
        .style(match fill {
            peniko::Fill::NonZero => swash::zeno::Fill::NonZero,
            peniko::Fill::EvenOdd => swash::zeno::Fill::EvenOdd,
        })
        .transform(zeno_transform)
        .render(&mut scaler, glyph_id)
    })?;

    let result = if image.placement.width == 0 || image.placement.height == 0 {
        // We can't create an empty `Pixmap`
        None
    } else {
        let mut pixmap = Pixmap::new(image.placement.width, image.placement.height)?;

        if image.content == SwashContent::Mask {
            for (a, &alpha) in pixmap.pixels_mut().iter_mut().zip(image.data.iter()) {
                *a = tiny_skia::Color::from_rgba8(c.r, c.g, c.b, alpha)
                    .premultiply()
                    .to_color_u8();
            }
        } else if image.content == SwashContent::Color {
            for (a, b) in pixmap.pixels_mut().iter_mut().zip(image.data.chunks(4)) {
                *a = tiny_skia::Color::from_rgba8(b[0], b[1], b[2], b[3])
                    .premultiply()
                    .to_color_u8();
            }
        } else {
            return None;
        }

        Some(Rc::new(Glyph {
            pixmap,
            left: image.placement.left as f32,
            top: image.placement.top as f32,
        }))
    };

    GLYPH_CACHE
        .with_borrow_mut(|gc| gc.insert((cache_key, c.to_u32()), (cache_color, result.clone())));

    result
}

macro_rules! try_ret {
    ($e:expr) => {
        if let Some(e) = $e {
            e
        } else {
            return;
        }
    };
}

struct Glyph {
    pixmap: Pixmap,
    left: f32,
    top: f32,
}

#[derive(PartialEq, Clone, Copy)]
struct CacheColor(bool);
impl std::ops::Not for CacheColor {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

pub(crate) struct Layer {
    pub(crate) pixmap: Pixmap,
    pub(crate) mask: Option<Mask>,
    /// this transform should generally only be used when making a draw call to skia
    transform: Affine,
    // the transform that the layer was pushed with that will be used when applying the layer
    combine_transform: Affine,
    blend_mode: BlendMode,
    alpha: f32,
    cache_color: CacheColor,
}
impl Layer {
    /// Renders the pixmap at the position and transforms it with the given transform.
    /// x and y should have already been scaled by the window scale
    fn render_pixmap_direct(&mut self, img_pixmap: &Pixmap, x: f32, y: f32, transform: Affine) {
        let img_rect = Rect::from_origin_size(
            (x, y),
            (img_pixmap.width() as f64, img_pixmap.height() as f64),
        );
        let paint = Paint {
            shader: Pattern::new(
                img_pixmap.as_ref(),
                SpreadMode::Pad,
                FilterQuality::Nearest,
                1.0,
                Transform::from_translate(x, y),
            ),
            ..Default::default()
        };

        let transform = transform.as_coeffs();
        let transform = Transform::from_row(
            transform[0] as f32,
            transform[1] as f32,
            transform[2] as f32,
            transform[3] as f32,
            transform[4] as f32,
            transform[5] as f32,
        );
        if let Some(rect) = to_skia_rect(img_rect) {
            self.pixmap
                .fill_rect(rect, &paint, transform, self.mask.as_ref());
        }
    }

    #[allow(dead_code)]
    fn render_pixmap_rect(&mut self, pixmap: &Pixmap, rect: tiny_skia::Rect) {
        let paint = Paint {
            shader: Pattern::new(
                pixmap.as_ref(),
                SpreadMode::Pad,
                FilterQuality::Bilinear,
                1.0,
                Transform::from_scale(
                    rect.width() / pixmap.width() as f32,
                    rect.height() / pixmap.height() as f32,
                ),
            ),
            ..Default::default()
        };

        self.pixmap
            .fill_rect(rect, &paint, self.skia_transform(), self.mask.as_ref());
    }

    #[allow(dead_code)]
    fn render_pixmap_with_paint(
        &mut self,
        pixmap: &Pixmap,
        rect: tiny_skia::Rect,
        paint: Option<Paint<'static>>,
    ) {
        let paint = if let Some(paint) = paint {
            paint
        } else {
            return self.render_pixmap_rect(pixmap, rect);
        };

        let mut colored_bg = try_ret!(Pixmap::new(pixmap.width(), pixmap.height()));
        colored_bg.fill_rect(
            try_ret!(tiny_skia::Rect::from_xywh(
                0.0,
                0.0,
                pixmap.width() as f32,
                pixmap.height() as f32
            )),
            &paint,
            Transform::identity(),
            None,
        );

        let mask = Mask::from_pixmap(pixmap.as_ref(), MaskType::Alpha);
        colored_bg.apply_mask(&mask);

        self.render_pixmap_rect(&colored_bg, rect);
    }

    fn skia_transform(&self) -> Transform {
        skia_transform(self.transform, 1.)
    }
}
impl Layer {
    /// The combine transform should be the transform that the layer is pushed with without combining with the previous transform. It will be used when combining layers to offset/transform this layer into the parent with the parent transform
    fn new(
        blend: impl Into<peniko::BlendMode>,
        alpha: f32,
        combine_transform: Affine,
        clip: &impl Shape,
        cache_color: CacheColor,
    ) -> Result<Self, anyhow::Error> {
        let transform = Affine::IDENTITY;
        let bbox = clip.bounding_box();
        let width = bbox.width() as u32;
        let height = bbox.height() as u32;
        let mut mask = Mask::new(width, height).ok_or_else(|| anyhow!("unable to create mask"))?;
        mask.fill_path(
            &shape_to_path(clip).ok_or_else(|| anyhow!("unable to create clip shape"))?,
            FillRule::Winding,
            false,
            Transform::identity(),
        );
        Ok(Self {
            pixmap: Pixmap::new(width, height).ok_or_else(|| anyhow!("unable to create pixmap"))?,
            mask: Some(mask),
            transform,
            combine_transform,
            blend_mode: blend.into(),
            alpha,
            cache_color,
        })
    }

    fn clip(&mut self, shape: &impl Shape, transform: Affine) {
        let transform = skia_transform(transform, 1.);
        let path = try_ret!(shape_to_path(shape));
        if let Some(ref mut mask) = self.mask {
            mask.fill_path(&path, FillRule::default(), false, transform);
        } else {
            // Create a new mask if none exists
            if let Some(mut mask) = Mask::new(self.pixmap.width(), self.pixmap.height()) {
                mask.fill_path(&path, FillRule::default(), false, transform);
                self.mask = Some(mask);
            }
        }
    }

    fn stroke<'b, 's>(
        &mut self,
        shape: &impl Shape,
        brush: impl Into<BrushRef<'b>>,
        stroke: &'s peniko::kurbo::Stroke,
        brush_transform: Option<Affine>,
    ) {
        let paint = try_ret!(brush_to_paint(brush, brush_transform));
        let path = try_ret!(shape_to_path(shape));
        let line_cap = match stroke.end_cap {
            peniko::kurbo::Cap::Butt => LineCap::Butt,
            peniko::kurbo::Cap::Square => LineCap::Square,
            peniko::kurbo::Cap::Round => LineCap::Round,
        };
        let line_join = match stroke.join {
            peniko::kurbo::Join::Bevel => LineJoin::Bevel,
            peniko::kurbo::Join::Miter => LineJoin::Miter,
            peniko::kurbo::Join::Round => LineJoin::Round,
        };
        let stroke = Stroke {
            width: stroke.width as f32,
            miter_limit: stroke.miter_limit as f32,
            line_cap,
            line_join,
            dash: (!stroke.dash_pattern.is_empty())
                .then_some(StrokeDash::new(
                    stroke.dash_pattern.iter().map(|v| *v as f32).collect(),
                    stroke.dash_offset as f32,
                ))
                .flatten(),
        };
        self.pixmap.stroke_path(
            &path,
            &paint,
            &stroke,
            self.skia_transform(),
            self.mask.as_ref(),
        );
    }

    fn fill<'b>(
        &mut self,
        shape: &impl Shape,
        brush: impl Into<BrushRef<'b>>,
        _blur_radius: f64,
        fill_style: Fill,
        brush_transform: Option<Affine>,
    ) {
        // FIXME: Handle _blur_radius
        let brush: BrushRef<'_> = brush.into();

        // Handle images specially
        if let BrushRef::Image(image) = brush {
            if let Some(cached_pixmap) = cache_image(self.cache_color, &image) {
                // Create a pattern from the cached pixmap
                let transform = brush_transform
                    .map(|t| skia_transform(t, 1.0))
                    .unwrap_or_else(Transform::identity);
                let pattern = Pattern::new(
                    cached_pixmap.as_ref().as_ref(),
                    SpreadMode::Pad,
                    FilterQuality::Nearest,
                    1.0,
                    transform,
                );
                let paint = Paint {
                    shader: pattern,
                    ..Default::default()
                };

                if let Some(rect) = shape.as_rect() {
                    let rect = try_ret!(to_skia_rect(rect));
                    self.pixmap
                        .fill_rect(rect, &paint, self.skia_transform(), self.mask.as_ref());
                } else {
                    let fill_rule = match fill_style {
                        Fill::NonZero => FillRule::Winding,
                        Fill::EvenOdd => FillRule::EvenOdd,
                    };
                    let path = try_ret!(shape_to_path(shape));
                    self.pixmap.fill_path(
                        &path,
                        &paint,
                        fill_rule,
                        self.skia_transform(),
                        self.mask.as_ref(),
                    );
                }
            }
        } else {
            // Handle non-image brushes
            let paint = try_ret!(brush_to_paint(brush, brush_transform));
            let fill_rule = match fill_style {
                Fill::NonZero => FillRule::Winding,
                Fill::EvenOdd => FillRule::EvenOdd,
            };
            if let Some(rect) = shape.as_rect() {
                let rect = try_ret!(to_skia_rect(rect));
                self.pixmap
                    .fill_rect(rect, &paint, self.skia_transform(), self.mask.as_ref());
            } else {
                let path = try_ret!(shape_to_path(shape));
                self.pixmap.fill_path(
                    &path,
                    &paint,
                    fill_rule,
                    self.skia_transform(),
                    self.mask.as_ref(),
                );
            }
        }
    }

    fn draw_glyph(
        &mut self,
        glyph_id: GlyphId,
        font_size: f32,
        color: Color,
        font: &FontData,
        x: f32,
        y: f32,
        normalized_coords: &[NormalizedCoord],
        hint: bool,
        glyph_transform: Option<Affine>,
        fill: peniko::Fill,
    ) {
        if let Some(cached_glyph) = cache_glyph(
            self.cache_color,
            glyph_id,
            font_size,
            color,
            font,
            normalized_coords,
            hint,
            fill,
            glyph_transform,
        ) {
            // Since transform and offset are now handled by swash, just render directly
            self.render_pixmap_direct(
                &cached_glyph.pixmap,
                cached_glyph.left + x,
                -cached_glyph.top + y,
                self.transform,
            );
        }
    }
}

#[allow(clippy::large_enum_variant)]
pub enum LayerOrClip {
    Layer(Layer),
    Clip { previous_mask: Option<Mask> },
}

pub struct TinySkiaScenePainter {
    pub(crate) layers: Vec<LayerOrClip>,
    pub(crate) last_non_clip_layer: usize,
    cache_color: CacheColor,
}

impl TinySkiaScenePainter {
    pub fn new(width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let pixmap = Pixmap::new(width, height).expect("Failed to create pixmap");
        let main_layer = Layer {
            pixmap,
            mask: None, // No clipping initially
            alpha: 1.0,
            transform: Affine::IDENTITY,
            combine_transform: Affine::IDENTITY,
            blend_mode: Mix::Normal.into(),
            cache_color: CacheColor(false),
        };
        let cache_color = CacheColor(false);
        Self {
            layers: vec![LayerOrClip::Layer(main_layer)],
            last_non_clip_layer: 0,
            cache_color,
        }
    }

    pub(crate) fn non_clip_layer(&mut self) -> &mut Layer {
        match self.layers.get_mut(self.last_non_clip_layer) {
            Some(LayerOrClip::Layer(layer)) => layer,
            _ => panic!("the base layer should aways exist and be a non clip layer"),
        }
    }
}

impl PaintScene for TinySkiaScenePainter {
    fn reset(&mut self) {
        let first_layer = self.non_clip_layer();
        first_layer.pixmap.fill(tiny_skia::Color::TRANSPARENT);
        first_layer.mask = None;
        first_layer.transform = Affine::IDENTITY;
        first_layer.cache_color = !first_layer.cache_color;
        self.layers.shrink_to(1);
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
    ) {
        let blend: BlendMode = blend.into();
        #[allow(deprecated)]
        if alpha == 1. && matches!(blend.mix, Mix::Normal | Mix::Clip) {
            let layer = self.non_clip_layer();

            // Capture the current mask state before applying new clip
            let previous_mask = layer.mask.clone();

            self.layers.push(LayerOrClip::Clip { previous_mask });
            let layer = self.non_clip_layer();
            layer.clip(clip, transform);
        } else if let Ok(layer) = Layer::new(blend, alpha, transform, clip, self.cache_color) {
            self.layers.push(LayerOrClip::Layer(layer));
            self.last_non_clip_layer = self.layers.len() - 1;
        }
    }

    fn pop_layer(&mut self) {
        if self.layers.len() <= 1 {
            return;
        }

        match self.layers.pop() {
            Some(LayerOrClip::Layer(layer)) => {
                // This was a real layer, apply it to the parent
                if let Some(LayerOrClip::Layer(parent)) = self.layers.last_mut() {
                    apply_layer(&layer, parent);
                }
                // Update last_non_clip_layer to point to the current top layer
                for (i, layer_or_clip) in self.layers.iter().enumerate().rev() {
                    if matches!(layer_or_clip, LayerOrClip::Layer(_)) {
                        self.last_non_clip_layer = i;
                        break;
                    }
                }
            }
            Some(LayerOrClip::Clip { previous_mask }) => {
                // This was just a clip, restore the previous mask state
                let layer = self.non_clip_layer();
                layer.mask = previous_mask;
            }
            None => {}
        }
    }

    fn stroke<'b>(
        &mut self,
        style: &kurbo::Stroke,
        transform: Affine,
        brush: impl Into<PaintRef<'b>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let layer = self.non_clip_layer();
        let paint_ref: PaintRef<'_> = brush.into();
        let brush_ref: BrushRef<'_> = paint_ref.into();

        let old_transform = layer.transform;
        layer.transform *= transform;

        layer.stroke(shape, brush_ref, style, brush_transform);

        layer.transform = old_transform;
    }

    fn fill<'b>(
        &mut self,
        style: Fill,
        transform: Affine,
        brush: impl Into<PaintRef<'b>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let layer = self.non_clip_layer();
        let paint_ref: PaintRef<'_> = brush.into();
        let brush_ref: BrushRef<'_> = paint_ref.into();

        let old_transform = layer.transform;
        layer.transform *= transform;

        layer.fill(shape, brush_ref, 0.0, style, brush_transform);

        layer.transform = old_transform;
    }

    fn draw_glyphs<'b, 's: 'b>(
        &'s mut self,
        font: &'b FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'b [NormalizedCoord],
        style: impl Into<StyleRef<'b>>,
        brush: impl Into<PaintRef<'b>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = anyrender::Glyph>,
    ) {
        let layer = self.non_clip_layer();
        let paint_ref: PaintRef<'_> = brush.into();
        let style_ref: StyleRef<'_> = style.into();

        // Extract color from paint and apply brush_alpha
        let base_color = match paint_ref {
            AnyRenderPaint::Solid(c) => c,
            _ => palette::css::BLACK,
        };

        let color = base_color.multiply_alpha(brush_alpha);

        let old_transform = layer.transform;
        layer.transform *= transform;

        for glyph in glyphs {
            match style_ref {
                StyleRef::Fill(fill) => {
                    // For fill styles, render the glyph normally
                    layer.draw_glyph(
                        glyph.id as GlyphId,
                        font_size,
                        color,
                        font,
                        glyph.x,
                        glyph.y,
                        normalized_coords,
                        hint,
                        glyph_transform,
                        fill,
                    );
                }
                StyleRef::Stroke(_stroke) => {
                    // TODO!
                    // For stroke styles, we need to render the glyph outline
                    // Since swash doesn't directly support stroke rendering,
                    // we just render the glyph normally for now
                    layer.draw_glyph(
                        glyph.id as GlyphId,
                        font_size,
                        color,
                        font,
                        glyph.x,
                        glyph.y,
                        normalized_coords,
                        hint,
                        glyph_transform,
                        peniko::Fill::default(), // Default fill for stroke fallback
                    );
                }
            }
        }

        layer.transform = old_transform;
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        brush: Color,
        radius: f64,
        std_dev: f64,
    ) {
        let layer = self.non_clip_layer();
        let old_transform = layer.transform;
        layer.transform = old_transform * transform;

        // Create a shadow using multiple passes with varying opacity
        // This approximates a Gaussian blur using box blur technique
        let blur_steps = (std_dev * 2.0).max(1.0) as i32;
        let base_alpha = brush.components[3] * 0.1; // Reduced alpha for accumulative effect
        for i in 0..blur_steps {
            let offset = i as f64 - (blur_steps as f64 / 2.0);
            let alpha_factor = 1.0 - (offset.abs() / (blur_steps as f64 / 2.0)).powf(0.5);
            let current_alpha = (base_alpha as f64 * alpha_factor).min(1.0);

            // Create shadow color with reduced alpha
            let shadow_color = Color::new([
                brush.components[0],
                brush.components[1],
                brush.components[2],
                current_alpha as f32,
            ]);

            // Create rounded rectangle for the shadow
            let shadow_rect = Rect::new(
                rect.x0 + offset,
                rect.y0 + offset,
                rect.x1 + offset,
                rect.y1 + offset,
            );

            let rounded_rect = kurbo::RoundedRect::new(
                shadow_rect.x0,
                shadow_rect.y0,
                shadow_rect.x1,
                shadow_rect.y1,
                radius,
            );

            // Create paint for this shadow layer
            let paint = Paint {
                shader: Shader::SolidColor(to_color(shadow_color)),
                ..Default::default()
            };

            // Draw the shadow layer
            if let Some(path) = shape_to_path(&rounded_rect) {
                layer.pixmap.fill_path(
                    &path,
                    &paint,
                    FillRule::Winding,
                    layer.skia_transform(),
                    layer.mask.as_ref(),
                );
            }
        }

        layer.transform = old_transform;
    }
}

fn to_color(color: Color) -> tiny_skia::Color {
    let c = color.to_rgba8();
    tiny_skia::Color::from_rgba8(c.r, c.g, c.b, c.a)
}

fn to_point(point: Point) -> tiny_skia::Point {
    tiny_skia::Point::from_xy(point.x as f32, point.y as f32)
}

fn shape_to_path(shape: &impl Shape) -> Option<Path> {
    let mut builder = PathBuilder::new();
    for element in shape.path_elements(0.1) {
        match element {
            PathEl::ClosePath => builder.close(),
            PathEl::MoveTo(p) => builder.move_to(p.x as f32, p.y as f32),
            PathEl::LineTo(p) => builder.line_to(p.x as f32, p.y as f32),
            PathEl::QuadTo(p1, p2) => {
                builder.quad_to(p1.x as f32, p1.y as f32, p2.x as f32, p2.y as f32)
            }
            PathEl::CurveTo(p1, p2, p3) => builder.cubic_to(
                p1.x as f32,
                p1.y as f32,
                p2.x as f32,
                p2.y as f32,
                p3.x as f32,
                p3.y as f32,
            ),
        }
    }
    builder.finish()
}

fn brush_to_paint<'b>(
    brush: impl Into<BrushRef<'b>>,
    brush_transform: Option<Affine>,
) -> Option<Paint<'static>> {
    let shader = match brush.into() {
        BrushRef::Solid(c) => Shader::SolidColor(to_color(c)),
        BrushRef::Gradient(g) => {
            let stops = g
                .stops
                .iter()
                .map(|s| GradientStop::new(s.offset, to_color(s.color.to_alpha_color())))
                .collect();
            match g.kind {
                GradientKind::Linear(linear_pos) => {
                    let transform = brush_transform
                        .map(|t| skia_transform(t, 1.0))
                        .unwrap_or_else(Transform::identity);
                    LinearGradient::new(
                        to_point(linear_pos.start),
                        to_point(linear_pos.end),
                        stops,
                        SpreadMode::Pad,
                        transform,
                    )?
                }
                GradientKind::Radial(radial) => {
                    let transform = brush_transform
                        .map(|t| skia_transform(t, 1.0))
                        .unwrap_or_else(Transform::identity);
                    RadialGradient::new(
                        to_point(radial.start_center),
                        to_point(radial.end_center),
                        radial.end_radius,
                        stops,
                        SpreadMode::Pad,
                        transform,
                    )?
                }
                GradientKind::Sweep { .. } => return None,
            }
        }
        BrushRef::Image(_) => return None,
    };
    Some(Paint {
        shader,
        ..Default::default()
    })
}

fn to_skia_rect(rect: Rect) -> Option<tiny_skia::Rect> {
    tiny_skia::Rect::from_ltrb(
        rect.x0 as f32,
        rect.y0 as f32,
        rect.x1 as f32,
        rect.y1 as f32,
    )
}

type TinyBlendMode = tiny_skia::BlendMode;

enum BlendStrategy {
    SinglePass(TinyBlendMode),
    MultiPass {
        first_pass: TinyBlendMode,
        second_pass: TinyBlendMode,
    },
}

fn determine_blend_strategy(peniko_mode: &BlendMode) -> BlendStrategy {
    match (peniko_mode.mix, peniko_mode.compose) {
        #[allow(deprecated)]
        (Mix::Normal | Mix::Clip, compose) => {
            BlendStrategy::SinglePass(compose_to_tiny_blend_mode(compose))
        }
        (mix, Compose::SrcOver) => BlendStrategy::SinglePass(mix_to_tiny_blend_mode(mix)),
        (mix, compose) => BlendStrategy::MultiPass {
            first_pass: compose_to_tiny_blend_mode(compose),
            second_pass: mix_to_tiny_blend_mode(mix),
        },
    }
}

fn compose_to_tiny_blend_mode(compose: Compose) -> TinyBlendMode {
    match compose {
        Compose::Clear => TinyBlendMode::Clear,
        Compose::Copy => TinyBlendMode::Source,
        Compose::Dest => TinyBlendMode::Destination,
        Compose::SrcOver => TinyBlendMode::SourceOver,
        Compose::DestOver => TinyBlendMode::DestinationOver,
        Compose::SrcIn => TinyBlendMode::SourceIn,
        Compose::DestIn => TinyBlendMode::DestinationIn,
        Compose::SrcOut => TinyBlendMode::SourceOut,
        Compose::DestOut => TinyBlendMode::DestinationOut,
        Compose::SrcAtop => TinyBlendMode::SourceAtop,
        Compose::DestAtop => TinyBlendMode::DestinationAtop,
        Compose::Xor => TinyBlendMode::Xor,
        Compose::Plus => TinyBlendMode::Plus,
        Compose::PlusLighter => TinyBlendMode::Plus,
    }
}

fn mix_to_tiny_blend_mode(mix: Mix) -> TinyBlendMode {
    match mix {
        Mix::Normal => TinyBlendMode::SourceOver,
        Mix::Multiply => TinyBlendMode::Multiply,
        Mix::Screen => TinyBlendMode::Screen,
        Mix::Overlay => TinyBlendMode::Overlay,
        Mix::Darken => TinyBlendMode::Darken,
        Mix::Lighten => TinyBlendMode::Lighten,
        Mix::ColorDodge => TinyBlendMode::ColorDodge,
        Mix::ColorBurn => TinyBlendMode::ColorBurn,
        Mix::HardLight => TinyBlendMode::HardLight,
        Mix::SoftLight => TinyBlendMode::SoftLight,
        Mix::Difference => TinyBlendMode::Difference,
        Mix::Exclusion => TinyBlendMode::Exclusion,
        Mix::Hue => TinyBlendMode::Hue,
        Mix::Saturation => TinyBlendMode::Saturation,
        Mix::Color => TinyBlendMode::Color,
        Mix::Luminosity => TinyBlendMode::Luminosity,
        #[allow(deprecated)]
        Mix::Clip => TinyBlendMode::SourceOver,
    }
}

fn apply_layer(layer: &Layer, parent: &mut Layer) {
    match determine_blend_strategy(&layer.blend_mode) {
        BlendStrategy::SinglePass(blend_mode) => {
            let transform = skia_transform_with_scaled_translation(
                parent.transform * layer.combine_transform,
                1.,
                1.,
            );

            parent.pixmap.draw_pixmap(
                0,
                0,
                layer.pixmap.as_ref(),
                &PixmapPaint {
                    opacity: layer.alpha,
                    blend_mode,
                    quality: FilterQuality::Bilinear,
                },
                transform,
                parent.mask.as_ref(),
            );
        }
        BlendStrategy::MultiPass {
            first_pass,
            second_pass,
        } => {
            let original_parent = parent.pixmap.clone();

            let transform = skia_transform_with_scaled_translation(
                parent.transform * layer.combine_transform,
                1.,
                1.,
            );

            parent.pixmap.draw_pixmap(
                0,
                0,
                layer.pixmap.as_ref(),
                &PixmapPaint {
                    opacity: 1.0,
                    blend_mode: first_pass,
                    quality: FilterQuality::Bilinear,
                },
                transform,
                parent.mask.as_ref(),
            );

            let intermediate = parent.pixmap.clone();
            parent.pixmap = original_parent;

            parent.pixmap.draw_pixmap(
                0,
                0,
                intermediate.as_ref(),
                &PixmapPaint {
                    opacity: layer.alpha,
                    blend_mode: second_pass,
                    quality: FilterQuality::Bilinear,
                },
                transform,
                parent.mask.as_ref(),
            );
        }
    }
    parent.transform *= layer.transform;
}

fn skia_transform(affine: Affine, window_scale: f32) -> Transform {
    let transform = affine.as_coeffs();
    Transform::from_row(
        transform[0] as f32,
        transform[1] as f32,
        transform[2] as f32,
        transform[3] as f32,
        transform[4] as f32,
        transform[5] as f32,
    )
    .post_scale(window_scale, window_scale)
}

fn skia_transform_with_scaled_translation(
    affine: Affine,
    translation_scale: f32,
    render_scale: f32,
) -> Transform {
    let transform = affine.as_coeffs();
    Transform::from_row(
        transform[0] as f32,
        transform[1] as f32,
        transform[2] as f32,
        transform[3] as f32,
        transform[4] as f32 * translation_scale,
        transform[5] as f32 * translation_scale,
    )
    .post_scale(render_scale, render_scale)
}
