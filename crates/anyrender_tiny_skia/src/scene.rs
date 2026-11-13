//! A tiny-skia implementation of [`PaintScene`].
//!
//! Every layer is a full-canvas premultiplied pixmap. Transforms are passed straight to tiny-skia
//! (nothing is accumulated on the layer stack), and a layer's clip shape becomes a coverage
//! [`Mask`] that is applied when the layer is composited back into its parent. This mirrors how
//! resvg renders SVG groups.

use std::{collections::HashMap, sync::Arc};

use anyrender::{
    BoxShadowGeometry, BoxShadowKind, Filter, Glyph, MaskMode, NonUniformRoundedRect,
    NormalizedCoord, Paint as AnyPaint, PaintRef, PaintScene, RenderContext,
};
use kurbo::{Affine, PathEl, Point, Shape, Stroke, Vec2};
use peniko::{
    BlendMode, Color, Compose, Extend, Fill, FontData, Gradient, GradientKind, ImageAlphaType,
    ImageBrushRef, ImageData, ImageFormat, ImageQuality, Mix, StyleRef, color::Srgb,
};
use skrifa::{
    GlyphId, MetadataProvider,
    instance::{LocationRef, Size},
    outline::{DrawSettings, OutlinePen},
    raw::types::F2Dot14,
};
use tiny_skia::{
    FillRule, FilterQuality, GradientStop, IntRect, LinearGradient, Mask, MaskType, Paint, Path,
    PathBuilder, Pattern, Pixmap, PixmapPaint, RadialGradient, Shader, SpreadMode, SweepGradient,
    Transform,
};

use crate::filter;

/// Tolerance used when flattening kurbo shapes into tiny-skia paths.
const TOLERANCE: f64 = 0.1;

enum LayerKind {
    Root,
    Group {
        blend: tiny_skia::BlendMode,
        alpha: f32,
        filter: Option<(Arc<Filter>, Affine)>,
    },
    Clip,
    Mask(MaskMode),
}

struct Layer {
    pixmap: Pixmap,
    /// Coverage of the layer's clip shape, in canvas space. `None` means "no clip".
    clip: Option<Mask>,
    /// Pixel bounds of the clip shape (used as the filter region).
    bounds: IntRect,
    kind: LayerKind,
}

#[derive(Hash, PartialEq, Eq)]
struct GlyphKey {
    font_id: u64,
    font_index: u32,
    glyph_id: u32,
    size_bits: u32,
    coords: Vec<NormalizedCoord>,
}

pub struct TinySkiaScenePainter {
    width: u32,
    height: u32,
    layers: Vec<Layer>,
    spare_pixmaps: Vec<Pixmap>,
    image_cache: HashMap<(u64, u32, u32), Arc<Pixmap>>,
    glyph_cache: HashMap<GlyphKey, Option<Path>>,
}

impl TinySkiaScenePainter {
    pub fn new(width: u32, height: u32) -> Self {
        let mut painter = Self {
            width: width.max(1),
            height: height.max(1),
            layers: Vec::new(),
            spare_pixmaps: Vec::new(),
            image_cache: HashMap::new(),
            glyph_cache: HashMap::new(),
        };
        painter.reset();
        painter
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
        self.spare_pixmaps.clear();
        self.layers.clear();
        self.reset();
    }

    /// The root pixmap (premultiplied RGBA8).
    pub fn pixmap(&self) -> &Pixmap {
        &self.layers[0].pixmap
    }

    fn full_rect(&self) -> IntRect {
        IntRect::from_xywh(0, 0, self.width, self.height).unwrap()
    }

    fn take_pixmap(&mut self) -> Pixmap {
        match self.spare_pixmaps.pop() {
            Some(mut pixmap) => {
                pixmap.fill(tiny_skia::Color::TRANSPARENT);
                pixmap
            }
            None => Pixmap::new(self.width, self.height).unwrap(),
        }
    }

    fn top(&mut self) -> &mut Pixmap {
        &mut self.layers.last_mut().unwrap().pixmap
    }

    fn push(&mut self, fill: Fill, transform: Affine, clip: &impl Shape, kind: LayerKind) {
        let (clip, bounds) = self.clip_mask(fill, transform, clip);
        let pixmap = self.take_pixmap();
        self.layers.push(Layer {
            pixmap,
            clip,
            bounds,
            kind,
        });
    }

    /// Rasterizes a clip shape into a coverage mask. Returns `None` for the mask when the shape
    /// covers the whole canvas (so no masking is needed).
    fn clip_mask(
        &self,
        fill: Fill,
        transform: Affine,
        shape: &impl Shape,
    ) -> (Option<Mask>, IntRect) {
        let full = self.full_rect();
        if let Some(rect) = shape.as_rect()
            && is_axis_aligned(transform)
        {
            let r = transform.transform_rect_bbox(rect);
            if r.x0 <= 0.0 && r.y0 <= 0.0 && r.x1 >= self.width as f64 && r.y1 >= self.height as f64
            {
                return (None, full);
            }
        }

        let mut mask = Mask::new(self.width, self.height).unwrap();
        let bbox = transform.transform_rect_bbox(shape.bounding_box());
        let bounds = IntRect::from_ltrb(
            bbox.x0.floor().max(0.0) as i32,
            bbox.y0.floor().max(0.0) as i32,
            bbox.x1.ceil().min(self.width as f64) as i32,
            bbox.y1.ceil().min(self.height as f64) as i32,
        )
        .unwrap_or(IntRect::from_xywh(0, 0, 1, 1).unwrap());
        if let Some(path) = to_path(shape) {
            mask.fill_path(&path, fill_rule(fill), true, skia_transform(transform));
        }
        (Some(mask), bounds)
    }

    fn image_pixmap(&mut self, image: &ImageData) -> Option<Arc<Pixmap>> {
        let key = (image.data.id(), image.width, image.height);
        if let Some(pixmap) = self.image_cache.get(&key) {
            return Some(pixmap.clone());
        }
        let pixmap = Arc::new(image_to_pixmap(image)?);
        self.image_cache.insert(key, pixmap.clone());
        Some(pixmap)
    }

    /// Builds a tiny-skia paint for `brush` and passes it, along with the current target pixmap,
    /// to `draw`.
    fn with_paint(
        &mut self,
        brush: PaintRef<'_>,
        brush_alpha: f32,
        brush_transform: Option<Affine>,
        draw: impl FnOnce(&mut Pixmap, &Paint<'_>),
    ) {
        let brush_transform = skia_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        let image;
        let shader = match brush {
            AnyPaint::Solid(color) => Shader::SolidColor(to_skia_color(color, brush_alpha)),
            AnyPaint::Gradient(gradient) => {
                match gradient_shader(gradient, brush_alpha, brush_transform) {
                    Some(shader) => shader,
                    None => return,
                }
            }
            AnyPaint::Image(brush) => {
                image = match self.image_pixmap(brush.image) {
                    Some(image) => image,
                    None => return,
                };
                image_shader(&image, &brush, brush_alpha, brush_transform)
            }
            AnyPaint::Resource(_) | AnyPaint::Custom(_) => return,
        };
        let paint = Paint {
            shader,
            anti_alias: true,
            ..Default::default()
        };
        draw(self.top(), &paint);
    }

    fn fill_path(
        &mut self,
        fill: Fill,
        transform: Affine,
        brush: PaintRef<'_>,
        brush_alpha: f32,
        brush_transform: Option<Affine>,
        path: &Path,
    ) {
        let transform = skia_transform(transform);
        self.with_paint(brush, brush_alpha, brush_transform, |pixmap, paint| {
            pixmap.fill_path(path, paint, fill_rule(fill), transform, None);
        });
    }

    fn stroke_path(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: PaintRef<'_>,
        brush_alpha: f32,
        brush_transform: Option<Affine>,
        path: &Path,
    ) {
        if style.width <= 0.0 {
            return;
        }
        let stroke = to_skia_stroke(style);
        let transform = skia_transform(transform);
        self.with_paint(brush, brush_alpha, brush_transform, |pixmap, paint| {
            pixmap.stroke_path(path, paint, &stroke, transform, None);
        });
    }

    fn glyph_path(
        &mut self,
        font: &FontData,
        font_size: f32,
        coords: &[NormalizedCoord],
        glyph_id: u32,
    ) -> Option<Path> {
        let key = GlyphKey {
            font_id: font.data.id(),
            font_index: font.index,
            glyph_id,
            size_bits: font_size.to_bits(),
            coords: coords.to_vec(),
        };
        self.glyph_cache
            .entry(key)
            .or_insert_with(|| {
                let font_ref = skrifa::FontRef::from_index(font.data.data(), font.index).ok()?;
                let outline = font_ref.outline_glyphs().get(GlyphId::new(glyph_id))?;
                let coords: Vec<F2Dot14> = coords.iter().map(|c| F2Dot14::from_bits(*c)).collect();
                let settings =
                    DrawSettings::unhinted(Size::new(font_size), LocationRef::new(&coords));
                let mut pen = SkiaPen(PathBuilder::new());
                outline.draw(settings, &mut pen).ok()?;
                pen.0.finish()
            })
            .clone()
    }
}

impl RenderContext for TinySkiaScenePainter {}

impl PaintScene for TinySkiaScenePainter {
    fn reset(&mut self) {
        while self.layers.len() > 1 {
            let layer = self.layers.pop().unwrap();
            self.spare_pixmaps.push(layer.pixmap);
        }
        if self.layers.is_empty() {
            let pixmap = self.take_pixmap();
            let bounds = self.full_rect();
            self.layers.push(Layer {
                pixmap,
                clip: None,
                bounds,
                kind: LayerKind::Root,
            });
        } else {
            self.layers[0].pixmap.fill(tiny_skia::Color::TRANSPARENT);
        }
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
        let kind = LayerKind::Group {
            blend: to_skia_blend(blend.into()),
            alpha,
            filter: filter.map(|filter| (filter, transform)),
        };
        self.push(fill, transform, clip, kind);
    }

    fn push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &impl Shape) {
        self.push(fill, transform, clip, LayerKind::Clip);
    }

    fn push_mask_layer(
        &mut self,
        fill: Fill,
        mode: MaskMode,
        transform: Affine,
        clip: &impl Shape,
    ) {
        self.push(fill, transform, clip, LayerKind::Mask(mode));
    }

    fn pop_layer(&mut self) {
        if self.layers.len() <= 1 {
            return;
        }
        let mut layer = self.layers.pop().unwrap();
        let parent = &mut self.layers.last_mut().unwrap().pixmap;
        match layer.kind {
            LayerKind::Root => unreachable!(),
            LayerKind::Clip => {
                parent.draw_pixmap(
                    0,
                    0,
                    layer.pixmap.as_ref(),
                    &PixmapPaint::default(),
                    Transform::identity(),
                    layer.clip.as_ref(),
                );
            }
            LayerKind::Group {
                blend,
                alpha,
                ref filter,
            } => {
                if let Some((filter, transform)) = filter {
                    filter::apply(filter, *transform, layer.bounds, &mut layer.pixmap);
                }
                parent.draw_pixmap(
                    0,
                    0,
                    layer.pixmap.as_ref(),
                    &PixmapPaint {
                        opacity: alpha.clamp(0.0, 1.0),
                        blend_mode: blend,
                        quality: FilterQuality::Nearest,
                    },
                    Transform::identity(),
                    layer.clip.as_ref(),
                );
            }
            LayerKind::Mask(mode) => {
                let mask_type = match mode {
                    MaskMode::Alpha => MaskType::Alpha,
                    MaskMode::Luminance => MaskType::Luminance,
                };
                let mut mask = Mask::from_pixmap(layer.pixmap.as_ref(), mask_type);
                if let Some(clip) = &layer.clip {
                    for (m, c) in mask.data_mut().iter_mut().zip(clip.data()) {
                        *m = ((*m as u16 * *c as u16 + 127) / 255) as u8;
                    }
                }
                parent.apply_mask(&mask);
            }
        }
        self.spare_pixmaps.push(layer.pixmap);
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let Some(path) = to_path(shape) else { return };
        self.stroke_path(style, transform, brush.into(), 1.0, brush_transform, &path);
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        brush: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let Some(path) = to_path(shape) else { return };
        self.fill_path(style, transform, brush.into(), 1.0, brush_transform, &path);
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'s mut self,
        font: &'a FontData,
        font_size: f32,
        _hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        _embolden: Vec2,
        style: impl Into<StyleRef<'a>>,
        brush: impl Into<PaintRef<'a>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
    ) {
        let style = style.into();
        let brush = brush.into();
        let glyph_transform = glyph_transform.unwrap_or(Affine::IDENTITY);
        for glyph in glyphs {
            let Some(path) = self.glyph_path(font, font_size, normalized_coords, glyph.id) else {
                continue;
            };
            let transform =
                transform * Affine::translate((glyph.x as f64, glyph.y as f64)) * glyph_transform;
            match style {
                StyleRef::Fill(fill) => {
                    self.fill_path(fill, transform, brush.clone(), brush_alpha, None, &path)
                }
                StyleRef::Stroke(stroke) => {
                    self.stroke_path(stroke, transform, brush.clone(), brush_alpha, None, &path)
                }
            }
        }
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        box_shape: &NonUniformRoundedRect,
        offset: Vec2,
        spread: f64,
        std_dev: f64,
        color: Color,
        kind: BoxShadowKind,
    ) {
        let geometry = BoxShadowGeometry::new(box_shape, offset, spread, std_dev, kind);
        if geometry.std_dev <= 0.0 {
            geometry.draw_unblurred(self, transform, color);
            return;
        }

        // The shape to blur. For inset shadows, the box (grown so that its outer edge is
        // never reached by the blur) with the shadow's hole cut out of it.
        let path = if geometry.is_inset() {
            let outer = geometry
                .box_shape
                .bounding_box()
                .union(geometry.shadow.bounding_box())
                .inflate(3.0 * geometry.std_dev + 1.0, 3.0 * geometry.std_dev + 1.0);
            let mut path = outer.to_path(TOLERANCE);
            path.extend(geometry.shadow.to_path(TOLERANCE).reverse_subpaths());
            path
        } else {
            geometry.shadow.to_path(TOLERANCE)
        };
        let Some(path) = to_path(&path) else { return };

        let mut shadow = self.take_pixmap();
        let paint = Paint {
            shader: Shader::SolidColor(to_skia_color(color, 1.0)),
            anti_alias: true,
            ..Default::default()
        };
        shadow.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            skia_transform(transform),
            None,
        );
        let (sx, sy) = transform_scale(transform);
        filter::blur(
            &mut shadow,
            self.full_rect(),
            geometry.std_dev * sx,
            geometry.std_dev * sy,
        );

        let (clip, _) = if geometry.needs_clip() {
            self.clip_mask(Fill::NonZero, transform, &geometry.area)
        } else {
            (None, self.full_rect())
        };
        self.top().draw_pixmap(
            0,
            0,
            shadow.as_ref(),
            &PixmapPaint::default(),
            Transform::identity(),
            clip.as_ref(),
        );
        self.spare_pixmaps.push(shadow);
    }
}

struct SkiaPen(PathBuilder);

impl OutlinePen for SkiaPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.0.move_to(x, -y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.0.line_to(x, -y);
    }
    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        self.0.quad_to(cx0, -cy0, x, -y);
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.0.cubic_to(cx0, -cy0, cx1, -cy1, x, -y);
    }
    fn close(&mut self) {
        self.0.close();
    }
}

fn is_axis_aligned(transform: Affine) -> bool {
    let [_, b, c, _, _, _] = transform.as_coeffs();
    b == 0.0 && c == 0.0
}

pub(crate) fn transform_scale(transform: Affine) -> (f64, f64) {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    (a.hypot(b), c.hypot(d))
}

pub(crate) fn skia_transform(affine: Affine) -> Transform {
    let [a, b, c, d, e, f] = affine.as_coeffs();
    Transform::from_row(a as f32, b as f32, c as f32, d as f32, e as f32, f as f32)
}

fn fill_rule(fill: Fill) -> FillRule {
    match fill {
        Fill::NonZero => FillRule::Winding,
        Fill::EvenOdd => FillRule::EvenOdd,
    }
}

pub(crate) fn to_skia_color(
    color: peniko::color::AlphaColor<Srgb>,
    alpha: f32,
) -> tiny_skia::Color {
    let [r, g, b, a] = color.components;
    tiny_skia::Color::from_rgba(
        r.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b.clamp(0.0, 1.0),
        (a * alpha).clamp(0.0, 1.0),
    )
    .unwrap_or(tiny_skia::Color::TRANSPARENT)
}

fn to_path(shape: &impl Shape) -> Option<Path> {
    let mut pb = PathBuilder::new();
    for el in shape.path_elements(TOLERANCE) {
        match el {
            PathEl::MoveTo(p) => pb.move_to(p.x as f32, p.y as f32),
            PathEl::LineTo(p) => pb.line_to(p.x as f32, p.y as f32),
            PathEl::QuadTo(p1, p2) => {
                pb.quad_to(p1.x as f32, p1.y as f32, p2.x as f32, p2.y as f32)
            }
            PathEl::CurveTo(p1, p2, p3) => pb.cubic_to(
                p1.x as f32,
                p1.y as f32,
                p2.x as f32,
                p2.y as f32,
                p3.x as f32,
                p3.y as f32,
            ),
            PathEl::ClosePath => pb.close(),
        }
    }
    pb.finish()
}

fn to_skia_stroke(style: &Stroke) -> tiny_skia::Stroke {
    let cap = match style.start_cap {
        kurbo::Cap::Butt => tiny_skia::LineCap::Butt,
        kurbo::Cap::Square => tiny_skia::LineCap::Square,
        kurbo::Cap::Round => tiny_skia::LineCap::Round,
    };
    let join = match style.join {
        kurbo::Join::Bevel => tiny_skia::LineJoin::Bevel,
        kurbo::Join::Miter => tiny_skia::LineJoin::Miter,
        kurbo::Join::Round => tiny_skia::LineJoin::Round,
    };
    let dash = if style.dash_pattern.is_empty() {
        None
    } else {
        let mut array: Vec<f32> = style.dash_pattern.iter().map(|d| *d as f32).collect();
        if array.len() % 2 == 1 {
            array.extend_from_within(..);
        }
        tiny_skia::StrokeDash::new(array, style.dash_offset as f32)
    };
    tiny_skia::Stroke {
        width: style.width as f32,
        miter_limit: style.miter_limit as f32,
        line_cap: cap,
        line_join: join,
        dash,
    }
}

fn spread_mode(extend: Extend) -> SpreadMode {
    match extend {
        Extend::Pad => SpreadMode::Pad,
        Extend::Repeat => SpreadMode::Repeat,
        Extend::Reflect => SpreadMode::Reflect,
    }
}

fn gradient_shader(
    gradient: &Gradient,
    alpha: f32,
    transform: Transform,
) -> Option<Shader<'static>> {
    let stops: Vec<GradientStop> = gradient
        .stops
        .iter()
        .map(|stop| {
            let color = stop.color.to_alpha_color::<Srgb>();
            GradientStop::new(stop.offset, to_skia_color(color, alpha))
        })
        .collect();
    if stops.is_empty() {
        return None;
    }
    if stops.len() == 1 {
        return Some(Shader::SolidColor(stops_color(&gradient.stops[0], alpha)));
    }
    let mode = spread_mode(gradient.extend);
    let point = |p: Point| tiny_skia::Point::from_xy(p.x as f32, p.y as f32);
    match gradient.kind {
        GradientKind::Linear(pos) => {
            LinearGradient::new(point(pos.start), point(pos.end), stops, mode, transform)
        }
        GradientKind::Radial(pos) => RadialGradient::new(
            point(pos.start_center),
            pos.start_radius,
            point(pos.end_center),
            pos.end_radius,
            stops,
            mode,
            transform,
        ),
        GradientKind::Sweep(pos) => SweepGradient::new(
            point(pos.center),
            pos.start_angle.to_degrees(),
            pos.end_angle.to_degrees(),
            stops,
            mode,
            transform,
        ),
    }
}

fn stops_color(stop: &peniko::ColorStop, alpha: f32) -> tiny_skia::Color {
    to_skia_color(stop.color.to_alpha_color::<Srgb>(), alpha)
}

fn image_shader<'a>(
    image: &'a Pixmap,
    brush: &ImageBrushRef<'_>,
    alpha: f32,
    transform: Transform,
) -> Shader<'a> {
    let quality = match brush.sampler.quality {
        ImageQuality::Low => FilterQuality::Nearest,
        ImageQuality::Medium => FilterQuality::Bilinear,
        ImageQuality::High => FilterQuality::Bicubic,
    };
    Pattern::new(
        image.as_ref(),
        spread_mode(brush.sampler.x_extend),
        quality,
        (brush.sampler.alpha * alpha).clamp(0.0, 1.0),
        transform,
    )
}

fn image_to_pixmap(image: &ImageData) -> Option<Pixmap> {
    let mut pixmap = Pixmap::new(image.width, image.height)?;
    let src = image.data.data();
    let len = image.width as usize * image.height as usize * 4;
    if src.len() < len {
        return None;
    }
    let premultiplied = matches!(image.alpha_type, ImageAlphaType::AlphaPremultiplied);
    for (dst, src) in pixmap
        .data_mut()
        .chunks_exact_mut(4)
        .zip(src[..len].chunks_exact(4))
    {
        let [r, g, b, a] = match image.format {
            ImageFormat::Bgra8 => [src[2], src[1], src[0], src[3]],
            _ => [src[0], src[1], src[2], src[3]],
        };
        if premultiplied {
            dst.copy_from_slice(&[r.min(a), g.min(a), b.min(a), a]);
        } else {
            let premul = |c: u8| ((c as u16 * a as u16 + 127) / 255) as u8;
            dst.copy_from_slice(&[premul(r), premul(g), premul(b), a]);
        }
    }
    Some(pixmap)
}

pub(crate) fn to_skia_blend(blend: BlendMode) -> tiny_skia::BlendMode {
    use tiny_skia::BlendMode as B;
    #[allow(deprecated)]
    match blend.mix {
        Mix::Normal => {}
        Mix::Multiply => return B::Multiply,
        Mix::Screen => return B::Screen,
        Mix::Overlay => return B::Overlay,
        Mix::Darken => return B::Darken,
        Mix::Lighten => return B::Lighten,
        Mix::ColorDodge => return B::ColorDodge,
        Mix::ColorBurn => return B::ColorBurn,
        Mix::HardLight => return B::HardLight,
        Mix::SoftLight => return B::SoftLight,
        Mix::Difference => return B::Difference,
        Mix::Exclusion => return B::Exclusion,
        Mix::Hue => return B::Hue,
        Mix::Saturation => return B::Saturation,
        Mix::Color => return B::Color,
        Mix::Luminosity => return B::Luminosity,
    }
    match blend.compose {
        Compose::Clear => B::Clear,
        Compose::Copy => B::Source,
        Compose::Dest => B::Destination,
        Compose::SrcOver => B::SourceOver,
        Compose::DestOver => B::DestinationOver,
        Compose::SrcIn => B::SourceIn,
        Compose::DestIn => B::DestinationIn,
        Compose::SrcOut => B::SourceOut,
        Compose::DestOut => B::DestinationOut,
        Compose::SrcAtop => B::SourceAtop,
        Compose::DestAtop => B::DestinationAtop,
        Compose::Xor => B::Xor,
        Compose::Plus => B::Plus,
        Compose::PlusLighter => B::Plus,
    }
}
