use std::collections::HashMap;

use anyrender::PaintScene;
use peniko::{StyleRef, color::DynamicColor};
use skia_safe::{
    AlphaType, BlendMode, Canvas, Color, Color4f, ColorType, Data, Font, FontArguments,
    FontHinting, FontMgr, GlyphId, ImageInfo, Matrix, Paint, PaintCap, PaintJoin, PaintStyle,
    Point, RRect, Rect, SamplingOptions, Shader, TileMode, Typeface,
    canvas::{GlyphPositions, SaveLayerRec},
    font::Edging,
    font_arguments::{VariationPosition, variation_position::Coordinate},
    gradient_shader::{Interpolation, interpolation},
    image_filters::{self, CropRect},
    shaders,
};

pub struct SkiaScenePainter<'a> {
    pub(crate) inner: &'a Canvas,
    pub(crate) font_mgr: &'a mut FontMgr,
    pub(crate) typeface_cache: &'a mut HashMap<(u64, u32), Typeface>,
}

impl PaintScene for SkiaScenePainter<'_> {
    fn reset(&mut self) {
        self.inner.clear(Color::WHITE);
    }

    fn push_layer(
        &mut self,
        blend: impl Into<peniko::BlendMode>,
        alpha: f32,
        transform: kurbo::Affine,
        clip: &impl kurbo::Shape,
    ) {
        let mut paint = Paint::default();
        paint.set_alpha_f(alpha);
        paint.set_anti_alias(true);
        paint.set_blend_mode(peniko_blend_to_skia_blend(blend.into()));

        self.inner
            .save_layer(&SaveLayerRec::default().paint(&paint));

        self.inner
            .set_matrix(&kurbo_affine_to_skia_matrix(transform).into());

        self.inner
            .clip_path(&kurbo_shape_to_skia_path(clip), None, None);
    }

    fn pop_layer(&mut self) {
        self.inner.restore();
    }

    fn stroke<'a>(
        &mut self,
        style: &kurbo::Stroke,
        transform: kurbo::Affine,
        brush: impl Into<anyrender::PaintRef<'a>>,
        brush_transform: Option<kurbo::Affine>,
        shape: &impl kurbo::Shape,
    ) {
        self.inner.save();
        self.inner
            .set_matrix(&kurbo_affine_to_skia_matrix(transform).into());

        let mut paint = anyrender_brush_to_skia_paint(brush.into(), brush_transform);
        apply_peniko_style_to_skia_paint(StyleRef::Stroke(style), &mut paint);
        paint.set_anti_alias(true);

        draw_kurbo_shape_to_skia_canvas(self.inner, shape, &paint, None);

        self.inner.restore();
    }

    fn fill<'a>(
        &mut self,
        style: peniko::Fill,
        transform: kurbo::Affine,
        brush: impl Into<anyrender::PaintRef<'a>>,
        brush_transform: Option<kurbo::Affine>,
        shape: &impl kurbo::Shape,
    ) {
        self.inner.save();
        self.inner
            .set_matrix(&kurbo_affine_to_skia_matrix(transform).into());

        let mut paint = anyrender_brush_to_skia_paint(brush.into(), brush_transform);
        paint.set_style(PaintStyle::Fill);
        paint.set_anti_alias(true);

        draw_kurbo_shape_to_skia_canvas(self.inner, shape, &paint, Some(style));

        self.inner.restore();
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'s mut self,
        font: &'a peniko::FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [anyrender::NormalizedCoord],
        style: impl Into<peniko::StyleRef<'a>>,
        brush: impl Into<anyrender::PaintRef<'a>>,
        brush_alpha: f32,
        transform: kurbo::Affine,
        glyph_transform: Option<kurbo::Affine>,
        glyphs: impl Iterator<Item = anyrender::Glyph>,
    ) {
        self.inner.save();
        self.inner
            .set_matrix(&kurbo_affine_to_skia_matrix(transform).into());

        if let Some(affine) = glyph_transform {
            self.inner.concat(&kurbo_affine_to_skia_matrix(affine));
        }

        let mut paint = anyrender_brush_to_skia_paint(brush.into(), None);
        apply_peniko_style_to_skia_paint(style.into(), &mut paint);
        paint.set_alpha_f(brush_alpha);
        paint.set_anti_alias(true);

        let font_key = (font.data.id(), font.index);

        if !self.typeface_cache.contains_key(&font_key) {
            let Some(typeface) = self
                .font_mgr
                .new_from_data(font.data.data(), font.index as usize)
            else {
                let tf = Typeface::make_deserialize(font.data.data(), None);
                eprintln!(
                    "WARNING: failed to load font {} {} {}",
                    font_key.0,
                    font_key.1,
                    tf.is_some()
                );
                return;
            };
            self.typeface_cache.insert(font_key, typeface);
        }

        let original_typeface = self.typeface_cache.get(&font_key).unwrap();
        let mut normalized_typeface: Option<Typeface> = None;

        fn f2dot14_to_f32(raw_value: i16) -> f32 {
            let int = (raw_value >> 14) as f32;
            let fract = (raw_value & !(!0 << 14)) as f32 / (1 << 14) as f32;
            int + fract
        }

        if !normalized_coords.is_empty() {
            let axes = original_typeface
                .variation_design_parameters()
                .unwrap_or_default();
            if !axes.is_empty() {
                let coordinates: Vec<Coordinate> = axes
                    .iter()
                    .zip(normalized_coords.iter().map(|c| f2dot14_to_f32(*c)))
                    .filter(|(_, value)| *value != 0.0)
                    .map(|(axis, factor)| {
                        let value = if factor < 0.0 {
                            lerp_f32(axis.min, axis.def, -factor)
                        } else {
                            lerp_f32(axis.def, axis.max, factor)
                        };

                        Coordinate {
                            axis: axis.tag,
                            value,
                        }
                    })
                    .collect();
                let variation_position = VariationPosition {
                    coordinates: &coordinates,
                };

                normalized_typeface = Some(
                    original_typeface
                        .clone_with_arguments(
                            &FontArguments::new().set_variation_design_position(variation_position),
                        )
                        .unwrap(),
                );
            }
        }

        let typeface = match &normalized_typeface {
            Some(it) => it,
            None => original_typeface,
        };

        let mut font = Font::from_typeface(typeface, font_size);
        font.set_hinting(if hint {
            FontHinting::Normal
        } else {
            FontHinting::None
        });
        font.set_edging(Edging::SubpixelAntiAlias);

        let mut draw_glyphs: Vec<GlyphId> = vec![];
        let mut draw_positions: Vec<Point> = vec![];

        for glyph in glyphs {
            draw_glyphs.push(GlyphId::from(glyph.id as u16));
            draw_positions.push(Point::new(glyph.x, glyph.y));
        }

        self.inner.draw_glyphs_at(
            &draw_glyphs[..],
            GlyphPositions::Points(&draw_positions[..]),
            Point::new(0.0, 0.0),
            &font,
            &paint,
        );

        self.inner.restore();
    }

    fn draw_box_shadow(
        &mut self,
        transform: kurbo::Affine,
        rect: kurbo::Rect,
        brush: peniko::Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.inner.save();
        self.inner
            .set_matrix(&kurbo_affine_to_skia_matrix(transform).into());

        let mut paint = Paint::default();
        paint.set_anti_alias(true);
        paint.set_color4f(
            Color4f::new(
                brush.components[0],
                brush.components[1],
                brush.components[2],
                brush.components[3],
            ),
            None,
        );
        paint.set_style(PaintStyle::Fill);

        paint.set_image_filter(
            image_filters::blur(
                (std_dev as f32, std_dev as f32),
                None,
                None,
                CropRect::NO_CROP_RECT,
            )
            .unwrap(),
        );

        let rrect = RRect::new_nine_patch(
            Rect::new(
                rect.x0 as f32,
                rect.y0 as f32,
                rect.x1 as f32,
                rect.y1 as f32,
            ),
            radius as f32,
            radius as f32,
            radius as f32,
            radius as f32,
        );

        self.inner.draw_rrect(rrect, &paint);

        self.inner.restore();
    }
}

fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn apply_peniko_style_to_skia_paint<'a>(style: peniko::StyleRef<'a>, paint: &mut Paint) {
    match style {
        peniko::StyleRef::Fill(_) => {
            paint.set_style(PaintStyle::Fill);
        }
        peniko::StyleRef::Stroke(stroke) => {
            paint.set_style(PaintStyle::Stroke);
            paint.set_stroke(true);
            paint.set_stroke_width(stroke.width as f32);
            paint.set_stroke_join(match stroke.join {
                kurbo::Join::Bevel => PaintJoin::Bevel,
                kurbo::Join::Miter => PaintJoin::Miter,
                kurbo::Join::Round => PaintJoin::Round,
            });
            paint.set_stroke_cap(match stroke.start_cap {
                kurbo::Cap::Butt => PaintCap::Butt,
                kurbo::Cap::Square => PaintCap::Square,
                kurbo::Cap::Round => PaintCap::Round,
            });
        }
    }
}

fn anyrender_brush_to_skia_paint<'a>(
    brush: anyrender::PaintRef<'a>,
    brush_transform: Option<kurbo::Affine>,
) -> Paint {
    match brush {
        anyrender::Paint::Solid(alpha_color) => Paint::new(
            Color4f::new(
                alpha_color.components[0],
                alpha_color.components[1],
                alpha_color.components[2],
                alpha_color.components[3],
            ),
            None,
        ),
        anyrender::Paint::Gradient(gradient) => {
            let shader = match gradient.kind {
                peniko::GradientKind::Linear(linear_gradient_position) => {
                    let mut colors: Vec<Color4f> = vec![];
                    let mut positions: Vec<f32> = vec![];

                    for color_stop in gradient.stops.iter() {
                        colors.push(peniko_to_skia_dyn_color(color_stop.color));
                        positions.push(color_stop.offset);
                    }
                    let start = skpt(linear_gradient_position.start);
                    let end = skpt(linear_gradient_position.end);

                    let interpolation = Interpolation {
                        color_space: peniko_to_skia_cs_tag_to_interpol_cs(
                            gradient.interpolation_cs,
                        ),
                        in_premul: interpolation::InPremul::Yes,
                        hue_method: peniko_to_skia_hue_direction_to_hue_method(
                            gradient.hue_direction,
                        ),
                    };

                    Shader::linear_gradient_with_interpolation(
                        (start, end),
                        (&colors[..], None),
                        &positions[..],
                        peniko_to_skia_extend_to_tile_mode(gradient.extend),
                        interpolation,
                        &brush_transform.map(kurbo_affine_to_skia_matrix),
                    )
                    .unwrap()
                }
                peniko::GradientKind::Radial(radial_gradient_position) => {
                    let mut colors: Vec<Color4f> = vec![];
                    let mut positions: Vec<f32> = vec![];

                    for color_stop in gradient.stops.iter() {
                        colors.push(peniko_to_skia_dyn_color(color_stop.color));
                        positions.push(color_stop.offset);
                    }

                    let start_center = skpt(radial_gradient_position.start_center);
                    let start_radius = radial_gradient_position.start_radius;
                    let end_center = skpt(radial_gradient_position.end_center);
                    let end_radius = radial_gradient_position.end_radius;

                    let interpolation = Interpolation {
                        color_space: peniko_to_skia_cs_tag_to_interpol_cs(
                            gradient.interpolation_cs,
                        ),
                        in_premul: interpolation::InPremul::Yes,
                        hue_method: peniko_to_skia_hue_direction_to_hue_method(
                            gradient.hue_direction,
                        ),
                    };

                    if start_center == end_center && start_radius == end_radius {
                        Shader::radial_gradient_with_interpolation(
                            (start_center, start_radius),
                            (&colors[..], None),
                            &positions[..],
                            peniko_to_skia_extend_to_tile_mode(gradient.extend),
                            interpolation,
                            &brush_transform.map(kurbo_affine_to_skia_matrix),
                        )
                        .unwrap()
                    } else {
                        Shader::two_point_conical_gradient_with_interpolation(
                            (start_center, start_radius),
                            (end_center, end_radius),
                            (&colors[..], None),
                            &positions[..],
                            peniko_to_skia_extend_to_tile_mode(gradient.extend),
                            interpolation,
                            &brush_transform.map(kurbo_affine_to_skia_matrix),
                        )
                        .unwrap()
                    }
                }
                peniko::GradientKind::Sweep(sweep_gradient_position) => {
                    let mut colors: Vec<Color4f> = vec![];
                    let mut positions: Vec<f32> = vec![];

                    for color_stop in gradient.stops.iter() {
                        colors.push(peniko_to_skia_dyn_color(color_stop.color));
                        positions.push(color_stop.offset);
                    }
                    let center = skpt(sweep_gradient_position.center);

                    let interpolation = Interpolation {
                        color_space: peniko_to_skia_cs_tag_to_interpol_cs(
                            gradient.interpolation_cs,
                        ),
                        in_premul: interpolation::InPremul::Yes,
                        hue_method: peniko_to_skia_hue_direction_to_hue_method(
                            gradient.hue_direction,
                        ),
                    };

                    Shader::sweep_gradient_with_interpolation(
                        center,
                        (&colors[..], None),
                        &positions[..],
                        peniko_to_skia_extend_to_tile_mode(gradient.extend),
                        (
                            rad_to_deg(sweep_gradient_position.start_angle),
                            rad_to_deg(sweep_gradient_position.end_angle),
                        ),
                        interpolation,
                        &brush_transform.map(kurbo_affine_to_skia_matrix),
                    )
                    .unwrap()
                }
            };

            let mut paint = Paint::default();
            paint.set_shader(Some(shader));
            paint
        }
        anyrender::Paint::Image(brush) => {
            let src_image = brush.image;

            let image_info = ImageInfo::new(
                (src_image.width as i32, src_image.height as i32),
                match src_image.format {
                    peniko::ImageFormat::Rgba8 => ColorType::RGBA8888,
                    peniko::ImageFormat::Bgra8 => ColorType::BGRA8888,
                    _ => unreachable!(),
                },
                match src_image.alpha_type {
                    peniko::ImageAlphaType::Alpha => AlphaType::Unpremul,
                    peniko::ImageAlphaType::AlphaPremultiplied => AlphaType::Premul,
                },
                None,
            );
            let pixels = unsafe {
                Data::new_bytes(src_image.data.data()) // We have to ensure the src image data lives long enough
            };
            let image = skia_safe::images::raster_from_data(
                &image_info,
                pixels,
                image_info.min_row_bytes(),
            )
            .unwrap();

            let shader = shaders::image(
                image,
                (
                    peniko_to_skia_extend_to_tile_mode(brush.sampler.x_extend),
                    peniko_to_skia_extend_to_tile_mode(brush.sampler.y_extend),
                ),
                &SamplingOptions::default(),
                &brush_transform.map(kurbo_affine_to_skia_matrix),
            );

            let mut paint = Paint::default();
            paint.set_shader(shader);
            paint
        }
        anyrender::Paint::Custom(_) => unreachable!(), // ToDo: figure out what to do with this
    }
}

fn rad_to_deg(rad: f32) -> f32 {
    if rad == 0.0 {
        return 0.0;
    }

    rad * 180.0 / std::f32::consts::PI
}

fn peniko_to_skia_extend_to_tile_mode(extend: peniko::Extend) -> TileMode {
    match extend {
        peniko::Extend::Pad => TileMode::Clamp,
        peniko::Extend::Repeat => TileMode::Repeat,
        peniko::Extend::Reflect => TileMode::Mirror,
    }
}

fn peniko_to_skia_hue_direction_to_hue_method(
    direction: peniko::color::HueDirection,
) -> interpolation::HueMethod {
    match direction {
        peniko::color::HueDirection::Shorter => interpolation::HueMethod::Shorter,
        peniko::color::HueDirection::Longer => interpolation::HueMethod::Longer,
        peniko::color::HueDirection::Increasing => interpolation::HueMethod::Increasing,
        peniko::color::HueDirection::Decreasing => interpolation::HueMethod::Decreasing,
        _ => unreachable!(),
    }
}

fn peniko_to_skia_cs_tag_to_interpol_cs(
    space: peniko::color::ColorSpaceTag,
) -> interpolation::ColorSpace {
    match space {
        peniko::color::ColorSpaceTag::Srgb => interpolation::ColorSpace::SRGB,
        peniko::color::ColorSpaceTag::LinearSrgb => interpolation::ColorSpace::SRGBLinear,
        peniko::color::ColorSpaceTag::Lab => interpolation::ColorSpace::Lab,
        peniko::color::ColorSpaceTag::Lch => interpolation::ColorSpace::LCH,
        peniko::color::ColorSpaceTag::Hsl => interpolation::ColorSpace::HSL,
        peniko::color::ColorSpaceTag::Hwb => interpolation::ColorSpace::HWB,
        peniko::color::ColorSpaceTag::Oklab => interpolation::ColorSpace::OKLab,
        peniko::color::ColorSpaceTag::Oklch => interpolation::ColorSpace::OKLCH,
        peniko::color::ColorSpaceTag::DisplayP3 => interpolation::ColorSpace::DisplayP3,
        peniko::color::ColorSpaceTag::A98Rgb => interpolation::ColorSpace::A98RGB,
        peniko::color::ColorSpaceTag::ProphotoRgb => interpolation::ColorSpace::ProphotoRGB,
        peniko::color::ColorSpaceTag::Rec2020 => interpolation::ColorSpace::Rec2020,
        _ => interpolation::ColorSpace::SRGB, // ToDo: overview unsupported color space tags and possibly document it, for now just fallback
    }
}

fn peniko_to_skia_dyn_color(color: DynamicColor) -> Color4f {
    Color4f::new(
        color.components[0],
        color.components[1],
        color.components[2],
        color.components[3],
    )
}

fn kurbo_affine_to_skia_matrix(affine: kurbo::Affine) -> Matrix {
    let m = affine.as_coeffs();
    let scale_x = m[0] as f32;
    let shear_y = m[1] as f32;
    let shear_x = m[2] as f32;
    let scale_y = m[3] as f32;
    let translate_x = m[4] as f32;
    let translate_y = m[5] as f32;

    Matrix::new_all(
        scale_x,
        shear_x,
        translate_x,
        shear_y,
        scale_y,
        translate_y,
        0.0,
        0.0,
        1.0,
    )
}

#[allow(deprecated)] // We need to support it even though it's deprecated
fn peniko_blend_to_skia_blend(blend_mode: peniko::BlendMode) -> BlendMode {
    if blend_mode.mix == peniko::Mix::Normal || blend_mode.mix == peniko::Mix::Clip {
        match blend_mode.compose {
            peniko::Compose::Clear => BlendMode::Clear,
            peniko::Compose::Copy => BlendMode::Src,
            peniko::Compose::Dest => BlendMode::Dst,
            peniko::Compose::SrcOver => BlendMode::SrcOver,
            peniko::Compose::DestOver => BlendMode::DstOver,
            peniko::Compose::SrcIn => BlendMode::SrcIn,
            peniko::Compose::DestIn => BlendMode::DstIn,
            peniko::Compose::SrcOut => BlendMode::SrcOut,
            peniko::Compose::DestOut => BlendMode::DstOut,
            peniko::Compose::SrcAtop => BlendMode::SrcATop,
            peniko::Compose::DestAtop => BlendMode::DstATop,
            peniko::Compose::Xor => BlendMode::Xor,
            peniko::Compose::Plus => BlendMode::Plus,
            peniko::Compose::PlusLighter => BlendMode::Plus,
        }
    } else {
        match blend_mode.mix {
            peniko::Mix::Normal => unreachable!(), // Handled above
            peniko::Mix::Multiply => BlendMode::Multiply,
            peniko::Mix::Screen => BlendMode::Screen,
            peniko::Mix::Overlay => BlendMode::Overlay,
            peniko::Mix::Darken => BlendMode::Darken,
            peniko::Mix::Lighten => BlendMode::Lighten,
            peniko::Mix::ColorDodge => BlendMode::ColorDodge,
            peniko::Mix::ColorBurn => BlendMode::ColorBurn,
            peniko::Mix::HardLight => BlendMode::HardLight,
            peniko::Mix::SoftLight => BlendMode::SoftLight,
            peniko::Mix::Difference => BlendMode::Difference,
            peniko::Mix::Exclusion => BlendMode::Exclusion,
            peniko::Mix::Hue => BlendMode::Hue,
            peniko::Mix::Saturation => BlendMode::Saturation,
            peniko::Mix::Color => BlendMode::Color,
            peniko::Mix::Luminosity => BlendMode::Luminosity,
            peniko::Mix::Clip => unreachable!(), // Handled above
        }
    }
}

fn draw_kurbo_shape_to_skia_canvas(
    canvas: &skia_safe::Canvas,
    shape: &impl kurbo::Shape,
    paint: &skia_safe::Paint,
    fill_type: Option<peniko::Fill>,
) {
    if let Some(rect) = shape.as_rect() {
        canvas.draw_rect(
            Rect::new(
                rect.x0 as f32,
                rect.y0 as f32,
                rect.x1 as f32,
                rect.y1 as f32,
            ),
            paint,
        );
    } else if let Some(rrect) = shape.as_rounded_rect() {
        let rect = Rect::new(
            rrect.rect().x0 as f32,
            rrect.rect().y0 as f32,
            rrect.rect().x1 as f32,
            rrect.rect().y1 as f32,
        );
        canvas.draw_rrect(
            RRect::new_nine_patch(
                rect,
                rrect.radii().bottom_left as f32,
                rrect.radii().top_left as f32,
                rrect.radii().top_right as f32,
                rrect.radii().bottom_right as f32,
            ),
            paint,
        );
    } else if let Some(line) = shape.as_line() {
        canvas.draw_line(
            (line.p0.x as f32, line.p0.y as f32),
            (line.p1.x as f32, line.p1.y as f32),
            paint,
        );
    } else if let Some(circle) = shape.as_circle() {
        canvas.draw_circle(
            (circle.center.x as f32, circle.center.y as f32),
            circle.radius as f32,
            paint,
        );
    } else if let Some(path_els) = shape.as_path_slice() {
        let mut path = kurbo_bezpath_els_to_skia_path(path_els);
        if let Some(fill_type) = fill_type {
            path.set_fill_type(to_skia_fill_type(fill_type));
        }
        canvas.draw_path(&path, paint);
    } else {
        let mut path = kurbo_shape_to_skia_path(shape);
        if let Some(fill_type) = fill_type {
            path.set_fill_type(to_skia_fill_type(fill_type));
        }
        canvas.draw_path(&path, paint);
    }
}

fn kurbo_shape_to_skia_path(shape: &impl kurbo::Shape) -> skia_safe::Path {
    let mut sk_path = skia_safe::Path::new();
    for el in shape.path_elements(0.1) {
        add_kurbo_bezpath_el_to_skia_path(&el, &mut sk_path);
    }
    sk_path
}

fn kurbo_bezpath_els_to_skia_path(path: &[kurbo::PathEl]) -> skia_safe::Path {
    let mut sk_path = skia_safe::Path::new();
    for el in path {
        add_kurbo_bezpath_el_to_skia_path(el, &mut sk_path);
    }
    sk_path
}

fn add_kurbo_bezpath_el_to_skia_path(path_el: &kurbo::PathEl, skia_path: &mut skia_safe::Path) {
    match path_el {
        kurbo::PathEl::MoveTo(p) => _ = skia_path.move_to(skpt(*p)),
        kurbo::PathEl::LineTo(p) => _ = skia_path.line_to(skpt(*p)),
        kurbo::PathEl::QuadTo(p1, p2) => _ = skia_path.quad_to(skpt(*p1), skpt(*p2)),
        kurbo::PathEl::CurveTo(p1, p2, p3) => {
            _ = skia_path.cubic_to(skpt(*p1), skpt(*p2), skpt(*p3))
        }
        kurbo::PathEl::ClosePath => _ = skia_path.close(),
    };
}

fn to_skia_fill_type(fill: peniko::Fill) -> skia_safe::PathFillType {
    match fill {
        peniko::Fill::NonZero => skia_safe::PathFillType::Winding,
        peniko::Fill::EvenOdd => skia_safe::PathFillType::EvenOdd,
    }
}

fn skpt(p: kurbo::Point) -> skia_safe::Point {
    (p.x as f32, p.y as f32).into()
}
