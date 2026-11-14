//! Evaluation of a [`Filter`] graph on tiny-skia pixmaps.
//!
//! Like resvg, each primitive operates in its color space (`color-interpolation-filters`) on the
//! pixels of the filter region.

use image_filters_types::composite::CompositeOperator;
use image_filters_types::{
    BlendMode, Filter, FilterColorSpace, FilterEffect, FilterInput, FilterSource,
    lighting::{LightSource, PointLightSource, SpotLightSource},
    turbulence::TurbulenceType,
};
use kurbo::Affine;
use peniko::color::{AlphaColor, Srgb};
use rgb::FromSlice;
use tiny_skia::{IntRect, Pixmap, PixmapPaint, Transform};

use super::kernels::{
    ImageRef, ImageRefMut, box_blur, color_matrix, component_transfer, composite, convolve_matrix,
    demultiply_alpha, displacement_map, from_linear_rgb, iir_blur, into_linear_rgb, lighting,
    morphology, multiply_alpha, turbulence,
};

#[derive(Clone)]
struct Image {
    pixmap: Pixmap,
    linear: bool,
}

impl Image {
    /// Converts the image into linear RGB if `linear` is set, or into sRGB otherwise.
    fn into_color_space(mut self, linear: bool) -> Pixmap {
        if linear && !self.linear {
            convert(&mut self.pixmap, into_linear_rgb);
        }
        if !linear && self.linear {
            convert(&mut self.pixmap, from_linear_rgb);
        }
        self.pixmap
    }

    fn into_srgb(mut self) -> Pixmap {
        if self.linear {
            convert(&mut self.pixmap, from_linear_rgb);
        }
        self.pixmap
    }
}

fn convert(pixmap: &mut Pixmap, f: fn(&mut [rgb::RGBA8])) {
    let data = pixmap.data_mut().as_rgba_mut();
    demultiply_alpha(data);
    f(data);
    multiply_alpha(data);
}

fn image_mut(pixmap: &mut Pixmap) -> ImageRefMut<'_> {
    let (w, h) = (pixmap.width(), pixmap.height());
    ImageRefMut::new(w, h, pixmap.data_mut().as_rgba_mut())
}

/// Blurs `region` of `pixmap` with the given device-space standard deviations.
pub fn blur(pixmap: &mut Pixmap, region: IntRect, std_dx: f64, std_dy: f64) {
    let Some(mut sub) = pixmap.clone_rect(region) else {
        return;
    };
    blur_in_place(&mut sub, std_dx, std_dy);
    replace_region(pixmap, region, &sub);
}

fn replace_region(pixmap: &mut Pixmap, region: IntRect, sub: &Pixmap) {
    pixmap.draw_pixmap(
        region.x(),
        region.y(),
        sub.as_ref(),
        &PixmapPaint {
            blend_mode: tiny_skia::BlendMode::Source,
            ..Default::default()
        },
        Transform::identity(),
        None,
    );
}

fn blur_in_place(pixmap: &mut Pixmap, std_dx: f64, std_dy: f64) {
    if !(std_dx > 0.05 || std_dy > 0.05) {
        return;
    }
    let (std_dx, std_dy) = (std_dx.max(0.0), std_dy.max(0.0));
    if std_dx >= 2.0 || std_dy >= 2.0 {
        box_blur::apply(std_dx, std_dy, image_mut(pixmap));
    } else {
        iir_blur::apply(std_dx, std_dy, image_mut(pixmap));
    }
}

/// Maps a user-space vector into pixel space with the linear part of `transform`.
fn transform_vector(transform: Affine, x: f64, y: f64) -> (f64, f64) {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    (a * x + c * y, b * x + d * y)
}

fn offset(pixmap: &Pixmap, dx: f64, dy: f64) -> Pixmap {
    let mut out = Pixmap::new(pixmap.width(), pixmap.height()).unwrap();
    out.draw_pixmap(
        dx as i32,
        dy as i32,
        pixmap.as_ref(),
        &PixmapPaint::default(),
        Transform::identity(),
        None,
    );
    out
}

/// Runs `f` on the pixmap with its alpha unpremultiplied.
fn unpremultiplied(pixmap: &mut Pixmap, f: impl FnOnce(ImageRefMut)) {
    demultiply_alpha(pixmap.data_mut().as_rgba_mut());
    f(image_mut(pixmap));
    multiply_alpha(pixmap.data_mut().as_rgba_mut());
}

/// Maps a light source from user space into the pixel space of `region`.
fn transform_light_source(source: &LightSource, region: IntRect, ts: Affine) -> LightSource {
    let [a, _, _, d, _, _] = ts.as_coeffs();
    let sz = ((a * a + d * d).sqrt() / std::f64::consts::SQRT_2) as f32;
    let map = |x: f32, y: f32| {
        let p = ts * kurbo::Point::new(x as f64, y as f64);
        (
            (p.x - region.x() as f64) as f32,
            (p.y - region.y() as f64) as f32,
        )
    };
    match source {
        LightSource::Distant(light) => LightSource::Distant(light.clone()),
        LightSource::Point(light) => {
            let (x, y) = map(light.x, light.y);
            LightSource::Point(PointLightSource {
                x,
                y,
                z: light.z * sz,
            })
        }
        LightSource::Spot(light) => {
            let (x, y) = map(light.x, light.y);
            let (points_at_x, points_at_y) = map(light.points_at_x, light.points_at_y);
            LightSource::Spot(SpotLightSource {
                x,
                y,
                z: light.z * sz,
                points_at_x,
                points_at_y,
                points_at_z: light.points_at_z * sz,
                ..light.clone()
            })
        }
    }
}

/// Applies `filter` to the `region` of `pixmap` (pixels outside the region become transparent).
/// `transform` maps user space into the pixel space of `pixmap`.
///
/// `render_image` renders the content of an `feImage` primitive into a new pixmap of the given
/// width and height, with the given transform from user space to the pixmap's pixels. If it
/// returns `None`, the primitive passes its input through unchanged.
pub fn apply<I>(
    filter: &Filter<I>,
    transform: Affine,
    region: IntRect,
    pixmap: &mut Pixmap,
    mut render_image: impl FnMut(&I, Affine, u32, u32) -> Option<Pixmap>,
) {
    let Some(source) = pixmap.clone_rect(region) else {
        return;
    };
    let (sx, sy) = transform_scale(transform);
    let (w, h) = (source.width(), source.height());
    let empty = || Image {
        pixmap: Pixmap::new(w, h).unwrap(),
        linear: false,
    };
    let mut source = Image {
        pixmap: source,
        linear: false,
    };

    let full = IntRect::from_xywh(0, 0, w, h).unwrap();
    // The filter region, in the pixel space of `source`. It clips the source graphic and every
    // result.
    let full = match filter.region() {
        Some(rect) => {
            let Some(filter_region) = pixel_region(rect, transform, region, full) else {
                pixmap.fill(tiny_skia::Color::TRANSPARENT);
                return;
            };
            clip_to(&mut source.pixmap, Some(filter_region));
            filter_region
        }
        None => full,
    };
    let mut results: Vec<Image> = Vec::with_capacity(filter.nodes().len());
    // Each result's primitive subregion, in the pixel space of `source`.
    let mut regions: Vec<IntRect> = Vec::with_capacity(filter.nodes().len());
    for node in filter.nodes() {
        let input_region = |input: &Option<FilterInput>| match input {
            None => regions.last().copied().unwrap_or(full),
            Some(FilterInput::Result(id)) => regions.get(id.0 as usize).copied().unwrap_or(full),
            Some(FilterInput::Source(_)) => full,
        };
        let subregion = match node.subregion {
            Some(rect) => pixel_region(rect, transform, region, full),
            None => match node.effect {
                FilterEffect::Flood(_) | FilterEffect::Turbulence(_) | FilterEffect::Image(_) => {
                    Some(full)
                }
                _ => {
                    let primary = input_region(&node.inputs.primary);
                    match &node.inputs.secondary {
                        Some(_) => union(primary, input_region(&node.inputs.secondary)),
                        None => Some(primary),
                    }
                }
            },
        };

        let resolve = |input: Option<FilterInput>, results: &[Image]| -> Image {
            match input {
                None => results.last().cloned().unwrap_or_else(|| source.clone()),
                Some(FilterInput::Source(FilterSource::SourceGraphic)) => source.clone(),
                Some(FilterInput::Source(FilterSource::SourceAlpha)) => {
                    let mut img = source.clone();
                    for p in img.pixmap.data_mut().as_chunks_mut::<4>().0 {
                        p[0] = 0;
                        p[1] = 0;
                        p[2] = 0;
                    }
                    img
                }
                Some(FilterInput::Source(_)) => empty(),
                Some(FilterInput::Result(id)) => {
                    results.get(id.0 as usize).cloned().unwrap_or_else(empty)
                }
            }
        };
        let linear = node.color_space == FilterColorSpace::LinearRgb;
        let input = || resolve(node.inputs.primary.clone(), &results);
        let input2 = || resolve(node.inputs.secondary.clone(), &results);

        let result = match &node.effect {
            FilterEffect::Flood(color) => {
                let mut pixmap = Pixmap::new(w, h).unwrap();
                pixmap.fill(to_skia_color(*color));
                Image {
                    pixmap,
                    linear: false,
                }
            }
            FilterEffect::GaussianBlur(blur) => {
                let mut pixmap = input().into_color_space(linear);
                blur_in_place(
                    &mut pixmap,
                    blur.std_deviation_x as f64 * sx,
                    blur.std_deviation_y as f64 * sy,
                );
                Image { pixmap, linear }
            }
            FilterEffect::DropShadow(shadow) => {
                let input = input().into_color_space(linear);
                let mut color_px = Pixmap::new(1, 1).unwrap();
                color_px.fill(to_skia_color(shadow.color));
                if linear {
                    convert(&mut color_px, into_linear_rgb);
                }
                let color = color_px.pixel(0, 0).unwrap().demultiply();

                let mut shadow_px = input.clone();
                for p in shadow_px.data_mut().as_chunks_mut::<4>().0 {
                    let a = p[3] as f32 / 255.0 * color.alpha() as f32 / 255.0;
                    p[0] = (color.red() as f32 * a + 0.5) as u8;
                    p[1] = (color.green() as f32 * a + 0.5) as u8;
                    p[2] = (color.blue() as f32 * a + 0.5) as u8;
                    p[3] = (a * 255.0 + 0.5) as u8;
                }
                blur_in_place(
                    &mut shadow_px,
                    shadow.std_deviation_x as f64 * sx,
                    shadow.std_deviation_y as f64 * sy,
                );
                let (dx, dy) = transform_vector(transform, shadow.dx as f64, shadow.dy as f64);
                let mut out = offset(&shadow_px, dx, dy);
                out.draw_pixmap(
                    0,
                    0,
                    input.as_ref(),
                    &PixmapPaint::default(),
                    Transform::identity(),
                    None,
                );
                Image {
                    pixmap: out,
                    linear,
                }
            }
            FilterEffect::Offset(v) => {
                let img = input();
                Image {
                    pixmap: {
                        let (dx, dy) = transform_vector(transform, v.x, v.y);
                        offset(&img.pixmap, dx, dy)
                    },
                    linear: img.linear,
                }
            }
            FilterEffect::ColorMatrix(matrix) => {
                let mut pixmap = input().into_color_space(linear);
                unpremultiplied(&mut pixmap, |img| color_matrix::apply(matrix, img));
                Image { pixmap, linear }
            }
            FilterEffect::ComponentTransfer(ct) => {
                let mut pixmap = input().into_color_space(linear);
                unpremultiplied(&mut pixmap, |img| component_transfer::apply(ct, img));
                Image { pixmap, linear }
            }
            FilterEffect::Composite(op) => {
                let in1 = input().into_color_space(linear);
                let mut in2 = input2().into_color_space(linear);
                let blend_mode = match op {
                    CompositeOperator::Over => Some(tiny_skia::BlendMode::SourceOver),
                    CompositeOperator::In => Some(tiny_skia::BlendMode::SourceIn),
                    CompositeOperator::Out => Some(tiny_skia::BlendMode::SourceOut),
                    CompositeOperator::Atop => Some(tiny_skia::BlendMode::SourceAtop),
                    CompositeOperator::Xor => Some(tiny_skia::BlendMode::Xor),
                    CompositeOperator::Lighter => Some(tiny_skia::BlendMode::Plus),
                    CompositeOperator::Arithmetic(_) => None,
                };
                let pixmap = match (blend_mode, op) {
                    (None, CompositeOperator::Arithmetic(k)) => {
                        let mut out = Pixmap::new(w, h).unwrap();
                        composite::arithmetic(
                            k.k1,
                            k.k2,
                            k.k3,
                            k.k4,
                            ImageRef::new(w, h, in1.data().as_rgba()),
                            ImageRef::new(w, h, in2.data().as_rgba()),
                            image_mut(&mut out),
                        );
                        out
                    }
                    (blend_mode, _) => {
                        in2.draw_pixmap(
                            0,
                            0,
                            in1.as_ref(),
                            &PixmapPaint {
                                blend_mode: blend_mode.unwrap_or_default(),
                                ..Default::default()
                            },
                            Transform::identity(),
                            None,
                        );
                        in2
                    }
                };
                Image { pixmap, linear }
            }
            FilterEffect::Blend(mode) => {
                let in1 = input().into_color_space(linear);
                let mut in2 = input2().into_color_space(linear);
                in2.draw_pixmap(
                    0,
                    0,
                    in1.as_ref(),
                    &PixmapPaint {
                        blend_mode: mix_to_skia(*mode),
                        ..Default::default()
                    },
                    Transform::identity(),
                    None,
                );
                Image {
                    pixmap: in2,
                    linear,
                }
            }
            FilterEffect::Morphology(m) => {
                let mut pixmap = input().into_color_space(linear);
                let rx = m.radius_x as f64 * sx;
                let ry = m.radius_y as f64 * sy;
                if rx > 0.0 && ry > 0.0 {
                    morphology::apply(m.operator, rx as f32, ry as f32, image_mut(&mut pixmap));
                } else {
                    pixmap.fill(tiny_skia::Color::TRANSPARENT);
                }
                Image { pixmap, linear }
            }
            FilterEffect::ConvolveMatrix(kernel) => {
                let mut pixmap = input().into_color_space(linear);
                if kernel.preserve_alpha {
                    demultiply_alpha(pixmap.data_mut().as_rgba_mut());
                }
                // Edge modes see the edges of the filter region, not of the rounded-out layer.
                let covers_pixmap = full.x() == 0
                    && full.y() == 0
                    && full.width() == pixmap.width()
                    && full.height() == pixmap.height();
                if covers_pixmap {
                    convolve_matrix::apply(kernel, image_mut(&mut pixmap));
                } else if let Some(mut sub) = pixmap.clone_rect(full) {
                    // TODO: convolve the region in place instead of copying it out and back.
                    convolve_matrix::apply(kernel, image_mut(&mut sub));
                    replace_region(&mut pixmap, full, &sub);
                }
                Image { pixmap, linear }
            }
            FilterEffect::DisplacementMap(fe) => {
                let source = input().into_color_space(linear);
                let mut map = input2().into_color_space(linear);
                // The displacement map is read unpremultiplied.
                demultiply_alpha(map.data_mut().as_rgba_mut());
                let mut pixmap = Pixmap::new(w, h).unwrap();
                displacement_map::apply(
                    fe,
                    fe.scale * sx as f32,
                    fe.scale * sy as f32,
                    ImageRef::new(w, h, source.data().as_rgba()),
                    ImageRef::new(w, h, map.data().as_rgba()),
                    image_mut(&mut pixmap),
                );
                Image { pixmap, linear }
            }
            FilterEffect::Turbulence(fe) => {
                let mut pixmap = Pixmap::new(w, h).unwrap();
                if sx > 0.0 && sy > 0.0 {
                    let [_, _, _, _, tx, ty] = transform.as_coeffs();
                    turbulence::apply(
                        region.x() as f64 - tx,
                        region.y() as f64 - ty,
                        sx,
                        sy,
                        fe.base_frequency_x as f64,
                        fe.base_frequency_y as f64,
                        fe.num_octaves,
                        fe.seed as i32,
                        fe.stitch_tiles,
                        fe.turbulence_type == TurbulenceType::FractalNoise,
                        image_mut(&mut pixmap),
                    );
                    multiply_alpha(pixmap.data_mut().as_rgba_mut());
                }
                Image { pixmap, linear }
            }
            FilterEffect::DiffuseLighting(fe) => {
                let input = input();
                let mut pixmap = Pixmap::new(w, h).unwrap();
                lighting::diffuse_lighting(
                    fe,
                    transform_light_source(&fe.light_source, region, transform),
                    ImageRef::new(w, h, input.pixmap.data().as_rgba()),
                    image_mut(&mut pixmap),
                );
                Image { pixmap, linear }
            }
            FilterEffect::SpecularLighting(fe) => {
                let input = input();
                let mut pixmap = Pixmap::new(w, h).unwrap();
                lighting::specular_lighting(
                    fe,
                    transform_light_source(&fe.light_source, region, transform),
                    ImageRef::new(w, h, input.pixmap.data().as_rgba()),
                    image_mut(&mut pixmap),
                );
                Image { pixmap, linear }
            }
            FilterEffect::Tile => {
                let input = input();
                Image {
                    pixmap: tile(&input.pixmap, input_region(&node.inputs.primary)),
                    linear: input.linear,
                }
            }
            FilterEffect::Image(image) => {
                let local = Affine::translate((-region.x() as f64, -region.y() as f64)) * transform;
                match render_image(image, local, w, h) {
                    Some(pixmap) => Image {
                        pixmap,
                        linear: false,
                    },
                    None => input(),
                }
            }
        };
        let mut result = result;
        // As in resvg, results are only clipped to explicit subregions (and the filter region).
        if node.subregion.is_some() {
            clip_to(&mut result.pixmap, subregion);
        } else if filter.region().is_some() {
            clip_to(&mut result.pixmap, Some(full));
        }
        results.push(result);
        regions.push(subregion.unwrap_or(full));
    }

    let output = results
        .get(filter.output().0 as usize)
        .cloned()
        .or_else(|| results.last().cloned())
        .unwrap_or(source);
    pixmap.fill(tiny_skia::Color::TRANSPARENT);
    replace_region(pixmap, region, &output.into_srgb());
}

/// Maps a user-space primitive subregion to pixels of the filter `region`, clipped to `full`
/// (`None` if nothing is left). Rounds outwards like resvg.
fn pixel_region(
    rect: kurbo::Rect,
    transform: Affine,
    region: IntRect,
    full: IntRect,
) -> Option<IntRect> {
    let rect = transform.transform_rect_bbox(rect);
    let x = rect.x0.floor() as i32;
    let y = rect.y0.floor() as i32;
    IntRect::from_xywh(
        x.checked_sub(region.x())?,
        y.checked_sub(region.y())?,
        rect.width().ceil() as u32,
        rect.height().ceil() as u32,
    )?
    .intersect(&full)
}

/// Clears the pixels of `pixmap` outside `keep` (all of them if `keep` is `None`).
fn clip_to(pixmap: &mut Pixmap, keep: Option<IntRect>) {
    let width = pixmap.width() as usize;
    for (y, row) in pixmap.data_mut().chunks_mut(width * 4).enumerate() {
        let (x0, x1) = match keep {
            Some(k) if (k.top()..k.bottom()).contains(&(y as i32)) => {
                (k.left() as usize, k.right() as usize)
            }
            _ => (0, 0),
        };
        row[..x0 * 4].fill(0);
        row[x1 * 4..].fill(0);
        if x1 == 0 {
            row.fill(0);
        }
    }
}

/// Repeats the `tile` part of `pixmap` across a pixmap of the same size, as resvg's `feTile`.
fn tile(pixmap: &Pixmap, tile: IntRect) -> Pixmap {
    let mut out = Pixmap::new(pixmap.width(), pixmap.height()).unwrap();
    let Some(tile_pixmap) = pixmap.clone_rect(tile) else {
        return out;
    };
    let paint = tiny_skia::Paint {
        shader: tiny_skia::Pattern::new(
            tile_pixmap.as_ref(),
            tiny_skia::SpreadMode::Repeat,
            tiny_skia::FilterQuality::Bicubic,
            1.0,
            tiny_skia::Transform::from_translate(tile.x() as f32, tile.y() as f32),
        ),
        ..Default::default()
    };
    let rect = tiny_skia::Rect::from_xywh(0.0, 0.0, out.width() as f32, out.height() as f32);
    if let Some(rect) = rect {
        out.fill_rect(rect, &paint, tiny_skia::Transform::identity(), None);
    }
    out
}

fn union(a: IntRect, b: IntRect) -> Option<IntRect> {
    IntRect::from_ltrb(
        a.left().min(b.left()),
        a.top().min(b.top()),
        a.right().max(b.right()),
        a.bottom().max(b.bottom()),
    )
}

fn transform_scale(transform: Affine) -> (f64, f64) {
    let [a, b, c, d, _, _] = transform.as_coeffs();
    (a.hypot(b), c.hypot(d))
}

fn to_skia_color(color: AlphaColor<Srgb>) -> tiny_skia::Color {
    let [r, g, b, a] = color.components;
    tiny_skia::Color::from_rgba(
        r.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b.clamp(0.0, 1.0),
        a.clamp(0.0, 1.0),
    )
    .unwrap_or(tiny_skia::Color::TRANSPARENT)
}

fn mix_to_skia(mode: BlendMode) -> tiny_skia::BlendMode {
    use tiny_skia::BlendMode as B;
    #[allow(deprecated)]
    match mode {
        BlendMode::Normal => B::SourceOver,
        BlendMode::Multiply => B::Multiply,
        BlendMode::Screen => B::Screen,
        BlendMode::Overlay => B::Overlay,
        BlendMode::Darken => B::Darken,
        BlendMode::Lighten => B::Lighten,
        BlendMode::ColorDodge => B::ColorDodge,
        BlendMode::ColorBurn => B::ColorBurn,
        BlendMode::HardLight => B::HardLight,
        BlendMode::SoftLight => B::SoftLight,
        BlendMode::Difference => B::Difference,
        BlendMode::Exclusion => B::Exclusion,
        BlendMode::Hue => B::Hue,
        BlendMode::Saturation => B::Saturation,
        BlendMode::Color => B::Color,
        BlendMode::Luminosity => B::Luminosity,
    }
}
