//! Evaluation of an anyrender [`Filter`] graph on tiny-skia pixmaps.
//!
//! Like resvg, primitives operate in linearRGB (the SVG default `color-interpolation-filters`)
//! on the pixels of the filter region.

use anyrender::Filter;
use anyrender::filters::composite::CompositeOperator;
use anyrender::filters::{
    FilterEffect, FilterInput, FilterSource, component_transfer::TransferFunction,
};
use kurbo::Affine;
use rgb::FromSlice;
use tiny_skia::{IntRect, Pixmap, PixmapPaint, Transform};

use super::{
    ImageRef, ImageRefMut, box_blur, composite, demultiply_alpha, from_linear_rgb, iir_blur,
    into_linear_rgb, morphology, multiply_alpha,
};
use crate::scene::{to_skia_blend, to_skia_color, transform_scale};

#[derive(Clone)]
struct Image {
    pixmap: Pixmap,
    linear: bool,
}

impl Image {
    fn into_linear(mut self) -> Pixmap {
        if !self.linear {
            convert(&mut self.pixmap, into_linear_rgb);
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
pub(crate) fn blur(pixmap: &mut Pixmap, region: IntRect, std_dx: f64, std_dy: f64) {
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

fn map_unpremultiplied(pixmap: &mut Pixmap, mut f: impl FnMut(&mut [f32; 4])) {
    let data = pixmap.data_mut().as_rgba_mut();
    demultiply_alpha(data);
    for p in data.iter_mut() {
        let mut c = [
            p.r as f32 / 255.0,
            p.g as f32 / 255.0,
            p.b as f32 / 255.0,
            p.a as f32 / 255.0,
        ];
        f(&mut c);
        let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        p.r = to_u8(c[0]);
        p.g = to_u8(c[1]);
        p.b = to_u8(c[2]);
        p.a = to_u8(c[3]);
    }
    multiply_alpha(data);
}

fn transfer(func: &TransferFunction, c: f32) -> f32 {
    match func {
        TransferFunction::Identity => c,
        TransferFunction::Table(values) => {
            let n = values.len();
            if n == 0 {
                return c;
            }
            if n == 1 {
                return values[0];
            }
            let c = c.clamp(0.0, 1.0);
            let k = ((c * (n - 1) as f32) as usize).min(n - 2);
            let vk = values[k];
            let vk1 = values[k + 1];
            let k = k as f32;
            let n = (n - 1) as f32;
            vk + (c - k / n) * n * (vk1 - vk)
        }
        TransferFunction::Discrete(values) => {
            let n = values.len();
            if n == 0 {
                return c;
            }
            let k = ((c.clamp(0.0, 1.0) * n as f32) as usize).min(n - 1);
            values[k]
        }
        TransferFunction::Linear(l) => l.slope * c + l.intercept,
        TransferFunction::Gamma(g) => g.amplitude * c.powf(g.exponent) + g.offset,
    }
}

/// Applies `filter` to the `region` of `pixmap` (pixels outside the region become transparent).
/// `transform` is the layer's user-space to device-space transform, used to scale lengths.
pub(crate) fn apply(filter: &Filter, transform: Affine, region: IntRect, pixmap: &mut Pixmap) {
    let Some(source) = pixmap.clone_rect(region) else {
        return;
    };
    let (sx, sy) = transform_scale(transform);
    let (w, h) = (source.width(), source.height());
    let empty = || Image {
        pixmap: Pixmap::new(w, h).unwrap(),
        linear: false,
    };
    let source = Image {
        pixmap: source,
        linear: false,
    };

    let mut results: Vec<Image> = Vec::with_capacity(filter.nodes().len());
    for node in filter.nodes() {
        let resolve = |input: Option<FilterInput>, results: &[Image]| -> Image {
            match input {
                None => results.last().cloned().unwrap_or_else(|| source.clone()),
                Some(FilterInput::Source(FilterSource::SourceGraphic)) => source.clone(),
                Some(FilterInput::Source(FilterSource::SourceAlpha)) => {
                    let mut img = source.clone();
                    for p in img.pixmap.data_mut().chunks_exact_mut(4) {
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
        let input = || resolve(node.inputs.primary.clone(), &results);
        let input2 = || resolve(node.inputs.secondary.clone(), &results);

        let result = match &node.effect {
            FilterEffect::Flood(color) => {
                let mut pixmap = Pixmap::new(w, h).unwrap();
                pixmap.fill(to_skia_color(*color, 1.0));
                Image {
                    pixmap,
                    linear: false,
                }
            }
            FilterEffect::GaussianBlur(blur) => {
                let mut pixmap = input().into_linear();
                let std = blur.std_deviation as f64;
                blur_in_place(&mut pixmap, std * sx, std * sy);
                Image {
                    pixmap,
                    linear: true,
                }
            }
            FilterEffect::DropShadow(shadow) => {
                let input = input().into_linear();
                let mut color_px = Pixmap::new(1, 1).unwrap();
                color_px.fill(to_skia_color(shadow.color, 1.0));
                convert(&mut color_px, into_linear_rgb);
                let color = color_px.pixel(0, 0).unwrap().demultiply();

                let mut shadow_px = input.clone();
                for p in shadow_px.data_mut().chunks_exact_mut(4) {
                    let a = p[3] as f32 / 255.0 * color.alpha() as f32 / 255.0;
                    p[0] = (color.red() as f32 * a + 0.5) as u8;
                    p[1] = (color.green() as f32 * a + 0.5) as u8;
                    p[2] = (color.blue() as f32 * a + 0.5) as u8;
                    p[3] = (a * 255.0 + 0.5) as u8;
                }
                let std = shadow.std_deviation as f64;
                blur_in_place(&mut shadow_px, std * sx, std * sy);
                let mut out = offset(&shadow_px, shadow.dx as f64 * sx, shadow.dy as f64 * sy);
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
                    linear: true,
                }
            }
            FilterEffect::Offset(v) => {
                let img = input();
                Image {
                    pixmap: offset(&img.pixmap, v.x * sx, v.y * sy),
                    linear: img.linear,
                }
            }
            FilterEffect::ColorMatrix(matrix) => {
                let mut pixmap = input().into_linear();
                let m = &matrix.0;
                map_unpremultiplied(&mut pixmap, |c| {
                    let [r, g, b, a] = *c;
                    for (i, out) in c.iter_mut().enumerate() {
                        let row = &m[i * 5..i * 5 + 5];
                        *out = row[0] * r + row[1] * g + row[2] * b + row[3] * a + row[4];
                    }
                });
                Image {
                    pixmap,
                    linear: true,
                }
            }
            FilterEffect::ComponentTransfer(ct) => {
                let mut pixmap = input().into_linear();
                map_unpremultiplied(&mut pixmap, |c| {
                    c[0] = transfer(&ct.red_function, c[0]);
                    c[1] = transfer(&ct.green_function, c[1]);
                    c[2] = transfer(&ct.blue_function, c[2]);
                    c[3] = transfer(&ct.alpha_function, c[3]);
                });
                Image {
                    pixmap,
                    linear: true,
                }
            }
            FilterEffect::Composite(op) => {
                let in1 = input().into_linear();
                let mut in2 = input2().into_linear();
                let blend_mode = match op {
                    CompositeOperator::Over => Some(tiny_skia::BlendMode::SourceOver),
                    CompositeOperator::In => Some(tiny_skia::BlendMode::SourceIn),
                    CompositeOperator::Out => Some(tiny_skia::BlendMode::SourceOut),
                    CompositeOperator::Atop => Some(tiny_skia::BlendMode::SourceAtop),
                    CompositeOperator::Xor => Some(tiny_skia::BlendMode::Xor),
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
                Image {
                    pixmap,
                    linear: true,
                }
            }
            FilterEffect::Blend(mode) => {
                let in1 = input().into_linear();
                let mut in2 = input2().into_linear();
                in2.draw_pixmap(
                    0,
                    0,
                    in1.as_ref(),
                    &PixmapPaint {
                        blend_mode: to_skia_blend((*mode).into()),
                        ..Default::default()
                    },
                    Transform::identity(),
                    None,
                );
                Image {
                    pixmap: in2,
                    linear: true,
                }
            }
            FilterEffect::Morphology(m) => {
                let mut pixmap = input().into_linear();
                let rx = m.radius as f64 * sx;
                let ry = m.radius as f64 * sy;
                if rx > 0.0 && ry > 0.0 {
                    morphology::apply(m.operator, rx as f32, ry as f32, image_mut(&mut pixmap));
                } else {
                    pixmap.fill(tiny_skia::Color::TRANSPARENT);
                }
                Image {
                    pixmap,
                    linear: true,
                }
            }
            // Not yet implemented: pass the input through unchanged.
            _ => input(),
        };
        results.push(result);
    }

    let output = results
        .get(filter.output().0 as usize)
        .cloned()
        .or_else(|| results.last().cloned())
        .unwrap_or(source);
    pixmap.fill(tiny_skia::Color::TRANSPARENT);
    replace_region(pixmap, region, &output.into_srgb());
}
