// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use anyrender::{Paint, PaintScene, Pattern, recording::Scene};
use kurbo::{Affine, BezPath, Point, Rect, Stroke};
use peniko::color::{self, DynamicColor};
use peniko::{BlendMode, Color, Compose, Fill, InterpolationAlphaSpace, Mix};

#[cfg(feature = "image")]
use peniko::{Blob, ImageBrush};

pub(crate) fn to_affine(ts: &usvg::Transform) -> Affine {
    let usvg::Transform {
        sx,
        kx,
        ky,
        sy,
        tx,
        ty,
    } = ts;
    Affine::new([sx, ky, kx, sy, tx, ty].map(|&x| f64::from(x)))
}

pub(crate) fn to_rect(rect: usvg::NonZeroRect) -> Rect {
    Rect::new(
        rect.left() as f64,
        rect.top() as f64,
        rect.right() as f64,
        rect.bottom() as f64,
    )
}

pub(crate) fn to_stroke(stroke: &usvg::Stroke) -> Stroke {
    let mut conv_stroke = Stroke::new(stroke.width().get() as f64)
        .with_caps(match stroke.linecap() {
            usvg::LineCap::Butt => kurbo::Cap::Butt,
            usvg::LineCap::Round => kurbo::Cap::Round,
            usvg::LineCap::Square => kurbo::Cap::Square,
        })
        .with_join(match stroke.linejoin() {
            usvg::LineJoin::Miter | usvg::LineJoin::MiterClip => kurbo::Join::Miter,
            usvg::LineJoin::Round => kurbo::Join::Round,
            usvg::LineJoin::Bevel => kurbo::Join::Bevel,
        })
        .with_miter_limit(stroke.miterlimit().get() as f64);
    if let Some(dash_array) = stroke.dasharray().as_ref() {
        conv_stroke = conv_stroke.with_dashes(
            stroke.dashoffset() as f64,
            dash_array.iter().map(|x| *x as f64),
        );
    }
    conv_stroke
}

pub(crate) fn to_blend_mode(blend_mode: usvg::BlendMode) -> BlendMode {
    let mix = match blend_mode {
        usvg::BlendMode::PlusLighter => return BlendMode::new(Mix::Normal, Compose::PlusLighter),
        usvg::BlendMode::Normal => Mix::Normal,
        usvg::BlendMode::Multiply => Mix::Multiply,
        usvg::BlendMode::Screen => Mix::Screen,
        usvg::BlendMode::Overlay => Mix::Overlay,
        usvg::BlendMode::Darken => Mix::Darken,
        usvg::BlendMode::Lighten => Mix::Lighten,
        usvg::BlendMode::ColorDodge => Mix::ColorDodge,
        usvg::BlendMode::ColorBurn => Mix::ColorBurn,
        usvg::BlendMode::HardLight => Mix::HardLight,
        usvg::BlendMode::SoftLight => Mix::SoftLight,
        usvg::BlendMode::Difference => Mix::Difference,
        usvg::BlendMode::Exclusion => Mix::Exclusion,
        usvg::BlendMode::Hue => Mix::Hue,
        usvg::BlendMode::Saturation => Mix::Saturation,
        usvg::BlendMode::Color => Mix::Color,
        usvg::BlendMode::Luminosity => Mix::Luminosity,
    };
    BlendMode::new(mix, Compose::SrcOver)
}

pub(crate) fn to_bez_path(path: &usvg::Path) -> BezPath {
    let mut local_path = BezPath::new();
    // The semantics of SVG paths don't line up with `BezPath`; we
    // must manually track initial points
    let mut just_closed = false;
    let mut most_recent_initial = (0., 0.);
    for elt in path.data().segments() {
        match elt {
            usvg::tiny_skia_path::PathSegment::MoveTo(p) => {
                if std::mem::take(&mut just_closed) {
                    local_path.move_to(most_recent_initial);
                }
                most_recent_initial = (p.x.into(), p.y.into());
                local_path.move_to(most_recent_initial);
            }
            usvg::tiny_skia_path::PathSegment::LineTo(p) => {
                if std::mem::take(&mut just_closed) {
                    local_path.move_to(most_recent_initial);
                }
                local_path.line_to(Point::new(p.x as f64, p.y as f64));
            }
            usvg::tiny_skia_path::PathSegment::QuadTo(p1, p2) => {
                if std::mem::take(&mut just_closed) {
                    local_path.move_to(most_recent_initial);
                }
                local_path.quad_to(
                    Point::new(p1.x as f64, p1.y as f64),
                    Point::new(p2.x as f64, p2.y as f64),
                );
            }
            usvg::tiny_skia_path::PathSegment::CubicTo(p1, p2, p3) => {
                if std::mem::take(&mut just_closed) {
                    local_path.move_to(most_recent_initial);
                }
                local_path.curve_to(
                    Point::new(p1.x as f64, p1.y as f64),
                    Point::new(p2.x as f64, p2.y as f64),
                    Point::new(p3.x as f64, p3.y as f64),
                );
            }
            usvg::tiny_skia_path::PathSegment::Close => {
                just_closed = true;
                local_path.close_path();
            }
        }
    }

    local_path
}

#[cfg(feature = "image")]
pub(crate) fn into_image(
    image: image::ImageBuffer<image::Rgba<u8>, Vec<u8>>,
    rendering_mode: usvg::ImageRendering,
) -> ImageBrush {
    use peniko::ImageData;

    let quality = match rendering_mode {
        usvg::ImageRendering::HighQuality => peniko::ImageQuality::High,
        usvg::ImageRendering::OptimizeQuality | usvg::ImageRendering::Smooth => {
            peniko::ImageQuality::Medium
        }
        usvg::ImageRendering::OptimizeSpeed
        | usvg::ImageRendering::CrispEdges
        | usvg::ImageRendering::Pixelated => peniko::ImageQuality::Low,
    };

    let (width, height) = (image.width(), image.height());
    let image_data: Vec<u8> = image.into_vec();
    ImageBrush::new(ImageData {
        data: Blob::new(std::sync::Arc::new(image_data)),
        format: peniko::ImageFormat::Rgba8,
        alpha_type: peniko::ImageAlphaType::Alpha,
        width,
        height,
    })
    .with_quality(quality)
}

pub(crate) fn to_brush(paint: &usvg::Paint, opacity: usvg::Opacity) -> Option<(Paint, Affine)> {
    match paint {
        usvg::Paint::Color(color) => Some((
            Paint::Solid(Color::from_rgba8(
                color.red,
                color.green,
                color.blue,
                opacity.to_u8(),
            )),
            Affine::IDENTITY,
        )),
        usvg::Paint::LinearGradient(gr) => {
            let stops: Vec<peniko::ColorStop> = gr
                .stops()
                .iter()
                .map(|stop| peniko::ColorStop {
                    offset: stop.offset().get(),
                    color: DynamicColor::from_alpha_color(Color::from_rgba8(
                        stop.color().red,
                        stop.color().green,
                        stop.color().blue,
                        (stop.opacity() * opacity).to_u8(),
                    )),
                })
                .collect();
            let start = Point::new(gr.x1() as f64, gr.y1() as f64);
            let end = Point::new(gr.x2() as f64, gr.y2() as f64);
            let arr = [
                gr.transform().sx,
                gr.transform().ky,
                gr.transform().kx,
                gr.transform().sy,
                gr.transform().tx,
                gr.transform().ty,
            ]
            .map(f64::from);
            let transform = Affine::new(arr);
            let gradient = peniko::Gradient::new_linear(start, end)
                .with_stops(stops.as_slice())
                .with_extend(to_extend(gr.spread_method()))
                .with_interpolation_alpha_space(InterpolationAlphaSpace::Unpremultiplied);
            Some((Paint::Gradient(gradient), transform))
        }
        usvg::Paint::RadialGradient(gr) => {
            let stops: Vec<peniko::ColorStop> = gr
                .stops()
                .iter()
                .map(|stop| peniko::ColorStop {
                    offset: stop.offset().get(),
                    color: DynamicColor::from_alpha_color(Color::from_rgba8(
                        stop.color().red,
                        stop.color().green,
                        stop.color().blue,
                        (stop.opacity() * opacity).to_u8(),
                    )),
                })
                .collect();

            let start_center = Point::new(gr.fx() as f64, gr.fy() as f64);
            let end_center = Point::new(gr.cx() as f64, gr.cy() as f64);
            let start_radius = gr.fr().get();
            let end_radius = gr.r().get();
            let arr = [
                gr.transform().sx,
                gr.transform().ky,
                gr.transform().kx,
                gr.transform().sy,
                gr.transform().tx,
                gr.transform().ty,
            ]
            .map(f64::from);
            let transform = Affine::new(arr);
            let gradient = peniko::Gradient::new_two_point_radial(
                start_center,
                start_radius,
                end_center,
                end_radius,
            )
            .with_stops(stops.as_slice())
            .with_extend(to_extend(gr.spread_method()))
            .with_interpolation_alpha_space(InterpolationAlphaSpace::Unpremultiplied);
            Some((Paint::Gradient(gradient), transform))
        }
        usvg::Paint::Pattern(pattern) => {
            // usvg resolves the pattern's units and view box, so its content is in the tile's
            // space, with the tile's top-left corner at the origin.
            let rect = pattern.rect();
            let tile = Rect::new(0.0, 0.0, rect.width() as f64, rect.height() as f64);
            let alpha = opacity.get();
            let mut scene = Scene::new();
            if alpha < 1.0 {
                scene.push_layer(
                    Fill::NonZero,
                    BlendMode::default(),
                    alpha,
                    Affine::IDENTITY,
                    &tile,
                    None,
                    None,
                );
            }
            crate::render::render_group(
                &mut scene,
                pattern.root(),
                Affine::IDENTITY,
                &mut default_error_handler,
            );
            if alpha < 1.0 {
                scene.pop_layer();
            }
            let transform = to_affine(&pattern.transform())
                * Affine::translate((rect.x() as f64, rect.y() as f64));
            Some((Paint::Pattern(Pattern::new(scene, tile)), transform))
        }
    }
}

fn to_extend(spread_method: usvg::SpreadMethod) -> peniko::Extend {
    match spread_method {
        usvg::SpreadMethod::Pad => peniko::Extend::Pad,
        usvg::SpreadMethod::Reflect => peniko::Extend::Reflect,
        usvg::SpreadMethod::Repeat => peniko::Extend::Repeat,
    }
}

/// Error handler for [`crate::render_svg_tree`] which draws a transparent red box over
/// unsupported SVG features. `transform` maps the node's object space into the scene.
pub(crate) fn default_error_handler<S: PaintScene>(
    scene: &mut S,
    node: &usvg::Node,
    transform: Affine,
) {
    let Some(bb) = node.bounding_box() else {
        return;
    };
    let rect = Rect {
        x0: bb.left() as f64,
        y0: bb.top() as f64,
        x1: bb.right() as f64,
        y1: bb.bottom() as f64,
    };
    scene.fill(
        Fill::NonZero,
        transform,
        color::palette::css::RED.multiply_alpha(0.5),
        None,
        &rect,
    );
}

#[cfg(feature = "image")]
pub(crate) fn decode_raw_raster_image(
    img: &usvg::ImageKind,
) -> Result<image::RgbaImage, image::ImageError> {
    // All `image::ImageFormat` variants exist even if the feature in the image crate is disabled,
    // but `image::load_from_memory_with_format` will fail with an Unsupported error if the
    // image crate feature flag is disabled. So we don't need any of our own feature handling here.
    let (data, format) = match img {
        usvg::ImageKind::JPEG(data) => (data, image::ImageFormat::Jpeg),
        usvg::ImageKind::PNG(data) => (data, image::ImageFormat::Png),
        usvg::ImageKind::GIF(data) => (data, image::ImageFormat::Gif),
        usvg::ImageKind::WEBP(data) => (data, image::ImageFormat::WebP),
        usvg::ImageKind::SVG(_) => unreachable!(),
    };

    let dyn_image = image::load_from_memory_with_format(data, format)?;
    Ok(dyn_image.into_rgba8())
}
