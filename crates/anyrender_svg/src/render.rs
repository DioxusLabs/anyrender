// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::util;
use anyrender::PaintScene;
use anyrender::filters::{
    Filter, FilterEffect, FilterInput, FilterInputs, FilterSource,
    color_transformation::ColorMatrix, composite::CompositeOperator,
};
use kurbo::{Affine, BezPath};
use peniko::{BlendMode, Compose, Fill, Mix};
use std::sync::Arc;
use usvg::{Node, Path};

pub(crate) fn render_group<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    group: &usvg::Group,
    transform: Affine,
    global_transform: Affine,
    error_handler: &mut F,
) {
    for node in group.children() {
        let transform = transform * util::to_affine(&node.abs_transform());
        match node {
            usvg::Node::Group(g) => {
                let alpha = g.opacity().get();
                let is_fully_opaque = alpha >= 1.0;
                let blend_mode = util::to_blend_mode(g.blend_mode());
                let group_transform = global_transform * transform;

                // Support clip-path with a single path
                let clip_path = g
                    .clip_path()
                    .and_then(|path| path.root().children().first());

                let did_push_layer = match clip_path {
                    // If there is a clip path, then push a layer that clips using it
                    Some(usvg::Node::Path(clip_path)) => {
                        let local_path = util::to_bez_path(clip_path);
                        scene.push_layer(
                            match clip_path.fill().map(|fill| fill.rule()) {
                                Some(usvg::FillRule::EvenOdd) => Fill::EvenOdd,
                                _ => Fill::NonZero,
                            },
                            blend_mode,
                            alpha,
                            group_transform,
                            &local_path,
                            None,
                            None,
                        );

                        true
                    }
                    // Else if there is blending to be done then push a layer with a rectangular clip
                    // A mask also needs an isolated layer for it to be applied to
                    #[allow(deprecated)]
                    _ if blend_mode != BlendMode::default()
                        || !is_fully_opaque
                        || g.mask().is_some() =>
                    {
                        // Use the layer bounding box as the clip path. It is in the group's
                        // content coordinate space, so map it to canvas space and outset it
                        // by 2px on each side (like resvg) so anti-aliased pixels at the
                        // layer edge aren't clipped.
                        // A group without a layer bounding box has nothing to draw.
                        let Some(bounding_box) = g.layer_bounding_box() else {
                            continue;
                        };
                        let rect = group_transform
                            .transform_rect_bbox(util::to_rect(bounding_box))
                            .inflate(2.0, 2.0);
                        scene.push_layer(
                            Fill::NonZero,
                            blend_mode,
                            alpha,
                            Affine::IDENTITY,
                            &rect,
                            None,
                            None,
                        );

                        true
                    }
                    // Else if there is no clip or blending then don't push a layer
                    _ => false,
                };

                render_group(scene, g, Affine::IDENTITY, global_transform, error_handler);

                // Apply the group's mask, and any masks linked from it with `<mask mask="...">`
                if let Some(bounding_box) = g.layer_bounding_box() {
                    let bounds = group_transform
                        .transform_rect_bbox(util::to_rect(bounding_box))
                        .inflate(2.0, 2.0);
                    for mask in std::iter::successors(g.mask(), |mask| mask.mask()) {
                        apply_mask(scene, mask, group_transform, bounds, error_handler);
                    }
                }

                if did_push_layer {
                    scene.pop_layer();
                }
            }
            usvg::Node::Path(path) => {
                if !path.is_visible() {
                    continue;
                }
                let local_path = util::to_bez_path(path);

                let transform = global_transform * transform;
                match path.paint_order() {
                    usvg::PaintOrder::FillAndStroke => {
                        fill(scene, error_handler, path, transform, &local_path, node);
                        stroke(scene, error_handler, path, transform, &local_path, node);
                    }
                    usvg::PaintOrder::StrokeAndFill => {
                        stroke(scene, error_handler, path, transform, &local_path, node);
                        fill(scene, error_handler, path, transform, &local_path, node);
                    }
                }
            }
            usvg::Node::Image(img) => {
                if !img.is_visible() {
                    continue;
                }
                match img.kind() {
                    usvg::ImageKind::JPEG(_)
                    | usvg::ImageKind::PNG(_)
                    | usvg::ImageKind::GIF(_)
                    | usvg::ImageKind::WEBP(_) => {
                        #[cfg(feature = "image")]
                        {
                            let Ok(decoded_image) = util::decode_raw_raster_image(img.kind())
                            else {
                                error_handler(scene, node);
                                continue;
                            };
                            let image = util::into_image(decoded_image, img.rendering_mode());
                            let image_ts = global_transform * util::to_affine(&img.abs_transform());
                            scene.draw_image(image.as_ref(), image_ts);
                        }

                        #[cfg(not(feature = "image"))]
                        {
                            error_handler(scene, node);
                            continue;
                        }
                    }
                    usvg::ImageKind::SVG(svg) => {
                        render_group(
                            scene,
                            svg.root(),
                            transform,
                            global_transform,
                            error_handler,
                        );
                    }
                }
            }
            usvg::Node::Text(text) => {
                // The children of the flattened text group already carry the
                // text node's absolute transform, so recurse with the identity
                // transform (like the group case) to avoid applying it twice.
                render_group(
                    scene,
                    text.flattened(),
                    Affine::IDENTITY,
                    global_transform,
                    error_handler,
                );
            }
        }
    }
}

/// Multiply the content drawn so far in the current layer by `mask`.
///
/// The mask's content is drawn into a layer that is composited with `DestIn`. `bounds` must cover
/// all of the content to be masked, as content outside of it is left untouched. Luminance masks
/// convert that layer to alpha with a filter, so backends without filter support treat them as
/// alpha masks.
fn apply_mask<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    mask: &usvg::Mask,
    transform: Affine,
    bounds: kurbo::Rect,
    error_handler: &mut F,
) {
    let filter = match mask.kind() {
        usvg::MaskType::Alpha => None,
        usvg::MaskType::Luminance => Some(Arc::new(luminance_to_alpha())),
    };
    scene.push_layer(
        Fill::NonZero,
        BlendMode::new(Mix::Normal, Compose::DestIn),
        1.0,
        Affine::IDENTITY,
        &bounds,
        filter,
        None,
    );
    // Content outside of the mask's rect is masked out
    scene.push_clip_layer(Fill::NonZero, transform, &util::to_rect(mask.rect()));
    render_group(
        scene,
        mask.root(),
        Affine::IDENTITY,
        transform,
        error_handler,
    );
    scene.pop_layer();
    scene.pop_layer();
}

/// A filter that outputs the luminance of its (unpremultiplied) input multiplied by its alpha
/// as alpha, as specified for luminance masks.
fn luminance_to_alpha() -> Filter {
    #[rustfmt::skip]
    let matrix = ColorMatrix([
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.2125, 0.7154, 0.0721, 0.0, 0.0,
    ]);
    let source = FilterInput::Source(FilterSource::SourceGraphic);
    let mut filter = Filter::empty();
    let luminance = filter.add(
        FilterEffect::ColorMatrix(matrix),
        FilterInputs::single(source.clone()),
    );
    let output = filter.add(
        FilterEffect::Composite(CompositeOperator::In),
        FilterInputs::dual(FilterInput::Result(luminance), source),
    );
    filter.set_output(output);
    filter
}

fn fill<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    error_handler: &mut F,
    path: &Path,
    transform: Affine,
    local_path: &BezPath,
    node: &Node,
) {
    if let Some(fill) = &path.fill() {
        if let Some((paint, brush_transform)) = util::to_brush(fill.paint(), fill.opacity()) {
            scene.fill(
                match fill.rule() {
                    usvg::FillRule::NonZero => Fill::NonZero,
                    usvg::FillRule::EvenOdd => Fill::EvenOdd,
                },
                transform,
                paint.as_ref(),
                Some(brush_transform),
                local_path,
            );
        } else {
            error_handler(scene, node);
        }
    }
}

fn stroke<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    error_handler: &mut F,
    path: &Path,
    transform: Affine,
    local_path: &BezPath,
    node: &Node,
) {
    if let Some(stroke) = &path.stroke() {
        if let Some((paint, brush_transform)) = util::to_brush(stroke.paint(), stroke.opacity()) {
            let conv_stroke = util::to_stroke(stroke);
            scene.stroke(
                &conv_stroke,
                transform,
                paint.as_ref(),
                Some(brush_transform),
                local_path,
            );
        } else {
            error_handler(scene, node);
        }
    }
}
