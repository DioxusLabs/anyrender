// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::util;
use anyrender::{MaskMode, PaintScene};
use kurbo::{Affine, BezPath};
use peniko::{BlendMode, Fill};
use usvg::{Node, Path};

/// `base_transform` maps the root of the current usvg (sub)tree into the SVG's coordinate
/// space: the identity for the document root, or the `<image>` placement for an embedded SVG.
/// `global_transform` maps the SVG's coordinate space into the scene.
///
/// The error handler is called with the node's transform into scene space.
pub(crate) fn render_group<S: PaintScene, F: FnMut(&mut S, &usvg::Node, Affine)>(
    scene: &mut S,
    group: &usvg::Group,
    base_transform: Affine,
    global_transform: Affine,
    error_handler: &mut F,
) {
    for node in group.children() {
        let transform = base_transform * util::to_affine(&node.abs_transform());
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

                // usvg's absolute transforms already include the ancestor groups
                render_group(scene, g, base_transform, global_transform, error_handler);

                // Apply the group's mask, and any masks linked from it with `<mask mask="...">`
                for mask in std::iter::successors(g.mask(), |mask| mask.mask()) {
                    apply_mask(scene, mask, group_transform, error_handler);
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
                                error_handler(scene, node, global_transform * transform);
                                continue;
                            };
                            let image = util::into_image(decoded_image, img.rendering_mode());
                            let image_ts = global_transform * transform;
                            scene.draw_image(image.as_ref(), image_ts);
                        }

                        #[cfg(not(feature = "image"))]
                        {
                            error_handler(scene, node, global_transform * transform);
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
                // text node's absolute transform, so recurse with the base
                // transform (like the group case) to avoid applying it twice.
                render_group(
                    scene,
                    text.flattened(),
                    base_transform,
                    global_transform,
                    error_handler,
                );
            }
        }
    }
}

/// Multiply the content drawn so far in the current layer by `mask`.
///
fn apply_mask<S: PaintScene, F: FnMut(&mut S, &usvg::Node, Affine)>(
    scene: &mut S,
    mask: &usvg::Mask,
    transform: Affine,
    error_handler: &mut F,
) {
    let mode = match mask.kind() {
        usvg::MaskType::Alpha => MaskMode::Alpha,
        usvg::MaskType::Luminance => MaskMode::Luminance,
    };
    // Content outside of the mask's rect is masked out
    let rect = util::to_rect(mask.rect());
    scene.push_mask_layer(Fill::NonZero, mode, transform, &rect);
    render_group(
        scene,
        mask.root(),
        Affine::IDENTITY,
        transform,
        error_handler,
    );
    scene.pop_layer();
}

fn fill<S: PaintScene, F: FnMut(&mut S, &usvg::Node, Affine)>(
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
            error_handler(scene, node, transform);
        }
    }
}

fn stroke<S: PaintScene, F: FnMut(&mut S, &usvg::Node, Affine)>(
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
            error_handler(scene, node, transform);
        }
    }
}
