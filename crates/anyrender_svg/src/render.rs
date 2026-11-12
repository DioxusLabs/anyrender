// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::{filter, util};
use anyrender::{MaskMode, PaintScene};
use kurbo::{Affine, BezPath, Rect};
use peniko::{BlendMode, Color, Fill};
use std::sync::Arc;
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

                let clip = g.clip_path();
                let clip_shape = clip.and_then(single_clip_shape);

                #[allow(deprecated)]
                let needs_layer = clip.is_some()
                    || blend_mode != BlendMode::default()
                    || !is_fully_opaque
                    || g.mask().is_some();

                // The canvas space bounds of the group's layer, when it is clipped by a mask
                let mut layer_bounds = None;
                let did_push_layer = if let Some((path, fill, path_transform)) = clip_shape {
                    // A clip-path made of a single shape can clip the layer directly
                    scene.push_layer(
                        fill,
                        blend_mode,
                        alpha,
                        group_transform * path_transform,
                        &util::to_bez_path(path),
                        None,
                        None,
                    );
                    true
                } else if needs_layer {
                    // Else push a layer with a rectangular clip. Blending, masks and
                    // any other clip-path are applied to this isolated layer.
                    //
                    // Use the layer bounding box as the clip path. It is in the group's
                    // content coordinate space, so map it to canvas space and outset it
                    // by 2px on each side (like resvg) so anti-aliased pixels at the
                    // layer edge aren't clipped. Round it out to whole pixels so that a
                    // clip mask applied with the same bounds fully covers the layer's edge.
                    // A group without a layer bounding box has nothing to draw.
                    let Some(bounding_box) = g.layer_bounding_box() else {
                        continue;
                    };
                    let rect = group_transform
                        .transform_rect_bbox(util::to_rect(bounding_box))
                        .inflate(2.0, 2.0)
                        .expand();
                    scene.push_layer(
                        Fill::NonZero,
                        blend_mode,
                        alpha,
                        Affine::IDENTITY,
                        &rect,
                        None,
                        None,
                    );
                    layer_bounds = Some(rect);
                    true
                } else {
                    // Else if there is no clip or blending then don't push a layer
                    false
                };

                // Apply filter effects by pushing a layer with an `anyrender::Filter`
                // attached, clipped to the SVG filter region. The filter is applied to
                // the group's content before the clip/opacity/blend of the outer layer,
                // matching the SVG rendering model.
                //
                // Groups whose filters cannot be expressed as an `anyrender::Filter`
                // are reported to the error handler and rendered unfiltered.
                let mut did_push_filter_layer = false;
                if !g.filters().is_empty() {
                    let converted = filter::to_anyrender_filter(g.filters());
                    match (converted, g.filters_bounding_box()) {
                        (Some(converted), Some(region)) => {
                            scene.push_layer(
                                Fill::NonZero,
                                BlendMode::default(),
                                1.0,
                                group_transform,
                                &util::to_rect(region),
                                Some(Arc::new(converted)),
                                None,
                            );
                            did_push_filter_layer = true;
                        }
                        _ => error_handler(scene, node, group_transform),
                    }
                }

                // usvg's absolute transforms already include the ancestor groups
                render_group(scene, g, base_transform, global_transform, error_handler);

                if did_push_filter_layer {
                    scene.pop_layer();
                }

                // Apply the group's mask, and any masks linked from it with `<mask mask="...">`
                for mask in std::iter::successors(g.mask(), |mask| mask.mask()) {
                    apply_mask(scene, mask, group_transform, error_handler);
                }

                // Apply any clip-path that couldn't clip the layer directly
                if let (Some(clip), Some(bounds)) = (clip, layer_bounds) {
                    apply_clip_mask(scene, clip, group_transform, bounds);
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

/// Returns the clip path's shape, its fill rule and its transform (relative to the clipped
/// element's user space) when `clip` is a single visible shape without nested clip paths.
fn single_clip_shape(clip: &usvg::ClipPath) -> Option<(&usvg::Path, Fill, Affine)> {
    if clip.clip_path().is_some() {
        return None;
    }
    let mut shape = None;
    if !find_single_clip_shape(clip.root(), &mut shape) {
        return None;
    }
    let path = shape?;
    let transform = util::to_affine(&clip.transform()) * util::to_affine(&path.abs_transform());
    Some((path, clip_fill_rule(path), transform))
}

/// Looks for the only visible shape in a clip path group. Returns false if there is more than
/// one, or if a nested group has its own clip path.
fn find_single_clip_shape<'a>(group: &'a usvg::Group, shape: &mut Option<&'a usvg::Path>) -> bool {
    for node in group.children() {
        match node {
            usvg::Node::Path(path) => {
                if !path.is_visible() {
                    continue;
                }
                if shape.is_some() {
                    return false;
                }
                *shape = Some(path);
            }
            usvg::Node::Group(g) => {
                if g.clip_path().is_some() || !find_single_clip_shape(g, shape) {
                    return false;
                }
            }
            usvg::Node::Text(text) => {
                if !find_single_clip_shape(text.flattened(), shape) {
                    return false;
                }
            }
            usvg::Node::Image(_) => {}
        }
    }
    true
}

fn clip_fill_rule(path: &usvg::Path) -> Fill {
    match path.fill().map(|fill| fill.rule()) {
        Some(usvg::FillRule::EvenOdd) => Fill::EvenOdd,
        _ => Fill::NonZero,
    }
}

/// Clip the content drawn so far in the current layer by `clip`, using it as an alpha mask.
///
/// `transform` maps the clipped element's user space to canvas space, and `bounds` are the
/// canvas space bounds of the current layer.
fn apply_clip_mask<S: PaintScene>(
    scene: &mut S,
    clip: &usvg::ClipPath,
    transform: Affine,
    bounds: Rect,
) {
    scene.push_mask_layer(Fill::NonZero, MaskMode::Alpha, Affine::IDENTITY, &bounds);
    fill_clip_group(
        scene,
        clip.root(),
        transform * util::to_affine(&clip.transform()),
        bounds,
    );
    scene.pop_layer();

    // A clip path's own clip-path is in the same user space as the clip path itself,
    // and intersects with it.
    if let Some(clip) = clip.clip_path() {
        apply_clip_mask(scene, clip, transform, bounds);
    }
}

/// Fill the shapes of a clip path group with an opaque color.
fn fill_clip_group<S: PaintScene>(
    scene: &mut S,
    group: &usvg::Group,
    transform: Affine,
    bounds: Rect,
) {
    for node in group.children() {
        match node {
            usvg::Node::Path(path) => {
                if !path.is_visible() {
                    continue;
                }
                scene.fill(
                    clip_fill_rule(path),
                    transform * util::to_affine(&path.abs_transform()),
                    Color::BLACK,
                    None,
                    &util::to_bez_path(path),
                );
            }
            usvg::Node::Group(g) => match g.clip_path() {
                // A clip path child with its own clip-path is drawn into a layer and clipped
                // before being added to the clip path.
                Some(clip) => {
                    scene.push_layer(
                        Fill::NonZero,
                        BlendMode::default(),
                        1.0,
                        Affine::IDENTITY,
                        &bounds,
                        None,
                        None,
                    );
                    fill_clip_group(scene, g, transform, bounds);
                    apply_clip_mask(
                        scene,
                        clip,
                        transform * util::to_affine(&g.abs_transform()),
                        bounds,
                    );
                    scene.pop_layer();
                }
                None => fill_clip_group(scene, g, transform, bounds),
            },
            usvg::Node::Text(text) => fill_clip_group(scene, text.flattened(), transform, bounds),
            usvg::Node::Image(_) => {}
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
