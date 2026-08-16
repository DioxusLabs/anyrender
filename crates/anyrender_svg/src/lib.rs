// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render an SVG into any impl of [`anyrender::PaintScene`].
//!
//! This currently lacks support for some important SVG features. Known missing features include: masking, filter effects, group backgrounds
//! path shape-rendering, and patterns.

// LINEBENDER LINT SET - lib.rs - v1
// See https://linebender.org/wiki/canonical-lints/
// These lints aren't included in Cargo.toml because they
// shouldn't apply to examples and tests
#![warn(unused_crate_dependencies)]
#![warn(clippy::print_stdout, clippy::print_stderr)]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
#![allow(missing_docs, clippy::shadow_unrelated, clippy::missing_errors_doc)]
#![cfg_attr(test, allow(unused_crate_dependencies))] // Some dev dependencies are only used in tests

mod error;
mod render;
mod util;

pub use error::Error;
pub use usvg;

use anyrender::PaintScene;
use kurbo::Affine;

/// Append an SVG to an [`anyrender::PaintScene`].
///
/// This will draw a red box over (some) unsupported elements.
pub fn render_svg_str<S: PaintScene>(
    scene: &mut S,
    svg: &str,
    transform: Affine,
) -> Result<(), Error> {
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_str(svg, &opt)?;
    render_svg_tree(scene, &tree, transform);
    Ok(())
}

/// Append an SVG to an [`anyrender::PaintScene`] (with custom error handling).
///
/// See the [module level documentation](crate#unsupported-features) for a list of some unsupported svg features
pub fn render_svg_str_with<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    svg: &str,
    transform: Affine,
    error_handler: &mut F,
) -> Result<(), Error> {
    let opt = usvg::Options::default();
    let tree = usvg::Tree::from_str(svg, &opt)?;
    render_svg_tree_with(scene, &tree, transform, error_handler);
    Ok(())
}

/// Append a [`usvg::Tree`] to an [`anyrender::PaintScene`].
///
/// This will draw a red box over (some) unsupported elements.
pub fn render_svg_tree<S: PaintScene>(scene: &mut S, svg: &usvg::Tree, transform: Affine) {
    render_svg_tree_with(scene, svg, transform, &mut util::default_error_handler);
}

/// Append a [`usvg::Tree`] to an [`anyrender::PaintScene`] (with custom error handling).
///
/// See the [module level documentation](crate#unsupported-features) for a list of some unsupported svg features
pub fn render_svg_tree_with<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    svg: &usvg::Tree,
    transform: Affine,
    error_handler: &mut F,
) {
    render::render_group(
        scene,
        svg.root(),
        Affine::IDENTITY,
        transform,
        error_handler,
    );
}

#[cfg(test)]
mod tests {
    use super::render_svg_str;
    use anyrender::{Scene, recording::RenderCommand};
    use kurbo::{Affine, Shape};

    #[test]
    fn opacity_layer_clip_rect_covers_group_in_canvas_space() {
        // A semi-transparent group nested inside a transformed parent group.
        // The fallback opacity layer's clip rect must land where the group's
        // content actually is on the canvas, outset by 2px on each side.
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
          <g transform="translate(50,30) scale(2)">
            <g opacity="0.5">
              <rect x="10" y="10" width="20" height="20" fill="red"/>
            </g>
          </g>
        </svg>"#;

        let mut scene = Scene::new();
        render_svg_str(&mut scene, svg, Affine::IDENTITY).unwrap();

        let layer = scene
            .commands
            .iter()
            .find_map(|cmd| match cmd {
                RenderCommand::PushLayer(layer) => Some(layer),
                _ => None,
            })
            .expect("expected an opacity layer to be pushed");

        // Group content occupies (70, 50) to (110, 90) on the canvas
        // (rect 10,10 20x20 mapped through translate(50,30) scale(2)),
        // outset by 2px on each side.
        let clip_bbox = layer
            .transform
            .transform_rect_bbox(layer.clip.bounding_box());
        let expected = kurbo::Rect::new(68.0, 48.0, 112.0, 92.0);
        assert!(
            (clip_bbox.x0 - expected.x0).abs() < 1e-3
                && (clip_bbox.y0 - expected.y0).abs() < 1e-3
                && (clip_bbox.x1 - expected.x1).abs() < 1e-3
                && (clip_bbox.y1 - expected.y1).abs() < 1e-3,
            "clip bbox {clip_bbox:?} != expected {expected:?}"
        );
    }
}
