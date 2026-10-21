// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render an SVG into any impl of [`anyrender::PaintScene`].
//!
//! This currently lacks support for some important SVG features. Known missing features include: masking, group backgrounds
//! path shape-rendering, and patterns.
//!
//! Filter effects are translated into [`anyrender::Filter`] graphs attached to layers.
//! Whether (and how faithfully) they are applied depends on the rendering backend.

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
mod filter;
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
    use super::*;
    use anyrender::recording::{RenderCommand, Scene};
    use peniko::Fill;

    fn clip_rule(clip_attrs: &str, path_attrs: &str) -> Fill {
        let svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <defs><clipPath id="clip" {clip_attrs}>
                    <path {path_attrs} d="M0 0H100V100H0Z M25 25H75V75H25Z"/>
                </clipPath></defs>
                <rect width="100" height="100" clip-path="url(#clip)"/>
            </svg>"##
        );
        let mut scene = Scene::new();
        render_svg_str(&mut scene, &svg, Affine::IDENTITY).unwrap();
        scene
            .commands
            .iter()
            .find_map(|command| match command {
                RenderCommand::PushLayer(layer) => Some(layer.fill),
                _ => None,
            })
            .expect("SVG should produce a clipping layer")
    }

    #[test]
    fn clip_rule_attribute_is_preserved() {
        assert_eq!(clip_rule("", ""), Fill::NonZero);
        assert_eq!(clip_rule("", r#"clip-rule="nonzero""#), Fill::NonZero);
        assert_eq!(clip_rule("", r#"clip-rule="evenodd""#), Fill::EvenOdd);
    }

    #[test]
    fn clip_rule_style_overrides_attribute_and_inherits() {
        assert_eq!(
            clip_rule("", r#"clip-rule="nonzero" style="clip-rule: evenodd""#),
            Fill::EvenOdd
        );
        assert_eq!(
            clip_rule(r#"style="clip-rule: evenodd""#, ""),
            Fill::EvenOdd
        );
        assert_eq!(
            clip_rule(r#"clip-rule="evenodd""#, r#"style="clip-rule: nonzero""#),
            Fill::NonZero
        );
    }

    #[test]
    fn fill_rule_does_not_control_clipping() {
        assert_eq!(clip_rule("", r#"fill-rule="evenodd""#), Fill::NonZero);
        assert_eq!(
            clip_rule("", r#"fill-rule="nonzero" clip-rule="evenodd""#),
            Fill::EvenOdd
        );
    }
}
