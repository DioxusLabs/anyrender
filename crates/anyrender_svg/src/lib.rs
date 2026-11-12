// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Render an SVG into any impl of [`anyrender::PaintScene`].
//!
//! This currently lacks support for some important SVG features. Known missing features include: filter effects, group backgrounds
//! path shape-rendering, and patterns. Luminance masks are rendered as alpha masks by backends without luminance mask support.

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
    render::render_group(
        scene,
        svg.root(),
        Affine::IDENTITY,
        transform,
        &mut util::default_error_handler,
    );
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
        &mut |scene, node, _| error_handler(scene, node),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyrender::MaskMode;
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

    fn render_mask(mask_attrs: &str) -> Vec<RenderCommand> {
        let svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <defs><mask id="mask" {mask_attrs}>
                    <circle cx="50" cy="50" r="40" fill="white"/>
                </mask></defs>
                <rect width="100" height="100" mask="url(#mask)"/>
            </svg>"##
        );
        let mut scene = Scene::new();
        render_svg_str(&mut scene, &svg, Affine::IDENTITY).unwrap();
        scene.commands
    }

    fn mask_modes(commands: &[RenderCommand]) -> Vec<MaskMode> {
        commands
            .iter()
            .filter_map(|command| match command {
                RenderCommand::PushMaskLayer(mask) => Some(mask.mode),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn mask_type_is_preserved() {
        let alpha = render_mask(r#"style="mask-type:alpha""#);
        assert_eq!(mask_modes(&alpha), [MaskMode::Alpha]);
        let luminance = render_mask("");
        assert_eq!(mask_modes(&luminance), [MaskMode::Luminance]);
    }

    fn fill_transforms(svg: &str, transform: Affine) -> Vec<Affine> {
        let mut scene = Scene::new();
        render_svg_str(&mut scene, svg, transform).unwrap();
        scene
            .commands
            .iter()
            .filter_map(|command| match command {
                RenderCommand::Fill(fill) => Some(fill.transform),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn embedded_svg_nested_groups_keep_image_transform() {
        let inner = r#"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/><g transform="translate(1 2)"><rect width="5" height="5"/></g></svg>"#;
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><image x="50" y="60" width="10" height="10" href="data:image/svg+xml;utf8,{}"/></svg>"#,
            inner
                .replace('"', "'")
                .replace('<', "%3C")
                .replace('>', "%3E")
        );
        assert_eq!(
            fill_transforms(&svg, Affine::scale(2.0)),
            [
                Affine::scale(2.0) * Affine::translate((50.0, 60.0)),
                Affine::scale(2.0) * Affine::translate((51.0, 62.0)),
            ]
        );
    }

    #[test]
    fn error_placeholder_uses_node_transform() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <defs><pattern id="p" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="2" height="2"/></pattern></defs>
            <g transform="translate(50 40)"><rect width="20" height="20" fill="url(#p)"/></g>
        </svg>"#;
        assert_eq!(
            fill_transforms(svg, Affine::scale(2.0)),
            [Affine::scale(2.0) * Affine::translate((50.0, 40.0))]
        );
    }

    #[test]
    fn mask_layer_is_isolated() {
        let commands = render_mask("");
        assert!(matches!(commands[0], RenderCommand::PushLayer(_)));
        assert!(matches!(commands[2], RenderCommand::PushMaskLayer(_)));
        assert!(matches!(commands.last(), Some(RenderCommand::PopLayer)));
    }
}
