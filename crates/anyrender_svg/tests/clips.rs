//! Clip paths clip to the same edges as filling their shape directly.

use anyrender::render_to_buffer;
use anyrender_tiny_skia::TinySkiaImageRenderer;
use kurbo::Affine;

const SIZE: u32 = 300;

fn render(body: &str) -> Vec<u8> {
    let svg =
        format!(r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">{body}</svg>"#);
    let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
    render_to_buffer::<TinySkiaImageRenderer, _>(
        |scene| anyrender_svg::render_svg_tree(scene, &tree, Affine::scale(1.5)),
        SIZE,
        SIZE,
    )
}

fn max_difference(a: &[u8], b: &[u8]) -> u8 {
    a.iter().zip(b).map(|(a, b)| a.abs_diff(*b)).max().unwrap()
}

/// The clip's straight edges lie on the bounds of its layer (resvg's
/// `masking/clipPath/clipPathUnits=objectBoundingBox`).
#[test]
fn clip_edges_on_the_layer_bounds_match_a_direct_fill() {
    let clipped = render(
        r#"<clipPath id="c" clipPathUnits="objectBoundingBox">
            <rect x="0.2" y="0.2" width="0.6" height="0.6" rx="0.1" ry="0.1"/>
        </clipPath>
        <rect width="200" height="200" fill="green" clip-path="url(#c)"/>"#,
    );
    let filled = render(r#"<rect x="40" y="40" width="120" height="120" rx="20" fill="green"/>"#);
    // Without margin around the clip's layer, its top row is under-covered by up to 48.
    let diff = max_difference(&clipped, &filled);
    assert!(diff <= 16, "max difference {diff}");
}
