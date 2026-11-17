//! SVG gradient stops interpolate unpremultiplied (SVG 1.1 § 13.2.4, as in resvg).

use anyrender::render_to_buffer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use kurbo::Affine;

const SIZE: u32 = 100;

fn render(svg: &str) -> Vec<u8> {
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).unwrap();
    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| anyrender_svg::render_svg_tree(scene, &tree, Affine::IDENTITY),
        SIZE,
        SIZE,
    )
}

fn pixel(buffer: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    buffer[i..i + 4].try_into().unwrap()
}

fn assert_close(actual: [u8; 4], expected: [u8; 4]) {
    assert!(
        actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 2),
        "expected {expected:?}, got {actual:?}"
    );
}

/// Opaque red to fully transparent lime, over blue. Halfway, unpremultiplied
/// interpolation gives 50% (127, 127, 0) (lime leaks in); premultiplied gives
/// 50% red.
fn svg_with(gradient: &str) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <defs>{gradient}</defs>
            <rect width="100" height="100" fill="blue"/>
            <rect width="100" height="100" fill="url(#g)"/>
        </svg>"#
    )
}

const STOPS: &str = r#"<stop offset="0" stop-color="red"/>
    <stop offset="1" stop-color="lime" stop-opacity="0"/>"#;

#[test]
fn linear_gradient_interpolates_unpremultiplied() {
    let buffer = render(&svg_with(&format!(
        r#"<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" x2="100">{STOPS}</linearGradient>"#
    )));
    assert_close(pixel(&buffer, 0, 50), [255, 0, 0, 255]);
    assert_close(pixel(&buffer, 50, 50), [64, 64, 128, 255]);
    assert_close(pixel(&buffer, 99, 50), [0, 1, 254, 255]);
}

#[test]
fn radial_gradient_interpolates_unpremultiplied() {
    let buffer = render(&svg_with(&format!(
        r#"<radialGradient id="g" gradientUnits="userSpaceOnUse" cx="0" cy="50" r="100">{STOPS}</radialGradient>"#
    )));
    assert_close(pixel(&buffer, 50, 50), [64, 64, 128, 255]);
}

/// A transparent white stop fades to an opaque black one through gray, not
/// through black (which darkens luminance masks).
#[test]
fn transparent_white_stop_does_not_fade_through_black() {
    let buffer = render(&svg_with(
        r#"<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" x2="100">
            <stop offset="0" stop-color="white" stop-opacity="0"/>
            <stop offset="1" stop-color="black"/>
        </linearGradient>"#,
    ));
    // 50% gray at 50% opacity over blue; premultiplied would give 50% black.
    assert_close(pixel(&buffer, 50, 50), [64, 64, 192, 255]);
}
