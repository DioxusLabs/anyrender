use std::sync::Arc;

use anyrender::filters::{
    Filter, FilterColorSpace, FilterEffect, FilterInputs, color_transformation::ColorMatrix,
};
use anyrender::{PaintScene, render_to_buffer};
use anyrender_skia::SkiaImageRenderer;
use kurbo::{Affine, Rect};
use peniko::{Color, Fill, Mix};

const SIZE: u32 = 16;

/// Halves the red channel of an opaque red square in `color_space`.
fn halve_red(color_space: FilterColorSpace) -> [u8; 4] {
    let mut filter = Filter::empty();
    #[rustfmt::skip]
    let matrix = ColorMatrix([
        0.5, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0, 0.0,
    ]);
    filter.add_with_color_space(
        FilterEffect::ColorMatrix(matrix),
        FilterInputs::NONE,
        color_space,
    );
    let filter = Arc::new(filter);
    let rect = Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64);
    let buffer = render_to_buffer::<SkiaImageRenderer, _>(
        |scene| {
            scene.push_layer(
                Fill::NonZero,
                Mix::Normal,
                1.0,
                Affine::IDENTITY,
                &rect,
                Some(filter.clone()),
                None,
            );
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::from_rgb8(255, 0, 0),
                None,
                &rect,
            );
            scene.pop_layer();
        },
        SIZE,
        SIZE,
    );
    let i = ((8 * SIZE + 8) * 4) as usize;
    buffer[i..i + 4].try_into().unwrap()
}

fn assert_close(actual: [u8; 4], expected: [u8; 4]) {
    assert!(
        actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 2),
        "expected {expected:?}, got {actual:?}"
    );
}

#[test]
fn filters_respect_color_space() {
    assert_close(halve_red(FilterColorSpace::Srgb), [128, 0, 0, 255]);
    // Linear 0.5 encodes to sRGB 188.
    assert_close(halve_red(FilterColorSpace::LinearRgb), [188, 0, 0, 255]);
}
