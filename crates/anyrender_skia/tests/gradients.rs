use anyrender::{PaintScene, render_to_buffer};
use anyrender_skia::SkiaImageRenderer;
use kurbo::{Affine, Rect};
use peniko::{Color, Fill, Gradient, InterpolationAlphaSpace};

const SIZE: u32 = 100;

/// Opaque red to fully transparent lime over blue, sampled halfway.
fn midpoint(alpha_space: InterpolationAlphaSpace) -> [u8; 4] {
    let gradient = Gradient::new_linear((0.0, 0.0), (SIZE as f64, 0.0))
        .with_stops([Color::from_rgb8(255, 0, 0), Color::from_rgba8(0, 255, 0, 0)])
        .with_interpolation_alpha_space(alpha_space);
    let rect = Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64);
    let buffer = render_to_buffer::<SkiaImageRenderer, _>(
        |scene| {
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::from_rgb8(0, 0, 255),
                None,
                &rect,
            );
            scene.fill(Fill::NonZero, Affine::IDENTITY, &gradient, None, &rect);
        },
        SIZE,
        SIZE,
    );
    let i = ((50 * SIZE + 50) * 4) as usize;
    buffer[i..i + 4].try_into().unwrap()
}

fn assert_close(actual: [u8; 4], expected: [u8; 4]) {
    assert!(
        actual.iter().zip(expected).all(|(a, e)| a.abs_diff(e) <= 2),
        "expected {expected:?}, got {actual:?}"
    );
}

#[test]
fn gradients_respect_interpolation_alpha_space() {
    assert_close(
        midpoint(InterpolationAlphaSpace::Premultiplied),
        [128, 0, 128, 255],
    );
    assert_close(
        midpoint(InterpolationAlphaSpace::Unpremultiplied),
        [64, 64, 128, 255],
    );
}
