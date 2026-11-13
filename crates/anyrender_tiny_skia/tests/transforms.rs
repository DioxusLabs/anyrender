use anyrender::{ImageRenderer, PaintScene};
use anyrender_tiny_skia::TinySkiaImageRenderer;
use kurbo::{Affine, Rect};
use peniko::{Color, Fill, Mix};

const SIZE: u32 = 40;

fn render(
    draw: impl FnOnce(&mut <TinySkiaImageRenderer as ImageRenderer>::ScenePainter<'_>),
) -> Vec<u8> {
    let mut renderer = TinySkiaImageRenderer::new(SIZE, SIZE);
    let mut buffer = Vec::new();
    renderer.render_to_vec(draw, &mut buffer);
    buffer
}

fn alpha_at(buffer: &[u8], x: u32, y: u32) -> u8 {
    buffer[((y * SIZE + x) * 4 + 3) as usize]
}

const RED: Color = Color::from_rgb8(255, 0, 0);

#[test]
fn fill_transform_is_applied_once() {
    let buffer = render(|scene| {
        let transform = Affine::translate((10.0, 10.0)) * Affine::scale(2.0);
        scene.fill(
            Fill::NonZero,
            transform,
            RED,
            None,
            &Rect::new(0.0, 0.0, 5.0, 5.0),
        );
    });
    assert_eq!(alpha_at(&buffer, 9, 9), 0);
    assert_eq!(alpha_at(&buffer, 10, 10), 255);
    assert_eq!(alpha_at(&buffer, 19, 19), 255);
    assert_eq!(alpha_at(&buffer, 20, 20), 0);
}

#[test]
fn layer_transform_only_affects_clip() {
    // The layer's transform positions its clip; content uses its own transform.
    let buffer = render(|scene| {
        let clip_transform = Affine::translate((20.0, 0.0));
        scene.push_layer(
            Fill::NonZero,
            Mix::Normal,
            1.0,
            clip_transform,
            &Rect::new(0.0, 0.0, 10.0, 40.0),
            None,
            None,
        );
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            RED,
            None,
            &Rect::new(0.0, 0.0, 40.0, 40.0),
        );
        scene.pop_layer();
    });
    assert_eq!(alpha_at(&buffer, 5, 5), 0);
    assert_eq!(alpha_at(&buffer, 25, 5), 255);
    assert_eq!(alpha_at(&buffer, 35, 5), 0);
}

#[test]
fn nested_layer_transforms_do_not_accumulate() {
    let buffer = render(|scene| {
        let outer = Affine::translate((10.0, 10.0));
        scene.push_clip_layer(Fill::NonZero, outer, &Rect::new(0.0, 0.0, 20.0, 20.0));
        let inner = outer * Affine::scale(2.0);
        scene.push_clip_layer(Fill::NonZero, inner, &Rect::new(0.0, 0.0, 5.0, 5.0));
        scene.fill(
            Fill::NonZero,
            inner,
            RED,
            None,
            &Rect::new(0.0, 0.0, 100.0, 100.0),
        );
        scene.pop_layer();
        scene.pop_layer();
    });
    assert_eq!(alpha_at(&buffer, 9, 9), 0);
    assert_eq!(alpha_at(&buffer, 10, 10), 255);
    assert_eq!(alpha_at(&buffer, 19, 19), 255);
    assert_eq!(alpha_at(&buffer, 21, 21), 0);
}

#[test]
fn rotated_brush_transform() {
    // A 90° rotation of a square about its centre should cover the same pixels.
    let buffer = render(|scene| {
        let transform = Affine::rotate_about(std::f64::consts::FRAC_PI_2, (20.0, 20.0));
        scene.fill(
            Fill::NonZero,
            transform,
            RED,
            None,
            &Rect::new(10.0, 10.0, 30.0, 30.0),
        );
    });
    assert_eq!(alpha_at(&buffer, 11, 11), 255);
    assert_eq!(alpha_at(&buffer, 28, 28), 255);
    assert_eq!(alpha_at(&buffer, 5, 5), 0);
}
