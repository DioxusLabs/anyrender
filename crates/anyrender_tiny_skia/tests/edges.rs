use anyrender::{BoxShadowKind, ImageRenderer, NonUniformRoundedRect, PaintScene};
use anyrender_tiny_skia::TinySkiaImageRenderer;
use kurbo::{Affine, Rect, Vec2};
use peniko::{Color, Fill, Gradient, InterpolationAlphaSpace, Mix};

const SIZE: u32 = 40;

fn render(
    draw: impl FnOnce(&mut <TinySkiaImageRenderer as ImageRenderer>::ScenePainter<'_>),
) -> Vec<u8> {
    let mut renderer = TinySkiaImageRenderer::new(SIZE, SIZE);
    let mut buffer = Vec::new();
    renderer.render_to_vec(draw, &mut buffer);
    buffer
}

fn pixel_at(buffer: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    buffer[i..i + 4].try_into().unwrap()
}

const RED: Color = Color::from_rgb8(255, 0, 0);

fn fill_in_clip(clip: Rect) -> Vec<u8> {
    render(|scene| {
        scene.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &clip);
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            RED,
            None,
            &Rect::new(0.0, 0.0, 40.0, 40.0),
        );
        scene.pop_layer();
    })
}

#[test]
fn huge_clip_keeps_content() {
    let buffer = fill_in_clip(Rect::new(-3e9, -3e9, 3e9, 3e9));
    assert_eq!(pixel_at(&buffer, 20, 20)[3], 255);
}

#[test]
fn infinite_clip_keeps_content() {
    let inf = f64::INFINITY;
    let buffer = fill_in_clip(Rect::new(-inf, -inf, inf, inf));
    assert_eq!(pixel_at(&buffer, 20, 20)[3], 255);

    let buffer = render(|scene| {
        scene.push_layer(
            Fill::NonZero,
            Mix::Normal,
            1.0,
            Affine::scale(2.0),
            &Rect::new(-inf, -inf, inf, inf),
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
    assert_eq!(pixel_at(&buffer, 20, 20)[3], 255);
}

#[test]
fn nested_layer_in_offset_parent() {
    // The parent layer's x range doesn't overlap its y range, so clamping either axis to the
    // other's range would drop the inner layer.
    let buffer = render(|scene| {
        scene.push_clip_layer(
            Fill::NonZero,
            Affine::IDENTITY,
            &Rect::new(20.0, 0.0, 40.0, 10.0),
        );
        scene.push_layer(
            Fill::NonZero,
            Mix::Normal,
            0.5,
            Affine::IDENTITY,
            &Rect::new(20.0, 0.0, 40.0, 10.0),
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
        scene.pop_layer();
    });
    assert!((120..136).contains(&pixel_at(&buffer, 30, 5)[3]));
}

#[test]
fn box_shadow_is_blurred_before_cropping() {
    // The box covers the canvas with a wide margin, so the shadow is opaque everywhere,
    // including at the edges of the canvas and of the clip layer.
    let shadow = |scene: &mut <TinySkiaImageRenderer as ImageRenderer>::ScenePainter<'_>| {
        scene.draw_box_shadow(
            Affine::IDENTITY,
            &NonUniformRoundedRect::from(Rect::new(-100.0, -100.0, 140.0, 140.0)),
            Vec2::ZERO,
            0.0,
            4.0,
            RED,
            BoxShadowKind::OutsetUnclipped,
        );
    };
    let buffer = render(shadow);
    for x in 0..SIZE {
        assert!(pixel_at(&buffer, x, 20)[3] >= 250, "x {x}");
    }

    let buffer = render(|scene| {
        scene.push_clip_layer(
            Fill::NonZero,
            Affine::IDENTITY,
            &Rect::new(4.0, 0.0, 40.0, 40.0),
        );
        shadow(scene);
        scene.pop_layer();
    });
    for x in 4..SIZE {
        assert!(pixel_at(&buffer, x, 20)[3] >= 250, "x {x}");
    }
}

fn gradient_midpoint(space: InterpolationAlphaSpace) -> [u8; 4] {
    let mut gradient = Gradient::new_linear((0.0, 0.0), (40.0, 0.0)).with_stops([
        (0.0, Color::from_rgba8(255, 0, 0, 0)),
        (1.0, Color::from_rgb8(0, 0, 255)),
    ]);
    gradient.interpolation_alpha_space = space;
    let buffer = render(|scene| {
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            &gradient,
            None,
            &Rect::new(0.0, 0.0, 40.0, 40.0),
        );
    });
    pixel_at(&buffer, 20, 20)
}

#[test]
fn gradient_interpolates_premultiplied() {
    // Output is premultiplied: half-transparent blue.
    let [r, _, b, a] = gradient_midpoint(InterpolationAlphaSpace::Premultiplied);
    assert!(r < 8, "red {r}");
    assert!((112..144).contains(&b), "blue {b}");
    assert!((112..144).contains(&a), "alpha {a}");
}

#[test]
fn gradient_interpolates_unpremultiplied() {
    // Purple at half alpha, premultiplied.
    let [r, _, b, _] = gradient_midpoint(InterpolationAlphaSpace::Unpremultiplied);
    assert!((48..80).contains(&r), "red {r}");
    assert!((48..80).contains(&b), "blue {b}");
}
