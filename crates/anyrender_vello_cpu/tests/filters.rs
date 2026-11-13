#![cfg(feature = "filters")]

use std::sync::Arc;

use anyrender::PaintScene;
use anyrender::filters::{Filter, FilterEffect, FilterInput, FilterInputs, FilterSource};
use anyrender_vello_cpu::VelloCpuScenePainter;
use kurbo::{Affine, Rect, Vec2};
use peniko::{BlendMode, Color, Fill};

const SIZE: u16 = 200;

fn render(draw: impl FnOnce(&mut VelloCpuScenePainter)) -> impl Fn(usize, usize) -> [u8; 4] {
    let mut scene = VelloCpuScenePainter::new_single_threaded(SIZE, SIZE);
    draw(&mut scene);
    let pixmap = scene.finish();
    move |x, y| {
        let p = pixmap.data()[y * SIZE as usize + x];
        [p.r, p.g, p.b, p.a]
    }
}

fn red_square(scene: &mut VelloCpuScenePainter, filter: Filter) {
    scene.push_layer(
        Fill::NonZero,
        BlendMode::default(),
        1.0,
        Affine::IDENTITY,
        &Rect::new(0.0, 0.0, 200.0, 200.0),
        Some(Arc::new(filter)),
        None,
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(255, 0, 0),
        None,
        &Rect::new(50.0, 50.0, 150.0, 150.0),
    );
    scene.pop_layer();
}

#[test]
fn blur_inside_rect_clip_is_soft() {
    let pixel = render(|scene| {
        scene.push_clip_layer(
            Fill::NonZero,
            Affine::IDENTITY,
            &Rect::new(40.0, 40.0, 190.0, 190.0),
        );
        red_square(scene, Filter::single(FilterEffect::blur(6.0)));
        scene.pop_layer();
    });
    // Blurred across the edge at x = 50, and still clipped at x = 40.
    assert_eq!(pixel(39, 100)[3], 0);
    assert!(pixel(42, 100)[3] > 0, "{:?}", pixel(42, 100));
    let edge = pixel(50, 100)[3];
    assert!(edge > 100 && edge < 160, "{edge}");
}

#[test]
fn multi_primitive_graph_is_unfiltered() {
    let mut filter = Filter::empty();
    let blur = filter.add(
        FilterEffect::blur(4.0),
        FilterInputs::single(FilterInput::Source(FilterSource::SourceGraphic)),
    );
    filter.add(
        FilterEffect::Offset(Vec2::new(30.0, 0.0)),
        FilterInputs::single(FilterInput::Result(blur)),
    );
    let pixel = render(|scene| red_square(scene, filter));
    assert_eq!(pixel(50, 100), [255, 0, 0, 255]);
    assert_eq!(pixel(160, 100)[3], 0);
}

#[test]
fn source_alpha_input_is_unfiltered() {
    let filter = {
        let mut filter = Filter::empty();
        filter.add(
            FilterEffect::blur(4.0),
            FilterInputs::single(FilterInput::Source(FilterSource::SourceAlpha)),
        );
        filter
    };
    let pixel = render(|scene| red_square(scene, filter));
    assert_eq!(pixel(50, 100), [255, 0, 0, 255]);
}

#[test]
fn flood_fills_the_filter_region() {
    let pixel = render(|scene| {
        scene.push_layer(
            Fill::NonZero,
            BlendMode::default(),
            1.0,
            Affine::IDENTITY,
            &Rect::new(20.0, 20.0, 180.0, 180.0),
            Some(Arc::new(Filter::single(FilterEffect::Flood(
                Color::from_rgb8(0, 0, 255),
            )))),
            None,
        );
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::from_rgb8(255, 0, 0),
            None,
            &Rect::new(90.0, 90.0, 110.0, 110.0),
        );
        scene.pop_layer();
    });
    assert_eq!(pixel(100, 100), [0, 0, 255, 255]);
    assert_eq!(pixel(25, 25), [0, 0, 255, 255]);
    assert_eq!(pixel(175, 175), [0, 0, 255, 255]);
    assert_eq!(pixel(10, 10)[3], 0);
}
