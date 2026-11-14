#![cfg(all(feature = "filters", not(feature = "multithreading")))]

use std::sync::Arc;

use anyrender::filters::{Filter, FilterEffect, FilterInputs};
use anyrender::{PaintScene, render_to_buffer};
use anyrender_vello_cpu::VelloCpuImageRenderer;
use kurbo::{Affine, Rect, Vec2};
use peniko::{Color, Fill, Mix};

const SIZE: u32 = 16;

/// Renders `filter` over a full-size layer containing an opaque red square.
fn render_filter(filter: Filter) -> Vec<u8> {
    let filter = Arc::new(filter);
    let rect = Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64);
    render_to_buffer::<VelloCpuImageRenderer, _>(
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
    )
}

fn pixel(buffer: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    buffer[i..i + 4].try_into().unwrap()
}

#[test]
fn results_are_clipped_to_the_filter_region() {
    let mut filter = Filter::empty().with_region(Some(Rect::new(0.0, 0.0, 4.0, 4.0)));
    filter.add(
        FilterEffect::Flood(Color::from_rgb8(0, 255, 0)),
        FilterInputs::NONE,
    );
    let buffer = render_filter(filter);
    assert_eq!(pixel(&buffer, 1, 1), [0, 255, 0, 255]);
    assert_eq!(pixel(&buffer, 9, 9), [0, 0, 0, 0]);
}

#[test]
fn source_graphic_is_clipped_to_the_filter_region() {
    let region = Rect::new(8.0, 0.0, SIZE as f64, SIZE as f64);
    let mut filter = Filter::empty().with_region(Some(region));
    filter.add(
        FilterEffect::Offset(Vec2::new(4.0, 0.0)),
        FilterInputs::NONE,
    );
    let buffer = render_filter(filter);
    assert_eq!(pixel(&buffer, 4, 8), [0, 0, 0, 0]);
    assert_eq!(pixel(&buffer, 10, 8), [0, 0, 0, 0]);
    assert_eq!(pixel(&buffer, 13, 8), [255, 0, 0, 255]);
}
