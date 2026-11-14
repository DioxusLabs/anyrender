use std::sync::Arc;

use anyrender::filters::{
    Filter, FilterColorSpace, FilterEffect, FilterGraphNode, FilterImage, FilterInput,
    FilterInputs, color_transformation::ColorMatrix,
};
use anyrender::{PaintScene, render_to_buffer};
use anyrender_skia::SkiaImageRenderer;
use kurbo::{Affine, Rect, Vec2};
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

/// Renders `filter` over a full-size layer containing an opaque red square.
fn render_filter(filter: Filter) -> Vec<u8> {
    let filter = Arc::new(filter);
    let rect = Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64);
    render_to_buffer::<SkiaImageRenderer, _>(
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

fn green_flood(subregion: Rect) -> FilterGraphNode {
    FilterGraphNode {
        effect: FilterEffect::Flood(Color::from_rgb8(0, 255, 0)),
        inputs: FilterInputs::NONE,
        color_space: FilterColorSpace::Srgb,
        subregion: Some(subregion),
    }
}

#[test]
fn results_are_clipped_to_subregions() {
    let mut filter = Filter::empty();
    filter.add_node(green_flood(Rect::new(0.0, 0.0, 4.0, 4.0)));
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 1, 1), [0, 255, 0, 255]);
    assert_close(pixel(&buffer, 9, 9), [0, 0, 0, 0]);
}

#[test]
fn results_are_clipped_to_the_filter_region() {
    let mut filter = Filter::empty().with_region(Some(Rect::new(0.0, 0.0, 4.0, 4.0)));
    filter.add(
        FilterEffect::Flood(Color::from_rgb8(0, 255, 0)),
        FilterInputs::NONE,
    );
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 1, 1), [0, 255, 0, 255]);
    assert_close(pixel(&buffer, 9, 9), [0, 0, 0, 0]);
}

#[test]
fn source_graphic_is_clipped_to_the_filter_region() {
    let half = Rect::new(0.0, 0.0, 8.0, SIZE as f64);
    let mut filter = Filter::empty().with_region(Some(half));
    filter.add(
        FilterEffect::Offset(Vec2::new(8.0, 0.0)),
        FilterInputs::NONE,
    );
    let buffer = render_filter(filter);
    // The source right of the region is not shifted into it.
    assert_close(pixel(&buffer, 4, 8), [0, 0, 0, 0]);

    let mut filter = Filter::empty().with_region(Some(half.with_origin((8.0, 0.0))));
    filter.add(
        FilterEffect::Offset(Vec2::new(4.0, 0.0)),
        FilterInputs::NONE,
    );
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 10, 8), [0, 0, 0, 0]);
    assert_close(pixel(&buffer, 13, 8), [255, 0, 0, 255]);
}

#[test]
fn tile_repeats_its_input_subregion() {
    let mut filter = Filter::empty();
    let flood = filter.add_node(green_flood(Rect::new(0.0, 0.0, 4.0, 4.0)));
    filter.add_node(FilterGraphNode {
        effect: FilterEffect::Tile,
        inputs: FilterInputs::single(FilterInput::Result(flood)),
        color_space: FilterColorSpace::Srgb,
        subregion: Some(Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64)),
    });
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 9, 9), [0, 255, 0, 255]);
    assert_close(pixel(&buffer, 14, 2), [0, 255, 0, 255]);
}

#[test]
fn image_renders_its_scene() {
    let mut content = anyrender::recording::Scene::new();
    content.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(0, 255, 0),
        None,
        &Rect::new(0.0, 0.0, 8.0, 8.0),
    );
    let mut filter = Filter::empty();
    filter.add_node(FilterGraphNode {
        effect: FilterEffect::Image(FilterImage { scene: content }),
        inputs: FilterInputs::NONE,
        color_space: FilterColorSpace::LinearRgb,
        subregion: None,
    });
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 2, 2), [0, 255, 0, 255]);
    assert_close(pixel(&buffer, 12, 12), [0, 0, 0, 0]);
}
