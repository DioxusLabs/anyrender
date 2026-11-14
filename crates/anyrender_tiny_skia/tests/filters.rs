use std::sync::Arc;

use anyrender::filters::{
    Filter, FilterColorSpace, FilterEffect, FilterGraphNode, FilterInput, FilterInputs,
    color_transformation::ColorMatrix,
    composite::CompositeOperator,
    turbulence::{TurbulenceFilter, TurbulenceType},
};
use anyrender::{PaintScene, render_to_buffer};
use anyrender_tiny_skia::TinySkiaImageRenderer;
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
    let buffer = render_to_buffer::<TinySkiaImageRenderer, _>(
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
    render_to_buffer::<TinySkiaImageRenderer, _>(
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

fn turbulence(seed: f32) -> Vec<u8> {
    let mut filter = Filter::empty();
    filter.add(
        FilterEffect::Turbulence(TurbulenceFilter {
            base_frequency_x: 0.1,
            base_frequency_y: 0.1,
            num_octaves: 2,
            seed,
            turbulence_type: TurbulenceType::Turbulence,
            stitch_tiles: false,
        }),
        FilterInputs::NONE,
    );
    render_filter(filter)
}

#[test]
fn turbulence_seed_is_truncated_toward_zero() {
    assert_eq!(turbulence(-5.7), turbulence(-5.0));
    assert_eq!(turbulence(3.9), turbulence(3.0));
    assert_ne!(turbulence(-5.0), turbulence(0.0));
    assert_ne!(turbulence(-5.0), turbulence(5.0));
}

#[test]
fn composite_lighter_adds_its_inputs() {
    let mut filter = Filter::empty();
    let flood = |filter: &mut Filter, color| {
        filter.add_node(FilterGraphNode {
            effect: FilterEffect::Flood(color),
            inputs: FilterInputs::NONE,
            color_space: FilterColorSpace::Srgb,
            subregion: None,
        })
    };
    let a = flood(&mut filter, Color::from_rgb8(100, 200, 0));
    let b = flood(&mut filter, Color::from_rgb8(100, 100, 50));
    filter.add_node(FilterGraphNode {
        effect: FilterEffect::Composite(CompositeOperator::Lighter),
        inputs: FilterInputs::dual(FilterInput::Result(a), FilterInput::Result(b)),
        color_space: FilterColorSpace::Srgb,
        subregion: None,
    });
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 8, 8), [200, 255, 50, 255]);
}

/// Renders a red square through an empty filter whose layer is clipped to `clip` under
/// `transform`, returning the number of partially transparent pixels.
fn partial_alpha_pixels(transform: Affine, clip: Rect) -> usize {
    let filter = Arc::new(Filter::empty());
    let size = SIZE as f64;
    let buffer = render_to_buffer::<TinySkiaImageRenderer, _>(
        |scene| {
            scene.push_layer(
                Fill::NonZero,
                Mix::Normal,
                1.0,
                transform,
                &clip,
                Some(filter.clone()),
                None,
            );
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::from_rgb8(255, 0, 0),
                None,
                &Rect::new(-size, -size, 2.0 * size, 2.0 * size),
            );
            scene.pop_layer();
        },
        SIZE,
        SIZE,
    );
    buffer
        .chunks(4)
        .filter(|p| p[3] != 0 && p[3] != 255)
        .count()
}

#[test]
fn axis_aligned_filter_region_is_a_hard_clip() {
    let clip = Rect::new(2.5, 2.5, 12.25, 12.75);
    assert_eq!(partial_alpha_pixels(Affine::IDENTITY, clip), 0);
}

#[test]
fn rotated_filter_region_keeps_antialiased_edges() {
    let clip = Rect::new(-4.0, -4.0, 4.0, 4.0);
    let transform = Affine::translate((8.0, 8.0)) * Affine::rotate(0.5);
    assert!(partial_alpha_pixels(transform, clip) > 0);
}
