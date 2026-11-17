use std::sync::Arc;

use anyrender::filters::{
    EdgeMode, Filter, FilterColorSpace, FilterEffect, FilterId, FilterImage, FilterInput,
    FilterInputs,
    blur::GaussianBlurFilter,
    color_transformation::ColorMatrix,
    composite::CompositeOperator,
    turbulence::{TurbulenceFilter, TurbulenceType},
};
use anyrender::{AntiAliasing, PaintScene, render_to_buffer};
use anyrender_tiny_skia::TinySkiaImageRenderer;
use kurbo::{Affine, Rect, Vec2};
use peniko::{Color, Fill, Mix};

const SIZE: u32 = 16;
const FULL: Rect = Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64);

/// Halves the red channel of an opaque red square in `color_space`.
fn halve_red(color_space: FilterColorSpace) -> [u8; 4] {
    let mut filter = Filter::new(FULL);
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
                AntiAliasing::Enabled,
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
                AntiAliasing::Enabled,
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

fn green_flood(filter: &mut Filter, subregion: Rect) -> FilterId {
    filter.add_with_subregion(
        FilterEffect::Flood(Color::from_rgb8(0, 255, 0)),
        FilterInputs::NONE,
        FilterColorSpace::Srgb,
        Some(subregion),
    )
}

#[test]
fn results_are_clipped_to_subregions() {
    let mut filter = Filter::new(FULL);
    green_flood(&mut filter, Rect::new(0.0, 0.0, 4.0, 4.0));
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 1, 1), [0, 255, 0, 255]);
    assert_close(pixel(&buffer, 9, 9), [0, 0, 0, 0]);
}

#[test]
fn results_are_clipped_to_the_filter_region() {
    let mut filter = Filter::new(Rect::new(0.0, 0.0, 4.0, 4.0));
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
    let mut filter = Filter::new(half);
    filter.add(
        FilterEffect::Offset(Vec2::new(8.0, 0.0)),
        FilterInputs::NONE,
    );
    let buffer = render_filter(filter);
    // The source right of the region is not shifted into it.
    assert_close(pixel(&buffer, 4, 8), [0, 0, 0, 0]);

    let mut filter = Filter::new(half.with_origin((8.0, 0.0)));
    filter.add(
        FilterEffect::Offset(Vec2::new(4.0, 0.0)),
        FilterInputs::NONE,
    );
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 10, 8), [0, 0, 0, 0]);
    assert_close(pixel(&buffer, 13, 8), [255, 0, 0, 255]);
}

#[test]
fn primitives_are_clipped_to_their_own_filter_region() {
    let mut filter = Filter::new(Rect::new(4.0, 0.0, 12.0, SIZE as f64));
    let shifted = filter.add(
        FilterEffect::Offset(Vec2::new(4.0, 0.0)),
        FilterInputs::NONE,
    );
    filter.set_region(FULL);
    filter.add(
        FilterEffect::Offset(Vec2::ZERO),
        FilterInputs::single(FilterInput::Result(shifted)),
    );
    let buffer = render_filter(filter);
    // The result is clipped to the region.
    assert_close(pixel(&buffer, 2, 8), [0, 0, 0, 0]);
    assert_close(pixel(&buffer, 13, 8), [0, 0, 0, 0]);
    // The source left of the region is not shifted into it.
    assert_close(pixel(&buffer, 6, 8), [0, 0, 0, 0]);
    assert_close(pixel(&buffer, 10, 8), [255, 0, 0, 255]);
}

#[test]
fn tile_repeats_its_input_subregion() {
    let mut filter = Filter::new(FULL);
    let flood = green_flood(&mut filter, Rect::new(0.0, 0.0, 4.0, 4.0));
    filter.add_with_subregion(
        FilterEffect::Tile,
        FilterInputs::single(FilterInput::Result(flood)),
        FilterColorSpace::Srgb,
        Some(FULL),
    );
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
        AntiAliasing::Enabled,
    );
    let mut filter = Filter::new(FULL);
    filter.add_with_color_space(
        FilterEffect::Image(FilterImage { scene: content }),
        FilterInputs::NONE,
        FilterColorSpace::LinearRgb,
    );
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 2, 2), [0, 255, 0, 255]);
    assert_close(pixel(&buffer, 12, 12), [0, 0, 0, 0]);
}

fn turbulence(seed: f32) -> Vec<u8> {
    let mut filter = Filter::new(FULL);
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
    let mut filter = Filter::new(FULL);
    let flood = |filter: &mut Filter, color| {
        filter.add_with_color_space(
            FilterEffect::Flood(color),
            FilterInputs::NONE,
            FilterColorSpace::Srgb,
        )
    };
    let a = flood(&mut filter, Color::from_rgb8(100, 200, 0));
    let b = flood(&mut filter, Color::from_rgb8(100, 100, 50));
    filter.add_with_color_space(
        FilterEffect::Composite(CompositeOperator::Lighter),
        FilterInputs::dual(FilterInput::Result(a), FilterInput::Result(b)),
        FilterColorSpace::Srgb,
    );
    let buffer = render_filter(filter);
    assert_close(pixel(&buffer, 8, 8), [200, 255, 50, 255]);
}

/// Renders a red square through an empty filter whose layer is clipped to `clip` under
/// `transform`, returning the number of partially transparent pixels.
fn partial_alpha_pixels(transform: Affine, clip: Rect) -> usize {
    let filter = Arc::new(Filter::new(FULL));
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
                AntiAliasing::Enabled,
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

/// Offsets a red square drawn left of the canvas into view, through a filter layer blended with
/// `blend` over a white background.
fn offset_from_outside_canvas(blend: Mix) -> Vec<u8> {
    let mut filter = Filter::new(Rect::new(-8.0, 0.0, 16.0, 16.0));
    filter.add(
        FilterEffect::Offset(Vec2::new(8.0, 0.0)),
        FilterInputs::NONE,
    );
    let filter = Arc::new(filter);
    render_to_buffer::<TinySkiaImageRenderer, _>(
        |scene| {
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::WHITE,
                None,
                &FULL,
                AntiAliasing::Enabled,
            );
            scene.push_layer(
                Fill::NonZero,
                blend,
                1.0,
                Affine::IDENTITY,
                &Rect::new(-8.0, 0.0, 16.0, 16.0),
                Some(filter.clone()),
                None,
            );
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::from_rgb8(255, 0, 0),
                None,
                &Rect::new(-8.0, 0.0, 0.0, 16.0),
                AntiAliasing::Enabled,
            );
            scene.pop_layer();
        },
        SIZE,
        SIZE,
    )
}

#[test]
fn filters_see_content_outside_the_canvas() {
    for blend in [Mix::Normal, Mix::Multiply] {
        let buffer = offset_from_outside_canvas(blend);
        assert_close(pixel(&buffer, 0, 8), [255, 0, 0, 255]);
        assert_close(pixel(&buffer, 7, 8), [255, 0, 0, 255]);
        assert_close(pixel(&buffer, 8, 8), [255, 255, 255, 255]);
    }
}

#[test]
fn filters_run_in_user_space_axes_under_rotation() {
    // A blur along user-space x, under a quarter turn, spreads along canvas y.
    let size = SIZE as f64;
    let mut filter = Filter::new(Rect::new(-size, -size, size, size));
    filter.add(
        FilterEffect::GaussianBlur(GaussianBlurFilter {
            std_deviation_x: 2.0,
            std_deviation_y: 0.0,
            edge_mode: EdgeMode::None,
        }),
        FilterInputs::NONE,
    );
    let filter = Arc::new(filter);
    let transform = Affine::translate((8.0, 8.0)) * Affine::rotate(std::f64::consts::FRAC_PI_2);
    let buffer = render_to_buffer::<TinySkiaImageRenderer, _>(
        |scene| {
            scene.push_layer(
                Fill::NonZero,
                Mix::Normal,
                1.0,
                transform,
                &Rect::new(-size, -size, size, size),
                Some(filter.clone()),
                None,
            );
            // A horizontal stripe on the canvas, along user-space y.
            scene.fill(
                Fill::NonZero,
                transform,
                Color::from_rgb8(255, 0, 0),
                None,
                &Rect::new(-1.0, -size, 1.0, size),
                AntiAliasing::Enabled,
            );
            scene.pop_layer();
        },
        SIZE,
        SIZE,
    );
    assert!(pixel(&buffer, 8, 4)[3] > 0);
    assert!(pixel(&buffer, 8, 8)[3] < 255);
}

/// Renders a red stripe under a skew and 1.5x scale, either moved by a filter offset of `dx`
/// along user-space x or, without a filter, by translating the stripe itself.
fn skewed_stripe(dx: f64, with_filter: bool) -> Vec<u8> {
    let size = SIZE as f64;
    // `scale(1.5) skewX(30)` as usvg hands it over. Dividing the scale back out of this in
    // floating point gives 1.4999999999999998 rather than 1.5.
    let transform = Affine::new([1.5, 0.0, 0.8660253882408142, 1.5, -6.0, 0.0]);
    let mut filter = Filter::new(Rect::new(-size, -size, 2.0 * size, 2.0 * size));
    filter.add(FilterEffect::Offset(Vec2::new(dx, 0.0)), FilterInputs::NONE);
    let filter = Arc::new(filter);
    let stripe = Rect::new(0.0, 0.0, 2.0, size);
    render_to_buffer::<TinySkiaImageRenderer, _>(
        |scene| {
            if with_filter {
                scene.push_layer(
                    Fill::NonZero,
                    Mix::Normal,
                    1.0,
                    transform,
                    &Rect::new(-size, -size, 2.0 * size, 2.0 * size),
                    Some(filter.clone()),
                    None,
                );
            }
            let shift = if with_filter { 0.0 } else { dx };
            scene.fill(
                Fill::NonZero,
                transform,
                Color::from_rgb8(255, 0, 0),
                None,
                &(stripe + Vec2::new(shift, 0.0)),
                AntiAliasing::Enabled,
            );
            if with_filter {
                scene.pop_layer();
            }
        },
        SIZE * 2,
        SIZE,
    )
}

#[test]
fn whole_pixel_offsets_stay_exact_under_skew() {
    // 4 user units at 1.5x is exactly 6 pixels, so the offset copy lands where the stripe would.
    let filtered = skewed_stripe(4.0, true);
    let moved = skewed_stripe(4.0, false);
    let max_diff = filtered
        .iter()
        .zip(&moved)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap();
    assert!(max_diff <= 64, "max channel difference {max_diff}");
}
