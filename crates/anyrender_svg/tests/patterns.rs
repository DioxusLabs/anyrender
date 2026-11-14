//! SVG patterns repeat their tile across the painted shape.

use anyrender::{ImageRenderer, render_to_buffer};
use anyrender_skia::SkiaImageRenderer;
use anyrender_tiny_skia::TinySkiaImageRenderer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use kurbo::Affine;

const SIZE: u32 = 100;

fn render<R: ImageRenderer>(svg: &str) -> Vec<u8> {
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).unwrap();
    render_to_buffer::<R, _>(
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

const RED: [u8; 4] = [255, 0, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const CLEAR: [u8; 4] = [0, 0, 0, 0];

/// A 20x20 tile offset by (5, 5), with a red square in its top-left quarter and
/// a blue one in its bottom-right quarter.
fn checkerboard(pattern_attrs: &str, paint: &str) -> String {
    format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <pattern id="p" patternUnits="userSpaceOnUse" x="5" y="5" width="20" height="20" {pattern_attrs}>
                <rect width="10" height="10" fill="red"/>
                <rect x="10" y="10" width="10" height="10" fill="blue"/>
            </pattern>
            <rect x="10" y="10" width="80" height="80" {paint}/>
        </svg>"#
    )
}

fn check_fill<R: ImageRenderer>() {
    let buffer = render::<R>(&checkerboard("", r#"fill="url(#p)""#));
    // Outside the shape
    assert_close(pixel(&buffer, 5, 5), CLEAR);
    // Tile (0, 0) is clipped to the shape
    assert_close(pixel(&buffer, 12, 12), RED);
    assert_close(pixel(&buffer, 20, 20), BLUE);
    assert_close(pixel(&buffer, 20, 12), CLEAR);
    // Repeated tiles
    assert_close(pixel(&buffer, 47, 67), RED);
    assert_close(pixel(&buffer, 80, 60), BLUE);
    assert_close(pixel(&buffer, 88, 88), RED);
}

#[test]
fn pattern_fill_tiny_skia() {
    check_fill::<TinySkiaImageRenderer>();
}

#[test]
fn pattern_fill_vello_cpu() {
    check_fill::<VelloCpuImageRenderer>();
}

#[test]
fn pattern_fill_skia() {
    check_fill::<SkiaImageRenderer>();
}

fn check_transform_and_opacity<R: ImageRenderer>() {
    let buffer = render::<R>(&checkerboard(
        r#"patternTransform="translate(10 0)""#,
        r#"fill="url(#p)" fill-opacity="0.5""#,
    ));
    assert_close(pixel(&buffer, 17, 12), [128, 0, 0, 128]);
    assert_close(pixel(&buffer, 30, 20), [0, 0, 128, 128]);
    assert_close(pixel(&buffer, 12, 12), CLEAR);
}

#[test]
fn pattern_transform_and_opacity_tiny_skia() {
    check_transform_and_opacity::<TinySkiaImageRenderer>();
}

#[test]
fn pattern_transform_and_opacity_vello_cpu() {
    check_transform_and_opacity::<VelloCpuImageRenderer>();
}

#[test]
fn pattern_transform_and_opacity_skia() {
    check_transform_and_opacity::<SkiaImageRenderer>();
}

fn check_stroke<R: ImageRenderer>() {
    let buffer = render::<R>(&checkerboard(
        "",
        r#"fill="none" stroke="url(#p)" stroke-width="4""#,
    ));
    assert_close(pixel(&buffer, 10, 12), RED);
    assert_close(pixel(&buffer, 20, 10), CLEAR);
    assert_close(pixel(&buffer, 50, 50), CLEAR);
    assert_close(pixel(&buffer, 90, 72), RED);
    assert_close(pixel(&buffer, 90, 80), CLEAR);
}

#[test]
fn pattern_stroke_tiny_skia() {
    check_stroke::<TinySkiaImageRenderer>();
}

#[test]
fn pattern_stroke_vello_cpu() {
    check_stroke::<VelloCpuImageRenderer>();
}

#[test]
fn pattern_stroke_skia() {
    check_stroke::<SkiaImageRenderer>();
}

/// The generic tile-by-tile fallback, which vello and vello_hybrid use.
#[test]
fn pattern_fill_fallback_tiles() {
    use anyrender::{PaintScene, Pattern, recording::Scene};
    use kurbo::Rect;
    use peniko::{Color, Fill};

    let mut tile_scene = Scene::new();
    tile_scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(255, 0, 0),
        None,
        &Rect::new(0.0, 0.0, 10.0, 10.0),
    );
    let pattern = Pattern::new(tile_scene, Rect::new(0.0, 0.0, 20.0, 20.0));
    let buffer = render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| {
            anyrender::fill_pattern_tiles(
                scene,
                &pattern,
                Fill::NonZero,
                Affine::IDENTITY,
                None,
                &Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64),
            );
        },
        SIZE,
        SIZE,
    );
    assert_close(pixel(&buffer, 45, 65), RED);
    assert_close(pixel(&buffer, 55, 65), CLEAR);
}

/// A pattern whose tile is solid red paints glyphs the same as solid red does.
fn check_glyphs<R: ImageRenderer>(
    draw: impl Fn(&mut R::ScenePainter<'_>, &anyrender::Pattern, &peniko::FontData),
) {
    use anyrender::{Glyph, PaintScene, Pattern, recording::Scene};
    use kurbo::Rect;
    use peniko::{Blob, Color, Fill, FontData};
    use std::sync::Arc;

    static ROBOTO: &[u8] = include_bytes!("../../../assets/fonts/roboto/Roboto.ttf");
    let font = FontData::new(Blob::new(Arc::new(ROBOTO)), 0);
    let tile = Rect::new(0.0, 0.0, 10.0, 10.0);
    let mut tile_scene = Scene::new();
    tile_scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(255, 0, 0),
        None,
        &tile,
    );
    let pattern = Pattern::new(tile_scene, tile);

    let solid = render_to_buffer::<R, _>(
        |scene| {
            scene.draw_glyphs(
                &font,
                40.0,
                true,
                &[],
                Default::default(),
                Fill::NonZero,
                Color::from_rgb8(255, 0, 0),
                1.0,
                GLYPHS_TRANSFORM,
                None,
                glyphs(),
            )
        },
        SIZE,
        SIZE,
    );
    let patterned = render_to_buffer::<R, _>(|scene| draw(scene, &pattern, &font), SIZE, SIZE);

    assert!(solid.chunks(4).any(|p| p == RED), "glyphs drew nothing");
    for (i, (s, p)) in solid.chunks(4).zip(patterned.chunks(4)).enumerate() {
        assert!(
            s.iter().zip(p).all(|(s, p)| s.abs_diff(*p) <= 8),
            "pixel {i}: solid {s:?}, pattern {p:?}"
        );
    }

    fn glyphs() -> impl Iterator<Item = Glyph> + Clone {
        (0..3).map(|i| Glyph {
            id: 40 + i,
            x: i as f32 * 25.0,
            y: 0.0,
        })
    }
}

const GLYPHS_TRANSFORM: Affine = Affine::new([1.0, 0.0, 0.0, 1.0, 10.0, 60.0]);

fn draw_patterned_glyphs<S: anyrender::PaintScene>(
    scene: &mut S,
    pattern: &anyrender::Pattern,
    font: &peniko::FontData,
) {
    scene.draw_glyphs(
        font,
        40.0,
        true,
        &[],
        Default::default(),
        peniko::Fill::NonZero,
        anyrender::PaintRef::Pattern(pattern),
        1.0,
        GLYPHS_TRANSFORM,
        None,
        (0..3).map(|i| anyrender::Glyph {
            id: 40 + i,
            x: i as f32 * 25.0,
            y: 0.0,
        }),
    );
}

// The closure keeps the painter's lifetime generic.
#[allow(clippy::redundant_closure)]
#[test]
fn pattern_glyphs_tiny_skia() {
    check_glyphs::<TinySkiaImageRenderer>(|s, p, f| draw_patterned_glyphs(s, p, f));
}

// The closure keeps the painter's lifetime generic.
#[allow(clippy::redundant_closure)]
#[test]
fn pattern_glyphs_vello_cpu() {
    check_glyphs::<VelloCpuImageRenderer>(|s, p, f| draw_patterned_glyphs(s, p, f));
}

// The closure keeps the painter's lifetime generic.
#[allow(clippy::redundant_closure)]
#[test]
fn pattern_glyphs_skia() {
    check_glyphs::<SkiaImageRenderer>(|s, p, f| draw_patterned_glyphs(s, p, f));
}

/// The generic glyph fallback, which vello and vello_hybrid use.
#[test]
fn pattern_glyphs_fallback_tiles() {
    check_glyphs::<VelloCpuImageRenderer>(|scene, pattern, font| {
        anyrender::draw_glyphs_pattern_tiles(
            scene,
            pattern,
            font,
            40.0,
            true,
            &[],
            Default::default(),
            peniko::Fill::NonZero,
            1.0,
            GLYPHS_TRANSFORM,
            None,
            (0..3).map(|i| anyrender::Glyph {
                id: 40 + i,
                x: i as f32 * 25.0,
                y: 0.0,
            }),
        )
    });
}
