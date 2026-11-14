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

    // Glyph antialiasing can differ between solid and shader paints (as Skia's does on macOS),
    // so compare colour per pixel and coverage overall.
    assert!(solid.chunks(4).any(|p| p == RED), "glyphs drew nothing");
    for (i, p) in patterned.chunks(4).enumerate() {
        assert!(
            p[1] <= 2 && p[2] <= 2 && p[0].abs_diff(p[3]) <= 2,
            "pixel {i} isn't red: {p:?}"
        );
    }
    let coverage = |b: &[u8]| b.chunks(4).map(|p| u64::from(p[3])).sum::<u64>() as f64;
    let ratio = coverage(&patterned) / coverage(&solid);
    assert!((0.95..1.05).contains(&ratio), "coverage ratio {ratio}");

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

/// A 20x16 tile at (3, 4): red in its left half, blue in its right half.
fn offset_tile() -> anyrender::Pattern {
    use anyrender::{PaintScene, recording::Scene};
    use kurbo::Rect;
    use peniko::{Color, Fill};

    let mut scene = Scene::new();
    let red = Color::from_rgb8(255, 0, 0);
    let blue = Color::from_rgb8(0, 0, 255);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        red,
        None,
        &Rect::new(3.0, 4.0, 13.0, 20.0),
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        blue,
        None,
        &Rect::new(13.0, 4.0, 23.0, 20.0),
    );
    anyrender::Pattern::new(scene, Rect::new(3.0, 4.0, 23.0, 20.0))
}

fn check_tile_origin<R: ImageRenderer>(fallback: bool) {
    use anyrender::PaintScene;
    use peniko::Fill;

    let pattern = offset_tile();
    let shape = kurbo::Rect::new(0.0, 0.0, SIZE as f64, SIZE as f64);
    let buffer = render_to_buffer::<R, _>(
        |scene| {
            if fallback {
                anyrender::fill_pattern_tiles(
                    scene,
                    &pattern,
                    Fill::NonZero,
                    Affine::IDENTITY,
                    None,
                    &shape,
                );
            } else {
                scene.fill(Fill::NonZero, Affine::IDENTITY, &pattern, None, &shape);
            }
        },
        SIZE,
        SIZE,
    );
    for (x, expected) in [(1, BLUE), (5, RED), (15, BLUE), (25, RED), (35, BLUE)] {
        assert_close(pixel(&buffer, x, 10), expected);
        assert_close(pixel(&buffer, x, 26), expected);
    }
}

#[test]
fn pattern_tile_origin_tiny_skia() {
    check_tile_origin::<TinySkiaImageRenderer>(false);
}

#[test]
fn pattern_tile_origin_vello_cpu() {
    check_tile_origin::<VelloCpuImageRenderer>(false);
}

#[test]
fn pattern_tile_origin_skia() {
    check_tile_origin::<SkiaImageRenderer>(false);
}

#[test]
fn pattern_tile_origin_fallback() {
    check_tile_origin::<VelloCpuImageRenderer>(true);
}

/// A pixel-aligned 1px line stays sharp.
#[test]
fn pattern_pixel_aligned_is_sharp_vello_cpu() {
    use anyrender::{PaintScene, recording::Scene};
    use kurbo::Rect;
    use peniko::{Color, Fill};

    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(255, 0, 0),
        None,
        &Rect::new(0.0, 0.0, 1.0, 10.0),
    );
    let pattern = anyrender::Pattern::new(scene, Rect::new(0.0, 0.0, 10.0, 10.0));
    let buffer = render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| {
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                &pattern,
                None,
                &Rect::new(0.0, 0.0, 100.0, 100.0),
            )
        },
        SIZE,
        SIZE,
    );
    assert_eq!(pixel(&buffer, 20, 5), RED);
    assert_eq!(pixel(&buffer, 21, 5), CLEAR);
}

/// A solid tile that overflows its tile rect, so any gap between tiles shows.
fn solid_tile(size: f64) -> anyrender::Pattern {
    use anyrender::{PaintScene, recording::Scene};
    use kurbo::Rect;
    use peniko::{Color, Fill};

    let mut scene = Scene::new();
    let content = Rect::new(-5.0, -5.0, size + 5.0, size + 5.0);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(255, 0, 0),
        None,
        &content,
    );
    anyrender::Pattern::new(scene, Rect::new(0.0, 0.0, size, size))
}

/// The fallback leaves no seams between tiles at fractional positions, and draws more than
/// 10,000 small tiles.
#[test]
fn pattern_fallback_seams_and_many_tiles() {
    for (size, transform) in [
        (7.5, Affine::IDENTITY),
        (10.0, Affine::scale(1.03)),
        (1.5, Affine::IDENTITY),
    ] {
        let pattern = solid_tile(size);
        let buffer = render_to_buffer::<VelloCpuImageRenderer, _>(
            |scene| {
                let shape = kurbo::Rect::new(0.0, 0.0, 200.0, 200.0);
                anyrender::fill_pattern_tiles(
                    scene,
                    &pattern,
                    peniko::Fill::NonZero,
                    transform,
                    None,
                    &shape,
                );
            },
            SIZE,
            SIZE,
        );
        for (i, p) in buffer.chunks(4).enumerate() {
            assert_eq!(p, RED, "tile {size} {transform:?}, pixel {i}");
        }
    }
}

/// With a glyph transform, the pattern stays in the glyph run's space: 8px red and blue stripes.
fn check_glyph_transform<R: ImageRenderer>(
    draw: impl Fn(&mut R::ScenePainter<'_>, &anyrender::Pattern, &peniko::FontData, Affine),
) {
    use anyrender::{PaintScene, recording::Scene};
    use kurbo::Rect;
    use peniko::{Blob, Color, Fill, FontData};
    use std::sync::Arc;

    static ROBOTO: &[u8] = include_bytes!("../../../assets/fonts/roboto/Roboto.ttf");
    let font = FontData::new(Blob::new(Arc::new(ROBOTO)), 0);
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(255, 0, 0),
        None,
        &Rect::new(0.0, 0.0, 8.0, 16.0),
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(0, 0, 255),
        None,
        &Rect::new(8.0, 0.0, 16.0, 16.0),
    );
    let pattern = anyrender::Pattern::new(scene, Rect::new(0.0, 0.0, 16.0, 16.0));
    let buffer = render_to_buffer::<R, _>(
        |scene| draw(scene, &pattern, &font, Affine::skew(-0.3, 0.0)),
        SIZE,
        SIZE,
    );
    let mut opaque = 0;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let p = pixel(&buffer, x, y);
            let stripe_x = (x as f64 + 0.5 - 10.0).rem_euclid(16.0);
            if p[3] < 255 || (stripe_x - 8.0).abs() < 1.5 || !(1.5..=14.5).contains(&stripe_x) {
                continue;
            }
            opaque += 1;
            assert_eq!(
                p,
                if stripe_x < 8.0 { RED } else { BLUE },
                "pixel ({x}, {y})"
            );
        }
    }
    assert!(opaque > 100, "glyphs drew nothing");
}

fn draw_skewed_glyphs<S: anyrender::PaintScene>(
    scene: &mut S,
    pattern: &anyrender::Pattern,
    font: &peniko::FontData,
    glyph_transform: Affine,
) {
    scene.draw_glyphs(
        font,
        60.0,
        false,
        &[],
        Default::default(),
        peniko::Fill::NonZero,
        anyrender::PaintRef::Pattern(pattern),
        1.0,
        GLYPHS_TRANSFORM,
        Some(glyph_transform),
        (0..2).map(|i| anyrender::Glyph {
            id: 40 + i,
            x: i as f32 * 40.0,
            y: 0.0,
        }),
    );
}

// The closure keeps the painter's lifetime generic.
#[allow(clippy::redundant_closure)]
#[test]
fn pattern_glyph_transform_tiny_skia() {
    check_glyph_transform::<TinySkiaImageRenderer>(|s, p, f, t| draw_skewed_glyphs(s, p, f, t));
}

// The closure keeps the painter's lifetime generic.
#[allow(clippy::redundant_closure)]
#[test]
fn pattern_glyph_transform_vello_cpu() {
    check_glyph_transform::<VelloCpuImageRenderer>(|s, p, f, t| draw_skewed_glyphs(s, p, f, t));
}

// The closure keeps the painter's lifetime generic.
#[allow(clippy::redundant_closure)]
#[test]
fn pattern_glyph_transform_skia() {
    check_glyph_transform::<SkiaImageRenderer>(|s, p, f, t| draw_skewed_glyphs(s, p, f, t));
}

#[test]
fn pattern_glyph_transform_fallback() {
    check_glyph_transform::<VelloCpuImageRenderer>(|scene, pattern, font, glyph_transform| {
        anyrender::draw_glyphs_pattern_tiles(
            scene,
            pattern,
            font,
            60.0,
            false,
            &[],
            Default::default(),
            peniko::Fill::NonZero,
            1.0,
            GLYPHS_TRANSFORM,
            Some(glyph_transform),
            (0..2).map(|i| anyrender::Glyph {
                id: 40 + i,
                x: i as f32 * 40.0,
                y: 0.0,
            }),
        )
    });
}

/// Shapes sharing an SVG pattern, in one render and across renders, share its pattern id.
#[test]
fn svg_pattern_is_shared() {
    use anyrender::recording::{RenderCommand, Scene};

    let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        <pattern id="p" patternUnits="userSpaceOnUse" width="20" height="20">
            <rect width="10" height="10" fill="red"/>
        </pattern>
        <rect width="40" height="40" fill="url(#p)"/>
        <rect x="50" width="40" height="40" fill="url(#p)"/>
        <rect y="50" width="40" height="40" fill="none" stroke="url(#p)"/>
    </svg>"#;
    let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).unwrap();
    let mut ids = Vec::new();
    for _ in 0..2 {
        let mut scene = Scene::new();
        anyrender_svg::render_svg_tree(&mut scene, &tree, Affine::IDENTITY);
        ids.extend(scene.commands.iter().filter_map(|command| match command {
            RenderCommand::Fill(fill) => match &fill.brush {
                anyrender::Paint::Pattern(pattern) => Some(pattern.id()),
                _ => None,
            },
            RenderCommand::Stroke(stroke) => match &stroke.brush {
                anyrender::Paint::Pattern(pattern) => Some(pattern.id()),
                _ => None,
            },
            _ => None,
        }));
    }
    assert_eq!(ids.len(), 6);
    assert!(ids.iter().all(|&id| id == ids[0]), "{ids:?}");
}
