//! Glyphs drawn from the raster cache must look the same as glyphs drawn directly.

use std::sync::Arc;

use anyrender::{AntiAliasing, Glyph, ImageRenderer, PaintScene};
use anyrender_vello_cpu::VelloCpuImageRenderer;
use kurbo::Affine;
use peniko::{Blob, Color, Fill, FontData};

const W: u32 = 400;
const H: u32 = 160;

static ROBOTO: &[u8] = include_bytes!("../../../assets/fonts/roboto/Roboto.ttf");
static NOTO_COLR: &[u8] =
    include_bytes!("../../anyrender_glyph/tests/fonts/NotoColorEmoji-Subset.ttf");
static NOTO_CBTF: &[u8] =
    include_bytes!("../../anyrender_glyph/tests/fonts/NotoColorEmoji-CBTF-Subset.ttf");

fn font(data: &'static [u8]) -> FontData {
    FontData::new(Blob::new(Arc::new(data)), 0)
}

/// Draws a row of glyphs upright in the top half and with `lower_glyph_transform` in the bottom
/// half. Each glyph is `offset` further from a whole pixel than the one before.
fn render(
    glyph_caching: bool,
    offset: f32,
    font: &FontData,
    glyph_ids: &[u32],
    lower_glyph_transform: Option<Affine>,
) -> Vec<u8> {
    let mut renderer = VelloCpuImageRenderer::new(W, H);
    renderer.set_glyph_caching(glyph_caching);
    let mut buf = vec![0; (W * H * 4) as usize];
    // Render twice so that the second frame can be served from the cache.
    for _ in 0..2 {
        renderer.reset();
        renderer.render(
            |scene| {
                for (y, glyph_transform) in [(60.0, None), (140.0, lower_glyph_transform)] {
                    scene.draw_glyphs(
                        font,
                        40.0,
                        true,
                        &[],
                        Default::default(),
                        Fill::NonZero,
                        Color::WHITE,
                        1.0,
                        Affine::translate((20.0, y)),
                        glyph_transform,
                        glyph_ids.iter().enumerate().map(|(i, id)| Glyph {
                            id: *id,
                            x: i as f32 * (45.0 + offset),
                            y: 0.0,
                        }),
                        AntiAliasing::Enabled,
                    );
                }
            },
            &mut buf,
        );
    }
    buf
}

fn pixels(buf: &[u8]) -> &[[u8; 4]] {
    buf.as_chunks().0
}

/// Counts the pixels of `a` that have no similar pixel in `b` within `slack` pixels.
fn differing(a: &[u8], b: &[u8], slack: usize) -> usize {
    let (a, b) = (pixels(a), pixels(b));
    let (w, h) = (W as usize, a.len() / W as usize);
    let similar = |a: &[u8; 4], b: &[u8; 4]| a.iter().zip(b).all(|(a, b)| a.abs_diff(*b) <= 64);
    let mut count = 0;
    for y in 0..h {
        for x in 0..w {
            let px = &a[y * w + x];
            let ys = y.saturating_sub(slack)..=(y + slack).min(h - 1);
            let found = ys.into_iter().any(|by| {
                let xs = x.saturating_sub(slack)..=(x + slack).min(w - 1);
                xs.into_iter().any(|bx| similar(px, &b[by * w + bx]))
            });
            count += usize::from(!found);
        }
    }
    count
}

fn ink(buf: &[u8]) -> usize {
    pixels(buf).iter().filter(|px| px[3] > 128).count()
}

/// Whether any pixel has a colour other than a shade of grey.
fn has_colour(buf: &[u8]) -> bool {
    pixels(buf)
        .iter()
        .any(|px| px[0].abs_diff(px[1]) > 32 || px[1].abs_diff(px[2]) > 32)
}

fn assert_cached_matches_uncached(
    offset: f32,
    slack: usize,
    font: &FontData,
    glyph_ids: &[u32],
    lower_glyph_transform: Option<Affine>,
) -> Vec<u8> {
    let off = render(false, offset, font, glyph_ids, lower_glyph_transform);
    let on = render(true, offset, font, glyph_ids, lower_glyph_transform);
    let (upper_off, lower_off) = off.split_at(off.len() / 2);
    let (upper_on, lower_on) = on.split_at(on.len() / 2);

    assert!(ink(upper_off) > 500 && ink(lower_off) > 500);
    // With slack, edges that are anti-aliased in one image and sharp in the other still differ:
    // allow 5% of the inked pixels.
    let allowed = if slack == 0 { 0 } else { ink(upper_off) / 20 };
    assert!(differing(upper_off, upper_on, slack) <= allowed);
    assert!(differing(upper_on, upper_off, slack) <= allowed);
    // The lower row may be skewed, which is never served from the cache.
    assert_eq!(differing(lower_off, lower_on, 0), 0);
    on
}

fn glyph_ids(font: &FontData, text: &str) -> Vec<u32> {
    let font = skrifa::FontRef::from_index(font.data.as_ref(), font.index).unwrap();
    let charmap = skrifa::MetadataProvider::charmap(&font);
    text.chars()
        .map(|c| charmap.map(c).unwrap().to_u32())
        .collect()
}

const EMOJI: &str = "\u{2705}\u{1f440}\u{1f389}\u{1f920}";

#[test]
fn outline_glyphs_match() {
    let font = font(ROBOTO);
    let ids = glyph_ids(&font, "Glyphs&g");
    let image = assert_cached_matches_uncached(0.3, 0, &font, &ids, None);
    assert!(!has_colour(&image));
}

/// Skewed glyphs can't be served from the cache, and must not be drawn from the upright entries
/// that the top row put there.
#[test]
fn skewed_glyphs_are_not_drawn_from_upright_cache_entries() {
    let font = font(ROBOTO);
    let ids = glyph_ids(&font, "Glyphs&g");
    assert_cached_matches_uncached(0.3, 0, &font, &ids, Some(Affine::skew(-0.5, 0.0)));
}

// A colour glyph's cached image is not aligned to the pixel grid and is drawn with nearest
// neighbour sampling, so it can be up to a pixel away from where it is drawn uncached.

#[test]
fn colr_glyphs_match() {
    let font = font(NOTO_COLR);
    let ids = glyph_ids(&font, EMOJI);
    let image = assert_cached_matches_uncached(0.3, 1, &font, &ids, Some(Affine::skew(-0.5, 0.0)));
    assert!(has_colour(&image));
}

#[test]
fn bitmap_glyphs_match() {
    let font = font(NOTO_CBTF);
    let ids = glyph_ids(&font, EMOJI);
    let image = assert_cached_matches_uncached(0.3, 1, &font, &ids, Some(Affine::skew(-0.5, 0.0)));
    assert!(has_colour(&image));
}
