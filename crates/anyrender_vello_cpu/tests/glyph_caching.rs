use std::sync::Arc;

use anyrender::{Glyph, ImageRenderer, PaintScene};
use anyrender_vello_cpu::VelloCpuImageRenderer;
use kurbo::Affine;
use peniko::{Blob, Color, Fill, FontData};

const W: u32 = 400;
const H: u32 = 160;

fn render(glyph_caching: bool, font: &FontData) -> Vec<u8> {
    let mut renderer = VelloCpuImageRenderer::new(W, H);
    renderer.set_glyph_caching(glyph_caching);
    let mut buf = vec![0; (W * H * 4) as usize];
    let skew = Affine::skew(-0.5, 0.0);
    // Render twice so that the second frame can be served from the atlas.
    for _ in 0..2 {
        renderer.reset();
        renderer.render(
            |scene| {
                // Upright in the top half, skewed in the bottom half.
                for (y, glyph_transform) in [(60.0, None), (140.0, Some(skew))] {
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
                        (0..8).map(|i| Glyph {
                            id: 40 + i,
                            x: i as f32 * 40.0,
                            y: 0.0,
                        }),
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

fn differing(a: &[u8], b: &[u8]) -> usize {
    pixels(a)
        .iter()
        .zip(pixels(b))
        .filter(|(a, b)| a.iter().zip(*b).any(|(a, b)| a.abs_diff(*b) > 64))
        .count()
}

/// Vello's glyph cache key doesn't include skew/rotation, so runs it can't cache must bypass it.
#[test]
fn skewed_glyphs_are_not_drawn_from_upright_cache_entries() {
    static ROBOTO: &[u8] = include_bytes!("../../../assets/fonts/roboto/Roboto.ttf");
    let font = FontData::new(Blob::new(Arc::new(ROBOTO)), 0);
    let off = render(false, &font);
    let on = render(true, &font);
    let (upright_off, skewed_off) = off.split_at(off.len() / 2);
    let (upright_on, skewed_on) = on.split_at(on.len() / 2);

    let ink = |buf: &[u8]| {
        buf.as_chunks::<4>()
            .0
            .iter()
            .filter(|px| px[3] > 128)
            .count()
    };
    assert!(ink(upright_off) > 500 && ink(skewed_off) > 500);

    assert_eq!(differing(upright_off, upright_on), 0);
    assert_eq!(differing(skewed_off, skewed_on), 0);
}
