use std::sync::Arc;

use anyrender::{AntiAliasing, Glyph, PaintScene, render_to_buffer};
use anyrender_skia::SkiaImageRenderer;
use kurbo::Affine;
use peniko::{Blob, Color, Fill, FontData};
use skrifa::{FontRef, MetadataProvider};

#[test]
fn glyph_transform_is_y_down_and_applied_before_placement() {
    static ROBOTO: &[u8] = include_bytes!("../../../assets/fonts/roboto/Roboto.ttf");
    let font = FontData::new(Blob::new(Arc::new(ROBOTO)), 0);
    let id = FontRef::new(ROBOTO)
        .unwrap()
        .charmap()
        .map('H')
        .unwrap()
        .to_u32();
    let glyphs = [
        Glyph {
            id,
            x: 20.0,
            y: 40.0,
        },
        Glyph {
            id,
            x: 70.0,
            y: 60.0,
        },
    ];
    let run_transform = Affine::translate((7.0, 11.0)) * Affine::scale_non_uniform(1.0, 1.5);
    let glyph_transform = Affine::new([2.0, 0.0, -0.5, 2.0, 4.0, -6.0]);

    let render = |apply_per_glyph: bool| {
        render_to_buffer::<SkiaImageRenderer, _>(
            |scene| {
                let mut draw = |transform, glyph_transform, glyphs: &[Glyph]| {
                    scene.draw_glyphs(
                        &font,
                        16.0,
                        false,
                        &[],
                        Default::default(),
                        Fill::NonZero,
                        Color::WHITE,
                        1.0,
                        transform,
                        glyph_transform,
                        glyphs.iter().copied(),
                        AntiAliasing::Disabled,
                    );
                };
                if apply_per_glyph {
                    for glyph in glyphs {
                        draw(
                            run_transform
                                * Affine::translate((f64::from(glyph.x), f64::from(glyph.y)))
                                * glyph_transform,
                            None,
                            &[Glyph { id, x: 0.0, y: 0.0 }],
                        );
                    }
                } else {
                    draw(run_transform, Some(glyph_transform), &glyphs);
                }
            },
            200,
            140,
        )
    };

    let expected = render(true);
    assert!(
        expected
            .as_chunks::<4>()
            .0
            .iter()
            .any(|pixel| pixel[0] == 255)
    );
    assert!(
        render(false) == expected,
        "glyph placement or shape differed"
    );
}
