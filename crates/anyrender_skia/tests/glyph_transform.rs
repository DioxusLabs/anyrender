use std::sync::Arc;

use anyrender::{AntiAliasing, Glyph, PaintRef, PaintScene, render_to_buffer};
use anyrender_skia::SkiaImageRenderer;
use kurbo::Affine;
use peniko::{Blob, Color, Fill, FontData, Gradient};
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
    let font_transform = Affine::new([1.5, 0.0, -0.5, 2.0, 4.0, -6.0]);
    let canvas_transform = Affine::new([1.5, 0.5, -0.5, 2.0, 4.0, -6.0]);

    let render = |apply_per_glyph: bool, glyph_transform, brush: PaintRef<'_>| {
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
                        brush.clone(),
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

    for glyph_transform in [
        font_transform,
        canvas_transform,
        Affine::scale_non_uniform(-1.0, 1.0),
        Affine::scale_non_uniform(1.0, -1.0),
        Affine::new([0.0, 1.0, -1.0, 0.0, 0.0, 0.0]),
    ] {
        let expected = render(true, glyph_transform, Color::WHITE.into());
        assert!(
            expected
                .as_chunks::<4>()
                .0
                .iter()
                .any(|pixel| pixel[0] == 255)
        );
        assert!(
            render(false, glyph_transform, Color::WHITE.into()) == expected,
            "glyph placement or shape differed for {glyph_transform:?}"
        );
    }

    for glyph_transform in [
        Affine::scale_non_uniform(0.0, 1.0),
        Affine::skew(f64::NAN, 0.0),
    ] {
        let actual = render(false, glyph_transform, Color::WHITE.into());
        assert!(
            actual.as_chunks::<4>().0.iter().all(|pixel| pixel[0] == 0),
            "invalid transform {glyph_transform:?} drew ink"
        );
    }

    let gradient = Gradient::new_linear((55.0, 0.0), (65.0, 0.0))
        .with_stops([Color::from_rgb8(255, 0, 0), Color::from_rgb8(0, 0, 255)]);
    let shaded = render(false, canvas_transform, (&gradient).into());
    let mask = render(true, canvas_transform, Color::WHITE.into());
    let mut left = 0;
    let mut right = 0;
    for (i, (pixel, mask)) in shaded
        .as_chunks::<4>()
        .0
        .iter()
        .zip(mask.as_chunks::<4>().0)
        .enumerate()
    {
        if mask[0] == 0 {
            assert_eq!(pixel[..3], [0, 0, 0]);
        } else if i % 200 < 62 {
            assert_eq!(pixel[..3], [255, 0, 0]);
            left += 1;
        } else if i % 200 >= 72 {
            assert_eq!(pixel[..3], [0, 0, 255]);
            right += 1;
        }
    }
    assert!(left > 0 && right > 0, "brush did not paint both glyphs");
}
