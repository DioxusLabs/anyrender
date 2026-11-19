use crate::BYTES_PER_PIXEL;

/// Exact rounded `(c * a) / 255` for `u8` channels, without division.
///
/// Matches `((c * a + 127) / 255)` and nearest-integer rounding of `(c * a) / 255.0` for all
/// `256 * 256` pairs.
#[inline]
pub(crate) fn mul_div_255(c: u8, a: u8) -> u8 {
    let prod = u16::from(c) * u16::from(a) + 128;
    ((prod + (prod >> 8)) >> 8) as u8
}

/// Fixed-point Q16 reciprocal table for unpremultiplying 8-bit channels:
/// `UNPREMUL_SCALE[a] = ceil(255 * 2^16 / a)` for `a > 0`, and `0` for `a == 0`.
///
/// With `c` clamped to `a`, `((c * UNPREMUL_SCALE[a] + 0x8000) >> 16)` matches exact
/// round-half-up `min(255, (2 * c * 255 + a) / (2 * a))` for every `c in 0..=255`, `a in 1..=255`,
/// yields `0` when `a == 0`, and never overflows `u32`.
const UNPREMUL_SCALE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut a = 1u32;
    while a < 256 {
        table[a as usize] = (255u32 << 16).div_ceil(a);
        a += 1;
    }
    table
};

/// Premultiplies straight-alpha RGBA8 pixel bytes in place.
///
/// Each `[r, g, b, a]` pixel has its colour channels scaled by `a / 255` with nearest-integer
/// rounding; pixels with `a == 0` become `[0, 0, 0, 0]`.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of [`BYTES_PER_PIXEL`] (4).
pub fn premultiply_rgba8(data: &mut [u8]) {
    let (pixels, remainder) = data.as_chunks_mut::<BYTES_PER_PIXEL>();
    assert!(
        remainder.is_empty(),
        "RGBA8 slice length ({}) must be a multiple of {BYTES_PER_PIXEL}",
        data.len()
    );
    for px in pixels {
        let a = px[3];
        px[0] = mul_div_255(px[0], a);
        px[1] = mul_div_255(px[1], a);
        px[2] = mul_div_255(px[2], a);
    }
}

/// Unpremultiplies premultiplied RGBA8 pixel bytes to straight alpha in place.
///
/// Each `[r, g, b, a]` pixel has its colour channels divided by `a / 255` with round-half-up
/// rounding, clamping any channel exceeding `a` to `255`. Pixels with `a == 0` become
/// `[0, 0, 0, 0]`.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of [`BYTES_PER_PIXEL`] (4).
pub fn unpremultiply_rgba8(data: &mut [u8]) {
    let (pixels, remainder) = data.as_chunks_mut::<BYTES_PER_PIXEL>();
    assert!(
        remainder.is_empty(),
        "RGBA8 slice length ({}) must be a multiple of {BYTES_PER_PIXEL}",
        data.len()
    );
    for px in pixels {
        let a = px[3];
        let scale = UNPREMUL_SCALE[a as usize];
        px[0] = ((u32::from(px[0].min(a)) * scale + 0x8000) >> 16) as u8;
        px[1] = ((u32::from(px[1].min(a)) * scale + 0x8000) >> 16) as u8;
        px[2] = ((u32::from(px[2].min(a)) * scale + 0x8000) >> 16) as u8;
    }
}

/// Multiplies every channel of premultiplied RGBA8 pixel bytes by `alpha / 255` in place, using
/// nearest-integer rounding.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of [`BYTES_PER_PIXEL`] (4).
pub fn multiply_alpha_rgba8(data: &mut [u8], alpha: u8) {
    assert!(
        data.as_chunks::<BYTES_PER_PIXEL>().1.is_empty(),
        "RGBA8 slice length ({}) must be a multiple of {BYTES_PER_PIXEL}",
        data.len()
    );
    if alpha == 255 {
        return;
    }
    if alpha == 0 {
        data.fill(0);
        return;
    }
    for byte in data {
        *byte = mul_div_255(*byte, alpha);
    }
}

/// Swaps the red and blue channels (bytes `0` and `2`) of every 4-byte pixel in place,
/// converting between RGBA8 and BGRA8.
///
/// # Panics
///
/// Panics if `data.len()` is not a multiple of [`BYTES_PER_PIXEL`] (4).
pub fn swap_rb_rgba8(data: &mut [u8]) {
    let (pixels, remainder) = data.as_chunks_mut::<BYTES_PER_PIXEL>();
    assert!(
        remainder.is_empty(),
        "RGBA8 slice length ({}) must be a multiple of {BYTES_PER_PIXEL}",
        data.len()
    );
    for px in pixels {
        px.swap(0, 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Pixmap, PixmapMut};
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn mul_div_255_is_exact_for_all_byte_pairs() {
        for a in 0u8..=255 {
            for c in 0u8..=255 {
                let expected = ((u32::from(c) * u32::from(a) + 127) / 255) as u8;
                assert_eq!(mul_div_255(c, a), expected, "mismatch for c={c}, a={a}");
            }
        }
    }

    #[test]
    fn unpremultiply_is_exact_and_round_trips_all_valid_premultiplied_pairs() {
        for a in 0u8..=255 {
            for c in 0u8..=255 {
                let mut px = [c, c, c, a];
                unpremultiply_rgba8(&mut px);
                let expected = if a == 0 {
                    0
                } else {
                    (((2 * u32::from(c) * 255 + u32::from(a)) / (2 * u32::from(a))).min(255)) as u8
                };
                assert_eq!(
                    px,
                    [expected, expected, expected, a],
                    "unpremultiply mismatch for c={c}, a={a}"
                );
                if c <= a {
                    premultiply_rgba8(&mut px);
                    assert_eq!(
                        px,
                        [c, c, c, a],
                        "round-trip mismatch for premul c={c}, a={a}"
                    );
                }
            }
        }
    }

    #[test]
    fn slice_alpha_and_channel_helpers() {
        let mut empty = [];
        premultiply_rgba8(&mut empty);
        unpremultiply_rgba8(&mut empty);
        multiply_alpha_rgba8(&mut empty, 128);
        swap_rb_rgba8(&mut empty);

        let mut buf = [
            255, 128, 64, 0, // transparent: RGB zeroed
            255, 128, 64, 255, // opaque: unchanged
            200, 100, 50, 128, // semi-transparent
        ];
        premultiply_rgba8(&mut buf);
        assert_eq!(
            buf,
            [
                0, 0, 0, 0, //
                255, 128, 64, 255, //
                100, 50, 25, 128,
            ]
        );

        let mut scaled = buf;
        multiply_alpha_rgba8(&mut scaled, 255);
        assert_eq!(scaled, buf);

        multiply_alpha_rgba8(&mut scaled, 128);
        assert_eq!(
            scaled,
            [
                0, 0, 0, 0, //
                128, 64, 32, 128, //
                50, 25, 13, 64,
            ]
        );

        multiply_alpha_rgba8(&mut scaled, 0);
        assert_eq!(scaled, [0; 12]);

        let mut swapped = buf;
        swap_rb_rgba8(&mut swapped);
        assert_eq!(
            swapped,
            [
                0, 0, 0, 0, //
                64, 128, 255, 255, //
                25, 50, 100, 128,
            ]
        );
        swap_rb_rgba8(&mut swapped);
        assert_eq!(swapped, buf);

        // Out-of-range premultiplied channels clamp to 255; a == 0 zeros RGB.
        let mut invalid = [10, 20, 30, 0, 250, 50, 100, 100];
        unpremultiply_rgba8(&mut invalid);
        assert_eq!(invalid, [0, 0, 0, 0, 255, 128, 255, 100]);
    }

    #[test]
    #[should_panic(expected = "multiple of 4")]
    fn premultiply_rgba8_panics_on_bad_len() {
        premultiply_rgba8(&mut [0u8; 3]);
    }

    #[test]
    #[should_panic(expected = "multiple of 4")]
    fn unpremultiply_rgba8_panics_on_bad_len() {
        unpremultiply_rgba8(&mut [0u8; 5]);
    }

    #[test]
    #[should_panic(expected = "multiple of 4")]
    fn multiply_alpha_rgba8_panics_on_bad_len_even_at_255() {
        multiply_alpha_rgba8(&mut [0u8; 3], 255);
    }

    #[test]
    #[should_panic(expected = "multiple of 4")]
    fn swap_rb_rgba8_panics_on_bad_len() {
        swap_rb_rgba8(&mut [0u8; 6]);
    }

    #[test]
    fn pixmap_alpha_and_channel_ops() {
        let straight = vec![
            200, 100, 50, 128, //
            255, 0, 10, 255, //
            40, 80, 120, 0, //
            10, 20, 30, 255,
        ];
        let mut pixmap = Pixmap::from_unpremultiplied(straight, 2, 2);
        assert_eq!(pixmap.pixel(0, 0), [100, 50, 25, 128]);
        assert_eq!(pixmap.pixel(1, 0), [255, 0, 10, 255]);
        assert_eq!(pixmap.pixel(0, 1), [0, 0, 0, 0]);
        assert_eq!(pixmap.pixel(1, 1), [10, 20, 30, 255]);

        let copied_straight = pixmap.to_unpremultiplied();
        assert_eq!(
            copied_straight,
            vec![
                199, 100, 50, 128, //
                255, 0, 10, 255, //
                0, 0, 0, 0, //
                10, 20, 30, 255,
            ]
        );
        assert_eq!(pixmap.clone().into_unpremultiplied(), copied_straight);

        pixmap.swap_rb();
        assert_eq!(pixmap.pixel(0, 0), [25, 50, 100, 128]);
        assert_eq!(pixmap.pixel(1, 0), [10, 0, 255, 255]);
        pixmap.swap_rb();

        pixmap.multiply_alpha(255);
        assert_eq!(pixmap.pixel(0, 0), [100, 50, 25, 128]);
        pixmap.multiply_alpha(128);
        assert_eq!(pixmap.pixel(0, 0), [50, 25, 13, 64]);
        assert_eq!(pixmap.pixel(1, 0), [128, 0, 5, 128]);

        pixmap.unpremultiply();
        assert_eq!(pixmap.pixel(1, 0), [255, 0, 10, 128]);
        pixmap.premultiply();
        assert_eq!(pixmap.pixel(1, 0), [128, 0, 5, 128]);

        pixmap.multiply_alpha(0);
        assert!(pixmap.data().iter().all(|&b| b == 0));

        // Empty pixmap operations are no-ops.
        let mut empty = Pixmap::from_unpremultiplied(Vec::new(), 0, 3);
        empty.premultiply();
        empty.unpremultiply();
        empty.multiply_alpha(128);
        empty.swap_rb();
        assert!(empty.to_unpremultiplied().is_empty());
        assert!(empty.into_unpremultiplied().is_empty());
    }

    #[test]
    fn pixmap_mut_alpha_and_channel_ops_preserve_stride_padding() {
        // 2x2 pixmap inside a 12-byte-strided buffer (4 bytes padding per row).
        let pad = [0xAA, 0xBB, 0xCC, 0xDD];
        let mut buf = vec![
            200, 100, 50, 128, 10, 20, 30, 255, pad[0], pad[1], pad[2], pad[3], // row 0
            40, 80, 120, 0, 255, 128, 0, 255, pad[0], pad[1], pad[2], pad[3], // row 1
        ];
        {
            let mut view = PixmapMut::from_parts_with_stride(&mut buf, 2, 2, 12);
            assert!(!view.is_contiguous());

            view.premultiply();
            assert_eq!(view.pixel(0, 0), [100, 50, 25, 128]);
            assert_eq!(view.pixel(1, 0), [10, 20, 30, 255]);
            assert_eq!(view.pixel(0, 1), [0, 0, 0, 0]);
            assert_eq!(view.pixel(1, 1), [255, 128, 0, 255]);

            assert_eq!(
                view.to_unpremultiplied(),
                vec![
                    199, 100, 50, 128, //
                    10, 20, 30, 255, //
                    0, 0, 0, 0, //
                    255, 128, 0, 255,
                ]
            );

            view.swap_rb();
            assert_eq!(view.pixel(0, 0), [25, 50, 100, 128]);
            assert_eq!(view.pixel(1, 1), [0, 128, 255, 255]);
            view.swap_rb();

            view.multiply_alpha(255);
            assert_eq!(view.pixel(0, 0), [100, 50, 25, 128]);

            view.multiply_alpha(128);
            assert_eq!(view.pixel(0, 0), [50, 25, 13, 64]);
            assert_eq!(view.pixel(1, 1), [128, 64, 0, 128]);

            view.unpremultiply();
            assert_eq!(view.pixel(1, 1), [255, 128, 0, 128]);

            view.multiply_alpha(0);
            assert_eq!(view.pixel(0, 0), [0, 0, 0, 0]);
            assert_eq!(view.pixel(1, 1), [0, 0, 0, 0]);
        }

        // Inter-row and trailing padding untouched throughout.
        assert_eq!(&buf[8..12], &pad);
        assert_eq!(&buf[20..24], &pad);
    }

    #[test]
    fn subregion_alpha_ops_leave_surrounding_pixels_untouched() {
        let mut pixmap = Pixmap::new(4, 3);
        pixmap.fill([200, 100, 50, 255]);
        let before = pixmap.clone();

        {
            let mut sub = pixmap.subregion_mut(1, 1, 2, 1);
            sub.swap_rb();
            sub.multiply_alpha(128);
        }

        for y in 0..3 {
            for x in 0..4 {
                if y == 1 && (1..3).contains(&x) {
                    assert_eq!(pixmap.pixel(x, y), [25, 50, 100, 128]);
                } else {
                    assert_eq!(pixmap.pixel(x, y), before.pixel(x, y));
                }
            }
        }
    }
}
