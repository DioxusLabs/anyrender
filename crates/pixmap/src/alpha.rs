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
