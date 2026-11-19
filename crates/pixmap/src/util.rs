use crate::BYTES_PER_PIXEL;

pub(crate) fn row_bytes(width: u16) -> usize {
    width as usize * BYTES_PER_PIXEL
}

pub(crate) fn checked_byte_len(width: u16, height: u16) -> Option<usize> {
    usize::from(width)
        .checked_mul(BYTES_PER_PIXEL)?
        .checked_mul(usize::from(height))
}

pub(crate) fn byte_len(width: u16, height: u16) -> usize {
    checked_byte_len(width, height).expect("pixmap dimensions overflow usize")
}

pub(crate) fn is_valid_packed_parts(data_len: usize, width: u16, height: u16) -> bool {
    checked_byte_len(width, height) == Some(data_len)
}

pub(crate) fn validate_packed_parts(data_len: usize, width: u16, height: u16) {
    assert_eq!(
        data_len,
        byte_len(width, height),
        "pixel data length must match the pixmap's dimensions"
    );
}

pub(crate) fn is_contiguous(width: u16, height: u16, stride: usize) -> bool {
    width == 0 || height <= 1 || stride == row_bytes(width)
}

pub(crate) fn fill_slice(data: &mut [u8], color: [u8; 4]) {
    if color[0] == color[1] && color[1] == color[2] && color[2] == color[3] {
        data.fill(color[0]);
    } else {
        let (pixels, remainder) = data.as_chunks_mut::<BYTES_PER_PIXEL>();
        debug_assert!(remainder.is_empty());
        pixels.fill(color);
    }
}

pub(crate) fn slice_is_opaque(data: &[u8]) -> bool {
    let (pixels, remainder) = data.as_chunks::<BYTES_PER_PIXEL>();
    debug_assert!(remainder.is_empty());
    pixels.iter().all(|px| px[3] == 255)
}

pub(crate) fn as_pixels(data: &[u8]) -> &[[u8; 4]] {
    let (pixels, remainder) = data.as_chunks::<BYTES_PER_PIXEL>();
    debug_assert!(remainder.is_empty());
    pixels
}

pub(crate) fn as_pixels_mut(data: &mut [u8]) -> &mut [[u8; 4]] {
    let (pixels, remainder) = data.as_chunks_mut::<BYTES_PER_PIXEL>();
    debug_assert!(remainder.is_empty());
    pixels
}

pub(crate) fn is_valid_strided_parts(
    data_len: usize,
    width: u16,
    height: u16,
    stride: usize,
) -> bool {
    let min_row_bytes = row_bytes(width);
    if stride < min_row_bytes {
        return false;
    }
    let Some(min_len) = (if height == 0 {
        Some(0)
    } else {
        (height as usize - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(min_row_bytes))
    }) else {
        return false;
    };
    data_len >= min_len
}

pub(crate) fn validate_strided_parts(data_len: usize, width: u16, height: u16, stride: usize) {
    let min_row_bytes = row_bytes(width);
    assert!(
        stride >= min_row_bytes,
        "stride ({stride}) must be at least width * 4 ({min_row_bytes})"
    );
    let min_len = if height == 0 {
        0
    } else {
        (height as usize - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(min_row_bytes))
            .expect("pixmap stride and dimensions overflow usize")
    };
    assert!(
        data_len >= min_len,
        "pixel data length ({data_len}) must be at least {min_len} for {width}x{height} with stride {stride}"
    );
}

pub(crate) fn subregion_byte_range(
    parent_width: u16,
    parent_height: u16,
    stride: usize,
    x: u16,
    y: u16,
    width: u16,
    height: u16,
) -> core::ops::Range<usize> {
    assert!(
        u32::from(x) + u32::from(width) <= u32::from(parent_width)
            && u32::from(y) + u32::from(height) <= u32::from(parent_height),
        "subregion ({x}, {y}, {width}x{height}) out of bounds for {parent_width}x{parent_height} pixmap"
    );
    if height == 0 {
        return 0..0;
    }
    let start = y as usize * stride + row_bytes(x);
    let len = (height as usize - 1) * stride + row_bytes(width);
    start..start + len
}

#[cfg(test)]
pub(crate) fn numbered_pixmap(width: u16, height: u16) -> crate::Pixmap {
    let mut data = alloc::vec::Vec::with_capacity(byte_len(width, height));
    for y in 0..height {
        for x in 0..width {
            data.extend_from_slice(&[x as u8, y as u8, (x + y) as u8, 255]);
        }
    }
    crate::Pixmap::from_parts(data, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_byte_len_handles_zero_and_large_dimensions() {
        assert_eq!(checked_byte_len(0, 0), Some(0));
        assert_eq!(checked_byte_len(u16::MAX, 0), Some(0));
        assert_eq!(checked_byte_len(0, u16::MAX), Some(0));
        assert_eq!(checked_byte_len(3, 2), Some(24));

        // 32768 * 32768 * 4 == 2^32, which wraps to 0 in unchecked 32-bit arithmetic.
        #[cfg(target_pointer_width = "32")]
        {
            assert_eq!(checked_byte_len(32768, 32768), None);
            assert_eq!(checked_byte_len(u16::MAX, u16::MAX), None);
        }
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(checked_byte_len(32768, 32768), Some(4_294_967_296));
            assert_eq!(
                checked_byte_len(u16::MAX, u16::MAX),
                Some(usize::from(u16::MAX) * usize::from(u16::MAX) * BYTES_PER_PIXEL)
            );
        }

        assert!(!is_valid_packed_parts(0, 32768, 32768));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn try_from_parts_rejects_32_bit_overflow_dimensions() {
        use crate::{Pixmap, PixmapMut, PixmapRef};
        use alloc::vec::Vec;

        assert!(Pixmap::try_from_parts(Vec::new(), 32768, 32768).is_none());
        assert!(Pixmap::try_from_unpremultiplied(Vec::new(), 32768, 32768).is_none());
        assert!(PixmapRef::try_from_parts(&[], 32768, 32768).is_none());
        assert!(PixmapMut::try_from_parts(&mut [], 32768, 32768).is_none());
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    #[should_panic(expected = "overflow")]
    fn from_parts_panics_on_32_bit_overflow_dimensions() {
        let _ = crate::Pixmap::from_parts(alloc::vec::Vec::new(), 32768, 32768);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    #[should_panic(expected = "overflow")]
    fn resize_buffer_panics_on_32_bit_overflow_dimensions() {
        let mut pixmap = crate::Pixmap::new(1, 1);
        pixmap.resize_buffer(32768, 32768);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    #[should_panic(expected = "overflow")]
    fn reset_panics_on_32_bit_overflow_dimensions() {
        let mut pixmap = crate::Pixmap::new(1, 1);
        pixmap.reset(32768, 32768);
    }
}
