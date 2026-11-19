//! Minimal RGBA8 pixel buffers: the owned [`Pixmap`] and the borrowed [`PixmapRef`] and
//! [`PixmapMut`] views.
//!
//! [`Pixmap`] is the pixel container shared by AnyRender's glyph and atlas crates, so that
//! neither has to depend on the other or on a renderer. It has no dependencies, and its buffer
//! is a plain `Vec<u8>` so that converting to a renderer's own image type can move the pixels
//! instead of copying them.
//!
//! Owned [`Pixmap`] buffers are always tightly packed. Borrowed [`PixmapRef`] and [`PixmapMut`]
//! views carry an explicit byte stride between rows, allowing zero-copy subregions and
//! interoperability with row-padded external or GPU buffers.

#![no_std]

extern crate alloc;

mod alpha;
mod blend;
mod blit;
mod pixmap;
mod pixmap_mut;
mod pixmap_ref;

pub use alpha::{multiply_alpha_rgba8, premultiply_rgba8, swap_rb_rgba8, unpremultiply_rgba8};
pub use blend::{BlendMode, Compose, Mix, composite_color_rgba8, composite_rgba8};
pub use pixmap::Pixmap;
pub use pixmap_mut::{PixmapMut, RowsMut};
pub use pixmap_ref::{PixmapRef, Rows};

/// The number of bytes one pixel occupies.
pub const BYTES_PER_PIXEL: usize = 4;

pub(crate) fn row_bytes(width: u16) -> usize {
    width as usize * BYTES_PER_PIXEL
}

pub(crate) fn byte_len(width: u16, height: u16) -> usize {
    row_bytes(width) * height as usize
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
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn numbered_pixmap(width: u16, height: u16) -> Pixmap {
        let mut data = Vec::with_capacity(byte_len(width, height));
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&[x as u8, y as u8, (x + y) as u8, 255]);
            }
        }
        Pixmap::from_parts(data, width, height)
    }

    #[test]
    fn new_is_transparent() {
        let pixmap = Pixmap::new(3, 2);
        assert_eq!(pixmap.byte_len(), 24);
        assert_eq!(pixmap.stride(), 12);
        assert!(!pixmap.is_empty());
        assert!(pixmap.data().iter().all(|&byte| byte == 0));
    }

    #[test]
    fn from_parts_round_trips_without_copying() {
        let data = vec![7_u8; 16];
        let ptr = data.as_ptr();
        let pixmap = Pixmap::from_parts(data, 2, 2);
        assert_eq!((pixmap.width(), pixmap.height()), (2, 2));
        let data = pixmap.into_data();
        assert_eq!(data.as_ptr(), ptr);
    }

    #[test]
    #[should_panic(expected = "pixel data length")]
    fn from_parts_rejects_wrong_length() {
        let _ = Pixmap::from_parts(vec![0; 15], 2, 2);
    }

    #[test]
    fn as_ref_and_pixmap_ref_from_parts() {
        let pixmap = numbered_pixmap(3, 2);
        let view = pixmap.as_ref();
        assert_eq!((view.width(), view.height(), view.stride()), (3, 2, 12));
        assert_eq!(view.byte_len(), 24);
        assert!(view.is_contiguous());
        assert_eq!(view.data(), Some(pixmap.data()));
        assert_eq!(view.raw_data(), pixmap.data());
        assert_eq!(view.to_owned(), pixmap);

        let from_parts = PixmapRef::from_parts(pixmap.data(), 3, 2);
        assert_eq!(from_parts, view);
        assert_eq!(PixmapRef::from(&pixmap), view);
    }

    #[test]
    #[should_panic(expected = "pixel data length")]
    fn pixmap_ref_rejects_wrong_length() {
        let _ = PixmapRef::from_parts(&[0; 7], 1, 2);
    }

    #[test]
    fn as_mut_writes_through_to_the_pixmap() {
        let mut pixmap = Pixmap::new(2, 1);
        let mut view = pixmap.as_mut();
        assert_eq!((view.width(), view.height(), view.stride()), (2, 1, 8));
        view.data_mut().unwrap()[4..].fill(9);
        assert_eq!(pixmap.data(), [0, 0, 0, 0, 9, 9, 9, 9]);
    }

    #[test]
    fn pixmap_mut_borrows_external_data() {
        let mut data = [1_u8; 8];
        let mut view = PixmapMut::from_parts(&mut data, 1, 2);
        view.reborrow().data_mut().unwrap()[0] = 5;
        assert_eq!(view.to_owned().data(), [5, 1, 1, 1, 1, 1, 1, 1]);
        view.into_raw_data()[7] = 6;
        assert_eq!(data, [5, 1, 1, 1, 1, 1, 1, 6]);
    }

    #[test]
    #[should_panic(expected = "pixel data length")]
    fn pixmap_mut_rejects_wrong_length() {
        let _ = PixmapMut::from_parts(&mut [0; 7], 1, 2);
    }

    #[test]
    fn strided_view_handles_row_padding_and_trimmed_last_row() {
        // 2x3 image with 16-byte stride (8 bytes active + 8 bytes padding per row).
        // Buffer has trailing padding on rows 0 and 1, and only active bytes on row 2.
        let mut buf = vec![0xAA_u8; 2 * 16 + 8];
        buf[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        buf[16..24].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
        buf[32..40].copy_from_slice(&[17, 18, 19, 20, 21, 22, 23, 24]);

        let view = PixmapRef::from_parts_with_stride(&buf, 2, 3, 16);
        assert!(!view.is_contiguous());
        assert_eq!(view.data(), None);
        assert_eq!(view.byte_len(), 24);
        assert_eq!(view.stride(), 16);
        assert_eq!(view.raw_data().len(), 40);
        assert_eq!(view.row(0), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(view.row(1), &[9, 10, 11, 12, 13, 14, 15, 16]);
        assert_eq!(view.row(2), &[17, 18, 19, 20, 21, 22, 23, 24]);

        let owned = view.to_owned();
        assert_eq!(owned.data(), &(1_u8..=24).collect::<Vec<_>>()[..]);
        // Equality compares active pixels, ignoring stride and padding bytes.
        assert_eq!(view, owned.as_ref());
    }

    #[test]
    fn single_row_strided_view_is_contiguous() {
        let buf = [1, 2, 3, 4, 5, 6, 7, 8, 0xFF, 0xFF, 0xFF, 0xFF];
        let view = PixmapRef::from_parts_with_stride(&buf, 2, 1, 12);
        assert!(view.is_contiguous());
        assert_eq!(view.data(), Some(&[1, 2, 3, 4, 5, 6, 7, 8][..]));
        assert_eq!(view.raw_data().len(), 12);
    }

    #[test]
    #[should_panic(expected = "stride")]
    fn from_parts_with_stride_rejects_small_stride() {
        let _ = PixmapRef::from_parts_with_stride(&[0; 32], 3, 2, 8);
    }

    #[test]
    #[should_panic(expected = "pixel data length")]
    fn from_parts_with_stride_rejects_short_buffer() {
        let _ = PixmapRef::from_parts_with_stride(&[0; 23], 2, 2, 16);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn from_parts_with_stride_rejects_overflow() {
        let _ = PixmapRef::from_parts_with_stride(&[0; 16], 1, 2, usize::MAX);
    }

    #[test]
    fn subregion_reads_and_nests_without_copying() {
        let pixmap = numbered_pixmap(4, 4);
        let sub = pixmap.subregion(1, 1, 2, 2);
        assert_eq!((sub.width(), sub.height(), sub.stride()), (2, 2, 16));
        assert!(!sub.is_contiguous());
        assert_eq!(sub.data(), None);
        assert_eq!(sub.row(0), &[1, 1, 2, 255, 2, 1, 3, 255]);
        assert_eq!(sub.row(1), &[1, 2, 3, 255, 2, 2, 4, 255]);

        let owned = sub.to_owned();
        assert_eq!(owned.as_ref(), sub);

        // Nested 1x1 subregion should point directly at pixel (2, 2).
        let nested = sub.subregion(1, 1, 1, 1);
        assert!(nested.is_contiguous());
        assert_eq!(nested.data(), Some(&[2, 2, 4, 255][..]));
    }

    #[test]
    fn subregion_mut_writes_only_inside_region() {
        let mut pixmap = Pixmap::new(4, 3);
        {
            let mut sub = pixmap.subregion_mut(1, 1, 2, 2);
            assert!(!sub.is_contiguous());
            assert_eq!(sub.data_mut(), None);
            for row in sub.rows_mut() {
                row.fill(42);
            }
        }
        assert_eq!(pixmap.row(0), &[0; 16]);
        assert_eq!(
            pixmap.row(1),
            &[0, 0, 0, 0, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0]
        );
        assert_eq!(
            pixmap.row(2),
            &[0, 0, 0, 0, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 0, 0]
        );
    }

    #[test]
    fn rows_and_rows_mut_support_double_ended_and_exact_size() {
        let mut pixmap = numbered_pixmap(4, 4);
        let sub = pixmap.subregion(1, 0, 2, 4);
        let mut rows = sub.rows();
        assert_eq!(rows.len(), 4);
        assert_eq!(rows.next(), Some(&[1, 0, 1, 255, 2, 0, 2, 255][..]));
        assert_eq!(rows.next_back(), Some(&[1, 3, 4, 255, 2, 3, 5, 255][..]));
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.next_back(), Some(&[1, 2, 3, 255, 2, 2, 4, 255][..]));
        assert_eq!(rows.next(), Some(&[1, 1, 2, 255, 2, 1, 3, 255][..]));
        assert_eq!(rows.len(), 0);
        assert_eq!(rows.next(), None);
        assert_eq!(rows.next_back(), None);

        let mut sub_mut = pixmap.subregion_mut(1, 1, 2, 3);
        let mut rows_mut = sub_mut.rows_mut();
        assert_eq!(rows_mut.len(), 3);
        rows_mut.next_back().unwrap().fill(99);
        rows_mut.next().unwrap().fill(11);
        rows_mut.next_back().unwrap().fill(55);
        assert_eq!(rows_mut.next(), None);
        assert_eq!(rows_mut.next_back(), None);

        assert_eq!(pixmap.subregion(1, 1, 2, 1).row(0), &[11; 8]);
        assert_eq!(pixmap.subregion(1, 2, 2, 1).row(0), &[55; 8]);
        assert_eq!(pixmap.subregion(1, 3, 2, 1).row(0), &[99; 8]);
    }

    #[test]
    fn empty_dimensions_and_edge_subregions() {
        let empty = Pixmap::new(0, 3);
        assert!(empty.is_empty());
        assert_eq!(empty.as_ref().data(), Some(&[][..]));
        assert_eq!(empty.rows().len(), 3);
        assert_eq!(empty.rows().collect::<Vec<_>>(), vec![&[][..]; 3]);

        let zero_h = Pixmap::new(3, 0);
        assert!(zero_h.is_empty());
        assert_eq!(zero_h.as_ref().data(), Some(&[][..]));
        assert_eq!(zero_h.rows().count(), 0);

        let pixmap = numbered_pixmap(4, 4);
        let right_edge = pixmap.subregion(4, 0, 0, 4);
        assert!(right_edge.is_empty());
        assert!(right_edge.is_contiguous());
        assert_eq!(right_edge.data(), Some(&[][..]));
        assert_eq!(right_edge.rows().len(), 4);

        let bottom_corner = pixmap.subregion(4, 4, 0, 0);
        assert!(bottom_corner.is_empty());
        assert_eq!(bottom_corner.rows().count(), 0);
        assert_eq!(bottom_corner.to_owned(), Pixmap::new(0, 0));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn subregion_rejects_out_of_bounds_and_u16_overflow() {
        let pixmap = Pixmap::new(4, 4);
        let _ = pixmap.subregion(u16::MAX, 0, 2, 1);
    }

    #[test]
    #[should_panic(expected = "row index")]
    fn row_rejects_out_of_bounds() {
        let pixmap = Pixmap::new(2, 2);
        let _ = pixmap.row(2);
    }

    #[test]
    fn equality_ignores_padding_bytes_and_distinguishes_pixels() {
        let mut buf_a = vec![0x11_u8; 16 + 4];
        let mut buf_b = vec![0x99_u8; 24 + 4];
        buf_a[0..4].copy_from_slice(&[1, 2, 3, 4]);
        buf_a[16..20].copy_from_slice(&[5, 6, 7, 8]);
        buf_b[0..4].copy_from_slice(&[1, 2, 3, 4]);
        buf_b[24..28].copy_from_slice(&[5, 6, 7, 8]);

        let view_a = PixmapMut::from_parts_with_stride(&mut buf_a, 1, 2, 16);
        let mut view_b = PixmapMut::from_parts_with_stride(&mut buf_b, 1, 2, 24);
        assert_eq!(view_a, view_b);

        view_b.row_mut(1)[0] = 9;
        assert_ne!(view_a, view_b);
    }

    #[test]
    fn pixel_and_set_pixel_work_on_owned_and_strided_views() {
        let mut pixmap = numbered_pixmap(4, 3);
        assert_eq!(pixmap.pixel(2, 1), [2, 1, 3, 255]);
        assert_eq!(pixmap.as_ref().pixel(3, 2), [3, 2, 5, 255]);

        pixmap.set_pixel(2, 1, [10, 20, 30, 40]);
        assert_eq!(pixmap.pixel(2, 1), [10, 20, 30, 40]);

        let mut sub = pixmap.subregion_mut(1, 1, 2, 2);
        assert_eq!(sub.pixel(1, 0), [10, 20, 30, 40]);
        sub.set_pixel(0, 1, [50, 60, 70, 80]);
        assert_eq!(sub.pixel(0, 1), [50, 60, 70, 80]);
        assert_eq!(pixmap.pixel(1, 2), [50, 60, 70, 80]);
        // Neighbouring pixel outside the subregion is untouched.
        assert_eq!(pixmap.pixel(0, 2), [0, 2, 2, 255]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn pixel_rejects_out_of_bounds_x() {
        let pixmap = Pixmap::new(2, 2);
        let _ = pixmap.pixel(2, 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_pixel_rejects_out_of_bounds_y() {
        let mut pixmap = Pixmap::new(2, 2);
        pixmap.set_pixel(0, 2, [1, 2, 3, 4]);
    }

    #[test]
    fn fill_and_clear_preserve_stride_padding() {
        // 2x2 strided buffer with 16-byte stride (8 active + 8 padding) and 4 trailing bytes.
        let mut buf = vec![0xAA_u8; 16 + 8 + 4];
        let mut view = PixmapMut::from_parts_with_stride(&mut buf, 2, 2, 16);

        view.fill([1, 2, 3, 4]);
        assert_eq!(view.row(0), &[1, 2, 3, 4, 1, 2, 3, 4]);
        assert_eq!(view.row(1), &[1, 2, 3, 4, 1, 2, 3, 4]);
        assert_eq!(&view.raw_data()[8..16], &[0xAA; 8]);
        assert_eq!(&view.raw_data()[24..28], &[0xAA; 4]);

        // Uniform-byte fast path.
        view.fill([0x55; 4]);
        assert_eq!(view.row(0), &[0x55; 8]);
        assert_eq!(view.row(1), &[0x55; 8]);
        assert_eq!(&view.raw_data()[8..16], &[0xAA; 8]);

        view.clear();
        assert_eq!(view.row(0), &[0; 8]);
        assert_eq!(view.row(1), &[0; 8]);
        assert_eq!(&view.raw_data()[8..16], &[0xAA; 8]);
        assert_eq!(&view.raw_data()[24..28], &[0xAA; 4]);

        // Single-row contiguous strided view also preserves trailing bytes.
        let mut row_buf = [0xBB_u8; 12];
        let mut single = PixmapMut::from_parts_with_stride(&mut row_buf, 2, 1, 12);
        single.fill([9, 8, 7, 6]);
        assert_eq!(&row_buf[..8], &[9, 8, 7, 6, 9, 8, 7, 6]);
        assert_eq!(&row_buf[8..], &[0xBB; 4]);
    }

    #[test]
    fn fill_rect_and_clear_rect_modify_only_target_region() {
        let mut pixmap = Pixmap::new(4, 4);
        pixmap.fill([10, 20, 30, 255]);
        pixmap.fill_rect(1, 1, 2, 2, [1, 2, 3, 4]);

        for y in 0..4 {
            for x in 0..4 {
                let expected = if (1..3).contains(&x) && (1..3).contains(&y) {
                    [1, 2, 3, 4]
                } else {
                    [10, 20, 30, 255]
                };
                assert_eq!(pixmap.pixel(x, y), expected, "at ({x}, {y})");
            }
        }

        // Zero-sized rects are no-ops, including at edges.
        pixmap.fill_rect(4, 4, 0, 0, [99, 99, 99, 99]);
        pixmap.clear_rect(0, 0, 0, 4);
        assert_eq!(pixmap.pixel(0, 0), [10, 20, 30, 255]);

        pixmap.as_mut().clear_rect(1, 2, 2, 1);
        assert_eq!(pixmap.pixel(1, 1), [1, 2, 3, 4]);
        assert_eq!(pixmap.pixel(1, 2), [0, 0, 0, 0]);
        assert_eq!(pixmap.pixel(2, 2), [0, 0, 0, 0]);
        assert_eq!(pixmap.pixel(3, 2), [10, 20, 30, 255]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn fill_rect_rejects_out_of_bounds() {
        let mut pixmap = Pixmap::new(4, 4);
        pixmap.fill_rect(3, 0, 2, 1, [1, 2, 3, 4]);
    }

    #[test]
    fn is_opaque_checks_active_pixels_and_ignores_padding() {
        assert!(Pixmap::new(0, 0).is_opaque());
        assert!(Pixmap::new(0, 3).is_opaque());
        assert!(Pixmap::new(3, 0).is_opaque());
        assert!(!Pixmap::new(2, 2).is_opaque());

        let mut pixmap = Pixmap::new(3, 2);
        pixmap.fill([10, 20, 30, 255]);
        assert!(pixmap.is_opaque());
        assert!(pixmap.as_ref().is_opaque());
        assert!(pixmap.as_mut().is_opaque());

        pixmap.set_pixel(2, 1, [10, 20, 30, 254]);
        assert!(!pixmap.is_opaque());
        assert!(pixmap.subregion(0, 0, 2, 2).is_opaque());

        // Strided buffer with zeroed padding bytes is still opaque if active pixels have alpha 255.
        let mut buf = vec![0_u8; 16 + 8];
        let mut view = PixmapMut::from_parts_with_stride(&mut buf, 2, 2, 16);
        view.fill([1, 2, 3, 255]);
        assert_eq!(&view.raw_data()[8..16], &[0; 8]);
        assert!(view.is_opaque());
        assert!(view.as_ref().is_opaque());

        view.set_pixel(1, 1, [1, 2, 3, 0]);
        assert!(!view.is_opaque());
    }

    #[test]
    #[allow(clippy::iter_nth_zero)]
    fn rows_and_rows_mut_fold_rfold_nth_and_nth_back() {
        // Test both full-stride (last row padded) and trimmed-last-row buffers.
        for extra_tail in [0usize, 8] {
            // 2x4 strided buffer with stride 16 (8 active + 8 padding per row).
            let mut buf = vec![0xEE_u8; 3 * 16 + 8 + extra_tail];
            for y in 0..4usize {
                let start = y * 16;
                buf[start..start + 8].fill(y as u8 + 1);
            }

            let view = PixmapRef::from_parts_with_stride(&buf, 2, 4, 16);
            assert_eq!(view.rows().count(), 4);
            assert_eq!(view.rows().last().unwrap(), &[4; 8]);

            let folded: Vec<u8> = view.rows().fold(Vec::new(), |mut acc, r| {
                acc.push(r[0]);
                acc
            });
            assert_eq!(folded, vec![1, 2, 3, 4]);

            let rfolded: Vec<u8> = view.rows().rfold(Vec::new(), |mut acc, r| {
                acc.push(r[0]);
                acc
            });
            assert_eq!(rfolded, vec![4, 3, 2, 1]);

            // nth from front across head and last.
            let mut it = view.rows();
            assert_eq!(it.nth(1).unwrap(), &[2; 8]);
            assert_eq!(it.len(), 2);
            assert_eq!(it.nth(1).unwrap(), &[4; 8]);
            assert_eq!(it.len(), 0);
            assert_eq!(it.nth(0), None);
            assert_eq!(it.next(), None);

            // nth past end empties iterator cleanly.
            let mut it = view.rows();
            assert_eq!(it.nth(10), None);
            assert_eq!(it.len(), 0);
            assert_eq!(it.next(), None);

            // nth_back across last and head.
            let mut it = view.rows();
            assert_eq!(it.nth_back(0).unwrap(), &[4; 8]);
            assert_eq!(it.len(), 3);
            assert_eq!(it.nth_back(1).unwrap(), &[2; 8]);
            assert_eq!(it.len(), 1);
            assert_eq!(it.nth_back(0).unwrap(), &[1; 8]);
            assert_eq!(it.len(), 0);
            assert_eq!(it.nth_back(0), None);

            // nth_back skipping last directly into head.
            let mut it = view.rows();
            assert_eq!(it.nth_back(2).unwrap(), &[2; 8]);
            assert_eq!(it.len(), 1);
            assert_eq!(it.next().unwrap(), &[1; 8]);
            assert_eq!(it.next_back(), None);

            // RowsMut fold, rfold, nth, nth_back preserve stride padding.
            let mut mut_view = PixmapMut::from_parts_with_stride(&mut buf, 2, 4, 16);
            assert_eq!(mut_view.rows_mut().count(), 4);
            assert_eq!(mut_view.rows_mut().last().unwrap(), &[4; 8]);

            mut_view.rows_mut().fold(10u8, |v, row| {
                row.fill(v);
                v + 1
            });
            assert_eq!(mut_view.row(0), &[10; 8]);
            assert_eq!(mut_view.row(3), &[13; 8]);

            mut_view.rows_mut().rfold(20u8, |v, row| {
                row[0] = v;
                v + 1
            });
            assert_eq!(mut_view.row(3)[0], 20);
            assert_eq!(mut_view.row(0)[0], 23);

            {
                let mut it = mut_view.rows_mut();
                it.nth(3).unwrap()[1] = 99;
                assert_eq!(it.len(), 0);
                assert_eq!(it.nth(0), None);
            }
            assert_eq!(mut_view.row(3)[1], 99);

            {
                let mut it = mut_view.rows_mut();
                it.nth_back(2).unwrap()[1] = 77;
                assert_eq!(it.len(), 1);
                assert_eq!(it.nth_back(5), None);
                assert_eq!(it.len(), 0);
            }
            assert_eq!(mut_view.row(1)[1], 77);

            // All inter-row padding and optional trailing padding stayed untouched.
            for y in 0..3 {
                assert_eq!(&buf[y * 16 + 8..(y + 1) * 16], &[0xEE; 8]);
            }
            if extra_tail > 0 {
                assert_eq!(&buf[3 * 16 + 8..], &[0xEE; 8]);
            }
        }
    }

    #[test]
    #[allow(clippy::iter_nth_zero)]
    fn rows_and_rows_mut_handle_zero_width_and_zero_height() {
        let mut empty_w = Pixmap::new(0, 3);
        assert_eq!(empty_w.rows().count(), 3);
        assert_eq!(
            empty_w.rows().fold(0usize, |n, r| {
                assert!(r.is_empty());
                n + 1
            }),
            3
        );
        assert_eq!(
            empty_w.rows().rfold(0usize, |n, r| {
                assert!(r.is_empty());
                n + 1
            }),
            3
        );

        let mut it = empty_w.rows();
        assert_eq!(it.nth(1), Some(&[][..]));
        assert_eq!(it.len(), 1);
        assert_eq!(it.nth_back(0), Some(&[][..]));
        assert_eq!(it.len(), 0);
        assert_eq!(it.nth(0), None);

        let mut mit = empty_w.rows_mut();
        assert_eq!(mit.nth_back(1), Some(&mut [][..]));
        assert_eq!(mit.len(), 1);
        assert_eq!(mit.nth(1), None);
        assert_eq!(mit.len(), 0);

        let mut empty_h = Pixmap::new(3, 0);
        assert_eq!(empty_h.rows().count(), 0);
        assert_eq!(empty_h.rows().last(), None);
        assert_eq!(empty_h.rows_mut().count(), 0);
        assert_eq!(empty_h.rows_mut().last(), None);
    }

    #[test]
    fn mul_div_255_is_exact_for_all_byte_pairs() {
        for a in 0u8..=255 {
            for c in 0u8..=255 {
                let expected = ((u32::from(c) * u32::from(a) + 127) / 255) as u8;
                assert_eq!(
                    alpha::mul_div_255(c, a),
                    expected,
                    "mismatch for c={c}, a={a}"
                );
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
