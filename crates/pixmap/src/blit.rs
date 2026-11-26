pub(crate) struct BlitRect {
    pub(crate) src_x: u16,
    pub(crate) src_y: u16,
    pub(crate) dst_x: u16,
    pub(crate) dst_y: u16,
    pub(crate) width: u16,
    pub(crate) height: u16,
}

pub(crate) fn clip_blit(
    dst_width: u16,
    dst_height: u16,
    src_width: u16,
    src_height: u16,
    x: i32,
    y: i32,
) -> Option<BlitRect> {
    let x = i64::from(x);
    let y = i64::from(y);
    let dst_w = i64::from(dst_width);
    let dst_h = i64::from(dst_height);
    let src_w = i64::from(src_width);
    let src_h = i64::from(src_height);

    let dst_x0 = x.max(0);
    let dst_y0 = y.max(0);
    let dst_x1 = (x + src_w).min(dst_w);
    let dst_y1 = (y + src_h).min(dst_h);

    if dst_x0 >= dst_x1 || dst_y0 >= dst_y1 {
        return None;
    }

    Some(BlitRect {
        src_x: (dst_x0 - x) as u16,
        src_y: (dst_y0 - y) as u16,
        dst_x: dst_x0 as u16,
        dst_y: dst_y0 as u16,
        width: (dst_x1 - dst_x0) as u16,
        height: (dst_y1 - dst_y0) as u16,
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::util::numbered_pixmap;
    use crate::{Pixmap, PixmapMut, PixmapRef};

    #[test]
    fn copy_from_full_and_interior() {
        let src = numbered_pixmap(3, 2);
        let mut dst = Pixmap::new(3, 2);
        dst.copy_from(src.as_ref(), 0, 0);
        assert_eq!(dst, src);

        let bg = [200, 150, 100, 255];
        let mut canvas = Pixmap::new(6, 5);
        canvas.fill(bg);
        canvas.copy_from(src.as_ref(), 2, 1);

        for y in 0..5 {
            for x in 0..6 {
                if (2..5).contains(&x) && (1..3).contains(&y) {
                    assert_eq!(canvas.pixel(x, y), src.pixel(x - 2, y - 1));
                } else {
                    assert_eq!(canvas.pixel(x, y), bg);
                }
            }
        }
    }

    #[test]
    fn copy_from_clips_all_edges_and_oversized_source() {
        let src = numbered_pixmap(4, 3);
        let bg = [10, 20, 30, 40];

        // Top-left clipping (negative x and y).
        let mut dst = Pixmap::new(4, 4);
        dst.fill(bg);
        dst.copy_from(src.as_ref(), -2, -1);
        for y in 0..4 {
            for x in 0..4 {
                if x < 2 && y < 2 {
                    assert_eq!(dst.pixel(x, y), src.pixel(x + 2, y + 1));
                } else {
                    assert_eq!(dst.pixel(x, y), bg);
                }
            }
        }

        // Bottom-right clipping.
        dst.fill(bg);
        dst.copy_from(src.as_ref(), 2, 2);
        for y in 0..4 {
            for x in 0..4 {
                if x >= 2 && y >= 2 {
                    assert_eq!(dst.pixel(x, y), src.pixel(x - 2, y - 2));
                } else {
                    assert_eq!(dst.pixel(x, y), bg);
                }
            }
        }

        // Source larger than destination on all four sides.
        let mut small = Pixmap::new(2, 1);
        small.fill(bg);
        small.copy_from(src.as_ref(), -1, -1);
        assert_eq!(small.pixel(0, 0), src.pixel(1, 1));
        assert_eq!(small.pixel(1, 0), src.pixel(2, 1));
    }

    #[test]
    fn copy_from_out_of_bounds_and_empty_are_noops() {
        let src = numbered_pixmap(3, 2);
        let bg = [9, 8, 7, 255];
        let mut dst = Pixmap::new(4, 3);
        dst.fill(bg);
        let expected = dst.clone();

        for (x, y) in [
            (-3, 0),
            (4, 0),
            (0, -2),
            (0, 3),
            (i32::MIN, 0),
            (0, i32::MIN),
            (i32::MAX, 0),
            (0, i32::MAX),
            (i32::MIN, i32::MAX),
        ] {
            dst.copy_from(src.as_ref(), x, y);
            assert_eq!(dst, expected);
        }

        for (w, h) in [(0, 0), (0, 3), (3, 0)] {
            let empty_src = Pixmap::new(w, h);
            dst.copy_from(empty_src.as_ref(), 0, 0);
            assert_eq!(dst, expected);

            let mut empty_dst = Pixmap::new(w, h);
            empty_dst.copy_from(src.as_ref(), 0, 0);
            assert!(empty_dst.is_empty());
        }
    }

    #[test]
    fn copy_from_strided_and_subregion_preserves_padding() {
        // 3x2 strided source with 4 bytes of padding per row (stride = 16).
        let src_pad = [0x11, 0x22, 0x33, 0x44];
        let mut src_buf = vec![0u8; 2 * 16];
        for y in 0..2u16 {
            for x in 0..3u16 {
                let off = y as usize * 16 + x as usize * 4;
                src_buf[off..off + 4].copy_from_slice(&[
                    (x + 1) as u8 * 10,
                    (y + 1) as u8 * 20,
                    30,
                    255,
                ]);
            }
            src_buf[y as usize * 16 + 12..y as usize * 16 + 16].copy_from_slice(&src_pad);
        }
        let src_view = PixmapRef::from_parts_with_stride(&src_buf, 3, 2, 16);

        // 3x3 strided destination with 4 bytes of padding per row (stride = 16).
        let dst_pad = [0xAA, 0xBB, 0xCC, 0xDD];
        let mut dst_buf = vec![0u8; 3 * 16];
        for row in 0..3 {
            dst_buf[row * 16 + 12..row * 16 + 16].copy_from_slice(&dst_pad);
        }

        {
            let mut dst_view = PixmapMut::from_parts_with_stride(&mut dst_buf, 3, 3, 16);
            // Blit a subregion of src_view starting at (1, 0) (2x2) into dst at (1, 1).
            dst_view.copy_from(src_view.subregion(1, 0, 2, 2), 1, 1);

            assert_eq!(dst_view.pixel(0, 0), [0, 0, 0, 0]);
            assert_eq!(dst_view.pixel(1, 1), src_view.pixel(1, 0));
            assert_eq!(dst_view.pixel(2, 1), src_view.pixel(2, 0));
            assert_eq!(dst_view.pixel(1, 2), src_view.pixel(1, 1));
            assert_eq!(dst_view.pixel(2, 2), src_view.pixel(2, 1));
        }

        for row in 0..3 {
            assert_eq!(&dst_buf[row * 16 + 12..row * 16 + 16], &dst_pad);
        }
    }
}
