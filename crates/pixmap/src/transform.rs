use alloc::vec;

use crate::util::{byte_len, fill_slice, row_bytes};
use crate::{BYTES_PER_PIXEL, Pixmap, PixmapMut, PixmapRef};

impl Pixmap {
    /// Flips the pixmap horizontally (left-to-right) in place.
    pub fn flip_horizontal(&mut self) {
        self.as_mut().flip_horizontal();
    }

    /// Flips the pixmap vertically (top-to-bottom) in place.
    pub fn flip_vertical(&mut self) {
        self.as_mut().flip_vertical();
    }

    /// Rotates the pixmap 180 degrees in place.
    pub fn rotate_180(&mut self) {
        self.as_mut().rotate_180();
    }

    /// Returns a new pixmap flipped horizontally (left-to-right).
    pub fn flipped_horizontal(&self) -> Self {
        self.as_ref().flipped_horizontal()
    }

    /// Returns a new pixmap flipped vertically (top-to-bottom).
    pub fn flipped_vertical(&self) -> Self {
        self.as_ref().flipped_vertical()
    }

    /// Returns a new pixmap rotated 90 degrees clockwise (`(x, y) -> (height - 1 - y, x)`).
    pub fn rotated_90(&self) -> Self {
        self.as_ref().rotated_90()
    }

    /// Returns a new pixmap rotated 180 degrees.
    pub fn rotated_180(&self) -> Self {
        self.as_ref().rotated_180()
    }

    /// Returns a new pixmap rotated 270 degrees clockwise (90 degrees counter-clockwise,
    /// `(x, y) -> (y, width - 1 - x)`).
    pub fn rotated_270(&self) -> Self {
        self.as_ref().rotated_270()
    }

    /// Returns a new pixmap transposed across the main diagonal (`(x, y) -> (y, x)`).
    pub fn transposed(&self) -> Self {
        self.as_ref().transposed()
    }

    /// Returns a new pixmap padded on each side by the given pixel counts, filling the border
    /// with `fill`.
    ///
    /// # Panics
    ///
    /// Panics if the resulting width or height overflows `u16`.
    pub fn padded(&self, top: u16, right: u16, bottom: u16, left: u16, fill: [u8; 4]) -> Self {
        self.as_ref().padded(top, right, bottom, left, fill)
    }

    /// Returns a new pixmap padded on each side by repeating (clamping) the outermost edge and
    /// corner pixels.
    ///
    /// If `self` is empty, the returned pixmap is filled with transparent black (`[0, 0, 0, 0]`).
    ///
    /// # Panics
    ///
    /// Panics if the resulting width or height overflows `u16`.
    pub fn padded_clamped(&self, top: u16, right: u16, bottom: u16, left: u16) -> Self {
        self.as_ref().padded_clamped(top, right, bottom, left)
    }
}

impl PixmapRef<'_> {
    /// Returns a new tightly packed [`Pixmap`] flipped horizontally (left-to-right).
    pub fn flipped_horizontal(&self) -> Pixmap {
        flipped_horizontal(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] flipped vertically (top-to-bottom).
    pub fn flipped_vertical(&self) -> Pixmap {
        flipped_vertical(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] rotated 90 degrees clockwise
    /// (`(x, y) -> (height - 1 - y, x)`).
    pub fn rotated_90(&self) -> Pixmap {
        rotated_90(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] rotated 180 degrees.
    pub fn rotated_180(&self) -> Pixmap {
        rotated_180(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] rotated 270 degrees clockwise (90 degrees
    /// counter-clockwise, `(x, y) -> (y, width - 1 - x)`).
    pub fn rotated_270(&self) -> Pixmap {
        rotated_270(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] transposed across the main diagonal
    /// (`(x, y) -> (y, x)`).
    pub fn transposed(&self) -> Pixmap {
        transposed(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] padded on each side by the given pixel counts,
    /// filling the border with `fill`.
    ///
    /// # Panics
    ///
    /// Panics if the resulting width or height overflows `u16`.
    pub fn padded(&self, top: u16, right: u16, bottom: u16, left: u16, fill: [u8; 4]) -> Pixmap {
        padded(*self, top, right, bottom, left, fill)
    }

    /// Returns a new tightly packed [`Pixmap`] padded on each side by repeating (clamping) the
    /// outermost edge and corner pixels.
    ///
    /// If `self` is empty, the returned pixmap is filled with transparent black (`[0, 0, 0, 0]`).
    ///
    /// # Panics
    ///
    /// Panics if the resulting width or height overflows `u16`.
    pub fn padded_clamped(&self, top: u16, right: u16, bottom: u16, left: u16) -> Pixmap {
        padded_clamped(*self, top, right, bottom, left)
    }
}

impl PixmapMut<'_> {
    /// Flips active pixels horizontally (left-to-right) in place, leaving any inter-row stride
    /// padding untouched.
    pub fn flip_horizontal(&mut self) {
        flip_horizontal_in_place(self);
    }

    /// Flips active pixels vertically (top-to-bottom) in place, leaving any inter-row stride
    /// padding untouched.
    pub fn flip_vertical(&mut self) {
        flip_vertical_in_place(self);
    }

    /// Rotates active pixels 180 degrees in place, leaving any inter-row stride padding untouched.
    pub fn rotate_180(&mut self) {
        rotate_180_in_place(self);
    }

    /// Returns a new tightly packed [`Pixmap`] flipped horizontally (left-to-right).
    pub fn flipped_horizontal(&self) -> Pixmap {
        self.as_ref().flipped_horizontal()
    }

    /// Returns a new tightly packed [`Pixmap`] flipped vertically (top-to-bottom).
    pub fn flipped_vertical(&self) -> Pixmap {
        self.as_ref().flipped_vertical()
    }

    /// Returns a new tightly packed [`Pixmap`] rotated 90 degrees clockwise
    /// (`(x, y) -> (height - 1 - y, x)`).
    pub fn rotated_90(&self) -> Pixmap {
        self.as_ref().rotated_90()
    }

    /// Returns a new tightly packed [`Pixmap`] rotated 180 degrees.
    pub fn rotated_180(&self) -> Pixmap {
        self.as_ref().rotated_180()
    }

    /// Returns a new tightly packed [`Pixmap`] rotated 270 degrees clockwise (90 degrees
    /// counter-clockwise, `(x, y) -> (y, width - 1 - x)`).
    pub fn rotated_270(&self) -> Pixmap {
        self.as_ref().rotated_270()
    }

    /// Returns a new tightly packed [`Pixmap`] transposed across the main diagonal
    /// (`(x, y) -> (y, x)`).
    pub fn transposed(&self) -> Pixmap {
        self.as_ref().transposed()
    }

    /// Returns a new tightly packed [`Pixmap`] padded on each side by the given pixel counts,
    /// filling the border with `fill`.
    ///
    /// # Panics
    ///
    /// Panics if the resulting width or height overflows `u16`.
    pub fn padded(&self, top: u16, right: u16, bottom: u16, left: u16, fill: [u8; 4]) -> Pixmap {
        self.as_ref().padded(top, right, bottom, left, fill)
    }

    /// Returns a new tightly packed [`Pixmap`] padded on each side by repeating (clamping) the
    /// outermost edge and corner pixels.
    ///
    /// If `self` is empty, the returned pixmap is filled with transparent black (`[0, 0, 0, 0]`).
    ///
    /// # Panics
    ///
    /// Panics if the resulting width or height overflows `u16`.
    pub fn padded_clamped(&self, top: u16, right: u16, bottom: u16, left: u16) -> Pixmap {
        self.as_ref().padded_clamped(top, right, bottom, left)
    }
}

/// Square tile edge length (in pixels) for cache-blocked 90°/270° rotations and transposition.
/// A 16×16 tile of RGBA8 pixels is 1 KiB, fitting comfortably in L1 cache and keeping both
/// source reads and destination writes sequential row-major.
const TILE: usize = 16;

/// Flips `dst` horizontally in place (mirroring left-to-right), preserving stride padding.
fn flip_horizontal_in_place(dst: &mut PixmapMut<'_>) {
    if dst.width() <= 1 || dst.height() == 0 {
        return;
    }
    for row in dst.rows_mut() {
        let (pixels, remainder) = row.as_chunks_mut::<BYTES_PER_PIXEL>();
        debug_assert!(remainder.is_empty());
        pixels.reverse();
    }
}

/// Flips `dst` vertically in place (mirroring top-to-bottom), preserving stride padding.
fn flip_vertical_in_place(dst: &mut PixmapMut<'_>) {
    if dst.width() == 0 || dst.height() <= 1 {
        return;
    }
    let mut rows = dst.rows_mut();
    while let (Some(top), Some(bottom)) = (rows.next(), rows.next_back()) {
        top.swap_with_slice(bottom);
    }
}

/// Rotates `dst` 180 degrees in place, preserving stride padding.
fn rotate_180_in_place(dst: &mut PixmapMut<'_>) {
    let w = dst.width();
    let h = dst.height();
    if w == 0 || h == 0 || (w == 1 && h == 1) {
        return;
    }
    if let Some(data) = dst.data_mut() {
        let (pixels, remainder) = data.as_chunks_mut::<BYTES_PER_PIXEL>();
        debug_assert!(remainder.is_empty());
        pixels.reverse();
        return;
    }
    let mut rows = dst.rows_mut();
    while let Some(top) = rows.next() {
        let (top_px, top_rem) = top.as_chunks_mut::<BYTES_PER_PIXEL>();
        debug_assert!(top_rem.is_empty());
        if let Some(bottom) = rows.next_back() {
            let (bot_px, bot_rem) = bottom.as_chunks_mut::<BYTES_PER_PIXEL>();
            debug_assert!(bot_rem.is_empty());
            for (a, b) in top_px.iter_mut().zip(bot_px.iter_mut().rev()) {
                core::mem::swap(a, b);
            }
        } else {
            top_px.reverse();
        }
    }
}

/// Returns a new tightly packed [`Pixmap`] flipped horizontally (mirrored left-to-right).
fn flipped_horizontal(src: PixmapRef<'_>) -> Pixmap {
    let w = src.width();
    let h = src.height();
    if w == 0 || h == 0 {
        return Pixmap::new(w, h);
    }
    let mut out = Pixmap::new(w, h);
    for (dst_row, src_row) in out.rows_mut().zip(src.rows()) {
        let (dst_px, dst_rem) = dst_row.as_chunks_mut::<BYTES_PER_PIXEL>();
        let (src_px, src_rem) = src_row.as_chunks::<BYTES_PER_PIXEL>();
        debug_assert!(dst_rem.is_empty() && src_rem.is_empty());
        for (d, &s) in dst_px.iter_mut().zip(src_px.iter().rev()) {
            *d = s;
        }
    }
    out
}

/// Returns a new tightly packed [`Pixmap`] flipped vertically (mirrored top-to-bottom).
fn flipped_vertical(src: PixmapRef<'_>) -> Pixmap {
    let w = src.width();
    let h = src.height();
    if w == 0 || h == 0 {
        return Pixmap::new(w, h);
    }
    let mut out = Pixmap::new(w, h);
    for (dst_row, src_row) in out.rows_mut().zip(src.rows().rev()) {
        dst_row.copy_from_slice(src_row);
    }
    out
}

/// Returns a new tightly packed [`Pixmap`] rotated 180 degrees.
fn rotated_180(src: PixmapRef<'_>) -> Pixmap {
    let w = src.width();
    let h = src.height();
    if w == 0 || h == 0 {
        return Pixmap::new(w, h);
    }
    let mut out = Pixmap::new(w, h);
    if let Some(src_data) = src.data() {
        let (dst_px, dst_rem) = out.data_mut().as_chunks_mut::<BYTES_PER_PIXEL>();
        let (src_px, src_rem) = src_data.as_chunks::<BYTES_PER_PIXEL>();
        debug_assert!(dst_rem.is_empty() && src_rem.is_empty());
        for (d, &s) in dst_px.iter_mut().zip(src_px.iter().rev()) {
            *d = s;
        }
        return out;
    }
    for (dst_row, src_row) in out.rows_mut().zip(src.rows().rev()) {
        let (dst_px, dst_rem) = dst_row.as_chunks_mut::<BYTES_PER_PIXEL>();
        let (src_px, src_rem) = src_row.as_chunks::<BYTES_PER_PIXEL>();
        debug_assert!(dst_rem.is_empty() && src_rem.is_empty());
        for (d, &s) in dst_px.iter_mut().zip(src_px.iter().rev()) {
            *d = s;
        }
    }
    out
}

/// Returns a new tightly packed [`Pixmap`] rotated 90 degrees clockwise
/// (`(x, y) -> (height - 1 - y, x)`).
fn rotated_90(src: PixmapRef<'_>) -> Pixmap {
    let src_w = src.width() as usize;
    let src_h = src.height() as usize;
    let dst_w = src_h;
    let dst_h = src_w;
    if src_w == 0 || src_h == 0 {
        return Pixmap::new(src.height(), src.width());
    }

    let mut data = vec![0u8; byte_len(src.height(), src.width())];
    let (dst_px, dst_rem) = data.as_chunks_mut::<BYTES_PER_PIXEL>();
    debug_assert!(dst_rem.is_empty());
    let src_data = src.raw_data();
    let src_stride = src.stride();

    let mut tile = [[0u8; 4]; TILE * TILE];
    for ty in (0..src_h).step_by(TILE) {
        let th = (src_h - ty).min(TILE);
        let dst_x0 = src_h - ty - th;
        for tx in (0..src_w).step_by(TILE) {
            let tw = (src_w - tx).min(TILE);
            for ly in 0..th {
                let row_start = (ty + ly) * src_stride + tx * BYTES_PER_PIXEL;
                let (src_px, _) = src_data[row_start..row_start + tw * BYTES_PER_PIXEL]
                    .as_chunks::<BYTES_PER_PIXEL>();
                let tile_col = th - 1 - ly;
                for (lx, &px) in src_px.iter().enumerate() {
                    tile[lx * TILE + tile_col] = px;
                }
            }
            for lx in 0..tw {
                let dst_start = (tx + lx) * dst_w + dst_x0;
                dst_px[dst_start..dst_start + th].copy_from_slice(&tile[lx * TILE..lx * TILE + th]);
            }
        }
    }

    debug_assert_eq!(dst_h, src_w);
    Pixmap::from_raw_parts(data, src.height(), src.width())
}

/// Returns a new tightly packed [`Pixmap`] rotated 270 degrees clockwise (90 degrees
/// counter-clockwise, `(x, y) -> (y, width - 1 - x)`).
fn rotated_270(src: PixmapRef<'_>) -> Pixmap {
    let src_w = src.width() as usize;
    let src_h = src.height() as usize;
    let dst_w = src_h;
    if src_w == 0 || src_h == 0 {
        return Pixmap::new(src.height(), src.width());
    }

    let mut data = vec![0u8; byte_len(src.height(), src.width())];
    let (dst_px, dst_rem) = data.as_chunks_mut::<BYTES_PER_PIXEL>();
    debug_assert!(dst_rem.is_empty());
    let src_data = src.raw_data();
    let src_stride = src.stride();

    let mut tile = [[0u8; 4]; TILE * TILE];
    for ty in (0..src_h).step_by(TILE) {
        let th = (src_h - ty).min(TILE);
        for tx in (0..src_w).step_by(TILE) {
            let tw = (src_w - tx).min(TILE);
            let dst_y0 = src_w - tx - tw;
            for ly in 0..th {
                let row_start = (ty + ly) * src_stride + tx * BYTES_PER_PIXEL;
                let (src_px, _) = src_data[row_start..row_start + tw * BYTES_PER_PIXEL]
                    .as_chunks::<BYTES_PER_PIXEL>();
                for (lx, &px) in src_px.iter().enumerate() {
                    tile[(tw - 1 - lx) * TILE + ly] = px;
                }
            }
            for row in 0..tw {
                let dst_start = (dst_y0 + row) * dst_w + ty;
                dst_px[dst_start..dst_start + th]
                    .copy_from_slice(&tile[row * TILE..row * TILE + th]);
            }
        }
    }

    Pixmap::from_raw_parts(data, src.height(), src.width())
}

/// Returns a new tightly packed [`Pixmap`] transposed across the main diagonal
/// (`(x, y) -> (y, x)`).
fn transposed(src: PixmapRef<'_>) -> Pixmap {
    let src_w = src.width() as usize;
    let src_h = src.height() as usize;
    let dst_w = src_h;
    if src_w == 0 || src_h == 0 {
        return Pixmap::new(src.height(), src.width());
    }

    let mut data = vec![0u8; byte_len(src.height(), src.width())];
    let (dst_px, dst_rem) = data.as_chunks_mut::<BYTES_PER_PIXEL>();
    debug_assert!(dst_rem.is_empty());
    let src_data = src.raw_data();
    let src_stride = src.stride();

    let mut tile = [[0u8; 4]; TILE * TILE];
    for ty in (0..src_h).step_by(TILE) {
        let th = (src_h - ty).min(TILE);
        for tx in (0..src_w).step_by(TILE) {
            let tw = (src_w - tx).min(TILE);
            for ly in 0..th {
                let row_start = (ty + ly) * src_stride + tx * BYTES_PER_PIXEL;
                let (src_px, _) = src_data[row_start..row_start + tw * BYTES_PER_PIXEL]
                    .as_chunks::<BYTES_PER_PIXEL>();
                for (lx, &px) in src_px.iter().enumerate() {
                    tile[lx * TILE + ly] = px;
                }
            }
            for lx in 0..tw {
                let dst_start = (tx + lx) * dst_w + ty;
                dst_px[dst_start..dst_start + th].copy_from_slice(&tile[lx * TILE..lx * TILE + th]);
            }
        }
    }

    Pixmap::from_raw_parts(data, src.height(), src.width())
}

fn padded_dimensions(
    width: u16,
    height: u16,
    top: u16,
    right: u16,
    bottom: u16,
    left: u16,
) -> (u16, u16) {
    let new_w = u16::try_from(u32::from(width) + u32::from(left) + u32::from(right))
        .expect("padded width overflows u16");
    let new_h = u16::try_from(u32::from(height) + u32::from(top) + u32::from(bottom))
        .expect("padded height overflows u16");
    (new_w, new_h)
}

/// Returns a new tightly packed [`Pixmap`] padded on each side with `fill`.
fn padded(
    src: PixmapRef<'_>,
    top: u16,
    right: u16,
    bottom: u16,
    left: u16,
    fill: [u8; 4],
) -> Pixmap {
    let (new_w, new_h) = padded_dimensions(src.width(), src.height(), top, right, bottom, left);
    if top == 0 && right == 0 && bottom == 0 && left == 0 {
        return src.to_owned();
    }
    let mut out = Pixmap::new(new_w, new_h);
    if src.is_empty() {
        if fill != [0, 0, 0, 0] {
            out.fill(fill);
        }
        return out;
    }

    let dst_stride = row_bytes(new_w);
    let left_bytes = row_bytes(left);
    let src_row_bytes = row_bytes(src.width());
    let top_bytes = top as usize * dst_stride;
    let mid_bytes = src.height() as usize * dst_stride;
    let data = out.data_mut();

    if fill != [0, 0, 0, 0] {
        fill_slice(&mut data[..top_bytes], fill);
        fill_slice(&mut data[top_bytes + mid_bytes..], fill);
    }

    for (dst_row, src_row) in data[top_bytes..top_bytes + mid_bytes]
        .chunks_exact_mut(dst_stride)
        .zip(src.rows())
    {
        if fill != [0, 0, 0, 0] {
            fill_slice(&mut dst_row[..left_bytes], fill);
            fill_slice(&mut dst_row[left_bytes + src_row_bytes..], fill);
        }
        dst_row[left_bytes..left_bytes + src_row_bytes].copy_from_slice(src_row);
    }

    out
}

/// Returns a new tightly packed [`Pixmap`] padded on each side by repeating (clamping) the
/// outermost edge and corner pixels of `src` (e.g. for atlas gutter extrusion).
///
/// If `src` is empty (`width == 0` or `height == 0`), the returned pixmap is filled with
/// transparent black (`[0, 0, 0, 0]`).
fn padded_clamped(src: PixmapRef<'_>, top: u16, right: u16, bottom: u16, left: u16) -> Pixmap {
    let (new_w, new_h) = padded_dimensions(src.width(), src.height(), top, right, bottom, left);
    if top == 0 && right == 0 && bottom == 0 && left == 0 {
        return src.to_owned();
    }
    let mut out = Pixmap::new(new_w, new_h);
    if src.is_empty() {
        return out;
    }

    let dst_stride = row_bytes(new_w);
    let left_bytes = row_bytes(left);
    let src_row_bytes = row_bytes(src.width());
    let top_bytes = top as usize * dst_stride;
    let mid_bytes = src.height() as usize * dst_stride;
    let data = out.data_mut();

    for (dst_row, src_row) in data[top_bytes..top_bytes + mid_bytes]
        .chunks_exact_mut(dst_stride)
        .zip(src.rows())
    {
        let first_px: [u8; 4] = src_row[..BYTES_PER_PIXEL].try_into().unwrap();
        let last_px: [u8; 4] = src_row[src_row_bytes - BYTES_PER_PIXEL..]
            .try_into()
            .unwrap();
        if left > 0 {
            fill_slice(&mut dst_row[..left_bytes], first_px);
        }
        dst_row[left_bytes..left_bytes + src_row_bytes].copy_from_slice(src_row);
        if right > 0 {
            fill_slice(&mut dst_row[left_bytes + src_row_bytes..], last_px);
        }
    }

    if top > 0 {
        let (top_part, rest) = data.split_at_mut(top_bytes);
        let first_row = &rest[..dst_stride];
        for dst_row in top_part.chunks_exact_mut(dst_stride) {
            dst_row.copy_from_slice(first_row);
        }
    }

    if bottom > 0 {
        let bottom_start = top_bytes + mid_bytes;
        let (head, bottom_part) = data.split_at_mut(bottom_start);
        let last_row = &head[bottom_start - dst_stride..bottom_start];
        for dst_row in bottom_part.chunks_exact_mut(dst_stride) {
            dst_row.copy_from_slice(last_row);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn numbered(width: u16, height: u16) -> Pixmap {
        let mut data = Vec::with_capacity(byte_len(width, height));
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&[
                    (x & 0xFF) as u8,
                    (y & 0xFF) as u8,
                    ((x.wrapping_mul(7) ^ y.wrapping_mul(13)) & 0xFF) as u8,
                    255,
                ]);
            }
        }
        Pixmap::from_parts(data, width, height)
    }

    #[test]
    fn flip_horizontal_and_vertical_on_owned_and_strided() {
        let orig = numbered(5, 3);

        let fh = orig.flipped_horizontal();
        assert_eq!((fh.width(), fh.height()), (5, 3));
        for y in 0..3 {
            for x in 0..5 {
                assert_eq!(fh.pixel(x, y), orig.pixel(4 - x, y));
            }
        }
        let mut in_place = orig.clone();
        in_place.flip_horizontal();
        assert_eq!(in_place, fh);
        in_place.flip_horizontal();
        assert_eq!(in_place, orig);

        let fv = orig.flipped_vertical();
        assert_eq!((fv.width(), fv.height()), (5, 3));
        for y in 0..3 {
            for x in 0..5 {
                assert_eq!(fv.pixel(x, y), orig.pixel(x, 2 - y));
            }
        }
        let mut in_place_v = orig.clone();
        in_place_v.flip_vertical();
        assert_eq!(in_place_v, fv);
        in_place_v.flip_vertical();
        assert_eq!(in_place_v, orig);

        // Strided view with padding bytes must preserve padding during in-place flips.
        let mut buf = vec![0xEEu8; 2 * 24 + 12];
        let mut view = PixmapMut::from_parts_with_stride(&mut buf, 3, 3, 24);
        view.copy_from(orig.subregion(1, 0, 3, 3), 0, 0);
        let expected_h = view.flipped_horizontal();
        view.flip_horizontal();
        assert_eq!(view.as_ref(), expected_h.as_ref());
        assert_eq!(&view.raw_data()[12..24], &[0xEE; 12]);
        assert_eq!(&view.raw_data()[36..48], &[0xEE; 12]);

        let expected_v = view.flipped_vertical();
        view.flip_vertical();
        assert_eq!(view.as_ref(), expected_v.as_ref());
        assert_eq!(&view.raw_data()[12..24], &[0xEE; 12]);
        assert_eq!(&view.raw_data()[36..48], &[0xEE; 12]);
    }

    #[test]
    fn rotations_and_transpose_match_coordinates_and_group_laws() {
        // Test both small odd dimensions and multi-tile dimensions (> TILE = 16).
        for &(w, h) in &[(1, 1), (3, 2), (2, 5), (16, 16), (37, 23)] {
            let orig = numbered(w, h);

            let r90 = orig.rotated_90();
            assert_eq!((r90.width(), r90.height()), (h, w));
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(r90.pixel(h - 1 - y, x), orig.pixel(x, y));
                }
            }

            let r180 = orig.rotated_180();
            assert_eq!((r180.width(), r180.height()), (w, h));
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(r180.pixel(w - 1 - x, h - 1 - y), orig.pixel(x, y));
                }
            }
            let mut r180_in_place = orig.clone();
            r180_in_place.rotate_180();
            assert_eq!(r180_in_place, r180);

            let r270 = orig.rotated_270();
            assert_eq!((r270.width(), r270.height()), (h, w));
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(r270.pixel(y, w - 1 - x), orig.pixel(x, y));
                }
            }

            let tr = orig.transposed();
            assert_eq!((tr.width(), tr.height()), (h, w));
            for y in 0..h {
                for x in 0..w {
                    assert_eq!(tr.pixel(y, x), orig.pixel(x, y));
                }
            }

            // Dihedral group identities.
            assert_eq!(r90.rotated_90(), r180);
            assert_eq!(r180.rotated_90(), r270);
            assert_eq!(r270.rotated_90(), orig);
            assert_eq!(tr.transposed(), orig);
            assert_eq!(tr.flipped_horizontal(), r90);
            assert_eq!(tr.flipped_vertical(), r270);
        }
    }

    #[test]
    fn strided_subregion_rotations_and_rotate_180_in_place_preserve_padding() {
        let mut parent = numbered(40, 30);
        let sub_owned = parent.subregion(5, 4, 21, 19).to_owned();
        let sub = parent.subregion(5, 4, 21, 19);

        assert_eq!(sub.rotated_90(), sub_owned.rotated_90());
        assert_eq!(sub.rotated_180(), sub_owned.rotated_180());
        assert_eq!(sub.rotated_270(), sub_owned.rotated_270());
        assert_eq!(sub.transposed(), sub_owned.transposed());

        // In-place rotate_180 on strided mutable subregion (odd height = 19, and even height = 4).
        let before = parent.clone();
        let expected_sub = sub_owned.rotated_180();
        parent.subregion_mut(5, 4, 21, 19).rotate_180();
        assert_eq!(parent.subregion(5, 4, 21, 19), expected_sub.as_ref());
        // Surrounding pixels untouched.
        assert_eq!(parent.pixel(4, 4), before.pixel(4, 4));
        assert_eq!(parent.pixel(26, 22), before.pixel(26, 22));
        assert_eq!(parent.pixel(5, 3), before.pixel(5, 3));
        assert_eq!(parent.pixel(5, 23), before.pixel(5, 23));

        // Even-height strided rotate_180.
        let expected_even = parent.subregion(1, 1, 5, 4).rotated_180();
        parent.subregion_mut(1, 1, 5, 4).rotate_180();
        assert_eq!(parent.subregion(1, 1, 5, 4), expected_even.as_ref());
    }

    #[test]
    fn empty_dimensions_are_handled_across_transforms() {
        let zero_w = Pixmap::new(0, 5);
        assert_eq!(
            (
                zero_w.flipped_horizontal().width(),
                zero_w.flipped_horizontal().height()
            ),
            (0, 5)
        );
        assert_eq!(
            (
                zero_w.flipped_vertical().width(),
                zero_w.flipped_vertical().height()
            ),
            (0, 5)
        );
        assert_eq!(
            (zero_w.rotated_180().width(), zero_w.rotated_180().height()),
            (0, 5)
        );
        assert_eq!(
            (zero_w.rotated_90().width(), zero_w.rotated_90().height()),
            (5, 0)
        );
        assert_eq!(
            (zero_w.rotated_270().width(), zero_w.rotated_270().height()),
            (5, 0)
        );
        assert_eq!(
            (zero_w.transposed().width(), zero_w.transposed().height()),
            (5, 0)
        );

        let mut mut_empty = zero_w.clone();
        mut_empty.flip_horizontal();
        mut_empty.flip_vertical();
        mut_empty.rotate_180();
        assert_eq!(mut_empty, zero_w);

        let zero_h = Pixmap::new(4, 0);
        assert_eq!(
            (zero_h.rotated_90().width(), zero_h.rotated_90().height()),
            (0, 4)
        );
    }

    #[test]
    fn padded_and_padded_clamped_work_with_fill_edges_and_empty_inputs() {
        let orig = numbered(3, 2);
        let fill = [10, 20, 30, 40];

        let p = orig.padded(1, 2, 3, 4, fill);
        assert_eq!((p.width(), p.height()), (9, 6));
        for y in 0..6 {
            for x in 0..9 {
                let expected = if (4..7).contains(&x) && (1..3).contains(&y) {
                    orig.pixel(x - 4, y - 1)
                } else {
                    fill
                };
                assert_eq!(p.pixel(x, y), expected, "padded mismatch at ({x}, {y})");
            }
        }

        // Transparent fill fast path.
        let p_zero = orig.padded(1, 1, 1, 1, [0, 0, 0, 0]);
        assert_eq!(p_zero.pixel(0, 0), [0, 0, 0, 0]);
        assert_eq!(p_zero.pixel(1, 1), orig.pixel(0, 0));
        assert_eq!(p_zero.pixel(4, 3), [0, 0, 0, 0]);

        // Zero padding is an exact copy.
        assert_eq!(orig.padded(0, 0, 0, 0, fill), orig);
        assert_eq!(orig.padded_clamped(0, 0, 0, 0), orig);

        // Edge-clamped padding repeats borders and corners.
        let clamped = orig.padded_clamped(2, 3, 1, 2);
        assert_eq!((clamped.width(), clamped.height()), (8, 5));
        for y in 0..5u16 {
            for x in 0..8u16 {
                let sx = x.saturating_sub(2).min(2);
                let sy = y.saturating_sub(2).min(1);
                assert_eq!(
                    clamped.pixel(x, y),
                    orig.pixel(sx, sy),
                    "clamped mismatch at ({x}, {y})"
                );
            }
        }

        // Strided subregion padding matches owned padding.
        let parent = numbered(6, 5);
        assert_eq!(
            parent.subregion(1, 1, 3, 2).padded_clamped(1, 2, 1, 2),
            parent
                .subregion(1, 1, 3, 2)
                .to_owned()
                .padded_clamped(1, 2, 1, 2)
        );

        // Empty source padding.
        let empty = Pixmap::new(0, 0);
        let padded_empty = empty.padded(1, 2, 1, 1, fill);
        assert_eq!((padded_empty.width(), padded_empty.height()), (3, 2));
        assert_eq!(padded_empty.pixel(1, 1), fill);
        let clamped_empty = empty.padded_clamped(1, 2, 1, 1);
        assert_eq!(clamped_empty, Pixmap::new(3, 2));
    }

    #[test]
    #[should_panic(expected = "padded width overflows u16")]
    fn padded_panics_on_width_overflow() {
        let p = Pixmap::new(10, 1);
        let _ = p.padded(0, u16::MAX, 0, 0, [0; 4]);
    }

    #[test]
    #[should_panic(expected = "padded height overflows u16")]
    fn padded_clamped_panics_on_height_overflow() {
        let p = Pixmap::new(1, 10);
        let _ = p.padded_clamped(u16::MAX, 0, 0, 0);
    }
}
