use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;
use core::slice::ChunksExact;

use crate::util::{
    as_pixels, byte_len, is_contiguous, is_valid_packed_parts, is_valid_strided_parts, row_bytes,
    slice_is_opaque, subregion_byte_range, validate_packed_parts, validate_strided_parts,
};
use crate::{BYTES_PER_PIXEL, Pixmap, PixmapMut};

/// An immutably borrowed buffer of RGBA8 pixels with rows stored top to bottom and an explicit
/// row stride in bytes.
///
/// The pixel layout is the same as [`Pixmap`]'s, except consecutive rows may be separated by
/// padding bytes (such as when borrowing a subregion or a row-aligned GPU buffer).
#[derive(Clone, Copy)]
pub struct PixmapRef<'a> {
    width: u16,
    height: u16,
    stride: usize,
    data: &'a [u8],
}

impl<'a> PixmapRef<'a> {
    pub(crate) fn from_raw_parts(data: &'a [u8], width: u16, height: u16, stride: usize) -> Self {
        Self {
            width,
            height,
            stride,
            data,
        }
    }

    /// Creates a tightly packed view borrowing existing pixel data.
    ///
    /// # Panics
    ///
    /// Panics if `width * height * 4` overflows `usize` or `data.len()` does not match.
    pub fn from_parts(data: &'a [u8], width: u16, height: u16) -> Self {
        validate_packed_parts(data.len(), width, height);
        Self {
            width,
            height,
            stride: row_bytes(width),
            data,
        }
    }

    /// Creates a tightly packed view borrowing existing pixel data, returning `None` if
    /// `width * height * 4` overflows `usize` or `data.len()` does not match.
    pub fn try_from_parts(data: &'a [u8], width: u16, height: u16) -> Option<Self> {
        if !is_valid_packed_parts(data.len(), width, height) {
            return None;
        }
        Some(Self {
            width,
            height,
            stride: row_bytes(width),
            data,
        })
    }

    /// Creates a strided view borrowing existing pixel data.
    ///
    /// `stride` is the distance in bytes between the start of consecutive rows. The last row
    /// only requires `width * 4` bytes, so `data.len()` must be at least
    /// `(height - 1) * stride + width * 4` (or `0` when `height == 0`).
    ///
    /// # Panics
    ///
    /// Panics if `stride < width * 4` or if `data` is shorter than the required byte length.
    pub fn from_parts_with_stride(data: &'a [u8], width: u16, height: u16, stride: usize) -> Self {
        validate_strided_parts(data.len(), width, height, stride);
        Self {
            width,
            height,
            stride,
            data,
        }
    }

    /// Creates a strided view borrowing existing pixel data, returning `None` if
    /// `stride < width * 4`, if the required byte length overflows `usize`, or if `data` is
    /// shorter than `(height - 1) * stride + width * 4` (or `0` when `height == 0`).
    pub fn try_from_parts_with_stride(
        data: &'a [u8],
        width: u16,
        height: u16,
        stride: usize,
    ) -> Option<Self> {
        if !is_valid_strided_parts(data.len(), width, height, stride) {
            return None;
        }
        Some(Self {
            width,
            height,
            stride,
            data,
        })
    }

    /// The width in pixels.
    pub fn width(&self) -> u16 {
        self.width
    }

    /// The height in pixels.
    pub fn height(&self) -> u16 {
        self.height
    }

    /// The distance in bytes between the start of consecutive rows.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Returns `true` if either dimension is zero.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// Returns `true` if every active pixel has alpha `255` (vacuously `true` when empty),
    /// ignoring any inter-row stride padding.
    pub fn is_opaque(&self) -> bool {
        if let Some(data) = self.data() {
            slice_is_opaque(data)
        } else {
            self.rows().all(slice_is_opaque)
        }
    }

    /// Returns the premultiplied `[r, g, b, a]` pixel at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width()` or `y >= self.height()`.
    pub fn pixel(&self, x: u16, y: u16) -> [u8; 4] {
        assert!(
            x < self.width && y < self.height,
            "pixel ({x}, {y}) out of bounds for {}x{} pixmap",
            self.width,
            self.height
        );
        let start = y as usize * self.stride + row_bytes(x);
        self.data[start..start + BYTES_PER_PIXEL]
            .try_into()
            .unwrap()
    }

    /// Returns the premultiplied `[r, g, b, a]` pixel at `(x, y)`, or `None` if out of bounds.
    pub fn get_pixel(&self, x: u16, y: u16) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let start = y as usize * self.stride + row_bytes(x);
        Some(
            self.data[start..start + BYTES_PER_PIXEL]
                .try_into()
                .unwrap(),
        )
    }

    /// The number of bytes of active pixel data (`width * height * 4`), excluding stride padding.
    pub fn byte_len(&self) -> usize {
        byte_len(self.width, self.height)
    }

    /// Returns `true` if all pixel rows are tightly packed in memory with no inter-row gaps.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(self.width, self.height, self.stride)
    }

    /// Returns the pixel bytes as a single contiguous slice (length `width * height * 4`) if
    /// [`is_contiguous`](Self::is_contiguous) is `true`, or `None` if rows have inter-row gaps.
    pub fn data(&self) -> Option<&'a [u8]> {
        self.is_contiguous().then(|| &self.data[..self.byte_len()])
    }

    /// Returns the pixels as a single contiguous slice of premultiplied `[r, g, b, a]` values
    /// (length `width * height`) if [`is_contiguous`](Self::is_contiguous) is `true`, or `None`
    /// if rows have inter-row gaps.
    #[inline]
    pub fn pixels(&self) -> Option<&'a [[u8; 4]]> {
        self.data().map(as_pixels)
    }

    /// Returns the underlying borrowed slice, which may include inter-row stride padding or
    /// trailing bytes beyond the active pixels.
    pub fn raw_data(&self) -> &'a [u8] {
        self.data
    }

    /// Returns the pixel bytes of row `y` (length `width * 4`).
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    pub fn row(&self, y: u16) -> &'a [u8] {
        assert!(
            y < self.height,
            "row index {y} out of bounds for height {}",
            self.height
        );
        let start = y as usize * self.stride;
        &self.data[start..start + row_bytes(self.width)]
    }

    /// Returns the premultiplied `[r, g, b, a]` pixels of row `y` (length `width`).
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    #[inline]
    pub fn row_pixels(&self, y: u16) -> &'a [[u8; 4]] {
        as_pixels(self.row(y))
    }

    /// Iterates over the rows of pixels from top to bottom.
    #[inline]
    pub fn rows(&self) -> Rows<'a> {
        Rows::new(self.data, self.width, self.height, self.stride)
    }

    /// Returns a view of a rectangular subregion, without copying.
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn subregion(&self, x: u16, y: u16, width: u16, height: u16) -> PixmapRef<'a> {
        let range = subregion_byte_range(self.width, self.height, self.stride, x, y, width, height);
        PixmapRef {
            width,
            height,
            stride: self.stride,
            data: &self.data[range],
        }
    }

    /// Copies the pixels into a new tightly packed owned [`Pixmap`].
    pub fn to_owned(&self) -> Pixmap {
        if let Some(bytes) = self.data() {
            return Pixmap::from_raw_parts(bytes.to_vec(), self.width, self.height);
        }
        let mut data = Vec::with_capacity(self.byte_len());
        for row in self.rows() {
            data.extend_from_slice(row);
        }
        Pixmap::from_raw_parts(data, self.width, self.height)
    }

    /// Copies the active pixels into a tightly packed straight-alpha RGBA8 buffer.
    pub fn to_unpremultiplied(&self) -> Vec<u8> {
        self.to_owned().into_unpremultiplied()
    }
}

/// An iterator over the rows of a pixmap, from top to bottom.
///
/// Each item is a slice of `width * 4` pixel bytes, excluding any inter-row stride padding.
#[derive(Clone, Debug)]
pub struct Rows<'a> {
    inner: RowsInner<'a>,
}

#[derive(Clone, Debug)]
enum RowsInner<'a> {
    /// Non-empty rows (`row_bytes > 0`, so `stride >= row_bytes > 0`).
    Strided {
        head: ChunksExact<'a, u8>,
        last: Option<&'a [u8]>,
        row_bytes: usize,
    },
    /// Zero-width or zero-height pixmap: yields `remaining` empty slices without touching `data`.
    Empty(usize),
}

impl<'a> Rows<'a> {
    #[inline]
    fn new(data: &'a [u8], width: u16, height: u16, stride: usize) -> Self {
        let row_len = row_bytes(width);
        let remaining = height as usize;
        if remaining == 0 || row_len == 0 {
            return Self {
                inner: RowsInner::Empty(remaining),
            };
        }
        // When the backing slice includes trailing padding on the final row (always true for
        // tightly packed pixmaps and full-stride buffers), every row fits in `ChunksExact`.
        if let Some(full_len) = remaining.checked_mul(stride)
            && full_len <= data.len()
        {
            return Self {
                inner: RowsInner::Strided {
                    head: data[..full_len].chunks_exact(stride),
                    last: None,
                    row_bytes: row_len,
                },
            };
        }
        let head_len = (remaining - 1) * stride;
        let (head_bytes, tail) = data[..head_len + row_len].split_at(head_len);
        Self {
            inner: RowsInner::Strided {
                head: head_bytes.chunks_exact(stride),
                last: Some(tail),
                row_bytes: row_len,
            },
        }
    }
}

impl<'a> Iterator for Rows<'a> {
    type Item = &'a [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            RowsInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                if let Some(chunk) = head.next() {
                    Some(&chunk[..*row_bytes])
                } else {
                    last.take()
                }
            }
            RowsInner::Empty(remaining) => {
                *remaining = remaining.checked_sub(1)?;
                Some(&[])
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    #[inline]
    fn count(self) -> usize {
        self.len()
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        match &mut self.inner {
            RowsInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                let head_len = head.len();
                if n < head_len {
                    head.nth(n).map(|chunk| &chunk[..*row_bytes])
                } else {
                    if head_len > 0 {
                        head.nth(head_len);
                    }
                    if n == head_len {
                        last.take()
                    } else {
                        *last = None;
                        None
                    }
                }
            }
            RowsInner::Empty(remaining) => {
                if n < *remaining {
                    *remaining -= n + 1;
                    Some(&[])
                } else {
                    *remaining = 0;
                    None
                }
            }
        }
    }

    #[inline]
    fn fold<B, F>(self, mut acc: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self.inner {
            RowsInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                acc = head.fold(acc, |acc, chunk| f(acc, &chunk[..row_bytes]));
                if let Some(row) = last {
                    acc = f(acc, row);
                }
                acc
            }
            RowsInner::Empty(remaining) => {
                for _ in 0..remaining {
                    acc = f(acc, &[]);
                }
                acc
            }
        }
    }
}

impl DoubleEndedIterator for Rows<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            RowsInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                if let Some(row) = last.take() {
                    Some(row)
                } else {
                    head.next_back().map(|chunk| &chunk[..*row_bytes])
                }
            }
            RowsInner::Empty(remaining) => {
                *remaining = remaining.checked_sub(1)?;
                Some(&[])
            }
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        match &mut self.inner {
            RowsInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                if let Some(row) = last.take() {
                    if n == 0 {
                        Some(row)
                    } else {
                        head.nth_back(n - 1).map(|chunk| &chunk[..*row_bytes])
                    }
                } else {
                    head.nth_back(n).map(|chunk| &chunk[..*row_bytes])
                }
            }
            RowsInner::Empty(remaining) => {
                if n < *remaining {
                    *remaining -= n + 1;
                    Some(&[])
                } else {
                    *remaining = 0;
                    None
                }
            }
        }
    }

    #[inline]
    fn rfold<B, F>(self, mut acc: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        match self.inner {
            RowsInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                if let Some(row) = last {
                    acc = f(acc, row);
                }
                head.rfold(acc, |acc, chunk| f(acc, &chunk[..row_bytes]))
            }
            RowsInner::Empty(remaining) => {
                for _ in 0..remaining {
                    acc = f(acc, &[]);
                }
                acc
            }
        }
    }
}

impl ExactSizeIterator for Rows<'_> {
    #[inline]
    fn len(&self) -> usize {
        match &self.inner {
            RowsInner::Strided { head, last, .. } => head.len() + usize::from(last.is_some()),
            RowsInner::Empty(remaining) => *remaining,
        }
    }
}

impl FusedIterator for Rows<'_> {}

impl<'a> From<&'a Pixmap> for PixmapRef<'a> {
    fn from(pixmap: &'a Pixmap) -> Self {
        pixmap.as_ref()
    }
}

impl<'a> From<PixmapMut<'a>> for PixmapRef<'a> {
    fn from(pixmap: PixmapMut<'a>) -> Self {
        pixmap.into_ref()
    }
}

impl<'a> From<&'a PixmapMut<'_>> for PixmapRef<'a> {
    fn from(pixmap: &'a PixmapMut<'_>) -> Self {
        pixmap.as_ref()
    }
}

impl PartialEq for PixmapRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.width != other.width || self.height != other.height {
            return false;
        }
        if let (Some(a), Some(b)) = (self.data(), other.data()) {
            return a == b;
        }
        self.rows().eq(other.rows())
    }
}

impl Eq for PixmapRef<'_> {}

impl fmt::Debug for PixmapRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PixmapRef")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::numbered_pixmap;
    use alloc::vec;

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
    #[should_panic(expected = "out of bounds")]
    fn subregion_rejects_out_of_bounds_and_u16_overflow() {
        let pixmap = Pixmap::new(4, 4);
        let _ = pixmap.subregion(u16::MAX, 0, 2, 1);
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
    fn try_from_parts_and_strided_validation() {
        let data = [1, 2, 3, 255, 4, 5, 6, 255];
        let view = PixmapRef::try_from_parts(&data, 2, 1).unwrap();
        assert_eq!(view.pixels(), Some(&[[1, 2, 3, 255], [4, 5, 6, 255]][..]));
        assert_eq!(view.get_pixel(1, 0), Some([4, 5, 6, 255]));
        assert_eq!(view.get_pixel(2, 0), None);
        assert_eq!(view.get_pixel(0, 1), None);

        assert!(PixmapRef::try_from_parts(&data[..7], 2, 1).is_none());
        assert!(PixmapRef::try_from_parts(&[], 0, 3).is_some());

        // Strided 2x2 with stride 16 and trimmed last row (16 + 8 = 24 bytes).
        let mut buf = vec![0xAA_u8; 24];
        buf[0..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        buf[16..24].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);

        let strided = PixmapRef::try_from_parts_with_stride(&buf, 2, 2, 16).unwrap();
        assert_eq!(strided.pixels(), None);
        assert_eq!(strided.row_pixels(0), &[[1, 2, 3, 4], [5, 6, 7, 8]]);
        assert_eq!(strided.row_pixels(1), &[[9, 10, 11, 12], [13, 14, 15, 16]]);
        assert_eq!(strided.get_pixel(1, 1), Some([13, 14, 15, 16]));
        assert_eq!(strided.get_pixel(2, 1), None);

        // Single-row strided view is contiguous, so `pixels()` succeeds without trailing bytes.
        let single = PixmapRef::try_from_parts_with_stride(&buf, 2, 1, 16).unwrap();
        assert_eq!(single.pixels(), Some(&[[1, 2, 3, 4], [5, 6, 7, 8]][..]));

        // Error cases: stride < width*4, buffer too short, stride overflow.
        assert!(PixmapRef::try_from_parts_with_stride(&buf, 2, 2, 4).is_none());
        assert!(PixmapRef::try_from_parts_with_stride(&buf[..23], 2, 2, 16).is_none());
        assert!(PixmapRef::try_from_parts_with_stride(&buf, 1, 2, usize::MAX).is_none());
        // Zero height needs 0 bytes as long as stride >= width*4.
        assert!(PixmapRef::try_from_parts_with_stride(&[], 2, 0, 8).is_some());
        assert!(PixmapRef::try_from_parts_with_stride(&[], 2, 0, 4).is_none());
    }
}
