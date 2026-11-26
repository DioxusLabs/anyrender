use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;
use core::slice::ChunksExactMut;

use crate::blit::clip_blit;
use crate::util::{
    as_pixels, as_pixels_mut, byte_len, fill_slice, is_contiguous, is_valid_packed_parts,
    is_valid_strided_parts, row_bytes, subregion_byte_range, validate_packed_parts,
    validate_strided_parts,
};
use crate::{
    BYTES_PER_PIXEL, Pixmap, PixmapRef, Rows, multiply_alpha_rgba8, premultiply_rgba8,
    swap_rb_rgba8, unpremultiply_rgba8,
};

/// A mutably borrowed buffer of RGBA8 pixels with rows stored top to bottom and an explicit
/// row stride in bytes.
///
/// The pixel layout is the same as [`Pixmap`]'s, except consecutive rows may be separated by
/// padding bytes (such as when borrowing a subregion or a row-aligned GPU buffer). This lets
/// code write pixels into a buffer it does not own, such as another library's image.
pub struct PixmapMut<'a> {
    width: u16,
    height: u16,
    stride: usize,
    data: &'a mut [u8],
}

impl<'a> PixmapMut<'a> {
    pub(crate) fn from_raw_parts(
        data: &'a mut [u8],
        width: u16,
        height: u16,
        stride: usize,
    ) -> Self {
        Self {
            width,
            height,
            stride,
            data,
        }
    }

    /// Creates a tightly packed mutable view borrowing existing pixel data.
    ///
    /// # Panics
    ///
    /// Panics if `width * height * 4` overflows `usize` or `data.len()` does not match.
    pub fn from_parts(data: &'a mut [u8], width: u16, height: u16) -> Self {
        validate_packed_parts(data.len(), width, height);
        Self {
            width,
            height,
            stride: row_bytes(width),
            data,
        }
    }

    /// Creates a tightly packed mutable view borrowing existing pixel data, returning `None` if
    /// `width * height * 4` overflows `usize` or `data.len()` does not match.
    pub fn try_from_parts(data: &'a mut [u8], width: u16, height: u16) -> Option<Self> {
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

    /// Creates a strided mutable view borrowing existing pixel data.
    ///
    /// `stride` is the distance in bytes between the start of consecutive rows. The last row
    /// only requires `width * 4` bytes, so `data.len()` must be at least
    /// `(height - 1) * stride + width * 4` (or `0` when `height == 0`).
    ///
    /// # Panics
    ///
    /// Panics if `stride < width * 4` or if `data` is shorter than the required byte length.
    pub fn from_parts_with_stride(
        data: &'a mut [u8],
        width: u16,
        height: u16,
        stride: usize,
    ) -> Self {
        validate_strided_parts(data.len(), width, height, stride);
        Self {
            width,
            height,
            stride,
            data,
        }
    }

    /// Creates a strided mutable view borrowing existing pixel data, returning `None` if
    /// `stride < width * 4`, if the required byte length overflows `usize`, or if `data` is
    /// shorter than `(height - 1) * stride + width * 4` (or `0` when `height == 0`).
    pub fn try_from_parts_with_stride(
        data: &'a mut [u8],
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
        self.as_ref().is_opaque()
    }

    /// Returns the premultiplied `[r, g, b, a]` pixel at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width()` or `y >= self.height()`.
    pub fn pixel(&self, x: u16, y: u16) -> [u8; 4] {
        self.as_ref().pixel(x, y)
    }

    /// Returns the premultiplied `[r, g, b, a]` pixel at `(x, y)`, or `None` if out of bounds.
    pub fn get_pixel(&self, x: u16, y: u16) -> Option<[u8; 4]> {
        self.as_ref().get_pixel(x, y)
    }

    /// Sets the premultiplied `[r, g, b, a]` pixel at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width()` or `y >= self.height()`.
    pub fn set_pixel(&mut self, x: u16, y: u16, color: [u8; 4]) {
        assert!(
            x < self.width && y < self.height,
            "pixel ({x}, {y}) out of bounds for {}x{} pixmap",
            self.width,
            self.height
        );
        let start = y as usize * self.stride + row_bytes(x);
        self.data[start..start + BYTES_PER_PIXEL].copy_from_slice(&color);
    }

    /// Sets every active pixel to transparent black (`[0, 0, 0, 0]`), leaving any inter-row
    /// stride padding untouched.
    pub fn clear(&mut self) {
        if let Some(data) = self.data_mut() {
            data.fill(0);
        } else {
            for row in self.rows_mut() {
                row.fill(0);
            }
        }
    }

    /// Sets every active pixel in a rectangular region to transparent black (`[0, 0, 0, 0]`).
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn clear_rect(&mut self, x: u16, y: u16, width: u16, height: u16) {
        self.subregion_mut(x, y, width, height).clear();
    }

    /// Sets every active pixel to the given premultiplied `[r, g, b, a]` colour, leaving any
    /// inter-row stride padding untouched.
    pub fn fill(&mut self, color: [u8; 4]) {
        if let Some(data) = self.data_mut() {
            fill_slice(data, color);
        } else {
            for row in self.rows_mut() {
                fill_slice(row, color);
            }
        }
    }

    /// Sets every active pixel in a rectangular region to the given premultiplied `[r, g, b, a]`
    /// colour.
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn fill_rect(&mut self, x: u16, y: u16, width: u16, height: u16, color: [u8; 4]) {
        self.subregion_mut(x, y, width, height).fill(color);
    }

    /// Premultiplies the colour channels of every active pixel by alpha in place, leaving any
    /// inter-row stride padding untouched.
    pub fn premultiply(&mut self) {
        if let Some(data) = self.data_mut() {
            premultiply_rgba8(data);
        } else {
            for row in self.rows_mut() {
                premultiply_rgba8(row);
            }
        }
    }

    /// Unpremultiplies the colour channels of every active pixel to straight alpha in place,
    /// leaving any inter-row stride padding untouched.
    ///
    /// Pixels with alpha `0` become `[0, 0, 0, 0]`, and any colour channel exceeding alpha is
    /// clamped to `255`.
    pub fn unpremultiply(&mut self) {
        if let Some(data) = self.data_mut() {
            unpremultiply_rgba8(data);
        } else {
            for row in self.rows_mut() {
                unpremultiply_rgba8(row);
            }
        }
    }

    /// Multiplies every channel of every active pixel by `alpha / 255` in place, using
    /// nearest-integer rounding and leaving any inter-row stride padding untouched.
    pub fn multiply_alpha(&mut self, alpha: u8) {
        if alpha == 255 {
            return;
        }
        if alpha == 0 {
            self.clear();
            return;
        }
        if let Some(data) = self.data_mut() {
            multiply_alpha_rgba8(data, alpha);
        } else {
            for row in self.rows_mut() {
                multiply_alpha_rgba8(row, alpha);
            }
        }
    }

    /// Swaps the red and blue channels of every active pixel in place (RGBA <-> BGRA), leaving
    /// any inter-row stride padding untouched.
    pub fn swap_rb(&mut self) {
        if let Some(data) = self.data_mut() {
            swap_rb_rgba8(data);
        } else {
            for row in self.rows_mut() {
                swap_rb_rgba8(row);
            }
        }
    }

    /// Copies pixels from `src` into `self` with `src`'s top-left corner placed at `(x, y)`,
    /// leaving any inter-row stride padding untouched.
    ///
    /// Source pixels are written verbatim (overwriting destination pixels without blending). Any
    /// part of `src` falling outside `self` is clipped; if there is no overlap, `self` is left
    /// unchanged.
    pub fn copy_from(&mut self, src: PixmapRef<'_>, x: i32, y: i32) {
        let Some(rect) = clip_blit(self.width, self.height, src.width(), src.height(), x, y) else {
            return;
        };
        let src_region = src.subregion(rect.src_x, rect.src_y, rect.width, rect.height);
        let mut dst_region = self.subregion_mut(rect.dst_x, rect.dst_y, rect.width, rect.height);
        if let (Some(dst_data), Some(src_data)) = (dst_region.data_mut(), src_region.data()) {
            dst_data.copy_from_slice(src_data);
        } else {
            for (dst_row, src_row) in dst_region.rows_mut().zip(src_region.rows()) {
                dst_row.copy_from_slice(src_row);
            }
        }
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
    pub fn data(&self) -> Option<&[u8]> {
        self.is_contiguous().then(|| &self.data[..self.byte_len()])
    }

    /// Returns the pixel bytes mutably as a single contiguous slice (length `width * height * 4`)
    /// if [`is_contiguous`](Self::is_contiguous) is `true`, or `None` if rows have inter-row gaps.
    pub fn data_mut(&mut self) -> Option<&mut [u8]> {
        let len = self.byte_len();
        self.is_contiguous().then(|| &mut self.data[..len])
    }

    /// Returns the pixels as a single contiguous slice of premultiplied `[r, g, b, a]` values
    /// (length `width * height`) if [`is_contiguous`](Self::is_contiguous) is `true`, or `None`
    /// if rows have inter-row gaps.
    #[inline]
    pub fn pixels(&self) -> Option<&[[u8; 4]]> {
        self.data().map(as_pixels)
    }

    /// Returns the pixels mutably as a single contiguous slice of premultiplied `[r, g, b, a]`
    /// values (length `width * height`) if [`is_contiguous`](Self::is_contiguous) is `true`, or
    /// `None` if rows have inter-row gaps.
    #[inline]
    pub fn pixels_mut(&mut self) -> Option<&mut [[u8; 4]]> {
        self.data_mut().map(as_pixels_mut)
    }

    /// Returns the underlying borrowed slice, which may include inter-row stride padding or
    /// trailing bytes beyond the active pixels.
    pub fn raw_data(&self) -> &[u8] {
        self.data
    }

    /// Returns the underlying borrowed slice mutably.
    ///
    /// For strided views (including subregions), this slice includes bytes between rows that
    /// lie outside this view's active pixels.
    pub fn raw_data_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Consumes the view and returns the underlying borrowed slice without copying.
    ///
    /// For strided views, this slice includes any inter-row stride padding or trailing bytes.
    pub fn into_raw_data(self) -> &'a mut [u8] {
        self.data
    }

    /// Returns the pixel bytes of row `y` (length `width * 4`).
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    pub fn row(&self, y: u16) -> &[u8] {
        self.as_ref().row(y)
    }

    /// Returns the mutable pixel bytes of row `y` (length `width * 4`).
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    pub fn row_mut(&mut self, y: u16) -> &mut [u8] {
        assert!(
            y < self.height,
            "row index {y} out of bounds for height {}",
            self.height
        );
        let start = y as usize * self.stride;
        &mut self.data[start..start + row_bytes(self.width)]
    }

    /// Returns the premultiplied `[r, g, b, a]` pixels of row `y` (length `width`).
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    #[inline]
    pub fn row_pixels(&self, y: u16) -> &[[u8; 4]] {
        as_pixels(self.row(y))
    }

    /// Returns the mutable premultiplied `[r, g, b, a]` pixels of row `y` (length `width`).
    ///
    /// # Panics
    ///
    /// Panics if `y >= self.height()`.
    #[inline]
    pub fn row_pixels_mut(&mut self, y: u16) -> &mut [[u8; 4]] {
        as_pixels_mut(self.row_mut(y))
    }

    /// Iterates over the rows of pixels from top to bottom.
    #[inline]
    pub fn rows(&self) -> Rows<'_> {
        self.as_ref().rows()
    }

    /// Iterates mutably over the rows of pixels from top to bottom.
    #[inline]
    pub fn rows_mut(&mut self) -> RowsMut<'_> {
        RowsMut::new(self.data, self.width, self.height, self.stride)
    }

    /// Consumes the view and iterates mutably over the rows of pixels with the original lifetime.
    #[inline]
    pub fn into_rows_mut(self) -> RowsMut<'a> {
        RowsMut::new(self.data, self.width, self.height, self.stride)
    }

    /// Borrows this view immutably as a [`PixmapRef`].
    pub fn as_ref(&self) -> PixmapRef<'_> {
        PixmapRef::from_raw_parts(self.data, self.width, self.height, self.stride)
    }

    /// Consumes this view and downgrades it to a [`PixmapRef`] with the original lifetime.
    pub fn into_ref(self) -> PixmapRef<'a> {
        PixmapRef::from_raw_parts(self.data, self.width, self.height, self.stride)
    }

    /// Reborrows the view for a shorter lifetime.
    pub fn reborrow(&mut self) -> PixmapMut<'_> {
        PixmapMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: self.data,
        }
    }

    /// Borrows a rectangular subregion immutably as a [`PixmapRef`], without copying.
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn subregion(&self, x: u16, y: u16, width: u16, height: u16) -> PixmapRef<'_> {
        self.as_ref().subregion(x, y, width, height)
    }

    /// Borrows a rectangular subregion mutably as a [`PixmapMut`], without copying.
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn subregion_mut(&mut self, x: u16, y: u16, width: u16, height: u16) -> PixmapMut<'_> {
        self.reborrow().into_subregion_mut(x, y, width, height)
    }

    /// Consumes this view and returns a mutable view of a rectangular subregion with the original
    /// lifetime.
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn into_subregion_mut(self, x: u16, y: u16, width: u16, height: u16) -> PixmapMut<'a> {
        let range = subregion_byte_range(self.width, self.height, self.stride, x, y, width, height);
        PixmapMut {
            width,
            height,
            stride: self.stride,
            data: &mut self.data[range],
        }
    }

    /// Copies the pixels into a new tightly packed owned [`Pixmap`].
    pub fn to_owned(&self) -> Pixmap {
        self.as_ref().to_owned()
    }

    /// Copies the active pixels into a tightly packed straight-alpha RGBA8 buffer.
    pub fn to_unpremultiplied(&self) -> Vec<u8> {
        self.as_ref().to_unpremultiplied()
    }
}

/// A mutable iterator over the rows of a pixmap, from top to bottom.
///
/// Each item is a mutable slice of `width * 4` pixel bytes, excluding any inter-row stride
/// padding.
#[derive(Debug)]
pub struct RowsMut<'a> {
    inner: RowsMutInner<'a>,
}

#[derive(Debug)]
enum RowsMutInner<'a> {
    /// Non-empty rows (`row_bytes > 0`, so `stride >= row_bytes > 0`).
    Strided {
        head: ChunksExactMut<'a, u8>,
        last: Option<&'a mut [u8]>,
        row_bytes: usize,
    },
    /// Zero-width or zero-height pixmap: yields `remaining` empty slices without touching `data`.
    Empty(usize),
}

impl<'a> RowsMut<'a> {
    #[inline]
    fn new(data: &'a mut [u8], width: u16, height: u16, stride: usize) -> Self {
        let row_len = row_bytes(width);
        let remaining = height as usize;
        if remaining == 0 || row_len == 0 {
            return Self {
                inner: RowsMutInner::Empty(remaining),
            };
        }
        // When the backing slice includes trailing padding on the final row (always true for
        // tightly packed pixmaps and full-stride buffers), every row fits in `ChunksExactMut`.
        if let Some(full_len) = remaining.checked_mul(stride)
            && full_len <= data.len()
        {
            return Self {
                inner: RowsMutInner::Strided {
                    head: data[..full_len].chunks_exact_mut(stride),
                    last: None,
                    row_bytes: row_len,
                },
            };
        }
        let head_len = (remaining - 1) * stride;
        let (head_bytes, tail) = data[..head_len + row_len].split_at_mut(head_len);
        Self {
            inner: RowsMutInner::Strided {
                head: head_bytes.chunks_exact_mut(stride),
                last: Some(tail),
                row_bytes: row_len,
            },
        }
    }
}

impl<'a> Iterator for RowsMut<'a> {
    type Item = &'a mut [u8];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            RowsMutInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                if let Some(chunk) = head.next() {
                    Some(&mut chunk[..*row_bytes])
                } else {
                    last.take()
                }
            }
            RowsMutInner::Empty(remaining) => {
                *remaining = remaining.checked_sub(1)?;
                Some(&mut [])
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
            RowsMutInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                let head_len = head.len();
                if n < head_len {
                    head.nth(n).map(|chunk| &mut chunk[..*row_bytes])
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
            RowsMutInner::Empty(remaining) => {
                if n < *remaining {
                    *remaining -= n + 1;
                    Some(&mut [])
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
            RowsMutInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                acc = head.fold(acc, |acc, chunk| f(acc, &mut chunk[..row_bytes]));
                if let Some(row) = last {
                    acc = f(acc, row);
                }
                acc
            }
            RowsMutInner::Empty(remaining) => {
                for _ in 0..remaining {
                    acc = f(acc, &mut []);
                }
                acc
            }
        }
    }
}

impl DoubleEndedIterator for RowsMut<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            RowsMutInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                if let Some(row) = last.take() {
                    Some(row)
                } else {
                    head.next_back().map(|chunk| &mut chunk[..*row_bytes])
                }
            }
            RowsMutInner::Empty(remaining) => {
                *remaining = remaining.checked_sub(1)?;
                Some(&mut [])
            }
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        match &mut self.inner {
            RowsMutInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                if let Some(row) = last.take() {
                    if n == 0 {
                        Some(row)
                    } else {
                        head.nth_back(n - 1).map(|chunk| &mut chunk[..*row_bytes])
                    }
                } else {
                    head.nth_back(n).map(|chunk| &mut chunk[..*row_bytes])
                }
            }
            RowsMutInner::Empty(remaining) => {
                if n < *remaining {
                    *remaining -= n + 1;
                    Some(&mut [])
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
            RowsMutInner::Strided {
                head,
                last,
                row_bytes,
            } => {
                if let Some(row) = last {
                    acc = f(acc, row);
                }
                head.rfold(acc, |acc, chunk| f(acc, &mut chunk[..row_bytes]))
            }
            RowsMutInner::Empty(remaining) => {
                for _ in 0..remaining {
                    acc = f(acc, &mut []);
                }
                acc
            }
        }
    }
}

impl ExactSizeIterator for RowsMut<'_> {
    #[inline]
    fn len(&self) -> usize {
        match &self.inner {
            RowsMutInner::Strided { head, last, .. } => head.len() + usize::from(last.is_some()),
            RowsMutInner::Empty(remaining) => *remaining,
        }
    }
}

impl FusedIterator for RowsMut<'_> {}

impl<'a> From<&'a mut Pixmap> for PixmapMut<'a> {
    fn from(pixmap: &'a mut Pixmap) -> Self {
        pixmap.as_mut()
    }
}

impl PartialEq for PixmapMut<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for PixmapMut<'_> {}

impl fmt::Debug for PixmapMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PixmapMut")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

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
    fn try_from_parts_pixels_mut_and_get_pixel() {
        let mut data = [0_u8; 8];
        {
            let mut view = PixmapMut::try_from_parts(&mut data, 2, 1).unwrap();
            view.pixels_mut().unwrap()[1] = [10, 20, 30, 255];
            assert_eq!(view.pixels(), Some(&[[0, 0, 0, 0], [10, 20, 30, 255]][..]));
            assert_eq!(view.get_pixel(1, 0), Some([10, 20, 30, 255]));
            assert_eq!(view.get_pixel(2, 0), None);
        }
        assert_eq!(data, [0, 0, 0, 0, 10, 20, 30, 255]);

        assert!(PixmapMut::try_from_parts(&mut data[..7], 2, 1).is_none());
        assert!(PixmapMut::try_from_parts(&mut [], 0, 2).is_some());

        // Strided mutable view preserves inter-row padding when mutating via `row_pixels_mut`.
        let mut buf = vec![0xAA_u8; 24];
        {
            let mut strided = PixmapMut::try_from_parts_with_stride(&mut buf, 2, 2, 16).unwrap();
            assert_eq!(strided.pixels(), None);
            assert_eq!(strided.pixels_mut(), None);
            strided.row_pixels_mut(0)[0] = [1, 2, 3, 4];
            strided.row_pixels_mut(1)[1] = [5, 6, 7, 8];
            assert_eq!(strided.row_pixels(0)[0], [1, 2, 3, 4]);
            assert_eq!(strided.get_pixel(1, 1), Some([5, 6, 7, 8]));
            assert_eq!(strided.get_pixel(0, 2), None);
        }
        assert_eq!(&buf[8..16], &[0xAA; 8]);

        assert!(PixmapMut::try_from_parts_with_stride(&mut buf, 2, 2, 4).is_none());
        assert!(PixmapMut::try_from_parts_with_stride(&mut buf[..23], 2, 2, 16).is_none());
        assert!(PixmapMut::try_from_parts_with_stride(&mut buf, 1, 2, usize::MAX).is_none());
    }
}
