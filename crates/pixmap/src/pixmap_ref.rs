use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;
use core::slice::ChunksExact;

use crate::{
    BYTES_PER_PIXEL, Pixmap, PixmapMut, ResampleFilter, byte_len, is_contiguous, row_bytes,
    slice_is_opaque, subregion_byte_range, validate_strided_parts,
};

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
    /// Panics if `data.len()` is not `width * height * 4`.
    pub fn from_parts(data: &'a [u8], width: u16, height: u16) -> Self {
        assert_eq!(
            data.len(),
            byte_len(width, height),
            "pixel data length must match the pixmap's dimensions"
        );
        Self {
            width,
            height,
            stride: row_bytes(width),
            data,
        }
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

    /// Returns a new tightly packed [`Pixmap`] flipped horizontally (left-to-right).
    pub fn flipped_horizontal(&self) -> Pixmap {
        crate::transform::flipped_horizontal(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] flipped vertically (top-to-bottom).
    pub fn flipped_vertical(&self) -> Pixmap {
        crate::transform::flipped_vertical(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] rotated 90 degrees clockwise
    /// (`(x, y) -> (height - 1 - y, x)`).
    pub fn rotated_90(&self) -> Pixmap {
        crate::transform::rotated_90(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] rotated 180 degrees.
    pub fn rotated_180(&self) -> Pixmap {
        crate::transform::rotated_180(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] rotated 270 degrees clockwise (90 degrees
    /// counter-clockwise, `(x, y) -> (y, width - 1 - x)`).
    pub fn rotated_270(&self) -> Pixmap {
        crate::transform::rotated_270(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] transposed across the main diagonal
    /// (`(x, y) -> (y, x)`).
    pub fn transposed(&self) -> Pixmap {
        crate::transform::transposed(*self)
    }

    /// Returns a new tightly packed [`Pixmap`] padded on each side by the given pixel counts,
    /// filling the border with `fill`.
    ///
    /// # Panics
    ///
    /// Panics if the resulting width or height overflows `u16`.
    pub fn padded(&self, top: u16, right: u16, bottom: u16, left: u16, fill: [u8; 4]) -> Pixmap {
        crate::transform::padded(*self, top, right, bottom, left, fill)
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
        crate::transform::padded_clamped(*self, top, right, bottom, left)
    }

    /// Resamples the pixmap view to `width × height` using `filter`, returning a new tightly
    /// packed [`Pixmap`].
    ///
    /// If `self` is empty and the target dimensions are non-zero, the returned pixmap is filled
    /// with transparent black (`[0, 0, 0, 0]`).
    pub fn resize(&self, width: u16, height: u16, filter: ResampleFilter) -> Pixmap {
        crate::resample::resize(*self, width, height, filter)
    }

    /// Resamples the pixmap view into `dst` using `filter`, leaving any inter-row stride padding
    /// on `dst` untouched.
    pub fn resize_into(&self, mut dst: PixmapMut<'_>, filter: ResampleFilter) {
        crate::resample::resize_into(*self, &mut dst, filter);
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
