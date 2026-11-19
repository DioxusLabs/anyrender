use core::fmt;
use core::iter::FusedIterator;

use crate::{
    Pixmap, PixmapRef, Rows, byte_len, is_contiguous, row_bytes, subregion_byte_range,
    validate_strided_parts,
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
    /// Panics if `data.len()` is not `width * height * 4`.
    pub fn from_parts(data: &'a mut [u8], width: u16, height: u16) -> Self {
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

    /// Iterates over the rows of pixels from top to bottom.
    pub fn rows(&self) -> Rows<'_> {
        self.as_ref().rows()
    }

    /// Iterates mutably over the rows of pixels from top to bottom.
    pub fn rows_mut(&mut self) -> RowsMut<'_> {
        RowsMut::new(self.data, self.width, self.height, self.stride)
    }

    /// Consumes the view and iterates mutably over the rows of pixels with the original lifetime.
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
}

/// A mutable iterator over the rows of a pixmap, from top to bottom.
///
/// Each item is a mutable slice of `width * 4` pixel bytes, excluding any inter-row stride
/// padding.
#[derive(Debug)]
pub struct RowsMut<'a> {
    data: &'a mut [u8],
    row_bytes: usize,
    stride: usize,
    remaining: usize,
}

impl<'a> RowsMut<'a> {
    fn new(data: &'a mut [u8], width: u16, height: u16, stride: usize) -> Self {
        let row_len = row_bytes(width);
        let remaining = height as usize;
        let data = if remaining == 0 {
            &mut []
        } else {
            &mut data[..(remaining - 1) * stride + row_len]
        };
        Self {
            data,
            row_bytes: row_len,
            stride,
            remaining,
        }
    }
}

impl<'a> Iterator for RowsMut<'a> {
    type Item = &'a mut [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let data = core::mem::take(&mut self.data);
        if self.remaining == 0 {
            Some(data)
        } else {
            let (head, rest) = data.split_at_mut(self.stride);
            self.data = rest;
            Some(&mut head[..self.row_bytes])
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl DoubleEndedIterator for RowsMut<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let data = core::mem::take(&mut self.data);
        if self.remaining == 0 {
            Some(data)
        } else {
            let last_start = self.remaining * self.stride;
            let prev_end = (self.remaining - 1) * self.stride + self.row_bytes;
            let (head, row) = data.split_at_mut(last_start);
            self.data = &mut head[..prev_end];
            Some(row)
        }
    }
}

impl ExactSizeIterator for RowsMut<'_> {
    fn len(&self) -> usize {
        self.remaining
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
