use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;

use crate::{
    Pixmap, PixmapMut, byte_len, is_contiguous, row_bytes, subregion_byte_range,
    validate_strided_parts,
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
}

/// An iterator over the rows of a pixmap, from top to bottom.
///
/// Each item is a slice of `width * 4` pixel bytes, excluding any inter-row stride padding.
#[derive(Clone, Debug)]
pub struct Rows<'a> {
    data: &'a [u8],
    row_bytes: usize,
    stride: usize,
    remaining: usize,
}

impl<'a> Rows<'a> {
    fn new(data: &'a [u8], width: u16, height: u16, stride: usize) -> Self {
        let row_len = row_bytes(width);
        let remaining = height as usize;
        let data = if remaining == 0 {
            &[]
        } else {
            &data[..(remaining - 1) * stride + row_len]
        };
        Self {
            data,
            row_bytes: row_len,
            stride,
            remaining,
        }
    }
}

impl<'a> Iterator for Rows<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        if self.remaining == 0 {
            let row = self.data;
            self.data = &[];
            Some(row)
        } else {
            let (head, rest) = self.data.split_at(self.stride);
            self.data = rest;
            Some(&head[..self.row_bytes])
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl DoubleEndedIterator for Rows<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        if self.remaining == 0 {
            let row = self.data;
            self.data = &[];
            Some(row)
        } else {
            let last_start = self.remaining * self.stride;
            let prev_end = (self.remaining - 1) * self.stride + self.row_bytes;
            let row = &self.data[last_start..];
            self.data = &self.data[..prev_end];
            Some(row)
        }
    }
}

impl ExactSizeIterator for Rows<'_> {
    fn len(&self) -> usize {
        self.remaining
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
