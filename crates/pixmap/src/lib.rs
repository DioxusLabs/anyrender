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

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use core::iter::FusedIterator;

/// The number of bytes one pixel occupies.
pub const BYTES_PER_PIXEL: usize = 4;

/// An owned, tightly packed buffer of RGBA8 pixels with rows stored top to bottom.
///
/// Each pixel is four bytes in R, G, B, A order, with colour premultiplied by alpha.
#[derive(Clone, PartialEq, Eq)]
pub struct Pixmap {
    width: u16,
    height: u16,
    data: Vec<u8>,
}

impl Pixmap {
    /// Creates a fully transparent pixmap.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            data: vec![0; byte_len(width, height)],
        }
    }

    /// Creates a pixmap from existing pixel data, without copying it.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not `width * height * 4`.
    pub fn from_parts(data: Vec<u8>, width: u16, height: u16) -> Self {
        assert_eq!(
            data.len(),
            byte_len(width, height),
            "pixel data length must match the pixmap's dimensions"
        );
        Self {
            width,
            height,
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

    /// The distance in bytes between the start of consecutive rows (`width * 4`).
    pub fn stride(&self) -> usize {
        row_bytes(self.width)
    }

    /// Returns `true` if either dimension is zero.
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// The pixel bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The pixel bytes, mutably.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
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
        let stride = self.stride();
        assert!(
            y < self.height,
            "row index {y} out of bounds for height {}",
            self.height
        );
        let start = y as usize * stride;
        &mut self.data[start..start + stride]
    }

    /// Iterates over the rows of pixels from top to bottom.
    pub fn rows(&self) -> Rows<'_> {
        self.as_ref().rows()
    }

    /// Iterates mutably over the rows of pixels from top to bottom.
    pub fn rows_mut(&mut self) -> RowsMut<'_> {
        self.as_mut().into_rows_mut()
    }

    /// Borrows the pixmap as a [`PixmapRef`].
    pub fn as_ref(&self) -> PixmapRef<'_> {
        PixmapRef {
            width: self.width,
            height: self.height,
            stride: row_bytes(self.width),
            data: &self.data,
        }
    }

    /// Borrows the pixmap as a [`PixmapMut`].
    pub fn as_mut(&mut self) -> PixmapMut<'_> {
        PixmapMut {
            width: self.width,
            height: self.height,
            stride: row_bytes(self.width),
            data: &mut self.data,
        }
    }

    /// Borrows a rectangular subregion as a [`PixmapRef`], without copying.
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn subregion(&self, x: u16, y: u16, width: u16, height: u16) -> PixmapRef<'_> {
        self.as_ref().subregion(x, y, width, height)
    }

    /// Borrows a rectangular subregion as a [`PixmapMut`], without copying.
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn subregion_mut(&mut self, x: u16, y: u16, width: u16, height: u16) -> PixmapMut<'_> {
        self.as_mut().into_subregion_mut(x, y, width, height)
    }

    /// Consumes the pixmap and returns its pixel bytes without copying them.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// The number of bytes of pixel data (`width * height * 4`).
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

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
            return Pixmap {
                width: self.width,
                height: self.height,
                data: bytes.to_vec(),
            };
        }
        let mut data = Vec::with_capacity(self.byte_len());
        for row in self.rows() {
            data.extend_from_slice(row);
        }
        Pixmap {
            width: self.width,
            height: self.height,
            data,
        }
    }
}

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
        PixmapRef {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: self.data,
        }
    }

    /// Consumes this view and downgrades it to a [`PixmapRef`] with the original lifetime.
    pub fn into_ref(self) -> PixmapRef<'a> {
        PixmapRef {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: self.data,
        }
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

impl<'a> From<&'a Pixmap> for PixmapRef<'a> {
    fn from(pixmap: &'a Pixmap) -> Self {
        pixmap.as_ref()
    }
}

impl<'a> From<&'a mut Pixmap> for PixmapMut<'a> {
    fn from(pixmap: &'a mut Pixmap) -> Self {
        pixmap.as_mut()
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

impl PartialEq for PixmapMut<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for PixmapMut<'_> {}

fn row_bytes(width: u16) -> usize {
    width as usize * BYTES_PER_PIXEL
}

fn byte_len(width: u16, height: u16) -> usize {
    row_bytes(width) * height as usize
}

fn is_contiguous(width: u16, height: u16, stride: usize) -> bool {
    width == 0 || height <= 1 || stride == row_bytes(width)
}

fn validate_strided_parts(data_len: usize, width: u16, height: u16, stride: usize) {
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

fn subregion_byte_range(
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

impl fmt::Debug for Pixmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pixmap")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for PixmapRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PixmapRef")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

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
}
