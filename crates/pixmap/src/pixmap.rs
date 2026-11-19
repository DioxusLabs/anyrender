use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::{
    PixmapMut, PixmapRef, Rows, RowsMut, byte_len, fill_slice, row_bytes, slice_is_opaque,
};

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
    pub(crate) fn from_raw_parts(data: Vec<u8>, width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

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

    /// Returns `true` if every pixel has alpha `255` (vacuously `true` when empty).
    pub fn is_opaque(&self) -> bool {
        slice_is_opaque(&self.data)
    }

    /// Returns the premultiplied `[r, g, b, a]` pixel at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width()` or `y >= self.height()`.
    pub fn pixel(&self, x: u16, y: u16) -> [u8; 4] {
        self.as_ref().pixel(x, y)
    }

    /// Sets the premultiplied `[r, g, b, a]` pixel at `(x, y)`.
    ///
    /// # Panics
    ///
    /// Panics if `x >= self.width()` or `y >= self.height()`.
    pub fn set_pixel(&mut self, x: u16, y: u16, color: [u8; 4]) {
        self.as_mut().set_pixel(x, y, color);
    }

    /// Sets every pixel to transparent black (`[0, 0, 0, 0]`).
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Sets every pixel in a rectangular region to transparent black (`[0, 0, 0, 0]`).
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn clear_rect(&mut self, x: u16, y: u16, width: u16, height: u16) {
        self.subregion_mut(x, y, width, height).clear();
    }

    /// Sets every pixel to the given premultiplied `[r, g, b, a]` colour.
    pub fn fill(&mut self, color: [u8; 4]) {
        fill_slice(&mut self.data, color);
    }

    /// Sets every pixel in a rectangular region to the given premultiplied `[r, g, b, a]` colour.
    ///
    /// # Panics
    ///
    /// Panics if `x + width > self.width()` or `y + height > self.height()`.
    pub fn fill_rect(&mut self, x: u16, y: u16, width: u16, height: u16, color: [u8; 4]) {
        self.subregion_mut(x, y, width, height).fill(color);
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
    #[inline]
    pub fn rows(&self) -> Rows<'_> {
        self.as_ref().rows()
    }

    /// Iterates mutably over the rows of pixels from top to bottom.
    #[inline]
    pub fn rows_mut(&mut self) -> RowsMut<'_> {
        self.as_mut().into_rows_mut()
    }

    /// Borrows the pixmap as a [`PixmapRef`].
    pub fn as_ref(&self) -> PixmapRef<'_> {
        PixmapRef::from_raw_parts(&self.data, self.width, self.height, row_bytes(self.width))
    }

    /// Borrows the pixmap as a [`PixmapMut`].
    pub fn as_mut(&mut self) -> PixmapMut<'_> {
        PixmapMut::from_raw_parts(
            &mut self.data,
            self.width,
            self.height,
            row_bytes(self.width),
        )
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

impl fmt::Debug for Pixmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pixmap")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}
