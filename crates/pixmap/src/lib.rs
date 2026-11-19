//! A minimal owned pixel buffer.
//!
//! [`Pixmap`] is the pixel container shared by AnyRender's glyph and atlas crates, so that
//! neither has to depend on the other or on a renderer. It has no dependencies, and its buffer
//! is a plain `Vec<u8>` so that converting to a renderer's own image type can move the pixels
//! instead of copying them.

#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

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

    /// The pixel bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The pixel bytes, mutably.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Consumes the pixmap and returns its pixel bytes without copying them.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    /// The number of bytes of pixel data.
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}

fn byte_len(width: u16, height: u16) -> usize {
    width as usize * height as usize * BYTES_PER_PIXEL
}

impl fmt::Debug for Pixmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pixmap")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_transparent() {
        let pixmap = Pixmap::new(3, 2);
        assert_eq!(pixmap.byte_len(), 24);
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
}
