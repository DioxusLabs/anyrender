//! Minimal RGBA8 pixel buffers: the owned [`Pixmap`] and the borrowed [`PixmapMut`].
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

    /// Borrows the pixmap as a [`PixmapMut`].
    pub fn as_mut(&mut self) -> PixmapMut<'_> {
        PixmapMut {
            width: self.width,
            height: self.height,
            data: &mut self.data,
        }
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

/// A mutably borrowed, tightly packed buffer of RGBA8 pixels with rows stored top to bottom.
///
/// The pixel layout is the same as [`Pixmap`]'s. This lets code write pixels into a buffer it
/// does not own, such as another library's image.
#[derive(PartialEq, Eq)]
pub struct PixmapMut<'a> {
    width: u16,
    height: u16,
    data: &'a mut [u8],
}

impl<'a> PixmapMut<'a> {
    /// Creates a pixmap that borrows existing pixel data.
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
        self.data
    }

    /// The pixel bytes, mutably.
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.data
    }

    /// Consumes the pixmap and returns the pixel bytes it borrowed.
    pub fn into_data(self) -> &'a mut [u8] {
        self.data
    }

    /// Reborrows the pixmap for a shorter lifetime.
    pub fn reborrow(&mut self) -> PixmapMut<'_> {
        PixmapMut {
            width: self.width,
            height: self.height,
            data: self.data,
        }
    }

    /// Copies the pixels into a new owned [`Pixmap`].
    pub fn to_owned(&self) -> Pixmap {
        Pixmap {
            width: self.width,
            height: self.height,
            data: self.data.to_vec(),
        }
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

impl fmt::Debug for PixmapMut<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PixmapMut")
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

    #[test]
    fn as_mut_writes_through_to_the_pixmap() {
        let mut pixmap = Pixmap::new(2, 1);
        let mut view = pixmap.as_mut();
        assert_eq!((view.width(), view.height()), (2, 1));
        view.data_mut()[4..].fill(9);
        assert_eq!(pixmap.data(), [0, 0, 0, 0, 9, 9, 9, 9]);
    }

    #[test]
    fn pixmap_mut_borrows_external_data() {
        let mut data = [1_u8; 8];
        let mut view = PixmapMut::from_parts(&mut data, 1, 2);
        view.reborrow().data_mut()[0] = 5;
        assert_eq!(view.to_owned().data(), [5, 1, 1, 1, 1, 1, 1, 1]);
        view.into_data()[7] = 6;
        assert_eq!(data, [5, 1, 1, 1, 1, 1, 1, 6]);
    }

    #[test]
    #[should_panic(expected = "pixel data length")]
    fn pixmap_mut_rejects_wrong_length() {
        let _ = PixmapMut::from_parts(&mut [0; 7], 1, 2);
    }
}
