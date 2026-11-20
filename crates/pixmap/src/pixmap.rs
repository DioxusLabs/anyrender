use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::util::{
    as_pixels, as_pixels_mut, byte_len, fill_slice, is_valid_packed_parts, row_bytes,
    slice_is_opaque, validate_packed_parts,
};
use crate::{
    BYTES_PER_PIXEL, PixmapMut, PixmapRef, Rows, RowsMut, multiply_alpha_rgba8, premultiply_rgba8,
    swap_rb_rgba8, unpremultiply_rgba8,
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
    ///
    /// # Panics
    ///
    /// Panics if `width * height * 4` overflows `usize` or exceeds the allocator's capacity limit.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            data: vec![0; byte_len(width, height)],
        }
    }

    /// Creates an empty (`0x0`) pixmap with backing storage pre-allocated for at least
    /// `pixel_capacity` pixels.
    ///
    /// # Panics
    ///
    /// Panics if `pixel_capacity * 4` overflows `usize` or exceeds the allocator's capacity limit.
    pub fn with_capacity(pixel_capacity: usize) -> Self {
        let byte_capacity = pixel_capacity
            .checked_mul(BYTES_PER_PIXEL)
            .expect("pixel capacity overflows usize");
        Self {
            width: 0,
            height: 0,
            data: Vec::with_capacity(byte_capacity),
        }
    }

    /// Creates a pixmap from existing pixel data, without copying it.
    ///
    /// # Panics
    ///
    /// Panics if `width * height * 4` overflows `usize` or `data.len()` does not match.
    pub fn from_parts(data: Vec<u8>, width: u16, height: u16) -> Self {
        validate_packed_parts(data.len(), width, height);
        Self {
            width,
            height,
            data,
        }
    }

    /// Creates a pixmap from existing pixel data, without copying it, returning `None` if
    /// `width * height * 4` overflows `usize` or `data.len()` does not match.
    pub fn try_from_parts(data: Vec<u8>, width: u16, height: u16) -> Option<Self> {
        if !is_valid_packed_parts(data.len(), width, height) {
            return None;
        }
        Some(Self {
            width,
            height,
            data,
        })
    }

    /// Creates a premultiplied pixmap from straight-alpha RGBA8 pixel data, premultiplying in
    /// place without copying.
    ///
    /// # Panics
    ///
    /// Panics if `width * height * 4` overflows `usize` or `data.len()` does not match.
    pub fn from_unpremultiplied(mut data: Vec<u8>, width: u16, height: u16) -> Self {
        validate_packed_parts(data.len(), width, height);
        premultiply_rgba8(&mut data);
        Self {
            width,
            height,
            data,
        }
    }

    /// Creates a premultiplied pixmap from straight-alpha RGBA8 pixel data, premultiplying in
    /// place without copying, or returning `None` if `width * height * 4` overflows `usize` or
    /// `data.len()` does not match.
    pub fn try_from_unpremultiplied(mut data: Vec<u8>, width: u16, height: u16) -> Option<Self> {
        if !is_valid_packed_parts(data.len(), width, height) {
            return None;
        }
        premultiply_rgba8(&mut data);
        Some(Self {
            width,
            height,
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

    /// Premultiplies the colour channels by alpha in place.
    ///
    /// [`Pixmap`] normally stores premultiplied pixels; this is useful after writing
    /// straight-alpha RGBA8 bytes directly into [`data_mut`](Self::data_mut).
    pub fn premultiply(&mut self) {
        premultiply_rgba8(&mut self.data);
    }

    /// Unpremultiplies the colour channels to straight alpha in place.
    ///
    /// Pixels with alpha `0` become `[0, 0, 0, 0]`, and any colour channel exceeding alpha is
    /// clamped to `255`.
    pub fn unpremultiply(&mut self) {
        unpremultiply_rgba8(&mut self.data);
    }

    /// Multiplies every channel of every pixel by `alpha / 255` in place, using nearest-integer
    /// rounding.
    pub fn multiply_alpha(&mut self, alpha: u8) {
        multiply_alpha_rgba8(&mut self.data, alpha);
    }

    /// Swaps the red and blue channels of every pixel in place (RGBA <-> BGRA).
    pub fn swap_rb(&mut self) {
        swap_rb_rgba8(&mut self.data);
    }

    /// Copies pixels from `src` into `self` with `src`'s top-left corner placed at `(x, y)`.
    ///
    /// Source pixels are written verbatim (overwriting destination pixels without blending). Any
    /// part of `src` falling outside `self` is clipped; if there is no overlap, `self` is left
    /// unchanged.
    pub fn copy_from(&mut self, src: PixmapRef<'_>, x: i32, y: i32) {
        self.as_mut().copy_from(src, x, y);
    }

    /// The pixel bytes.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The pixel bytes, mutably.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// The pixels as a tightly packed slice of premultiplied `[r, g, b, a]` values.
    #[inline]
    pub fn pixels(&self) -> &[[u8; 4]] {
        as_pixels(&self.data)
    }

    /// The pixels as a mutable tightly packed slice of premultiplied `[r, g, b, a]` values.
    #[inline]
    pub fn pixels_mut(&mut self) -> &mut [[u8; 4]] {
        as_pixels_mut(&mut self.data)
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

    /// Consumes the pixmap, unpremultiplying its pixels to straight-alpha RGBA8 in place without
    /// allocating.
    pub fn into_unpremultiplied(mut self) -> Vec<u8> {
        unpremultiply_rgba8(&mut self.data);
        self.data
    }

    /// Copies the pixels into a tightly packed straight-alpha RGBA8 buffer.
    pub fn to_unpremultiplied(&self) -> Vec<u8> {
        self.as_ref().to_unpremultiplied()
    }

    /// The number of bytes of pixel data (`width * height * 4`).
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// The number of pixels the backing allocation can hold without reallocating.
    ///
    /// Calling [`resize_buffer`](Self::resize_buffer) or [`reset`](Self::reset) with
    /// `width as usize * height as usize <= self.capacity()` reuses the existing allocation.
    pub fn capacity(&self) -> usize {
        self.data.capacity() / BYTES_PER_PIXEL
    }

    /// Shrinks the capacity of the backing buffer as much as possible to fit the current
    /// dimensions.
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// Changes the pixmap's dimensions without resampling the image, reusing the backing
    /// allocation whenever `width as usize * height as usize <= self.capacity()`.
    ///
    /// If the new pixel count is larger than the current one, newly added pixels are initialized
    /// to transparent black (`[0, 0, 0, 0]`); if smaller, the buffer is truncated while keeping
    /// its reserved capacity. Existing bytes are not re-laid out for the new width, so callers
    /// that want a clean transparent surface should use [`reset`](Self::reset).
    ///
    /// # Panics
    ///
    /// Panics if `width * height * 4` overflows `usize` or exceeds the allocator's capacity limit.
    pub fn resize_buffer(&mut self, width: u16, height: u16) {
        let new_len = byte_len(width, height);
        self.data.resize(new_len, 0);
        self.width = width;
        self.height = height;
    }

    /// Changes the pixmap's dimensions and clears every pixel to transparent black
    /// (`[0, 0, 0, 0]`), reusing the backing allocation whenever
    /// `width as usize * height as usize <= self.capacity()`.
    ///
    /// If the backing buffer must grow beyond its current capacity, stale pixel data is
    /// discarded before reallocating so it is not copied into the new allocation.
    ///
    /// # Panics
    ///
    /// Panics if `width * height * 4` overflows `usize` or exceeds the allocator's capacity limit.
    pub fn reset(&mut self, width: u16, height: u16) {
        let new_len = byte_len(width, height);
        self.width = 0;
        self.height = 0;
        self.data.clear();
        self.data.resize(new_len, 0);
        self.width = width;
        self.height = height;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::numbered_pixmap;
    use alloc::vec;

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
    #[should_panic(expected = "row index")]
    fn row_rejects_out_of_bounds() {
        let pixmap = Pixmap::new(2, 2);
        let _ = pixmap.row(2);
    }

    #[test]
    fn pixel_and_set_pixel_work_on_owned_and_strided_views() {
        let mut pixmap = numbered_pixmap(4, 3);
        assert_eq!(pixmap.pixel(2, 1), [2, 1, 3, 255]);
        assert_eq!(pixmap.as_ref().pixel(3, 2), [3, 2, 5, 255]);

        pixmap.set_pixel(2, 1, [10, 20, 30, 40]);
        assert_eq!(pixmap.pixel(2, 1), [10, 20, 30, 40]);

        let mut sub = pixmap.subregion_mut(1, 1, 2, 2);
        assert_eq!(sub.pixel(1, 0), [10, 20, 30, 40]);
        sub.set_pixel(0, 1, [50, 60, 70, 80]);
        assert_eq!(sub.pixel(0, 1), [50, 60, 70, 80]);
        assert_eq!(pixmap.pixel(1, 2), [50, 60, 70, 80]);
        // Neighbouring pixel outside the subregion is untouched.
        assert_eq!(pixmap.pixel(0, 2), [0, 2, 2, 255]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn pixel_rejects_out_of_bounds_x() {
        let pixmap = Pixmap::new(2, 2);
        let _ = pixmap.pixel(2, 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn set_pixel_rejects_out_of_bounds_y() {
        let mut pixmap = Pixmap::new(2, 2);
        pixmap.set_pixel(0, 2, [1, 2, 3, 4]);
    }

    #[test]
    fn fill_rect_and_clear_rect_modify_only_target_region() {
        let mut pixmap = Pixmap::new(4, 4);
        pixmap.fill([10, 20, 30, 255]);
        pixmap.fill_rect(1, 1, 2, 2, [1, 2, 3, 4]);

        for y in 0..4 {
            for x in 0..4 {
                let expected = if (1..3).contains(&x) && (1..3).contains(&y) {
                    [1, 2, 3, 4]
                } else {
                    [10, 20, 30, 255]
                };
                assert_eq!(pixmap.pixel(x, y), expected, "at ({x}, {y})");
            }
        }

        // Zero-sized rects are no-ops, including at edges.
        pixmap.fill_rect(4, 4, 0, 0, [99, 99, 99, 99]);
        pixmap.clear_rect(0, 0, 0, 4);
        assert_eq!(pixmap.pixel(0, 0), [10, 20, 30, 255]);

        pixmap.as_mut().clear_rect(1, 2, 2, 1);
        assert_eq!(pixmap.pixel(1, 1), [1, 2, 3, 4]);
        assert_eq!(pixmap.pixel(1, 2), [0, 0, 0, 0]);
        assert_eq!(pixmap.pixel(2, 2), [0, 0, 0, 0]);
        assert_eq!(pixmap.pixel(3, 2), [10, 20, 30, 255]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn fill_rect_rejects_out_of_bounds() {
        let mut pixmap = Pixmap::new(4, 4);
        pixmap.fill_rect(3, 0, 2, 1, [1, 2, 3, 4]);
    }

    #[test]
    fn is_opaque_checks_active_pixels_and_ignores_padding() {
        assert!(Pixmap::new(0, 0).is_opaque());
        assert!(Pixmap::new(0, 3).is_opaque());
        assert!(Pixmap::new(3, 0).is_opaque());
        assert!(!Pixmap::new(2, 2).is_opaque());

        let mut pixmap = Pixmap::new(3, 2);
        pixmap.fill([10, 20, 30, 255]);
        assert!(pixmap.is_opaque());
        assert!(pixmap.as_ref().is_opaque());
        assert!(pixmap.as_mut().is_opaque());

        pixmap.set_pixel(2, 1, [10, 20, 30, 254]);
        assert!(!pixmap.is_opaque());
        assert!(pixmap.subregion(0, 0, 2, 2).is_opaque());

        // Strided buffer with zeroed padding bytes is still opaque if active pixels have alpha 255.
        let mut buf = vec![0_u8; 16 + 8];
        let mut view = PixmapMut::from_parts_with_stride(&mut buf, 2, 2, 16);
        view.fill([1, 2, 3, 255]);
        assert_eq!(&view.raw_data()[8..16], &[0; 8]);
        assert!(view.is_opaque());
        assert!(view.as_ref().is_opaque());

        view.set_pixel(1, 1, [1, 2, 3, 0]);
        assert!(!view.is_opaque());
    }

    #[test]
    fn try_from_parts_and_try_from_unpremultiplied() {
        let data = vec![10, 20, 30, 255, 40, 50, 60, 255];
        let ptr = data.as_ptr();
        let pixmap = Pixmap::try_from_parts(data, 2, 1).unwrap();
        assert_eq!(pixmap.data().as_ptr(), ptr);
        assert_eq!(pixmap.pixels(), &[[10, 20, 30, 255], [40, 50, 60, 255]]);

        assert!(Pixmap::try_from_parts(vec![0; 7], 2, 1).is_none());
        assert!(Pixmap::try_from_parts(vec![0; 9], 2, 1).is_none());
        assert!(Pixmap::try_from_parts(vec![], 0, 3).is_some());

        let straight = vec![255, 128, 0, 128];
        let premul = Pixmap::try_from_unpremultiplied(straight, 1, 1).unwrap();
        assert_eq!(premul.pixel(0, 0), [128, 64, 0, 128]);
        assert!(Pixmap::try_from_unpremultiplied(vec![0; 3], 1, 1).is_none());
    }

    #[test]
    fn pixels_row_pixels_and_get_pixel() {
        let mut pixmap = Pixmap::new(2, 2);
        assert_eq!(pixmap.pixels().len(), 4);
        pixmap.pixels_mut()[0] = [1, 2, 3, 255];
        pixmap.row_pixels_mut(1)[1] = [4, 5, 6, 255];

        assert_eq!(pixmap.row_pixels(0), &[[1, 2, 3, 255], [0, 0, 0, 0]]);
        assert_eq!(pixmap.row_pixels(1), &[[0, 0, 0, 0], [4, 5, 6, 255]]);
        assert_eq!(pixmap.get_pixel(0, 0), Some([1, 2, 3, 255]));
        assert_eq!(pixmap.get_pixel(1, 1), Some([4, 5, 6, 255]));
        assert_eq!(pixmap.get_pixel(2, 0), None);
        assert_eq!(pixmap.get_pixel(0, 2), None);

        let empty = Pixmap::new(0, 2);
        assert!(empty.pixels().is_empty());
        assert!(empty.row_pixels(0).is_empty());
        assert_eq!(empty.get_pixel(0, 0), None);
    }

    #[test]
    fn with_capacity_reuses_allocation_for_resize_buffer_and_reset() {
        let mut pixmap = Pixmap::with_capacity(16);
        assert_eq!((pixmap.width(), pixmap.height()), (0, 0));
        assert!(pixmap.is_empty());
        assert!(pixmap.capacity() >= 16);

        let initial_ptr = pixmap.data().as_ptr();
        pixmap.reset(4, 4);
        assert_eq!((pixmap.width(), pixmap.height()), (4, 4));
        assert_eq!(pixmap.byte_len(), 64);
        assert_eq!(pixmap.data().as_ptr(), initial_ptr);
        assert!(pixmap.data().iter().all(|&b| b == 0));

        pixmap.fill([10, 20, 30, 255]);

        // Shrinking truncates `len` without reallocating or dropping capacity.
        pixmap.resize_buffer(2, 2);
        assert_eq!((pixmap.width(), pixmap.height()), (2, 2));
        assert_eq!(pixmap.byte_len(), 16);
        assert!(pixmap.capacity() >= 16);
        assert_eq!(pixmap.data().as_ptr(), initial_ptr);
        assert_eq!(pixmap.pixel(0, 0), [10, 20, 30, 255]);

        // Growing back within capacity reuses the allocation and zeroes newly added pixels.
        pixmap.resize_buffer(3, 4);
        assert_eq!((pixmap.width(), pixmap.height()), (3, 4));
        assert_eq!(pixmap.byte_len(), 48);
        assert_eq!(pixmap.data().as_ptr(), initial_ptr);
        assert_eq!(&pixmap.data()[..16], &[10, 20, 30, 255].repeat(4)[..]);
        assert!(pixmap.data()[16..].iter().all(|&b| b == 0));

        // `reset` within capacity clears all pixels without reallocating.
        pixmap.reset(4, 3);
        assert_eq!((pixmap.width(), pixmap.height()), (4, 3));
        assert_eq!(pixmap.data().as_ptr(), initial_ptr);
        assert!(pixmap.data().iter().all(|&b| b == 0));
    }

    #[test]
    fn grow_beyond_capacity_and_shrink_to_fit() {
        let mut pixmap = Pixmap::new(2, 2);
        pixmap.fill([1, 2, 3, 255]);

        pixmap.reset(10, 10);
        assert_eq!((pixmap.width(), pixmap.height()), (10, 10));
        assert_eq!(pixmap.byte_len(), 400);
        assert!(pixmap.capacity() >= 100);
        assert!(pixmap.data().iter().all(|&b| b == 0));

        pixmap.fill([9, 8, 7, 255]);
        pixmap.resize_buffer(1, 2);
        assert_eq!(pixmap.byte_len(), 8);
        assert!(pixmap.capacity() >= 100);

        pixmap.shrink_to_fit();
        assert!(pixmap.capacity() >= 2);
        assert!(pixmap.capacity() < 100);
        assert_eq!(pixmap.pixel(0, 0), [9, 8, 7, 255]);
        assert_eq!(pixmap.pixel(0, 1), [9, 8, 7, 255]);

        // Unaligned capacity from `from_parts` reports whole pixels that fit.
        let mut raw = Vec::with_capacity(11);
        raw.extend_from_slice(&[0; 8]);
        let pixmap = Pixmap::from_parts(raw, 2, 1);
        assert!(pixmap.capacity() >= 2);
        assert_eq!(pixmap.capacity(), pixmap.data.capacity() / BYTES_PER_PIXEL);

        // Resetting to empty dimensions works cleanly.
        let mut empty = Pixmap::new(3, 3);
        empty.reset(0, 5);
        assert_eq!((empty.width(), empty.height()), (0, 5));
        assert!(empty.data().is_empty());
        assert!(empty.capacity() >= 9);
    }
}
