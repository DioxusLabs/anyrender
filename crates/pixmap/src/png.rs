use alloc::vec;
use alloc::vec::Vec;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Cursor, Seek, Write};
use std::path::Path;

use ::png::{BitDepth, ColorType, Decoder, DecodingError, Encoder, EncodingError, Transformations};

use crate::util::checked_byte_len;
use crate::{
    BYTES_PER_PIXEL, Pixmap, PixmapMut, PixmapRef, premultiply_rgba8, unpremultiply_rgba8,
};

impl Pixmap {
    /// Decodes a PNG image from a seekable buffered reader into a premultiplied RGBA8 [`Pixmap`].
    ///
    /// All standard PNG color types (grayscale, grayscale+alpha, RGB, RGBA, and indexed palettes
    /// with optional `tRNS` transparency) and bit depths (1, 2, 4, 8, and 16-bit) are normalized
    /// to 8-bit RGBA and premultiplied by alpha.
    ///
    /// Returns [`DecodingError::LimitsExceeded`] if the image dimensions exceed `u16::MAX` or its
    /// decoded RGBA byte length overflows `usize`.
    pub fn from_png(data: impl BufRead + Seek) -> Result<Self, DecodingError> {
        let mut decoder = Decoder::new(data);
        decoder
            .set_transformations(Transformations::normalize_to_color8() | Transformations::ALPHA);

        let mut reader = decoder.read_info()?;
        let info = reader.info();
        let width: u16 = info
            .width
            .try_into()
            .map_err(|_| DecodingError::LimitsExceeded)?;
        let height: u16 = info
            .height
            .try_into()
            .map_err(|_| DecodingError::LimitsExceeded)?;

        let rgba_len = checked_byte_len(width, height).ok_or(DecodingError::LimitsExceeded)?;

        let (color_type, bit_depth) = reader.output_color_type();
        debug_assert_eq!(
            bit_depth,
            BitDepth::Eight,
            "normalize_to_color8 ensures 8-bit output"
        );

        let mut rgba = vec![0_u8; rgba_len];
        match color_type {
            ColorType::Rgba => {
                debug_assert_eq!(reader.output_buffer_size(), Some(rgba.len()));
                reader.next_frame(&mut rgba)?;
            }
            ColorType::GrayscaleAlpha => {
                let gray_len = reader
                    .output_buffer_size()
                    .ok_or(DecodingError::LimitsExceeded)?;
                let mut gray_buf = vec![0_u8; gray_len];
                reader.next_frame(&mut gray_buf)?;

                let (gray_pixels, gray_rem) = gray_buf.as_chunks::<2>();
                let (rgba_pixels, rgba_rem) = rgba.as_chunks_mut::<BYTES_PER_PIXEL>();
                debug_assert!(gray_rem.is_empty() && rgba_rem.is_empty());
                for (&[gray, alpha], dst) in gray_pixels.iter().zip(rgba_pixels.iter_mut()) {
                    *dst = [gray, gray, gray, alpha];
                }
            }
            ColorType::Rgb | ColorType::Grayscale => {
                unreachable!("Transformations::ALPHA always produces an alpha channel");
            }
            ColorType::Indexed => {
                unreachable!("Transformations::normalize_to_color8 always expands indexed PNGs");
            }
        }

        premultiply_rgba8(&mut rgba);
        Ok(Self::from_raw_parts(rgba, width, height))
    }

    /// Decodes a PNG image from an in-memory byte slice into a premultiplied RGBA8 [`Pixmap`].
    pub fn decode_png(data: &[u8]) -> Result<Self, DecodingError> {
        Self::from_png(Cursor::new(data))
    }

    /// Loads and decodes a PNG file from disk into a premultiplied RGBA8 [`Pixmap`].
    pub fn load_png(path: impl AsRef<Path>) -> Result<Self, DecodingError> {
        let file = File::open(path)?;
        Self::from_png(BufReader::new(file))
    }

    /// Consumes the pixmap and encodes it as a PNG byte vector.
    ///
    /// Fully opaque pixmaps are encoded as 8-bit RGB; pixmaps with transparency are unpremultiplied
    /// in place (without copying the pixel buffer) and encoded as 8-bit RGBA.
    pub fn into_png(self) -> Result<Vec<u8>, EncodingError> {
        if self.is_opaque() {
            return self.encode_png();
        }
        let width = u32::from(self.width());
        let height = u32::from(self.height());
        let straight = self.into_unpremultiplied();

        let mut out = Vec::new();
        let mut encoder = Encoder::new(&mut out, width, height);
        encoder.set_color(ColorType::Rgba);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&straight)?;
        writer.finish()?;
        Ok(out)
    }

    /// Encodes the pixmap as a PNG byte vector.
    ///
    /// Fully opaque pixmaps are encoded as 8-bit RGB; pixmaps with transparency are unpremultiplied
    /// to straight-alpha 8-bit RGBA one row at a time.
    pub fn encode_png(&self) -> Result<Vec<u8>, EncodingError> {
        self.as_ref().encode_png()
    }

    /// Encodes the pixmap as a PNG directly to `writer`.
    pub fn write_png(&self, writer: impl Write) -> Result<(), EncodingError> {
        self.as_ref().write_png(writer)
    }

    /// Saves the pixmap to `path` as a PNG file.
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), EncodingError> {
        self.as_ref().save_png(path)
    }
}

impl PixmapRef<'_> {
    /// Encodes this view as a PNG byte vector.
    ///
    /// Fully opaque views are encoded as 8-bit RGB; views with transparency are unpremultiplied to
    /// straight-alpha 8-bit RGBA one row at a time, using only a single row of scratch memory.
    pub fn encode_png(&self) -> Result<Vec<u8>, EncodingError> {
        let mut out = Vec::new();
        self.write_png(&mut out)?;
        Ok(out)
    }

    /// Encodes this view as a PNG directly to `writer`.
    ///
    /// Handles arbitrary row strides and subregions without copying the full image.
    pub fn write_png(&self, out: impl Write) -> Result<(), EncodingError> {
        let width = usize::from(self.width());
        let opaque = self.is_opaque();

        let mut encoder = Encoder::new(out, u32::from(self.width()), u32::from(self.height()));
        encoder.set_color(if opaque {
            ColorType::Rgb
        } else {
            ColorType::Rgba
        });
        encoder.set_depth(BitDepth::Eight);

        let mut writer = encoder.write_header()?;
        let mut stream = writer.stream_writer()?;

        if opaque {
            let mut row_rgb = vec![0_u8; width * 3];
            for row in self.rows() {
                let (src_pixels, src_rem) = row.as_chunks::<BYTES_PER_PIXEL>();
                let (dst_pixels, dst_rem) = row_rgb.as_chunks_mut::<3>();
                debug_assert!(src_rem.is_empty() && dst_rem.is_empty());
                for (&[r, g, b, _], dst) in src_pixels.iter().zip(dst_pixels.iter_mut()) {
                    *dst = [r, g, b];
                }
                stream.write_all(&row_rgb)?;
            }
        } else {
            let mut row_rgba = vec![0_u8; width * BYTES_PER_PIXEL];
            for row in self.rows() {
                row_rgba.copy_from_slice(row);
                unpremultiply_rgba8(&mut row_rgba);
                stream.write_all(&row_rgba)?;
            }
        }

        stream.finish()?;
        writer.finish()?;
        Ok(())
    }

    /// Saves this view to `path` as a PNG file.
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), EncodingError> {
        let file = File::create(path)?;
        self.write_png(BufWriter::new(file))
    }
}

impl PixmapMut<'_> {
    /// Encodes this view as a PNG byte vector.
    pub fn encode_png(&self) -> Result<Vec<u8>, EncodingError> {
        self.as_ref().encode_png()
    }

    /// Encodes this view as a PNG directly to `writer`.
    pub fn write_png(&self, writer: impl Write) -> Result<(), EncodingError> {
        self.as_ref().write_png(writer)
    }

    /// Saves this view to `path` as a PNG file.
    pub fn save_png(&self, path: impl AsRef<Path>) -> Result<(), EncodingError> {
        self.as_ref().save_png(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec};
    use std::io::Cursor;

    #[test]
    fn round_trip_semitransparent_and_opaque_png() {
        // Semitransparent pixmap (encoded as RGBA, unpremultiplied -> premultiplied round-trip).
        let mut pixmap = Pixmap::new(3, 2);
        pixmap.set_pixel(0, 0, [255, 0, 0, 255]);
        pixmap.set_pixel(1, 0, [128, 64, 0, 128]);
        pixmap.set_pixel(2, 0, [0, 0, 0, 0]);
        pixmap.set_pixel(0, 1, [10, 20, 30, 50]);
        pixmap.set_pixel(1, 1, [200, 150, 100, 255]);
        pixmap.set_pixel(2, 1, [17, 17, 17, 17]);

        let encoded_ref = pixmap.encode_png().unwrap();
        let encoded_owned = pixmap.clone().into_png().unwrap();

        // Verify both encoded PNGs use RGBA since there is transparency and decode identically.
        for encoded in [&encoded_ref, &encoded_owned] {
            let info = Decoder::new(Cursor::new(encoded))
                .read_info()
                .unwrap()
                .info()
                .clone();
            assert_eq!(info.color_type, ColorType::Rgba);
            assert_eq!(Pixmap::decode_png(encoded).unwrap(), pixmap);
        }

        // Fully opaque pixmap is encoded as compact RGB and decodes identically.
        let mut opaque = Pixmap::new(2, 2);
        opaque.set_pixel(0, 0, [10, 20, 30, 255]);
        opaque.set_pixel(1, 0, [40, 50, 60, 255]);
        opaque.set_pixel(0, 1, [70, 80, 90, 255]);
        opaque.set_pixel(1, 1, [100, 110, 120, 255]);

        let opaque_png = opaque.encode_png().unwrap();
        assert_eq!(opaque_png, opaque.clone().into_png().unwrap());
        let opaque_info = Decoder::new(Cursor::new(&opaque_png))
            .read_info()
            .unwrap()
            .info()
            .clone();
        assert_eq!(opaque_info.color_type, ColorType::Rgb);
        assert_eq!(Pixmap::decode_png(&opaque_png).unwrap(), opaque);
    }

    #[test]
    fn encode_strided_subregion_and_save_load_file() {
        let mut backing = vec![0xAA_u8; 4 * 20];
        let mut view = PixmapMut::from_parts_with_stride(&mut backing, 4, 4, 20);
        for y in 0..4 {
            for x in 0..4 {
                let a = if (x + y) % 2 == 0 { 255 } else { 128 };
                let r = ((x * 50) as u8).min(a);
                let g = ((y * 50) as u8).min(a);
                view.set_pixel(x, y, [r, g, 0, a]);
            }
        }

        let sub = view.subregion(1, 1, 2, 2);
        let bytes = sub.encode_png().unwrap();
        assert_eq!(Pixmap::decode_png(&bytes).unwrap(), sub.to_owned());

        let path = std::env::temp_dir().join(format!("pixmap-png-test-{}.png", std::process::id()));
        view.save_png(&path).unwrap();
        let loaded = Pixmap::load_png(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded, view.as_ref().to_owned());
    }

    #[test]
    fn decode_grayscale_grayscale_alpha_indexed_and_16bit_pngs() {
        // Grayscale (8-bit)
        let mut gray_png = Vec::new();
        {
            let mut enc = Encoder::new(&mut gray_png, 2, 1);
            enc.set_color(ColorType::Grayscale);
            enc.set_depth(BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[42, 200]).unwrap();
        }
        let decoded = Pixmap::decode_png(&gray_png).unwrap();
        assert_eq!(decoded.pixels(), &[[42, 42, 42, 255], [200, 200, 200, 255]]);

        // GrayscaleAlpha (8-bit)
        let mut ga_png = Vec::new();
        {
            let mut enc = Encoder::new(&mut ga_png, 2, 1);
            enc.set_color(ColorType::GrayscaleAlpha);
            enc.set_depth(BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[255, 128, 100, 0]).unwrap();
        }
        let decoded = Pixmap::decode_png(&ga_png).unwrap();
        assert_eq!(decoded.pixels(), &[[128, 128, 128, 128], [0, 0, 0, 0]]);

        // Indexed palette with tRNS
        let mut idx_png = Vec::new();
        {
            let mut enc = Encoder::new(&mut idx_png, 2, 1);
            enc.set_color(ColorType::Indexed);
            enc.set_depth(BitDepth::Eight);
            enc.set_palette(&[255, 0, 0, 0, 255, 0]);
            enc.set_trns(&[128, 255]);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[0, 1]).unwrap();
        }
        let decoded = Pixmap::decode_png(&idx_png).unwrap();
        assert_eq!(decoded.pixels(), &[[128, 0, 0, 128], [0, 255, 0, 255]]);

        // 16-bit RGBA
        let mut rgba16_png = Vec::new();
        {
            let mut enc = Encoder::new(&mut rgba16_png, 1, 1);
            enc.set_color(ColorType::Rgba);
            enc.set_depth(BitDepth::Sixteen);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[0xFF, 0xFF, 0x00, 0x00, 0x80, 0x80, 0xFF, 0xFF])
                .unwrap();
        }
        let decoded = Pixmap::decode_png(&rgba16_png).unwrap();
        assert_eq!(decoded.pixel(0, 0), [255, 0, 128, 255]);
    }

    #[test]
    fn empty_encode_and_invalid_png_return_errors() {
        assert!(Pixmap::new(0, 4).encode_png().is_err());
        assert!(Pixmap::new(4, 0).into_png().is_err());
        assert!(Pixmap::decode_png(b"not a png").is_err());
    }
}
