//! Conversion of [`peniko::ImageData`] to vello_cpu [`Pixmap`]s.

use std::sync::Arc;
use vello_common::pixmap::PixelMetadata;
use vello_cpu::Pixmap;

/// Convert a [`peniko::ImageData`] to a premultiplied RGBA8 [`Pixmap`].
///
/// Equivalent to `ImageSource::from_peniko_image_data`, but computes an exact
/// transparency hint for already-premultiplied images.
///
/// # Panics
///
/// Panics if `image` has a `width` or `height` greater than `u16::MAX`.
pub(crate) fn convert_image(image: &peniko::ImageData) -> Arc<Pixmap> {
    assert!(
        image.width <= u16::MAX as u32 && image.height <= u16::MAX as u32,
        "The image is too big. Its width and height can be no larger than {} pixels.",
        u16::MAX,
    );
    let width = image.width.try_into().unwrap();
    let height = image.height.try_into().unwrap();

    let data = image.data.data();
    let pixel_bytes = data.len() & !3;
    let mut bytes = Vec::with_capacity(pixel_bytes);
    bytes.extend_from_slice(&data[..pixel_bytes]);

    match image.format {
        peniko::ImageFormat::Rgba8 => {}
        peniko::ImageFormat::Bgra8 => {
            for pixel in bytes.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }
        format => unimplemented!("Unsupported image format: {format:?}"),
    }

    // `Pixmap::from_parts` premultiplies (and computes an exact transparency
    // hint for) non-premultiplied data, but trusts the hint for premultiplied data.
    let may_have_transparency = match image.alpha_type {
        peniko::ImageAlphaType::AlphaPremultiplied => {
            bytes.as_chunks::<4>().0.iter().any(|p| p[3] != 255)
        }
        _ => true,
    };

    Arc::new(Pixmap::from_parts(
        bytes,
        width,
        height,
        PixelMetadata::new(image.alpha_type, may_have_transparency),
    ))
}

/// Prefilter premultiplied pixels: straight-alpha resizing creates dark fringes.
pub(crate) fn minify_image(image: &peniko::ImageData, width: u32, height: u32) -> Arc<Pixmap> {
    let source = convert_image(image);
    let bytes = source
        .data()
        .iter()
        .flat_map(|p| [p.r, p.g, p.b, p.a])
        .collect();
    let rgba = image::RgbaImage::from_raw(image.width, image.height, bytes).unwrap();
    let mut resized =
        image::imageops::resize(&rgba, width, height, image::imageops::FilterType::Lanczos3)
            .into_raw();
    for p in resized.chunks_exact_mut(4) {
        p[0] = p[0].min(p[3]);
        p[1] = p[1].min(p[3]);
        p[2] = p[2].min(p[3]);
    }
    let transparent = resized.chunks_exact(4).any(|p| p[3] != 255);
    Arc::new(Pixmap::from_parts(
        resized,
        width as u16,
        height as u16,
        PixelMetadata::new(peniko::ImageAlphaType::AlphaPremultiplied, transparent),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use peniko::{Blob, ImageAlphaType, ImageData, ImageFormat};

    fn image(pixels: &[[u8; 4]]) -> ImageData {
        ImageData {
            data: Blob::new(Arc::new(pixels.concat())),
            format: ImageFormat::Rgba8,
            alpha_type: ImageAlphaType::Alpha,
            width: pixels.len() as u32,
            height: 1,
        }
    }

    #[test]
    fn minification_does_not_bleed_transparent_rgb() {
        let image = image(&[[255, 0, 0, 255], [0, 0, 255, 0]]);
        let pixmap = minify_image(&image, 1, 1);
        let p = pixmap.data()[0];
        assert_eq!((p.g, p.b), (0, 0));
        assert_eq!(p.r, p.a);
        assert!((120..=135).contains(&p.a));
    }

    #[test]
    fn premultiply() {
        let pixmap = convert_image(&image(&[[100, 150, 200, 128], [10, 20, 30, 255]]));
        assert!(pixmap.may_have_transparency());
        let px = pixmap.data()[0];
        assert_eq!((px.r, px.g, px.b, px.a), (50, 75, 100, 128));
        let px = pixmap.data()[1];
        assert_eq!((px.r, px.g, px.b, px.a), (10, 20, 30, 255));
    }

    #[test]
    fn premultiply_bgra_opaque() {
        let mut img = image(&[[1, 2, 3, 255]; 20]);
        img.format = ImageFormat::Bgra8;
        let pixmap = convert_image(&img);
        assert!(!pixmap.may_have_transparency());
        let px = pixmap.data()[0];
        assert_eq!((px.r, px.g, px.b, px.a), (3, 2, 1, 255));
    }
}
