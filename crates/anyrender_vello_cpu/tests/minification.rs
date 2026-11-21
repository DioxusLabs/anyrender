use anyrender::{AntiAliasing, PaintScene};
use anyrender_vello_cpu::VelloCpuScenePainter;
use kurbo::{Affine, Rect, Shape};
use peniko::{
    Blob, Extend, Fill, ImageAlphaType, ImageBrush, ImageData, ImageFormat, ImageQuality,
};
use std::sync::Arc;

fn stripes(quality: ImageQuality, extend: Extend) -> ImageBrush {
    let mut bytes = Vec::new();
    for _y in 0..64 {
        for x in 0..64 {
            let v = if x % 16 < 4 { 0 } else { 255 };
            bytes.extend_from_slice(&[v, v, v, 255]);
        }
    }
    let mut image = ImageBrush::new(ImageData {
        data: Blob::new(Arc::new(bytes)),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width: 64,
        height: 64,
    });
    image.sampler.quality = quality;
    image.sampler.x_extend = extend;
    image.sampler.y_extend = extend;
    image
}

fn render(image: &ImageBrush, transform: Affine) -> Vec<[u8; 4]> {
    let mut scene = VelloCpuScenePainter::new(12, 12);
    scene.draw_image(image.as_ref(), transform);
    scene
        .finish()
        .data()
        .iter()
        .map(|p| [p.r, p.g, p.b, p.a])
        .collect()
}

// Each interior output footprint covers 75% white. Exclude finite-image
// boundaries, whose extension changes the Lanczos filter support.
#[test]
fn quality_minification_preserves_stripe_coverage() {
    for quality in [ImageQuality::Medium, ImageQuality::High] {
        let image = stripes(quality, Extend::Pad);
        let pixels = render(&image, Affine::scale(1.0 / 16.0));
        for y in 0..4 {
            for x in [1, 2] {
                let p = pixels[y * 12 + x];
                assert!((175..=205).contains(&p[0]), "expected ~75% coverage: {p:?}");
                assert_eq!(p, [p[0], p[0], p[0], 255]);
            }
        }
    }
}

#[test]
fn nearest_neighbor_minification_stays_unfiltered() {
    let pixels = render(
        &stripes(ImageQuality::Low, Extend::Pad),
        Affine::scale(1.0 / 16.0),
    );
    assert_eq!(pixels[12 + 1], [255, 255, 255, 255]);
}

#[test]
fn periodic_samplers_keep_the_existing_sampling_path() {
    for extend in [Extend::Repeat, Extend::Reflect] {
        let pixels = render(
            &stripes(ImageQuality::High, extend),
            Affine::scale(1.0 / 16.0),
        );
        assert_eq!(pixels[12 + 1], [255, 255, 255, 255]);
    }
}

#[test]
fn rotations_and_reflections_preserve_minified_coverage() {
    let image = stripes(ImageQuality::High, Extend::Pad);
    for orientation in [
        Affine::rotate(std::f64::consts::FRAC_PI_2),
        Affine::scale_non_uniform(-1.0, 1.0),
    ] {
        let pixels = render(
            &image,
            Affine::translate((6.0, 2.0)) * orientation * Affine::scale(1.0 / 16.0),
        );
        // These samples are strictly inside both transformed rectangles.
        for (x, y) in [(3, 3), (4, 4)] {
            let p = pixels[y * 12 + x];
            assert!((175..=205).contains(&p[0]), "transformed coverage: {p:?}");
            assert_eq!(p[3], 255);
        }
    }
}

#[test]
fn geometry_and_brush_transforms_produce_equivalent_pixels() {
    let image = stripes(ImageQuality::High, Extend::Pad);
    for transform in [
        Affine::translate((2.0, 2.0)) * Affine::scale(1.0 / 16.0),
        Affine::new([0.0625, 0.0, 0.015625, 0.0625, 2.0, 2.0]),
        Affine::translate((2.0, 2.0)) * Affine::scale_non_uniform(0.0625, 0.125),
    ] {
        let expected = render(&image, transform);
        let mut scene = VelloCpuScenePainter::new(12, 12);
        let shape = transform * Rect::new(0.0, 0.0, 64.0, 64.0).to_path(0.01);
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            image.as_ref(),
            Some(transform),
            &shape,
            AntiAliasing::Enabled,
        );
        let actual: Vec<_> = scene
            .finish()
            .data()
            .iter()
            .map(|p| [p.r, p.g, p.b, p.a])
            .collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn native_size_and_upscaling_preserve_solid_opaque_pixels() {
    let mut image = stripes(ImageQuality::High, Extend::Pad);
    image.image.width = 1;
    image.image.height = 1;
    image.image.data = Blob::new(Arc::new(vec![100, 50, 25, 255]));
    for scale in [1.0, 4.0] {
        let pixels = render(&image, Affine::scale(scale));
        assert_eq!(pixels[0], [100, 50, 25, 255]);
    }
}

#[test]
fn empty_images_do_not_enter_the_resampler() {
    let mut image = stripes(ImageQuality::High, Extend::Pad);
    image.image.width = 0;
    image.image.height = 0;
    image.image.data = Blob::new(Arc::new(Vec::new()));
    let pixels = render(&image, Affine::scale(1.0 / 16.0));
    assert!(pixels.iter().all(|p| *p == [0, 0, 0, 0]));
}
