use anyrender::{ImageRenderer, render_to_buffer};
use anyrender_tiny_skia::TinySkiaImageRenderer;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use kurbo::Affine;

fn check_clip_shape_rendering<R: ImageRenderer>() {
    for rendering in ["auto", "crispEdges", "optimizeSpeed"] {
        let svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">
                <clipPath id="clip">
                    <circle cx="30.3" cy="30.7" r="18.2" shape-rendering="{rendering}"/>
                </clipPath>
                <rect width="64" height="64" fill="white" clip-path="url(#clip)"/>
            </svg>"#
        );
        let tree = usvg::Tree::from_str(&svg, &usvg::Options::default()).unwrap();
        let buffer = render_to_buffer::<R, _>(
            |scene| anyrender_svg::render_svg_tree(scene, &tree, Affine::IDENTITY),
            64,
            64,
        );
        let partial = buffer
            .chunks_exact(4)
            .filter(|pixel| pixel[3] != 0 && pixel[3] != 255)
            .count();
        let opaque = buffer
            .chunks_exact(4)
            .filter(|pixel| pixel[3] == 255)
            .count();
        eprintln!("{rendering}: {partial} partial, {opaque} opaque pixels");
        assert!(opaque > 0, "clip rendered nothing");
        assert_eq!(&buffer[..4], &[0, 0, 0, 0], "clip did not apply");
        if rendering == "auto" {
            assert!(partial > 0, "default clip lost antialiasing");
        } else {
            assert_eq!(partial, 0, "clip ignored shape-rendering={rendering}");
        }
    }
}

#[test]
fn clip_shape_rendering_tiny_skia() {
    check_clip_shape_rendering::<TinySkiaImageRenderer>();
}

#[test]
fn clip_shape_rendering_vello_cpu() {
    check_clip_shape_rendering::<VelloCpuImageRenderer>();
}
