use anyrender::PaintScene;
use anyrender_vello_cpu::VelloCpuScenePainter;
use kurbo::{Affine, Rect};
use peniko::{Color, Fill};

#[test]
fn finish_renders_scene() {
    let mut scene = VelloCpuScenePainter::new(16, 16);
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(255, 0, 0),
        None,
        &Rect::new(0.0, 0.0, 8.0, 16.0),
    );
    let pixmap = scene.finish();
    let pixel = |x: usize, y: usize| {
        let p = pixmap.data()[y * 16 + x];
        [p.r, p.g, p.b, p.a]
    };
    assert_eq!(pixel(2, 8), [255, 0, 0, 255]);
    assert_eq!(pixel(12, 8), [0, 0, 0, 0]);
}
