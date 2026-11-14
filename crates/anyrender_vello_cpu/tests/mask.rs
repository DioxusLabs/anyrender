use anyrender::{MaskMode, PaintScene};
use anyrender_vello_cpu::VelloCpuScenePainter;
use kurbo::{Affine, BezPath, Rect, Shape};
use peniko::{BlendMode, Color, Fill};

const RED: [u8; 4] = [255, 0, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
const CLEAR: [u8; 4] = [0, 0, 0, 0];

fn fill(scene: &mut VelloCpuScenePainter, color: Color, rect: Rect) {
    scene.fill(Fill::NonZero, Affine::IDENTITY, color, None, &rect);
}

/// The pixels of a 16x16 scene, row by row.
fn render(scene: VelloCpuScenePainter) -> Vec<[u8; 4]> {
    let pixmap = scene.finish();
    pixmap.data().iter().map(|p| [p.r, p.g, p.b, p.a]).collect()
}

/// Masks the left half of a red canvas with half-opaque content covering the whole canvas,
/// clipped to the left half by `clip`.
fn half_masked(clip: &impl Shape) -> Vec<[u8; 4]> {
    let mut scene = VelloCpuScenePainter::new(16, 16);
    let canvas = Rect::new(0.0, 0.0, 16.0, 16.0);
    fill(&mut scene, Color::from_rgb8(255, 0, 0), canvas);
    scene.push_mask_layer(Fill::NonZero, MaskMode::Alpha, Affine::IDENTITY, clip);
    fill(&mut scene, Color::from_rgba8(0, 0, 0, 128), canvas);
    scene.pop_layer();
    render(scene)
}

#[test]
fn mask_with_rect_clip_removes_content_outside_it() {
    let pixels = half_masked(&Rect::new(0.0, 0.0, 8.0, 16.0));
    assert_eq!(pixels[8 * 16 + 2], [128, 0, 0, 128]);
    assert_eq!(pixels[8 * 16 + 12], CLEAR);
}

#[test]
fn mask_with_path_clip_removes_content_outside_it() {
    let clip: BezPath = Rect::new(0.0, 0.0, 8.0, 16.0).to_path(0.1);
    let pixels = half_masked(&clip);
    assert_eq!(pixels[8 * 16 + 2], [128, 0, 0, 128]);
    assert_eq!(pixels[8 * 16 + 12], CLEAR);
}

#[test]
fn mask_only_affects_its_enclosing_layer() {
    let mut scene = VelloCpuScenePainter::new(16, 16);
    let canvas = Rect::new(0.0, 0.0, 16.0, 16.0);
    fill(&mut scene, Color::from_rgb8(255, 0, 0), canvas);
    scene.push_layer(
        Fill::NonZero,
        BlendMode::default(),
        1.0,
        Affine::IDENTITY,
        &Rect::new(0.0, 0.0, 12.0, 16.0),
        None,
        None,
    );
    fill(&mut scene, Color::from_rgb8(0, 0, 255), canvas);
    let clip: BezPath = Rect::new(0.0, 0.0, 8.0, 16.0).to_path(0.1);
    scene.push_mask_layer(Fill::NonZero, MaskMode::Alpha, Affine::IDENTITY, &clip);
    fill(&mut scene, Color::BLACK, canvas);
    scene.pop_layer();
    scene.pop_layer();
    let pixels = render(scene);
    assert_eq!(pixels[8 * 16 + 2], BLUE);
    assert_eq!(pixels[8 * 16 + 10], RED);
    assert_eq!(pixels[8 * 16 + 14], RED);
}
