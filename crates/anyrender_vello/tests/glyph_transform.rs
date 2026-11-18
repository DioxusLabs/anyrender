#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use anyrender::{AntiAliasing, Glyph, ImageRenderer, PaintScene};
use anyrender_tiny_skia::TinySkiaImageRenderer;
use anyrender_vello::VelloScenePainter;
use anyrender_vello_cpu::VelloCpuImageRenderer;
use kurbo::{Affine, Vec2};
use peniko::{Blob, Color, Fill, FontData};
use wgpu_context::{BufferRendererConfig, WGPUContext};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 160;

fn draw_glyph(painter: &mut impl PaintScene, font: &FontData, glyph_transform: Affine) {
    painter.draw_glyphs(
        font,
        80.0,
        false,
        &[],
        Vec2::ZERO,
        Fill::NonZero,
        Color::WHITE,
        1.0,
        Affine::translate((30.0, 20.0)),
        Some(glyph_transform),
        std::iter::once(Glyph {
            id: 45, // 'I' in Roboto
            x: 50.0,
            y: 95.0,
        }),
        AntiAliasing::Enabled,
    );
}

fn ink_range(pixels: &[u8], y: usize) -> Option<[usize; 2]> {
    let row_width = WIDTH as usize * 4;
    let row = &pixels[y * row_width..(y + 1) * row_width];
    let mut ink = row
        .chunks_exact(4)
        .enumerate()
        .filter_map(|(x, pixel)| (pixel[3] > 128).then_some(x));
    let left = ink.next()?;
    Some([left, ink.next_back().unwrap_or(left)])
}

#[test]
#[cfg_attr(
    target_os = "windows",
    ignore = "Vello GPU rendering is not supported on hosted Windows runners"
)]
fn glyph_transforms_use_y_down_glyph_local_space() {
    let mut context = WGPUContext::new();
    let Ok(target) = pollster::block_on(context.create_buffer_renderer(BufferRendererConfig {
        width: WIDTH,
        height: HEIGHT,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
    })) else {
        eprintln!("Skipping glyph transform comparison: no adapter/device available");
        return;
    };
    let mut vello = vello::Renderer::new(
        target.device(),
        vello::RendererOptions {
            antialiasing_support: vello::AaSupport::area_only(),
            ..Default::default()
        },
    )
    .unwrap();
    let mut render_vello = |scene: &vello::Scene, pixels: &mut [u8]| {
        vello
            .render_to_texture(
                target.device(),
                target.queue(),
                scene,
                &target.target_texture_view(),
                &vello::RenderParams {
                    base_color: Color::TRANSPARENT,
                    width: WIDTH,
                    height: HEIGHT,
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .unwrap();
        target.copy_texture_to_buffer(pixels);
    };
    let mut cpu = VelloCpuImageRenderer::new(WIDTH, HEIGHT);
    let mut tiny = TinySkiaImageRenderer::new(WIDTH, HEIGHT);
    let mut scene = vello::Scene::new();
    let mut actual = vec![0; (WIDTH * HEIGHT * 4) as usize];
    let mut expected_cpu = actual.clone();
    let mut expected_tiny = actual.clone();
    let font = FontData::new(
        Blob::new(Arc::new(include_bytes!(
            "../../../assets/fonts/roboto/Roboto.ttf"
        ))),
        0,
    );

    for glyph_transform in [
        Affine::skew(0.5, 0.0),
        Affine::skew(-0.5, 0.0),
        Affine::scale_non_uniform(2.0, 1.0),
        Affine::translate((3.0, 9.0)),
    ] {
        scene.reset();
        draw_glyph(
            &mut VelloScenePainter::new(&mut scene),
            &font,
            glyph_transform,
        );
        render_vello(&scene, &mut actual);
        cpu.reset();
        cpu.render(
            |painter| draw_glyph(painter, &font, glyph_transform),
            &mut expected_cpu,
        );
        tiny.reset();
        tiny.render(
            |painter| draw_glyph(painter, &font, glyph_transform),
            &mut expected_tiny,
        );

        for y in [70, 110, 120] {
            for (backend, pixels) in [("Vello CPU", &expected_cpu), ("TinySkia", &expected_tiny)] {
                let expected = ink_range(pixels, y);
                let actual = ink_range(&actual, y);
                assert!(
                    match (actual, expected) {
                        (Some(a), Some(b)) => a.into_iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 1),
                        (None, None) => y == 120,
                        _ => false,
                    },
                    "{glyph_transform:?}, row {y}: classic Vello {actual:?}, {backend} {expected:?}"
                );
            }
        }
    }
}
