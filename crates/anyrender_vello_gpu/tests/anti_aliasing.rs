#![cfg(not(target_arch = "wasm32"))]

use anyrender::{AntiAliasing, PaintScene};
use anyrender_vello_gpu::{ExternalTextures, ImageCache, ImageManager, VelloGpuScenePainter};
use kurbo::{Affine, Circle, Stroke, Vec2};
use peniko::{Blob, Color, Fill, FontData};
use vello_gpu::{RenderOptions, RenderSize, RenderTarget, RenderTargetConfig, Renderer, Scene};
use wgpu_context::{BufferRenderer, BufferRendererConfig, DeviceHandle};

#[test]
fn disabled_anti_aliasing_has_binary_coverage() {
    let instance = wgpu::Instance::default();
    let Ok(handle) = pollster::block_on(DeviceHandle::new_from_compatible_surface(
        instance, None, None, None,
    )) else {
        eprintln!("Skipping GPU coverage test: no adapter/device available");
        return;
    };
    let target = BufferRenderer::new(
        BufferRendererConfig {
            width: 64,
            height: 64,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        },
        handle.clone(),
        0,
    );
    let (mut renderer, mut resources) = Renderer::new(
        &handle.device,
        &handle.queue,
        &RenderTargetConfig {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: 64,
            height: 64,
        },
    );
    let depth = renderer.create_depth_texture_view(&RenderSize {
        width: 64,
        height: 64,
    });
    let mut cache = ImageCache::default();
    let mut textures = ExternalTextures::default();
    let font = FontData::new(
        Blob::new(std::sync::Arc::new(
            include_bytes!("../../../assets/fonts/roboto/Roboto.ttf").to_vec(),
        )),
        0,
    );
    for kind in ["fill", "stroke", "glyphs"] {
        for aa in [AntiAliasing::Disabled, AntiAliasing::Enabled] {
            let mut scene = Scene::new(64, 64);
            let mut painter = VelloGpuScenePainter::new(
                &mut scene,
                ImageManager::new(
                    &mut resources,
                    &handle.device,
                    &handle.queue,
                    &mut cache,
                    &mut textures,
                ),
                &handle,
            )
            .with_glyph_caching(true);
            let circle = Circle::new((30.3, 30.7), 18.2);
            match kind {
                "fill" => painter.fill(
                    Fill::NonZero,
                    Affine::IDENTITY,
                    Color::WHITE,
                    None,
                    &circle,
                    aa,
                ),
                "stroke" => painter.stroke(
                    &Stroke::new(3.2),
                    Affine::IDENTITY,
                    Color::WHITE,
                    None,
                    &circle,
                    aa,
                ),
                _ => painter.draw_glyphs(
                    &font,
                    40.0,
                    false,
                    &[],
                    Vec2::ZERO,
                    Fill::NonZero,
                    Color::WHITE,
                    1.0,
                    Affine::IDENTITY,
                    None,
                    [anyrender::Glyph {
                        id: 36,
                        x: 10.3,
                        y: 48.7,
                    }]
                    .into_iter(),
                    aa,
                ),
            }
            drop(painter);
            let mut encoder = handle.device.create_command_encoder(&Default::default());
            renderer
                .render(
                    &scene,
                    &mut resources,
                    &mut encoder,
                    RenderTarget {
                        view: &target.target_texture_view(),
                        depth_view: Some(&depth),
                    },
                    textures.bindings(),
                    RenderOptions::default(),
                )
                .unwrap();
            handle.queue.submit([encoder.finish()]);
            let mut pixels = vec![0; 64 * 64 * 4];
            target.copy_texture_to_buffer(&mut pixels);
            let partial = pixels
                .chunks_exact(4)
                .filter(|p| p[3] != 0 && p[3] != 255)
                .count();
            let opaque = pixels.chunks_exact(4).filter(|p| p[3] == 255).count();
            eprintln!("{kind} {aa:?}: {partial} partial, {opaque} opaque pixels");
            assert!(opaque > 0, "{kind} rendered nothing");
            if aa == AntiAliasing::Disabled {
                assert_eq!(partial, 0, "{kind} ignored AntiAliasing::Disabled");
            } else {
                assert!(partial > 0, "{kind} lost antialiasing");
            }
        }
    }
}
