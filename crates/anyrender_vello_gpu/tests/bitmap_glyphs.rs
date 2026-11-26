#![cfg(not(target_arch = "wasm32"))]

use anyrender::{AntiAliasing, Glyph, PaintScene};
use anyrender_vello_gpu::{ExternalTextures, ImageCache, ImageManager, VelloGpuScenePainter};
use kurbo::{Affine, Stroke, Vec2};
use peniko::{Blob, Color, Fill, FontData, StyleRef};
use vello_gpu::{RenderOptions, RenderSize, RenderTarget, RenderTargetConfig, Renderer, Scene};
use wgpu_context::{BufferRenderer, BufferRendererConfig, DeviceHandle};

/// Minimal 4x4 opaque red RGBA PNG.
const RED_4X4_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 4, 0, 0, 0, 4, 8, 6, 0,
    0, 0, 169, 241, 158, 126, 0, 0, 0, 18, 73, 68, 65, 84, 120, 156, 99, 248, 207, 192, 240, 31,
    25, 51, 144, 46, 0, 0, 60, 64, 31, 225, 224, 129, 119, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174,
    66, 96, 130,
];

/// Build a minimal valid OpenType font with an `sbix` PNG bitmap strike for glyph ID 1.
fn make_sbix_bitmap_font() -> FontData {
    let mut head = [0u8; 54];
    head[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    head[18..20].copy_from_slice(&1000u16.to_be_bytes()); // unitsPerEm

    let mut hhea = [0u8; 36];
    hhea[0..4].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    hhea[34..36].copy_from_slice(&2u16.to_be_bytes()); // numberOfHMetrics

    let mut hmtx = Vec::new();
    for _ in 0..2 {
        hmtx.extend_from_slice(&1000u16.to_be_bytes()); // advanceWidth
        hmtx.extend_from_slice(&0i16.to_be_bytes()); // lsb
    }

    let mut maxp = [0u8; 6];
    maxp[0..4].copy_from_slice(&0x0000_5000u32.to_be_bytes()); // version 0.5
    maxp[4..6].copy_from_slice(&2u16.to_be_bytes()); // numGlyphs

    let mut sbix = Vec::new();
    sbix.extend_from_slice(&1u16.to_be_bytes()); // version
    sbix.extend_from_slice(&1u16.to_be_bytes()); // flags
    sbix.extend_from_slice(&1u32.to_be_bytes()); // numStrikes
    sbix.extend_from_slice(&12u32.to_be_bytes()); // strikeOffset[0]
    // Strike header (ppem = 16, ppi = 72)
    sbix.extend_from_slice(&16u16.to_be_bytes());
    sbix.extend_from_slice(&72u16.to_be_bytes());
    let glyph1_start = 4u32 + 3 * 4;
    let glyph1_end = glyph1_start + 8 + RED_4X4_PNG.len() as u32;
    sbix.extend_from_slice(&glyph1_start.to_be_bytes()); // glyph 0 start (empty)
    sbix.extend_from_slice(&glyph1_start.to_be_bytes()); // glyph 1 start
    sbix.extend_from_slice(&glyph1_end.to_be_bytes()); // glyph 1 end
    sbix.extend_from_slice(&0i16.to_be_bytes()); // originOffsetX
    sbix.extend_from_slice(&0i16.to_be_bytes()); // originOffsetY
    sbix.extend_from_slice(b"png ");
    sbix.extend_from_slice(RED_4X4_PNG);

    let tables: [(&[u8; 4], &[u8]); 5] = [
        (b"head", &head),
        (b"hhea", &hhea),
        (b"hmtx", &hmtx),
        (b"maxp", &maxp),
        (b"sbix", &sbix),
    ];

    let num_tables = tables.len() as u16;
    let mut font = Vec::new();
    font.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // sfntVersion
    font.extend_from_slice(&num_tables.to_be_bytes());
    font.extend_from_slice(&64u16.to_be_bytes()); // searchRange
    font.extend_from_slice(&2u16.to_be_bytes()); // entrySelector
    font.extend_from_slice(&16u16.to_be_bytes()); // rangeShift

    let mut offset = 12 + tables.len() * 16;
    for (tag, data) in tables {
        font.extend_from_slice(tag);
        font.extend_from_slice(&0u32.to_be_bytes()); // checksum
        font.extend_from_slice(&(offset as u32).to_be_bytes());
        font.extend_from_slice(&(data.len() as u32).to_be_bytes());
        offset += (data.len() + 3) & !3;
    }
    for (_, data) in tables {
        font.extend_from_slice(data);
        while font.len() % 4 != 0 {
            font.push(0);
        }
    }

    FontData::new(Blob::new(std::sync::Arc::new(font)), 0)
}

#[test]
fn bitmap_glyphs_render_without_panicking() {
    let instance = wgpu::Instance::default();
    let Ok(handle) = pollster::block_on(DeviceHandle::new_from_compatible_surface(
        instance, None, None, None,
    )) else {
        eprintln!("Skipping GPU bitmap glyph test: no adapter/device available");
        return;
    };
    let width = 128u16;
    let height = 128u16;
    let target = BufferRenderer::new(
        BufferRendererConfig {
            width: u32::from(width),
            height: u32::from(height),
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
            width,
            height,
        },
    );
    let depth = renderer.create_depth_texture_view(&RenderSize { width, height });
    let mut cache = ImageCache::default();
    let mut textures = ExternalTextures::default();
    let font = make_sbix_bitmap_font();

    struct Case {
        name: &'static str,
        glyph_caching: bool,
        font_size: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        stroke: bool,
        aa: AntiAliasing,
    }

    let cases = [
        Case {
            name: "default_uncached",
            glyph_caching: false,
            font_size: 32.0,
            transform: Affine::translate((24.0, 80.0)),
            glyph_transform: None,
            stroke: false,
            aa: AntiAliasing::Enabled,
        },
        Case {
            name: "cached",
            glyph_caching: true,
            font_size: 32.0,
            transform: Affine::translate((24.0, 80.0)),
            glyph_transform: None,
            stroke: false,
            aa: AntiAliasing::Enabled,
        },
        Case {
            name: "large_font_size",
            glyph_caching: false,
            font_size: 160.0,
            transform: Affine::translate((16.0, 110.0)),
            glyph_transform: None,
            stroke: false,
            aa: AntiAliasing::Enabled,
        },
        Case {
            name: "scaled_over_max_cached_size",
            glyph_caching: true,
            font_size: 80.0,
            transform: Affine::translate((16.0, 110.0)) * Affine::scale(2.0),
            glyph_transform: None,
            stroke: false,
            aa: AntiAliasing::Enabled,
        },
        Case {
            name: "skewed",
            glyph_caching: true,
            font_size: 32.0,
            transform: Affine::translate((32.0, 80.0)),
            glyph_transform: Some(Affine::skew(0.25, 0.0)),
            stroke: false,
            aa: AntiAliasing::Enabled,
        },
        Case {
            name: "rotated",
            glyph_caching: false,
            font_size: 32.0,
            transform: Affine::translate((64.0, 64.0)) * Affine::rotate(0.4),
            glyph_transform: None,
            stroke: false,
            aa: AntiAliasing::Enabled,
        },
        Case {
            name: "flipped",
            glyph_caching: true,
            font_size: 32.0,
            transform: Affine::translate((80.0, 80.0)) * Affine::scale_non_uniform(-1.0, 1.0),
            glyph_transform: None,
            stroke: false,
            aa: AntiAliasing::Enabled,
        },
        Case {
            name: "stroked",
            glyph_caching: false,
            font_size: 32.0,
            transform: Affine::translate((24.0, 80.0)),
            glyph_transform: None,
            stroke: true,
            aa: AntiAliasing::Enabled,
        },
        Case {
            name: "disabled_aa",
            glyph_caching: false,
            font_size: 32.0,
            transform: Affine::translate((24.0, 80.0)),
            glyph_transform: None,
            stroke: false,
            aa: AntiAliasing::Disabled,
        },
    ];

    for case in cases {
        let mut scene = Scene::new(width, height);
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
        .with_glyph_caching(case.glyph_caching);

        let stroke = Stroke::new(2.0);
        let style = if case.stroke {
            StyleRef::Stroke(&stroke)
        } else {
            StyleRef::Fill(Fill::NonZero)
        };
        painter.draw_glyphs(
            &font,
            case.font_size,
            false,
            &[],
            Vec2::ZERO,
            style,
            Color::WHITE,
            1.0,
            case.transform,
            case.glyph_transform,
            [Glyph {
                id: 1,
                x: 0.0,
                y: 0.0,
            }]
            .into_iter(),
            case.aa,
        );
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

        let mut pixels = vec![0u8; usize::from(width) * usize::from(height) * 4];
        target.copy_texture_to_buffer(&mut pixels);
        let red_pixels = pixels
            .chunks_exact(4)
            .filter(|p| p[0] > 200 && p[1] < 20 && p[2] < 20 && p[3] > 200)
            .count();
        assert!(
            red_pixels > 0,
            "case `{}` rendered no red bitmap glyph pixels",
            case.name
        );

        ImageManager::new(
            &mut resources,
            &handle.device,
            &handle.queue,
            &mut cache,
            &mut textures,
        )
        .finish_frame();
    }
}
