//! Glyphs drawn from the glyph atlas must look the same as glyphs drawn as paths.
#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;

use anyrender::{AntiAliasing, Glyph, PaintScene};
use anyrender_vello_gpu::{
    ExternalTextures, GlyphAtlas, ImageCache, ImageManager, VelloGpuScenePainter,
};
use kurbo::Affine;
use peniko::{Blob, Color, Fill, FontData};
use vello_gpu::{RenderOptions, RenderSize, RenderTarget, RenderTargetConfig, Renderer, Scene};
use wgpu_context::{BufferRenderer, BufferRendererConfig, DeviceHandle};

const W: u32 = 400;
const H: u32 = 160;

static ROBOTO: &[u8] = include_bytes!("../../../assets/fonts/roboto/Roboto.ttf");
static NOTO_COLR: &[u8] =
    include_bytes!("../../anyrender_glyph/tests/fonts/NotoColorEmoji-Subset.ttf");
static NOTO_CBTF: &[u8] =
    include_bytes!("../../anyrender_glyph/tests/fonts/NotoColorEmoji-CBTF-Subset.ttf");

fn font(data: &'static [u8]) -> FontData {
    FontData::new(Blob::new(Arc::new(data)), 0)
}

/// Draws a row of glyphs upright in the top half and with `lower_glyph_transform` in the bottom
/// half. Each glyph is `offset` further from a whole pixel than the one before.
///
/// Returns `None` if there is no GPU to render with.
fn render(
    glyph_caching: bool,
    offset: f32,
    font: &FontData,
    glyph_ids: &[u32],
    lower_glyph_transform: Option<Affine>,
) -> Option<Vec<u8>> {
    let instance = wgpu::Instance::default();
    let Ok(handle) = pollster::block_on(DeviceHandle::new_from_compatible_surface(
        instance, None, None, None,
    )) else {
        eprintln!("Skipping GPU glyph test: no adapter/device available");
        return None;
    };
    let target = BufferRenderer::new(
        BufferRendererConfig {
            width: W,
            height: H,
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
            width: W as u16,
            height: H as u16,
        },
    );
    let depth = renderer.create_depth_texture_view(&RenderSize {
        width: W as u16,
        height: H as u16,
    });
    let mut cache = ImageCache::default();
    let mut textures = ExternalTextures::default();
    let mut glyphs = GlyphAtlas::new();
    let mut buf = vec![0; (W * H * 4) as usize];
    // Render twice so that the second frame can be served from the atlas.
    for _ in 0..2 {
        let mut scene = Scene::new(W as u16, H as u16);
        let mut painter = VelloGpuScenePainter::new(
            &mut scene,
            ImageManager::new(
                &handle.device,
                &handle.queue,
                &mut cache,
                &mut textures,
                &mut glyphs,
            ),
            &handle,
        )
        .with_glyph_caching(glyph_caching);
        for (y, glyph_transform) in [(60.0, None), (140.0, lower_glyph_transform)] {
            painter.draw_glyphs(
                font,
                40.0,
                true,
                &[],
                Default::default(),
                Fill::NonZero,
                Color::WHITE,
                1.0,
                Affine::translate((20.0, y)),
                glyph_transform,
                glyph_ids.iter().enumerate().map(|(i, id)| Glyph {
                    id: *id,
                    x: i as f32 * (45.0 + offset),
                    y: 0.0,
                }),
                AntiAliasing::Enabled,
            );
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
        ImageManager::new(
            &handle.device,
            &handle.queue,
            &mut cache,
            &mut textures,
            &mut glyphs,
        )
        .finish_frame();
        handle.queue.submit([encoder.finish()]);
        target.copy_texture_to_buffer(&mut buf);
    }
    Some(buf)
}

fn pixels(buf: &[u8]) -> &[[u8; 4]] {
    buf.as_chunks().0
}

/// Counts the pixels of `a` that have no similar pixel in `b` within `slack` pixels.
fn differing(a: &[u8], b: &[u8], slack: usize) -> usize {
    let (a, b) = (pixels(a), pixels(b));
    let (w, h) = (W as usize, a.len() / W as usize);
    let similar = |a: &[u8; 4], b: &[u8; 4]| a.iter().zip(b).all(|(a, b)| a.abs_diff(*b) <= 64);
    let mut count = 0;
    for y in 0..h {
        for x in 0..w {
            let px = &a[y * w + x];
            let ys = y.saturating_sub(slack)..=(y + slack).min(h - 1);
            let found = ys.into_iter().any(|by| {
                let xs = x.saturating_sub(slack)..=(x + slack).min(w - 1);
                xs.into_iter().any(|bx| similar(px, &b[by * w + bx]))
            });
            count += usize::from(!found);
        }
    }
    count
}

fn ink(buf: &[u8]) -> usize {
    pixels(buf).iter().filter(|px| px[3] > 128).count()
}

/// Whether any pixel has a colour other than a shade of grey.
fn has_colour(buf: &[u8]) -> bool {
    pixels(buf)
        .iter()
        .any(|px| px[0].abs_diff(px[1]) > 32 || px[1].abs_diff(px[2]) > 32)
}

/// Cached glyphs are rasterized by `vello_cpu` and the others by `vello_gpu`, whose
/// anti-aliasing differs slightly, so the upright row is compared with one pixel of slack and
/// 5% of the inked pixels may differ.
fn assert_cached_matches_uncached(
    font: &FontData,
    glyph_ids: &[u32],
    lower_glyph_transform: Option<Affine>,
) -> Option<Vec<u8>> {
    let off = render(false, 0.3, font, glyph_ids, lower_glyph_transform)?;
    let on = render(true, 0.3, font, glyph_ids, lower_glyph_transform)?;
    let (upper_off, upper_on) = (&off[..off.len() / 2], &on[..on.len() / 2]);

    assert!(ink(upper_off) > 500);
    let allowed = ink(upper_off) / 20;
    assert!(differing(upper_off, upper_on, 1) <= allowed);
    assert!(differing(upper_on, upper_off, 1) <= allowed);
    Some(on)
}

fn glyph_ids(font: &FontData, text: &str) -> Vec<u32> {
    let font = skrifa::FontRef::from_index(font.data.as_ref(), font.index).unwrap();
    let charmap = skrifa::MetadataProvider::charmap(&font);
    text.chars()
        .map(|c| charmap.map(c).unwrap().to_u32())
        .collect()
}

const EMOJI: &str = "\u{2705}\u{1f440}\u{1f389}\u{1f920}";

#[test]
fn outline_glyphs_match() {
    let font = font(ROBOTO);
    let ids = glyph_ids(&font, "Glyphs&g");
    let Some(image) = assert_cached_matches_uncached(&font, &ids, None) else {
        return;
    };
    assert!(!has_colour(&image));
}

/// Skewed glyphs can't be served from the atlas, and must not be drawn from the upright entries
/// that the top row put there.
#[test]
fn skewed_glyphs_are_not_drawn_from_upright_atlas_entries() {
    let font = font(ROBOTO);
    let ids = glyph_ids(&font, "Glyphs&g");
    let skew = Some(Affine::skew(-0.5, 0.0));
    let (Some(off), Some(on)) = (
        render(false, 0.3, &font, &ids, skew),
        render(true, 0.3, &font, &ids, skew),
    ) else {
        return;
    };
    let (lower_off, lower_on) = (&off[off.len() / 2..], &on[on.len() / 2..]);
    assert!(ink(lower_off) > 500);
    assert_eq!(differing(lower_off, lower_on, 0), 0);
}

#[test]
fn colr_glyphs_match() {
    let font = font(NOTO_COLR);
    let ids = glyph_ids(&font, EMOJI);
    let Some(image) = assert_cached_matches_uncached(&font, &ids, None) else {
        return;
    };
    assert!(has_colour(&image));
}

/// Vello can't draw a pixmap, so bitmap glyphs are drawn from the atlas with caching on or off,
/// upright or not.
#[test]
fn bitmap_glyphs_are_drawn() {
    let font = font(NOTO_CBTF);
    let ids = glyph_ids(&font, EMOJI);
    let skew = Some(Affine::skew(-0.5, 0.0));
    for glyph_caching in [false, true] {
        let Some(image) = render(glyph_caching, 0.3, &font, &ids, skew) else {
            return;
        };
        let (upper, lower) = image.split_at(image.len() / 2);
        assert!(ink(upper) > 500 && ink(lower) > 500);
        assert!(has_colour(upper) && has_colour(lower));
    }
}
