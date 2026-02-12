//! Integration tests for scene serialization.

use std::io::{Cursor, Read};

use anyrender::recording::{RecordingRenderContext, RenderCommand, Scene};
use anyrender::{Glyph, ImageResource, PaintScene, RenderContext};
use anyrender_serialize::{
    ArchiveError, ResourceManifest, SceneArchive, SerializableRenderCommand, SerializeConfig,
};
use kurbo::{Affine, Rect, Stroke};
use peniko::{
    Blob, Brush, Color, Compose, Fill, FontData, ImageAlphaType, ImageBrush, ImageData,
    ImageFormat, Mix,
};
use read_fonts::TableProvider;
use zip::ZipArchive;

#[test]
fn test_empty_scene_roundtrip() {
    assert_scene_roundtrip(&RecordingRenderContext::new(), &Scene::new());
}

/// Tests that all non-image and non-font command types survive a roundtrip.
#[test]
fn test_all_command_types_roundtrip() {
    let mut scene = Scene::new();

    // Layer with blend mode
    scene.push_layer(
        Mix::Multiply,
        0.75,
        Affine::translate((5.0, 5.0)),
        &Rect::new(0.0, 0.0, 500.0, 500.0),
    );

    // Fill (NonZero)
    scene.fill(
        Fill::NonZero,
        Affine::translate((10.0, 20.0)),
        Color::from_rgb8(255, 0, 0),
        None,
        &Rect::new(0.0, 0.0, 100.0, 50.0),
    );

    // Fill (EvenOdd) with brush transform
    scene.fill(
        Fill::EvenOdd,
        Affine::IDENTITY,
        Color::from_rgb8(0, 0, 255),
        Some(Affine::rotate(std::f64::consts::PI / 4.0)),
        &Rect::new(0.0, 0.0, 50.0, 50.0),
    );

    // Stroke
    scene.stroke(
        &Stroke::new(3.5),
        Affine::scale(2.0),
        Color::from_rgb8(0, 255, 0),
        None,
        &Rect::new(10.0, 10.0, 90.0, 90.0),
    );

    // Box shadow
    scene.draw_box_shadow(
        Affine::translate((0.0, 100.0)),
        Rect::new(0.0, 0.0, 100.0, 50.0),
        Color::from_rgba8(0, 0, 0, 100),
        5.0,
        3.0,
    );

    // Clip layer
    scene.push_clip_layer(Affine::scale(1.5), &Rect::new(50.0, 50.0, 150.0, 150.0));
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(255, 255, 0),
        None,
        &Rect::new(0.0, 0.0, 200.0, 200.0),
    );
    scene.pop_layer();

    // Layer with compose blend mode
    scene.push_layer(
        Compose::SrcOver,
        1.0,
        Affine::IDENTITY,
        &Rect::new(0.0, 0.0, 500.0, 500.0),
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(128, 128, 128),
        None,
        &Rect::new(0.0, 0.0, 100.0, 100.0),
    );
    scene.pop_layer();

    scene.pop_layer();

    assert_scene_roundtrip(&RecordingRenderContext::new(), &scene);
}

#[test]
fn test_image_data_roundtrip() {
    let pixels: Vec<u8> = vec![
        255, 0, 0, 255, // red
        0, 255, 0, 255, // green
        0, 0, 255, 255, // blue
        255, 255, 255, 255, // white
    ];
    let image_data = ImageData {
        data: Blob::from(pixels.clone()),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width: 2,
        height: 2,
    };

    let mut ctx = RecordingRenderContext::new();
    let resource = ctx.register_image(image_data);
    let image_brush = ImageBrush {
        image: resource,
        sampler: Default::default(),
    };

    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        image_brush,
        None,
        &Rect::new(0.0, 0.0, 100.0, 100.0),
    );

    let data = serialize_to_vec(&ctx, &scene, &default_config()).unwrap();
    let archive = archive_deserialize_from_slice(&data).unwrap();

    // Verify manifest metadata
    assert_eq!(archive.manifest.images.len(), 1);
    assert_eq!(archive.manifest.images[0].width, 2);
    assert_eq!(archive.manifest.images[0].height, 2);
    assert_eq!(archive.manifest.images[0].entry.size, 16);

    // Verify pixel data survives the roundtrip
    let mut restore_ctx = RecordingRenderContext::new();
    let restored = archive.to_scene(&mut restore_ctx).unwrap();
    assert_eq!(extract_image_pixels(&restored, &restore_ctx, 0), pixels);
}

#[test]
fn test_image_deduplication() {
    let mut ctx = RecordingRenderContext::new();
    let resource = ctx.register_image(make_1x1_image(255, 0, 0, 255));
    let image_brush: ImageBrush<ImageResource> = ImageBrush {
        image: resource,
        sampler: Default::default(),
    };

    let mut scene = Scene::new();
    // Same image drawn twice
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        image_brush.clone(),
        None,
        &Rect::new(0.0, 0.0, 50.0, 50.0),
    );
    scene.fill(
        Fill::NonZero,
        Affine::translate((50.0, 0.0)),
        image_brush,
        None,
        &Rect::new(0.0, 0.0, 50.0, 50.0),
    );

    let archive = SceneArchive::from_scene(&ctx, &scene, &default_config()).unwrap();
    assert_eq!(archive.commands.len(), 2);
    assert_eq!(archive.images.len(), 1); // deduplicated
}

#[test]
fn test_multiple_different_images() {
    let red_pixels = vec![255, 0, 0, 255];
    let blue_pixels = vec![0, 0, 255, 255];

    let mut ctx = RecordingRenderContext::new();
    let red_resource = ctx.register_image(make_1x1_image(255, 0, 0, 255));
    let blue_resource = ctx.register_image(make_1x1_image(0, 0, 255, 255));

    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        ImageBrush::<ImageResource> {
            image: red_resource,
            sampler: Default::default(),
        },
        None,
        &Rect::new(0.0, 0.0, 50.0, 50.0),
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        ImageBrush::<ImageResource> {
            image: blue_resource,
            sampler: Default::default(),
        },
        None,
        &Rect::new(0.0, 0.0, 50.0, 50.0),
    );

    let archive = SceneArchive::from_scene(&ctx, &scene, &default_config()).unwrap();
    assert_eq!(archive.commands.len(), 2);
    assert_eq!(archive.images.len(), 2);

    // Verify pixel data survives roundtrip
    let data = archive_serialize_to_vec(&archive).unwrap();
    let mut restore_ctx = RecordingRenderContext::new();
    let restored = archive_deserialize_from_slice(&data)
        .unwrap()
        .to_scene(&mut restore_ctx)
        .unwrap();
    assert_eq!(extract_image_pixels(&restored, &restore_ctx, 0), red_pixels);
    assert_eq!(
        extract_image_pixels(&restored, &restore_ctx, 1),
        blue_pixels
    );
}

#[test]
fn test_glyph_run_roundtrip() {
    let font = roboto_font();

    let scene = build_glyph_scene(&font);
    let data = serialize_to_vec(&RecordingRenderContext::new(), &scene, &default_config()).unwrap();
    let archive = archive_deserialize_from_slice(&data).unwrap();

    // Verify font metadata
    assert_eq!(archive.manifest.fonts.len(), 1);

    assert!(archive.manifest.fonts[0].entry.path.ends_with(".ttf"));
    assert_eq!(archive.manifest.fonts[0].entry.size, font.data.data().len(),);
    let restored = archive
        .to_scene(&mut RecordingRenderContext::new())
        .unwrap();
    assert_glyph_run_preserved(&restored);
}

#[test]
fn test_glyph_run_roundtrip_with_subsetting_and_woff2() {
    let font = roboto_font();
    let original_font_size = font.data.data().len();

    let scene = build_glyph_scene(&font);
    let config = subset_and_woff2_config();
    let data = serialize_to_vec(&RecordingRenderContext::new(), &scene, &config).unwrap();
    let archive = archive_deserialize_from_slice(&data).unwrap();

    assert_eq!(archive.manifest.fonts.len(), 1);
    assert!(archive.manifest.fonts[0].entry.path.ends_with(".woff2"));
    assert!(
        archive.manifest.fonts[0].entry.size < original_font_size,
        "Subsetted font ({} bytes) should be smaller than original ({} bytes)",
        archive.manifest.fonts[0].entry.size,
        original_font_size
    );

    // Verify subsetting
    {
        let ttf_data = wuff::decompress_woff2(archive.fonts[0].data()).unwrap();
        let font_ref = read_fonts::FontRef::new(&ttf_data).unwrap();
        let loca = font_ref.loca(None).unwrap();
        let glyf = font_ref.glyf().unwrap();

        // The used glyph IDs (43, 72, 79) should have outlines in the subsetted font
        for &gid in &[43u32, 72, 79] {
            let glyph = loca
                .get_glyf(read_fonts::types::GlyphId::new(gid), &glyf)
                .unwrap();
            assert!(
                glyph.is_some(),
                "Glyph {gid} should have an outline in the subsetted font"
            );
        }

        // An unused glyph ID should be an empty slot (RETAIN_GIDS preserves IDs
        // but removes outlines for glyphs not in the subset)
        let unused_glyph = loca
            .get_glyf(read_fonts::types::GlyphId::new(50), &glyf)
            .unwrap();
        assert!(
            unused_glyph.is_none(),
            "Glyph 50 should be an empty slot in the subsetted font"
        );
    }

    // Verify the scene roundtrip
    let restored = archive
        .to_scene(&mut RecordingRenderContext::new())
        .unwrap();
    assert_glyph_run_preserved(&restored);
}

#[test]
fn test_font_deduplication() {
    let font = roboto_font();

    let mut scene = Scene::new();
    // Two glyph runs with the same font
    for x_offset in [0.0, 100.0] {
        scene.draw_glyphs(
            &font,
            12.0,
            false,
            &[],
            Fill::NonZero,
            Color::from_rgb8(0, 0, 0),
            1.0,
            Affine::translate((x_offset, 0.0)),
            None,
            [Glyph {
                id: 1,
                x: 0.0,
                y: 0.0,
            }]
            .into_iter(),
        );
    }

    let archive =
        SceneArchive::from_scene(&RecordingRenderContext::new(), &scene, &default_config())
            .unwrap();
    assert_eq!(archive.commands.len(), 2);
    assert_eq!(archive.fonts.len(), 1); // deduplicated
}

#[test]
fn test_resource_manifest_version() {
    assert_eq!(ResourceManifest::CURRENT_VERSION, 1);
}

#[test]
fn test_archive_contains_expected_files() {
    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(255, 0, 0),
        None,
        &Rect::new(0.0, 0.0, 100.0, 100.0),
    );

    let data = serialize_to_vec(&RecordingRenderContext::new(), &scene, &default_config()).unwrap();
    let mut zip = ZipArchive::new(Cursor::new(&data)).unwrap();

    // Verify resources.json
    let mut resources_json = String::new();
    zip.by_name("resources.json")
        .unwrap()
        .read_to_string(&mut resources_json)
        .unwrap();
    let manifest: ResourceManifest = serde_json::from_str(&resources_json).unwrap();
    assert_eq!(manifest.version, 1);
    assert!(manifest.images.is_empty());
    assert!(manifest.fonts.is_empty());

    // Verify draw_commands.json
    let mut commands_json = String::new();
    zip.by_name("draw_commands.json")
        .unwrap()
        .read_to_string(&mut commands_json)
        .unwrap();
    let commands: Vec<SerializableRenderCommand> = serde_json::from_str(&commands_json).unwrap();
    assert_eq!(commands.len(), 1);
}

// Helpers

fn default_config() -> SerializeConfig {
    SerializeConfig::new()
}

fn subset_and_woff2_config() -> SerializeConfig {
    SerializeConfig::new()
        .with_subset_fonts(true)
        .with_woff2_fonts(true)
}

fn serialize_to_vec(
    ctx: &RecordingRenderContext,
    scene: &Scene,
    config: &SerializeConfig,
) -> Result<Vec<u8>, ArchiveError> {
    let mut buf = Cursor::new(Vec::new());
    SceneArchive::from_scene(ctx, scene, config)?.serialize(&mut buf)?;
    Ok(buf.into_inner())
}

fn archive_serialize_to_vec(archive: &SceneArchive) -> Result<Vec<u8>, ArchiveError> {
    let mut buf = Cursor::new(Vec::new());
    archive.serialize(&mut buf)?;
    Ok(buf.into_inner())
}

fn archive_deserialize_from_slice(data: &[u8]) -> Result<SceneArchive, ArchiveError> {
    SceneArchive::deserialize(Cursor::new(data))
}

fn assert_scene_roundtrip(ctx: &RecordingRenderContext, scene: &Scene) {
    let data = serialize_to_vec(ctx, scene, &default_config()).unwrap();
    let restored = archive_deserialize_from_slice(&data)
        .unwrap()
        .to_scene(&mut RecordingRenderContext::new())
        .unwrap();
    assert_eq!(*scene, restored);
}

fn make_1x1_image(r: u8, g: u8, b: u8, a: u8) -> ImageData {
    ImageData {
        data: Blob::from(vec![r, g, b, a]),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width: 1,
        height: 1,
    }
}

fn extract_image_pixels(
    scene: &Scene,
    render_ctx: &RecordingRenderContext,
    command_index: usize,
) -> Vec<u8> {
    match &scene.commands[command_index] {
        RenderCommand::Fill(f) => match &f.brush {
            Brush::Image(img) => {
                let data = render_ctx
                    .image_data()
                    .get(&img.image.id)
                    .expect("Image data not found for resource");
                data.data.data().to_vec()
            }
            other => panic!("Expected image brush, got {other:?}"),
        },
        other => panic!("Expected Fill command, got {other:?}"),
    }
}

fn roboto_font() -> FontData {
    static ROBOTO_BYTES: &[u8] = include_bytes!("../../../assets/fonts/roboto/Roboto.ttf");
    FontData::new(Blob::from(ROBOTO_BYTES.to_vec()), 0)
}

fn build_glyph_scene(font: &FontData) -> Scene {
    let mut scene = Scene::new();
    let glyphs = [
        Glyph {
            id: 43,
            x: 0.0,
            y: 0.0,
        },
        Glyph {
            id: 72,
            x: 10.0,
            y: 0.0,
        },
        Glyph {
            id: 79,
            x: 20.0,
            y: 0.0,
        },
    ];
    scene.draw_glyphs(
        font,
        16.0,
        false,
        &[],
        Fill::NonZero,
        Color::from_rgb8(0, 0, 0),
        1.0,
        Affine::translate((10.0, 50.0)),
        None,
        glyphs.into_iter(),
    );
    scene
}

fn assert_glyph_run_preserved(restored: &Scene) {
    assert_eq!(restored.commands.len(), 1);

    match &restored.commands[0] {
        RenderCommand::GlyphRun(glyph_run) => {
            assert_eq!(glyph_run.font_size, 16.0);
            assert_eq!(glyph_run.hint, false);
            assert_eq!(glyph_run.brush_alpha, 1.0);
            assert_eq!(glyph_run.transform, Affine::translate((10.0, 50.0)));
            assert_eq!(glyph_run.glyph_transform, None);
            assert_eq!(glyph_run.glyphs.len(), 3);
            // Glyph positions are preserved
            assert_eq!(glyph_run.glyphs[0].x, 0.0);
            assert_eq!(glyph_run.glyphs[1].x, 10.0);
            assert_eq!(glyph_run.glyphs[2].x, 20.0);
            // Glyph IDs are preserved (RETAIN_GIDS keeps original IDs)
            assert_eq!(glyph_run.glyphs[0].id, 43);
            assert_eq!(glyph_run.glyphs[1].id, 72);
            assert_eq!(glyph_run.glyphs[2].id, 79);
        }
        other => panic!("Expected GlyphRun command, got {other:?}"),
    }
}
