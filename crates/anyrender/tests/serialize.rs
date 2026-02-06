//! Integration tests for scene serialization.

use std::io::{Cursor, Read};

use anyrender::recording::{RenderCommand, Scene};
use anyrender::serialize::{
    ArchiveError, ResourceManifest, SceneArchive, SerializableRenderCommand,
};
use anyrender::{Glyph, PaintScene};
use kurbo::{Affine, Rect, Stroke};
use peniko::{
    Blob, Brush, Color, Compose, Fill, FontData, ImageAlphaType, ImageBrush, ImageData,
    ImageFormat, Mix,
};
use zip::ZipArchive;

#[test]
fn test_empty_scene_roundtrip() {
    assert_scene_roundtrip(&Scene::new());
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

    assert_scene_roundtrip(&scene);
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

    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        ImageBrush::new(image_data).as_ref(),
        None,
        &Rect::new(0.0, 0.0, 100.0, 100.0),
    );

    let data = serialize_to_vec(&scene).unwrap();
    let archive = archive_deserialize_from_slice(&data).unwrap();

    // Verify manifest metadata
    assert_eq!(archive.manifest.images.len(), 1);
    assert_eq!(archive.manifest.images[0].width, 2);
    assert_eq!(archive.manifest.images[0].height, 2);
    assert_eq!(archive.manifest.images[0].entry.size, 16);

    // Verify pixel data survives the roundtrip
    let restored = archive.to_scene().unwrap();
    assert_eq!(extract_image_pixels(&restored, 0), pixels);
}

#[test]
fn test_image_deduplication() {
    let image_brush = ImageBrush::new(make_1x1_image(255, 0, 0, 255));

    let mut scene = Scene::new();
    // Same image drawn twice
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        image_brush.as_ref(),
        None,
        &Rect::new(0.0, 0.0, 50.0, 50.0),
    );
    scene.fill(
        Fill::NonZero,
        Affine::translate((50.0, 0.0)),
        image_brush.as_ref(),
        None,
        &Rect::new(0.0, 0.0, 50.0, 50.0),
    );

    let archive = SceneArchive::from_scene(&scene).unwrap();
    assert_eq!(archive.commands.len(), 2);
    assert_eq!(archive.images.len(), 1); // deduplicated
}

#[test]
fn test_multiple_different_images() {
    let red_pixels = vec![255, 0, 0, 255];
    let blue_pixels = vec![0, 0, 255, 255];

    let mut scene = Scene::new();
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        ImageBrush::new(make_1x1_image(255, 0, 0, 255)).as_ref(),
        None,
        &Rect::new(0.0, 0.0, 50.0, 50.0),
    );
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        ImageBrush::new(make_1x1_image(0, 0, 255, 255)).as_ref(),
        None,
        &Rect::new(0.0, 0.0, 50.0, 50.0),
    );

    let archive = SceneArchive::from_scene(&scene).unwrap();
    assert_eq!(archive.commands.len(), 2);
    assert_eq!(archive.images.len(), 2);

    // Verify pixel data survives roundtrip
    let data = archive_serialize_to_vec(&archive).unwrap();
    let restored = archive_deserialize_from_slice(&data)
        .unwrap()
        .to_scene()
        .unwrap();
    assert_eq!(extract_image_pixels(&restored, 0), red_pixels);
    assert_eq!(extract_image_pixels(&restored, 1), blue_pixels);
}

#[test]
fn test_glyph_run_roundtrip() {
    let font = roboto_font();
    let font_bytes = font.data.data().to_vec();

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
    let font_size = 16.0;
    let hint = false;
    let normalized_coords = [];
    let style = Fill::NonZero;
    let brush = Color::from_rgb8(0, 0, 0);
    let brush_alpha = 1.0;
    let transform = Affine::translate((10.0, 50.0));
    let glyph_transform = None;
    scene.draw_glyphs(
        &font,
        font_size,
        hint,
        &normalized_coords,
        style,
        brush,
        brush_alpha,
        transform,
        glyph_transform,
        glyphs.into_iter(),
    );

    let data = serialize_to_vec(&scene).unwrap();
    let archive = archive_deserialize_from_slice(&data).unwrap();

    // Verify font metadata
    assert_eq!(archive.manifest.fonts.len(), 1);
    assert_eq!(archive.manifest.fonts[0].entry.size, font_bytes.len());
    assert_eq!(archive.manifest.fonts[0].index, 0);

    // Verify font bytes
    assert_eq!(archive.fonts[0].data.data(), font_bytes.as_slice());

    // Verify the scene roundtrip
    let restored = archive.to_scene().unwrap();
    assert_eq!(restored.commands.len(), 1);

    match &restored.commands[0] {
        RenderCommand::GlyphRun(glyph_run) => {
            assert_eq!(glyph_run.font_data.data.data(), font_bytes.as_slice());
            assert_eq!(glyph_run.font_size, font_size);
            assert_eq!(glyph_run.hint, hint);
            assert_eq!(glyph_run.brush_alpha, brush_alpha);
            assert_eq!(glyph_run.transform, transform);
            assert_eq!(glyph_run.glyph_transform, glyph_transform);
            assert_eq!(glyph_run.glyphs, glyphs);
        }
        other => panic!("Expected GlyphRun command, got {other:?}"),
    }
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

    let archive = SceneArchive::from_scene(&scene).unwrap();
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

    let data = serialize_to_vec(&scene).unwrap();
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

fn serialize_to_vec(scene: &Scene) -> Result<Vec<u8>, ArchiveError> {
    let mut buf = Cursor::new(Vec::new());
    scene.serialize(&mut buf)?;
    Ok(buf.into_inner())
}

fn deserialize_from_slice(data: &[u8]) -> Result<Scene, ArchiveError> {
    Scene::deserialize(Cursor::new(data))
}

fn archive_serialize_to_vec(archive: &SceneArchive) -> Result<Vec<u8>, ArchiveError> {
    let mut buf = Cursor::new(Vec::new());
    archive.serialize(&mut buf)?;
    Ok(buf.into_inner())
}

fn archive_deserialize_from_slice(data: &[u8]) -> Result<SceneArchive, ArchiveError> {
    SceneArchive::deserialize(Cursor::new(data))
}

fn assert_scene_roundtrip(scene: &Scene) {
    let data = serialize_to_vec(scene).unwrap();
    let restored = deserialize_from_slice(&data).unwrap();
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

fn extract_image_pixels(scene: &Scene, command_index: usize) -> Vec<u8> {
    match &scene.commands[command_index] {
        RenderCommand::Fill(f) => match &f.brush {
            Brush::Image(img) => img.image.data.data().to_vec(),
            other => panic!("Expected image brush, got {other:?}"),
        },
        other => panic!("Expected Fill command, got {other:?}"),
    }
}

fn roboto_font() -> FontData {
    static ROBOTO_BYTES: &[u8] = include_bytes!("../../../assets/fonts/roboto/Roboto.ttf");
    FontData::new(Blob::from(ROBOTO_BYTES.to_vec()), 0)
}
