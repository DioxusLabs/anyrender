//! Example demonstrating scene serialization and deserialization.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyrender::recording::Scene;
use anyrender::{Glyph, PaintScene, render_to_buffer};
use anyrender_serialize::{SceneArchive, SerializeConfig};
use anyrender_vello_cpu::VelloCpuImageRenderer;
use image::{ImageBuffer, RgbaImage};
use kurbo::{Affine, Circle, Point, Rect, RoundedRect, Stroke};
use parley::style::{FontFamily, FontStack};
use parley::{Alignment, AlignmentOptions, FontContext, Layout, LayoutContext, StyleProperty};
use peniko::{
    Blob, Color, Fill, FontData, ImageAlphaType, ImageBrush, ImageData, ImageFormat, Mix,
};

const WIDTH: u32 = 400;
const HEIGHT: u32 = 300;

const OUTPUT_DIR: &str = "examples/serialize/_output";

fn main() {
    let original_scene = create_demo_scene();

    // Render original prior to serialization/deserialization roundtrip
    let pixels = render_scene_to_buffer(&original_scene);
    let img: RgbaImage = ImageBuffer::from_raw(WIDTH, HEIGHT, pixels.to_vec()).unwrap();
    img.save(Path::new(OUTPUT_DIR).join("original.png"))
        .unwrap();

    // Serialize
    let archive_path = Path::new(OUTPUT_DIR).join("demo_scene.anyrender.zip");
    let file = File::create(&archive_path).unwrap();
    let writer = BufWriter::new(file);
    let config = SerializeConfig::new()
        .with_subset_fonts(true)
        .with_woff2_fonts(true);
    SceneArchive::from_scene(&original_scene, &config)
        .unwrap()
        .serialize(writer)
        .unwrap();

    // Deserialize
    let file = File::open(&archive_path).unwrap();
    let deserialized_scene = SceneArchive::deserialize(file).unwrap().to_scene().unwrap();

    // Render deserialized scene to verify against original
    let pixels = render_scene_to_buffer(&deserialized_scene);
    let img: RgbaImage = ImageBuffer::from_raw(WIDTH, HEIGHT, pixels.to_vec()).unwrap();
    img.save(Path::new(OUTPUT_DIR).join("roundtrip.png"))
        .unwrap();

    // Assert that `original.png` and `roundtrip.png` are the same
    let original_img = image::open(Path::new(OUTPUT_DIR).join("original.png")).unwrap();
    let roundtrip_img = image::open(Path::new(OUTPUT_DIR).join("roundtrip.png")).unwrap();
    assert_eq!(original_img.to_rgba8(), roundtrip_img.to_rgba8());
}

fn create_demo_scene() -> Scene {
    let mut scene = Scene::new();

    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgb8(240, 240, 245),
        None,
        &Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64),
    );

    let card_rect = Rect::new(20.0, 20.0, 180.0, 140.0);
    let rounded_card = RoundedRect::from_rect(card_rect, 12.0);

    scene.draw_box_shadow(
        Affine::translate((4.0, 4.0)),
        card_rect,
        Color::from_rgba8(0, 0, 0, 60),
        12.0,
        8.0,
    );

    // Card background
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::WHITE,
        None,
        &rounded_card,
    );

    // Card border
    scene.stroke(
        &Stroke::new(1.5),
        Affine::IDENTITY,
        Color::from_rgb8(200, 200, 210),
        None,
        &rounded_card,
    );

    // Text
    draw_text_with_parley(&mut scene);

    // Draw some circles using layers with blend modes
    scene.push_layer(
        Mix::Multiply,
        0.8,
        Affine::IDENTITY,
        &Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64),
    );

    // Red circle
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgba8(255, 100, 100, 200),
        None,
        &Circle::new(Point::new(280.0, 80.0), 50.0),
    );

    // Blue circle (overlapping)
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgba8(100, 100, 255, 200),
        None,
        &Circle::new(Point::new(320.0, 80.0), 50.0),
    );

    // Green circle (overlapping)
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        Color::from_rgba8(100, 255, 100, 200),
        None,
        &Circle::new(Point::new(300.0, 120.0), 50.0),
    );

    scene.pop_layer();

    // Draw a clipped region
    scene.push_clip_layer(
        Affine::IDENTITY,
        &Circle::new(Point::new(100.0, 220.0), 60.0),
    );

    // Stripes inside the clipped circle
    for i in 0..10 {
        let color = if i % 2 == 0 {
            Color::from_rgb8(255, 200, 100)
        } else {
            Color::from_rgb8(100, 200, 255)
        };
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            color,
            None,
            &Rect::new(
                40.0 + (i as f64 * 12.0),
                160.0,
                40.0 + ((i + 1) as f64 * 12.0),
                280.0,
            ),
        );
    }

    scene.pop_layer();

    // Draw with an image brush (checkerboard pattern)
    let checkerboard = create_checkerboard_image(8, 8);
    let image_brush = ImageBrush::new(checkerboard);

    scene.fill(
        Fill::NonZero,
        Affine::translate((220.0, 180.0)) * Affine::scale(8.0),
        image_brush.as_ref(),
        None,
        &Rect::new(0.0, 0.0, 16.0, 12.0),
    );

    // Border around the checkerboard
    scene.stroke(
        &Stroke::new(2.0),
        Affine::IDENTITY,
        Color::from_rgb8(60, 60, 80),
        None,
        &Rect::new(220.0, 180.0, 348.0, 276.0),
    );

    scene
}

/// Lay out text with parley using the Roboto font and draw it onto the scene.
fn draw_text_with_parley(scene: &mut Scene) {
    let mut font_cx = FontContext::new();
    let mut layout_cx = LayoutContext::new();

    let font_blob = Blob::from(include_bytes!("../../../assets/fonts/roboto/Roboto.ttf").to_vec());
    font_cx.collection.register_fonts(font_blob.clone(), None);

    // Title
    {
        let text = "Hello World!";
        let mut builder = layout_cx.ranged_builder(&mut font_cx, text, 1.0, true);
        builder.push_default(StyleProperty::FontSize(18.0));
        builder.push_default(StyleProperty::FontStack(FontStack::Single(
            FontFamily::Named("Roboto".into()),
        )));
        let mut layout: Layout<()> = builder.build(text);
        layout.break_all_lines(Some(140.0));
        layout.align(Some(140.0), Alignment::Start, AlignmentOptions::default());
        render_layout(
            scene,
            &layout,
            &font_blob,
            Affine::translate((32.0, 50.0)),
            Color::from_rgb8(40, 40, 60),
        );
    }
    // Paragraph
    {
        let text =
            "Serialization roundtrip test: fonts are subsetted, compressed to WOFF2, and restored.";
        let mut builder = layout_cx.ranged_builder(&mut font_cx, text, 1.0, true);
        builder.push_default(StyleProperty::FontSize(13.0));
        builder.push_default(StyleProperty::FontStack(FontStack::Single(
            FontFamily::Named("Roboto".into()),
        )));
        let mut layout: Layout<()> = builder.build(text);
        layout.break_all_lines(Some(150.0));
        layout.align(Some(150.0), Alignment::Start, AlignmentOptions::default());
        render_layout(
            scene,
            &layout,
            &font_blob,
            Affine::translate((32.0, 76.0)),
            Color::from_rgb8(80, 80, 100),
        );
    }
}

fn render_layout(
    scene: &mut Scene,
    layout: &Layout<()>,
    font_blob: &Blob<u8>,
    transform: Affine,
    color: Color,
) {
    for line in layout.lines() {
        for item in line.items() {
            if let parley::PositionedLayoutItem::GlyphRun(glyph_run) = item {
                let run = glyph_run.run();
                let parley_font = run.font();
                let font_data = FontData::new(font_blob.clone(), parley_font.index);
                let font_size = run.font_size();
                let normalized_coords = run.normalized_coords();
                let glyphs = glyph_run.positioned_glyphs().map(|g| Glyph {
                    id: g.id,
                    x: g.x,
                    y: g.y,
                });

                scene.draw_glyphs(
                    &font_data,
                    font_size,
                    false,
                    normalized_coords,
                    Fill::NonZero,
                    color,
                    1.0,
                    transform,
                    None,
                    glyphs.into_iter(),
                );
            }
        }
    }
}

/// Create a checkerboard image for demonstrating image brushes.
fn create_checkerboard_image(width: u32, height: u32) -> ImageData {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);

    for y in 0..height {
        for x in 0..width {
            let is_light = (x + y) % 2 == 0;
            if is_light {
                pixels.extend_from_slice(&[220, 220, 230, 255]);
            } else {
                pixels.extend_from_slice(&[80, 80, 100, 255]);
            }
        }
    }

    ImageData {
        data: Blob::from(pixels),
        format: ImageFormat::Rgba8,
        alpha_type: ImageAlphaType::Alpha,
        width,
        height,
    }
}

/// Render a scene to an RGBA buffer using Vello CPU.
fn render_scene_to_buffer(scene: &Scene) -> Vec<u8> {
    render_to_buffer::<VelloCpuImageRenderer, _>(
        |painter| {
            painter.append_scene(scene.clone(), Affine::IDENTITY);
        },
        WIDTH,
        HEIGHT,
    )
}
