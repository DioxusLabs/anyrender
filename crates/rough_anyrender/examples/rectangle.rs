use std::ops::DerefMut;

use bevy::prelude::*;
use bevy_vello::{prelude::*, VelloPlugin};
use palette::Srgba;
use rough_vello::VelloGenerator;
use roughr::core::{FillStyle, OptionsBuilder};

const CANVAS_WIDTH: u32 = 800;
const CANVAS_HEIGHT: u32 = 600;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VelloPlugin::default())
        .add_systems(Startup, setup_vector_graphics)
        .run();
}

fn setup_vector_graphics(mut commands: Commands) {
    // Set clear color for the background
    commands.insert_resource(ClearColor(Color::srgb(
        150.0 / 255.0,
        192.0 / 255.0,
        183.0 / 255.0,
    )));

    // Create rough rectangle using Vello
    let options = OptionsBuilder::default()
        .stroke(Srgba::from_components((114u8, 87u8, 82u8, 255u8)).into_format())
        .fill(Srgba::from_components((254u8, 246u8, 201u8, 255)).into_format())
        .fill_style(FillStyle::ZigZagLine)
        .fill_weight(96.0 * 0.01)
        .bowing(0.8)
        .build()
        .unwrap();

    let generator = VelloGenerator::new(options);
    let rect_width = 300.0;
    let rect_height = 200.0;
    // Position rectangle at the center of the canvas
    // For centering: (canvas_size - rect_size) / 2
    let rect = generator.rectangle::<f32>(
        (CANVAS_WIDTH as f32 - rect_width) / 2.0,
        (CANVAS_HEIGHT as f32 - rect_height) / 2.0,
        rect_width,
        rect_height,
    );

    // Create Vello scene and add the rough rectangle
    let mut scene = vello::Scene::new();

    // Draw the rough rectangle to the scene
    rect.draw(&mut scene);

    commands.spawn((Camera2d, VelloView));
    commands.spawn(VelloSceneBundle {
        scene: VelloScene::from(scene),

        transform: Transform::from_translation(Vec3::new(
            -(CANVAS_WIDTH as f32) / 2.0,
            (CANVAS_HEIGHT as f32) / 2.0,
            0.0,
        )),
        ..default()
    });
}
