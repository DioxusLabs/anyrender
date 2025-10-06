use anyrender_vello::VelloScenePainter;
use bevy::prelude::*;
use bevy_vello::{prelude::*, VelloPlugin};
use palette::Srgba;
use rough_anyrender::VelloGenerator;
use roughr::core::{FillStyle, OptionsBuilder};

#[derive(Component)]
struct AnimatedLogo {
    velocity: Vec2,
    rotation_speed: f32,
    scale_oscillation: f32,
    scale_phase: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VelloPlugin::default())
        .add_systems(Startup, setup_vector_graphics)
        .add_systems(Update, animate_logo)
        .run();
}

fn setup_vector_graphics(mut commands: Commands) {
    // Set clear color for the background
    commands.insert_resource(ClearColor(Color::srgb(
        30.0 / 255.0,
        30.0 / 255.0,
        40.0 / 255.0,
    )));

    let rust_logo_svg_path: String  = "M 149.98 37.69 a 9.51 9.51 90 0 1 4.755 -8.236 c 2.9425 -1.6985 6.5675 -1.6985 9.51 0 A 9.51 9.51 90 0 1 169 37.69 c 0 5.252 -4.258 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 M 36.52 123.79 c 0 -5.252 4.2575 -9.51 9.51 -9.51 s 9.51 4.258 9.51 9.51 s -4.258 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 m 226.92 0.44 c 0 -5.252 4.258 -9.51 9.51 -9.51 s 9.51 4.258 9.51 9.51 s -4.2575 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 m -199.4 13.06 c 4.375 -1.954 6.3465 -7.0775 4.41 -11.46 l -4.22 -9.54 h 16.6 v 74.8 H 47.34 a 117.11 117.11 90 0 1 -3.79 -44.7 z m 69.42 1.84 v -22.05 h 39.52 c 2.04 0 14.4 2.36 14.4 11.6 c 0 7.68 -9.5 10.44 -17.3 10.44 z M 79.5 257.84 a 9.51 9.51 90 0 1 4.755 -8.236 c 2.9425 -1.6985 6.5675 -1.6985 9.51 0 a 9.51 9.51 90 0 1 4.755 8.236 c 0 5.252 -4.258 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 m 140.93 0.44 c 0 -5.252 4.2575 -9.51 9.51 -9.51 s 9.51 4.258 9.51 9.51 s -4.258 9.51 -9.51 9.51 s -9.51 -4.2575 -9.51 -9.51 m 2.94 -21.57 c -4.7 -1 -9.3 1.98 -10.3 6.67 l -4.77 22.28 c -31.0655 14.07 -66.7215 13.8985 -97.65 -0.47 l -4.77 -22.28 c -1 -4.7 -5.6 -7.68 -10.3 -6.67 l -19.67 4.22 c -3.655 -3.7645 -7.0525 -7.77 -10.17 -11.99 h 95.7 c 1.08 0 1.8 -0.2 1.8 -1.18 v -33.85 c 0 -1 -0.72 -1.18 -1.8 -1.18 h -28 V 170.8 h 30.27 c 2.76 0 14.77 0.8 18.62 16.14 l 5.65 25 c 1.8 5.5 9.13 16.53 16.93 16.53 h 49.4 c -3.3155 4.4345 -6.941 8.6285 -10.85 12.55 z m 53.14 -89.38 c 0.6725 6.7565 0.7565 13.559 0.25 20.33 h -12 c -1.2 0 -1.7 0.8 -1.7 1.97 v 5.52 c 0 13 -7.32 15.8 -13.74 16.53 c -6.1 0.7 -12.9 -2.56 -13.72 -6.3 c -3.6 -20.28 -9.6 -24.6 -19 -32.1 c 11.77 -7.48 24.02 -18.5 24.02 -33.27 c 0 -15.94 -10.93 -25.98 -18.38 -30.9 c -10.45 -6.9 -22.02 -8.27 -25.14 -8.27 H 72.75 a 117.1 117.1 90 0 1 65.51 -36.97 l 14.65 15.37 c 3.3 3.47 8.8 3.6 12.26 0.28 l 16.4 -15.67 c 33.8115 6.331 63.129 27.2085 80.17 57.09 l -11.22 25.34 c -1.9365 4.3825 0.035 9.506 4.41 11.46 z m 27.98 0.4 l -0.38 -3.92 l 11.56 -10.78 c 2.35 -2.2 1.47 -6.6 -1.53 -7.72 l -14.77 -5.52 l -1.16 -3.8 l 9.2 -12.8 c 1.88 -2.6 0.15 -6.75 -3 -7.27 l -15.58 -2.53 l -1.87 -3.5 l 6.55 -14.37 c 1.34 -2.93 -1.15 -6.67 -4.37 -6.55 l -15.8 0.55 l -2.5 -3.03 l 3.63 -15.4 c 0.73 -3.13 -2.44 -6.3 -5.57 -5.57 l -15.4 3.63 l -3.04 -2.5 l 0.55 -15.8 c 0.12 -3.2 -3.62 -5.7 -6.54 -4.37 l -14.36 6.55 l -3.5 -1.88 l -2.54 -15.58 c -0.5 -3.16 -4.67 -4.88 -7.27 -3 l -12.8 9.2 l -3.8 -1.15 l -5.52 -14.77 c -1.12 -3 -5.53 -3.88 -7.72 -1.54 l -10.78 11.56 l -3.92 -0.38 l -8.32 -13.45 c -1.68 -2.72 -6.2 -2.72 -7.87 0 l -8.32 13.45 l -3.92 0.38 l -10.8 -11.58 c -2.2 -2.34 -6.6 -1.47 -7.72 1.54 L 119.79 20.6 l -3.8 1.15 l -12.8 -9.2 c -2.6 -1.88 -6.76 -0.15 -7.27 3 l -2.54 15.58 l -3.5 1.88 l -14.36 -6.55 c -2.92 -1.33 -6.67 1.17 -6.54 4.37 l 0.55 15.8 l -3.04 2.5 l -15.4 -3.63 c -3.13 -0.73 -6.3 2.44 -5.57 5.57 l 3.63 15.4 l -2.5 3.03 l -15.8 -0.55 c -3.2 -0.1 -5.7 3.62 -4.37 6.55 l 6.55 14.37 l -1.88 3.5 l -15.58 2.53 c -3.16 0.5 -4.88 4.67 -3 7.27 l 9.2 12.8 l -1.16 3.8 l -14.77 5.52 c -3 1.12 -3.88 5.53 -1.53 7.72 l 11.56 10.78 l -0.38 3.92 l -13.45 8.32 c -2.72 1.68 -2.72 6.2 0 7.87 l 13.45 8.32 l 0.38 3.92 l -11.59 10.82 c -2.34 2.2 -1.47 6.6 1.53 7.72 l 14.77 5.52 l 1.16 3.8 l -9.2 12.8 c -1.87 2.6 -0.15 6.76 3 7.27 l 15.57 2.53 l 1.88 3.5 l -6.55 14.36 c -1.33 2.92 1.18 6.67 4.37 6.55 l 15.8 -0.55 l 2.5 3.04 l -3.63 15.4 c -0.73 3.12 2.44 6.3 5.57 5.56 l 15.4 -3.63 l 3.04 2.5 l -0.55 15.8 c -0.12 3.2 3.62 5.7 6.54 4.37 l 14.36 -6.55 l 3.5 1.88 l 2.54 15.57 c 0.5 3.17 4.67 4.88 7.27 3.02 l 12.8 -9.22 l 3.8 1.16 l 5.52 14.77 c 1.12 3 5.53 3.88 7.72 1.53 l 10.78 -11.56 l 3.92 0.4 l 8.32 13.45 c 1.68 2.7 6.18 2.72 7.87 0 l 8.32 -13.45 l 3.92 -0.4 l 10.78 11.56 c 2.2 2.35 6.6 1.47 7.72 -1.53 l 5.52 -14.77 l 3.8 -1.16 l 12.8 9.22 c 2.6 1.87 6.76 0.15 7.27 -3.02 l 2.54 -15.57 l 3.5 -1.88 l 14.36 6.55 c 2.92 1.33 6.66 -1.16 6.54 -4.37 l -0.55 -15.8 l 3.03 -2.5 l 15.4 3.63 c 3.13 0.73 6.3 -2.44 5.57 -5.56 l -3.63 -15.4 l 2.5 -3.04 l 15.8 0.55 c 3.2 0.13 5.7 -3.63 4.37 -6.55 l -6.55 -14.36 l 1.87 -3.5 l 15.58 -2.53 c 3.17 -0.5 4.9 -4.66 3 -7.27 l -9.2 -12.8 l 1.16 -3.8 l 14.77 -5.52 c 3 -1.13 3.88 -5.53 1.53 -7.72 l -11.56 -10.78 l 0.38 -3.92 l 13.45 -8.32 c 2.72 -1.68 2.73 -6.18 0 -7.87 z".into();

    commands.spawn((Camera2d, VelloView));

    // Create 10 different logo instances with varying options and movement patterns
    for i in 0..10 {
        let angle = (i as f32) * std::f32::consts::TAU / 10.0; // Distribute in circle
        let radius = 200.0;
        let x = angle.cos() * radius;
        let y = angle.sin() * radius;

        // Generate different visual styles for each logo
        let hue = (i as f32) * 36.0; // Different hue for each logo (0-360 degrees)
        let stroke_color = Srgba::from_components((
            ((hue.to_radians().cos() * 127.0 + 128.0) as u8),
            (((hue + 120.0).to_radians().cos() * 127.0 + 128.0) as u8),
            (((hue + 240.0).to_radians().cos() * 127.0 + 128.0) as u8),
            255u8,
        ))
        .into_format();

        let fill_color = Srgba::from_components((
            ((hue.to_radians().sin() * 100.0 + 155.0) as u8),
            (((hue + 120.0).to_radians().sin() * 100.0 + 155.0) as u8),
            (((hue + 240.0).to_radians().sin() * 100.0 + 155.0) as u8),
            180u8,
        ))
        .into_format();

        let fill_styles = [
            FillStyle::Hachure,
            FillStyle::Solid,
            FillStyle::ZigZag,
            FillStyle::CrossHatch,
            FillStyle::Dots,
        ];

        let options = OptionsBuilder::default()
            .stroke(stroke_color)
            .fill(fill_color)
            .fill_style(fill_styles[i % fill_styles.len()])
            .fill_weight(0.5 + (i as f32) * 0.3)
            .bowing(0.2 + (i as f32) * 0.1)
            .roughness(0.5 + (i as f32) * 0.2)
            .stroke_width(1.0 + (i as f32) * 0.5)
            .build()
            .unwrap();

        let generator = VelloGenerator::new(options);
        let rust_logo_drawing = generator.path::<f32>(rust_logo_svg_path.clone());
        let mut scene = vello::Scene::new();
        rust_logo_drawing.draw(&mut VelloScenePainter::new(&mut scene));

        // Create different movement patterns for each logo
        let velocity = match i % 5 {
            0 => Vec2::new(50.0 + i as f32 * 10.0, 30.0 + i as f32 * 5.0), // Linear movement
            1 => Vec2::new(-40.0 - i as f32 * 8.0, 60.0 + i as f32 * 7.0), // Different linear
            2 => Vec2::new(80.0 + i as f32 * 12.0, -50.0 - i as f32 * 6.0), // Another linear
            3 => Vec2::new(-70.0 - i as f32 * 9.0, -40.0 - i as f32 * 4.0), // Diagonal
            _ => Vec2::new(45.0 + i as f32 * 11.0, 70.0 + i as f32 * 8.0), // Default
        };

        let scale = 0.3 + (i as f32) * 0.07; // Different scales from 0.3 to 0.93

        commands.spawn((
            VelloSceneBundle {
                scene: VelloScene::from(scene),
                transform: Transform::from_translation(Vec3::new(x, y, 0.0))
                    .with_scale(Vec3::splat(scale)),
                ..default()
            },
            AnimatedLogo {
                velocity,
                rotation_speed: (i as f32 + 1.0) * 0.3, // Different rotation speeds
                scale_oscillation: 0.1 + (i as f32) * 0.02, // Different scale oscillation
                scale_phase: (i as f32) * std::f32::consts::PI / 5.0, // Different phases
            },
        ));
    }
}

fn animate_logo(time: Res<Time>, mut query: Query<(&mut Transform, &mut AnimatedLogo)>) {
    let dt = time.delta_secs();

    for (mut transform, logo) in query.iter_mut() {
        // Update position based on velocity
        transform.translation.x += logo.velocity.x * dt;
        transform.translation.y += logo.velocity.y * dt;

        // Apply rotation
        transform.rotation *= Quat::from_rotation_z(logo.rotation_speed * dt);

        // Apply scale oscillation
        let scale_factor =
            1.0 + logo.scale_oscillation * (time.elapsed_secs() + logo.scale_phase).sin();
        let base_scale =
            0.3 + (transform.translation.x.abs() + transform.translation.y.abs()) * 0.0001; // Vary base scale slightly
        transform.scale = Vec3::splat(scale_factor * base_scale);

        // Simple screen wrapping (optional boundary handling)
        if transform.translation.x > 800.0 {
            transform.translation.x = -800.0;
        }
        if transform.translation.x < -800.0 {
            transform.translation.x = 800.0;
        }
        if transform.translation.y > 600.0 {
            transform.translation.y = -600.0;
        }
        if transform.translation.y < -600.0 {
            transform.translation.y = 600.0;
        }
    }
}
