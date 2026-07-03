use std::f64::consts::{PI, TAU};
use std::sync::Arc;
use std::time::Instant;

use anyrender::{PaintScene, Scene, WindowRenderer};
use anyrender_vello_hybrid::VelloHybridWindowRenderer;
use kurbo::{Affine, Rect};
use palette::Srgba;
use peniko::{Color, Fill};
use rough_anyrender::VelloGenerator;
use roughr::core::{FillStyle, OptionsBuilder};
use svg_path_ops::pt::PathTransformer;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const CANVAS_WIDTH: u32 = 1000;
const CANVAS_HEIGHT: u32 = 1000;
const BACKGROUND: Color = Color::from_rgb8(30, 30, 40);
const NUM_LOGOS: usize = 10;
// Margin (in pixels) used when wrapping logos around the screen edges.
const WRAP_MARGIN: f64 = 300.0;

const RUST_LOGO_SVG_PATH: &str = "M 149.98 37.69 a 9.51 9.51 90 0 1 4.755 -8.236 c 2.9425 -1.6985 6.5675 -1.6985 9.51 0 A 9.51 9.51 90 0 1 169 37.69 c 0 5.252 -4.258 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 M 36.52 123.79 c 0 -5.252 4.2575 -9.51 9.51 -9.51 s 9.51 4.258 9.51 9.51 s -4.258 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 m 226.92 0.44 c 0 -5.252 4.258 -9.51 9.51 -9.51 s 9.51 4.258 9.51 9.51 s -4.2575 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 m -199.4 13.06 c 4.375 -1.954 6.3465 -7.0775 4.41 -11.46 l -4.22 -9.54 h 16.6 v 74.8 H 47.34 a 117.11 117.11 90 0 1 -3.79 -44.7 z m 69.42 1.84 v -22.05 h 39.52 c 2.04 0 14.4 2.36 14.4 11.6 c 0 7.68 -9.5 10.44 -17.3 10.44 z M 79.5 257.84 a 9.51 9.51 90 0 1 4.755 -8.236 c 2.9425 -1.6985 6.5675 -1.6985 9.51 0 a 9.51 9.51 90 0 1 4.755 8.236 c 0 5.252 -4.258 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 m 140.93 0.44 c 0 -5.252 4.2575 -9.51 9.51 -9.51 s 9.51 4.258 9.51 9.51 s -4.258 9.51 -9.51 9.51 s -9.51 -4.2575 -9.51 -9.51 m 2.94 -21.57 c -4.7 -1 -9.3 1.98 -10.3 6.67 l -4.77 22.28 c -31.0655 14.07 -66.7215 13.8985 -97.65 -0.47 l -4.77 -22.28 c -1 -4.7 -5.6 -7.68 -10.3 -6.67 l -19.67 4.22 c -3.655 -3.7645 -7.0525 -7.77 -10.17 -11.99 h 95.7 c 1.08 0 1.8 -0.2 1.8 -1.18 v -33.85 c 0 -1 -0.72 -1.18 -1.8 -1.18 h -28 V 170.8 h 30.27 c 2.76 0 14.77 0.8 18.62 16.14 l 5.65 25 c 1.8 5.5 9.13 16.53 16.93 16.53 h 49.4 c -3.3155 4.4345 -6.941 8.6285 -10.85 12.55 z m 53.14 -89.38 c 0.6725 6.7565 0.7565 13.559 0.25 20.33 h -12 c -1.2 0 -1.7 0.8 -1.7 1.97 v 5.52 c 0 13 -7.32 15.8 -13.74 16.53 c -6.1 0.7 -12.9 -2.56 -13.72 -6.3 c -3.6 -20.28 -9.6 -24.6 -19 -32.1 c 11.77 -7.48 24.02 -18.5 24.02 -33.27 c 0 -15.94 -10.93 -25.98 -18.38 -30.9 c -10.45 -6.9 -22.02 -8.27 -25.14 -8.27 H 72.75 a 117.1 117.1 90 0 1 65.51 -36.97 l 14.65 15.37 c 3.3 3.47 8.8 3.6 12.26 0.28 l 16.4 -15.67 c 33.8115 6.331 63.129 27.2085 80.17 57.09 l -11.22 25.34 c -1.9365 4.3825 0.035 9.506 4.41 11.46 z m 27.98 0.4 l -0.38 -3.92 l 11.56 -10.78 c 2.35 -2.2 1.47 -6.6 -0.75 -8.44 l -6.55 -1.32 c 2.4695 -11.4425 2.6795 -23.2585 0.62 -34.78 l 6.02 -5.62 c 2.35 -2.2 3.22 -6.24 0.87 -8.44 l -0.28 -2.84 l -13.45 8.32 c -0.9 -6.03 -3.71 -11.6 -8 -15.9 h 8.02 c 3.19 0 5.77 -3.7 5.77 -8.28 c 0 -4.58 -2.58 -8.28 -5.77 -8.28 h -8.83 c -1.44 -6.72 -6.32 -12.14 -12.68 -14.1 l -3.11 -13.29 c -0.88 -3.75 -4.63 -6.09 -8.38 -5.22 c -3.75 0.88 -6.09 4.63 -5.22 8.38 l 2.87 12.28 h -14.13 l 2.98 -12.28 c 0.91 -3.74 -1.38 -7.5 -5.12 -8.42 c -3.74 -0.91 -7.5 1.38 -8.42 5.12 l -3.32 13.63 c -6.15 2.04 -10.87 7.36 -12.28 13.9 h -8.83 c -3.19 0 -5.77 3.7 -5.77 8.28 c 0 4.58 2.58 8.28 5.77 8.28 h 8.02 c -4.09 4.1 -6.86 9.35 -7.92 15.04 l -12.85 -7.95 l -0.38 2.84 c -2.35 2.2 -1.48 6.24 0.87 8.44 l 6.44 6.01 c -1.9855 11.4295 -1.7825 23.1385 0.6 34.49 l -6.16 5.75 c -2.35 2.2 -3.22 6.24 -0.87 8.44 l 11.56 10.78 l -0.38 3.92 l 13.45 -8.32 c 2.72 -1.68 2.73 -6.18 0 -7.87 z";

/// Per-logo animation state.
struct AnimatedLogo {
    position: (f64, f64),
    velocity: (f64, f64),
    rotation: f64,
    rotation_speed: f64,
    base_scale: f64,
    scale_oscillation: f64,
    scale_phase: f64,
    scale: f64,
}

impl AnimatedLogo {
    fn update(&mut self, dt: f64, elapsed: f64, width: f64, height: f64) {
        // Update position based on velocity
        self.position.0 += self.velocity.0 * dt;
        self.position.1 += self.velocity.1 * dt;

        // Apply rotation
        self.rotation += self.rotation_speed * dt;

        // Apply scale oscillation
        let scale_factor = 1.0 + self.scale_oscillation * (elapsed + self.scale_phase).sin();
        self.scale = self.base_scale * scale_factor;

        // Simple screen wrapping
        if self.position.0 > width + WRAP_MARGIN {
            self.position.0 = -WRAP_MARGIN;
        } else if self.position.0 < -WRAP_MARGIN {
            self.position.0 = width + WRAP_MARGIN;
        }
        if self.position.1 > height + WRAP_MARGIN {
            self.position.1 = -WRAP_MARGIN;
        } else if self.position.1 < -WRAP_MARGIN {
            self.position.1 = height + WRAP_MARGIN;
        }
    }
}

enum RenderState {
    Active {
        window: Arc<Window>,
        renderer: Box<VelloHybridWindowRenderer>,
    },
    Suspended(Option<Arc<Window>>),
}

struct App {
    render_state: RenderState,
    width: u32,
    height: u32,
    scenes: Vec<Scene>,
    logos: Vec<AnimatedLogo>,
    logo_center: (f64, f64),
    start_time: Instant,
    last_frame: Instant,
}

impl App {
    fn new() -> Self {
        let (scenes, logos, logo_center) = build_logos(CANVAS_WIDTH as f64, CANVAS_HEIGHT as f64);
        let now = Instant::now();
        App {
            render_state: RenderState::Suspended(None),
            width: CANVAS_WIDTH,
            height: CANVAS_HEIGHT,
            scenes,
            logos,
            logo_center,
            start_time: now,
            last_frame: now,
        }
    }

    fn request_redraw(&self) {
        if let RenderState::Active { window, renderer } = &self.render_state {
            if renderer.is_active() {
                window.request_redraw();
            }
        }
    }
}

/// Build the rough logo scenes (one per instance, each with its own style) and
/// their initial animation state. Returns the scenes, the states and the shared
/// logo center (used as the rotation/scale pivot).
fn build_logos(width: f64, height: f64) -> (Vec<Scene>, Vec<AnimatedLogo>, (f64, f64)) {
    let fill_styles = [
        FillStyle::Hachure,
        FillStyle::Solid,
        FillStyle::ZigZag,
        FillStyle::CrossHatch,
        FillStyle::Dots,
    ];

    let mut scenes = Vec::with_capacity(NUM_LOGOS);
    let mut logos = Vec::with_capacity(NUM_LOGOS);

    for i in 0..NUM_LOGOS {
        let fi = i as f64;

        // Generate a distinct color palette for each logo.
        let hue = fi * 36.0; // degrees
        let stroke_color = Srgba::from_components((
            (hue.to_radians().cos() * 127.0 + 128.0) as u8,
            ((hue + 120.0).to_radians().cos() * 127.0 + 128.0) as u8,
            ((hue + 240.0).to_radians().cos() * 127.0 + 128.0) as u8,
            255u8,
        ))
        .into_format();

        let fill_color = Srgba::from_components((
            (hue.to_radians().sin() * 100.0 + 155.0) as u8,
            ((hue + 120.0).to_radians().sin() * 100.0 + 155.0) as u8,
            ((hue + 240.0).to_radians().sin() * 100.0 + 155.0) as u8,
            180u8,
        ))
        .into_format();

        let options = OptionsBuilder::default()
            .stroke(stroke_color)
            .fill(fill_color)
            .fill_style(fill_styles[i % fill_styles.len()])
            .fill_weight(0.5 + fi as f32 * 0.3)
            .bowing(0.2 + fi as f32 * 0.1)
            .roughness(0.5 + fi as f32 * 0.2)
            .stroke_width(1.0 + fi as f32 * 0.5)
            .build()
            .unwrap();

        let generator = VelloGenerator::new(options);
        let drawing = generator.path::<f32>(RUST_LOGO_SVG_PATH.to_string());
        let mut scene = Scene::new();
        drawing.draw(&mut scene);
        scenes.push(scene);

        // Distribute the logos on a circle around the canvas center.
        let angle = fi * TAU / NUM_LOGOS as f64;
        let radius = 200.0;
        let position = (
            width / 2.0 + angle.cos() * radius,
            height / 2.0 + angle.sin() * radius,
        );

        // Different movement patterns per logo.
        let velocity = match i % 5 {
            0 => (50.0 + fi * 10.0, 30.0 + fi * 5.0),
            1 => (-40.0 - fi * 8.0, 60.0 + fi * 7.0),
            2 => (80.0 + fi * 12.0, -50.0 - fi * 6.0),
            3 => (-70.0 - fi * 9.0, -40.0 - fi * 4.0),
            _ => (45.0 + fi * 11.0, 70.0 + fi * 8.0),
        };

        let base_scale = 0.3 + fi * 0.07;
        logos.push(AnimatedLogo {
            position,
            velocity,
            rotation: 0.0,
            rotation_speed: (fi + 1.0) * 0.3,
            base_scale,
            scale_oscillation: 0.1 + fi * 0.02,
            scale_phase: fi * PI / 5.0,
            scale: base_scale,
        });
    }

    let bbox = PathTransformer::new(RUST_LOGO_SVG_PATH.to_string()).to_box(None);
    let center = (
        bbox.min_x.unwrap_or(0.0) + bbox.width() / 2.0,
        bbox.min_y.unwrap_or(0.0) + bbox.height() / 2.0,
    );

    (scenes, logos, center)
}

impl ApplicationHandler for App {
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active { window, .. } = &self.render_state {
            self.render_state = RenderState::Suspended(Some(window.clone()));
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = match &self.render_state {
            RenderState::Suspended(Some(window)) => window.clone(),
            _ => {
                let attr = Window::default_attributes()
                    .with_inner_size(winit::dpi::LogicalSize::new(self.width, self.height))
                    .with_resizable(true)
                    .with_title("rough_anyrender: animate");
                Arc::new(event_loop.create_window(attr).unwrap())
            }
        };

        let size = window.inner_size();
        self.width = size.width;
        self.height = size.height;

        let mut renderer = VelloHybridWindowRenderer::new();
        renderer.resume(window.clone(), self.width, self.height, || {});
        let _ = renderer.complete_resume();

        // Reset the frame clock so the first frame doesn't jump.
        self.last_frame = Instant::now();
        self.render_state = RenderState::Active {
            window,
            renderer: Box::new(renderer),
        };
        self.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let window_matches = match &self.render_state {
            RenderState::Active { window, .. } => window.id() == window_id,
            RenderState::Suspended(_) => false,
        };
        if !window_matches {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                self.width = physical_size.width;
                self.height = physical_size.height;
                if let RenderState::Active { renderer, .. } = &mut self.render_state {
                    renderer.set_size(self.width, self.height);
                }
                self.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f64();
                self.last_frame = now;
                let elapsed = (now - self.start_time).as_secs_f64();
                let (width, height) = (self.width as f64, self.height as f64);

                // Advance the animation.
                for logo in &mut self.logos {
                    logo.update(dt, elapsed, width, height);
                }

                let scenes = &self.scenes;
                let logos = &self.logos;
                let center = self.logo_center;
                if let RenderState::Active { window, renderer } = &mut self.render_state {
                    renderer.render(|scene| {
                        // Draw the background
                        scene.fill(
                            Fill::NonZero,
                            Affine::IDENTITY,
                            BACKGROUND,
                            None,
                            &Rect::new(0.0, 0.0, width, height),
                        );

                        // Draw each animated logo around its own center.
                        for (logo_scene, logo) in scenes.iter().zip(logos) {
                            let transform = Affine::translate(logo.position)
                                * Affine::rotate(logo.rotation)
                                * Affine::scale(logo.scale)
                                * Affine::translate((-center.0, -center.1));
                            scene.append_scene(logo_scene.clone(), transform);
                        }
                    });

                    // Keep animating.
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let mut app = App::new();
    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}
