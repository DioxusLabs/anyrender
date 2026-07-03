use std::sync::Arc;

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

const CANVAS_WIDTH: u32 = 600;
const CANVAS_HEIGHT: u32 = 600;
const BACKGROUND: Color = Color::from_rgb8(150, 192, 183);

const RUST_LOGO_SVG_PATH: &str = "M 149.98 37.69 a 9.51 9.51 90 0 1 4.755 -8.236 c 2.9425 -1.6985 6.5675 -1.6985 9.51 0 A 9.51 9.51 90 0 1 169 37.69 c 0 5.252 -4.258 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 M 36.52 123.79 c 0 -5.252 4.2575 -9.51 9.51 -9.51 s 9.51 4.258 9.51 9.51 s -4.258 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 m 226.92 0.44 c 0 -5.252 4.258 -9.51 9.51 -9.51 s 9.51 4.258 9.51 9.51 s -4.2575 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 m -199.4 13.06 c 4.375 -1.954 6.3465 -7.0775 4.41 -11.46 l -4.22 -9.54 h 16.6 v 74.8 H 47.34 a 117.11 117.11 90 0 1 -3.79 -44.7 z m 69.42 1.84 v -22.05 h 39.52 c 2.04 0 14.4 2.36 14.4 11.6 c 0 7.68 -9.5 10.44 -17.3 10.44 z M 79.5 257.84 a 9.51 9.51 90 0 1 4.755 -8.236 c 2.9425 -1.6985 6.5675 -1.6985 9.51 0 a 9.51 9.51 90 0 1 4.755 8.236 c 0 5.252 -4.258 9.51 -9.51 9.51 s -9.51 -4.258 -9.51 -9.51 m 140.93 0.44 c 0 -5.252 4.2575 -9.51 9.51 -9.51 s 9.51 4.258 9.51 9.51 s -4.258 9.51 -9.51 9.51 s -9.51 -4.2575 -9.51 -9.51 m 2.94 -21.57 c -4.7 -1 -9.3 1.98 -10.3 6.67 l -4.77 22.28 c -31.0655 14.07 -66.7215 13.8985 -97.65 -0.47 l -4.77 -22.28 c -1 -4.7 -5.6 -7.68 -10.3 -6.67 l -19.67 4.22 c -3.655 -3.7645 -7.0525 -7.77 -10.17 -11.99 h 95.7 c 1.08 0 1.8 -0.2 1.8 -1.18 v -33.85 c 0 -1 -0.72 -1.18 -1.8 -1.18 h -28 V 170.8 h 30.27 c 2.76 0 14.77 0.8 18.62 16.14 l 5.65 25 c 1.8 5.5 9.13 16.53 16.93 16.53 h 49.4 c -3.3155 4.4345 -6.941 8.6285 -10.85 12.55 z m 53.14 -89.38 c 0.6725 6.7565 0.7565 13.559 0.25 20.33 h -12 c -1.2 0 -1.7 0.8 -1.7 1.97 v 5.52 c 0 13 -7.32 15.8 -13.74 16.53 c -6.1 0.7 -12.9 -2.56 -13.72 -6.3 c -3.6 -20.28 -9.6 -24.6 -19 -32.1 c 11.77 -7.48 24.02 -18.5 24.02 -33.27 c 0 -15.94 -10.93 -25.98 -18.38 -30.9 c -10.45 -6.9 -22.02 -8.27 -25.14 -8.27 H 72.75 a 117.1 117.1 90 0 1 65.51 -36.97 l 14.65 15.37 c 3.3 3.47 8.8 3.6 12.26 0.28 l 16.4 -15.67 c 33.8115 6.331 63.129 27.2085 80.17 57.09 l -11.22 25.34 c -1.9365 4.3825 0.035 9.506 4.41 11.46 z m 27.98 0.4 l -0.38 -3.92 l 11.56 -10.78 c 2.35 -2.2 1.47 -6.6 -1.53 -7.72 l -14.77 -5.52 l -1.16 -3.8 l 9.2 -12.8 c 1.88 -2.6 0.15 -6.75 -3 -7.27 l -15.58 -2.53 l -1.87 -3.5 l 6.55 -14.37 c 1.34 -2.93 -1.15 -6.67 -4.37 -6.55 l -15.8 0.55 l -2.5 -3.03 l 3.63 -15.4 c 0.73 -3.13 -2.44 -6.3 -5.57 -5.57 l -15.4 3.63 l -3.04 -2.5 l 0.55 -15.8 c 0.12 -3.2 -3.62 -5.7 -6.54 -4.37 l -14.36 6.55 l -3.5 -1.88 l -2.54 -15.58 c -0.5 -3.16 -4.67 -4.88 -7.27 -3 l -12.8 9.2 l -3.8 -1.15 l -5.52 -14.77 c -1.12 -3 -5.53 -3.88 -7.72 -1.54 l -10.78 11.56 l -3.92 -0.38 l -8.32 -13.45 c -1.68 -2.72 -6.2 -2.72 -7.87 0 l -8.32 13.45 l -3.92 0.38 l -10.8 -11.58 c -2.2 -2.34 -6.6 -1.47 -7.72 1.54 L 119.79 20.6 l -3.8 1.15 l -12.8 -9.2 c -2.6 -1.88 -6.76 -0.15 -7.27 3 l -2.54 15.58 l -3.5 1.88 l -14.36 -6.55 c -2.92 -1.33 -6.67 1.17 -6.54 4.37 l 0.55 15.8 l -3.04 2.5 l -15.4 -3.63 c -3.13 -0.73 -6.3 2.44 -5.57 5.57 l 3.63 15.4 l -2.5 3.03 l -15.8 -0.55 c -3.2 -0.1 -5.7 3.62 -4.37 6.55 l 6.55 14.37 l -1.88 3.5 l -15.58 2.53 c -3.16 0.5 -4.88 4.67 -3 7.27 l 9.2 12.8 l -1.16 3.8 l -14.77 5.52 c -3 1.12 -3.88 5.53 -1.53 7.72 l 11.56 10.78 l -0.38 3.92 l -13.45 8.32 c -2.72 1.68 -2.72 6.2 0 7.87 l 13.45 8.32 l 0.38 3.92 l -11.59 10.82 c -2.34 2.2 -1.47 6.6 1.53 7.72 l 14.77 5.52 l 1.16 3.8 l -9.2 12.8 c -1.87 2.6 -0.15 6.76 3 7.27 l 15.57 2.53 l 1.88 3.5 l -6.55 14.36 c -1.33 2.92 1.18 6.67 4.37 6.55 l 15.8 -0.55 l 2.5 3.04 l -3.63 15.4 c -0.73 3.12 2.44 6.3 5.57 5.56 l 15.4 -3.63 l 3.04 2.5 l -0.55 15.8 c -0.12 3.2 3.62 5.7 6.54 4.37 l 14.36 -6.55 l 3.5 1.88 l 2.54 15.57 c 0.5 3.17 4.67 4.88 7.27 3.02 l 12.8 -9.22 l 3.8 1.16 l 5.52 14.77 c 1.12 3 5.53 3.88 7.72 1.53 l 10.78 -11.56 l 3.92 0.4 l 8.32 13.45 c 1.68 2.7 6.18 2.72 7.87 0 l 8.32 -13.45 l 3.92 -0.4 l 10.78 11.56 c 2.2 2.35 6.6 1.47 7.72 -1.53 l 5.52 -14.77 l 3.8 -1.16 l 12.8 9.22 c 2.6 1.87 6.76 0.15 7.27 -3.02 l 2.54 -15.57 l 3.5 -1.88 l 14.36 6.55 c 2.92 1.33 6.66 -1.16 6.54 -4.37 l -0.55 -15.8 l 3.03 -2.5 l 15.4 3.63 c 3.13 0.73 6.3 -2.44 5.57 -5.56 l -3.63 -15.4 l 2.5 -3.04 l 15.8 0.55 c 3.2 0.13 5.7 -3.63 4.37 -6.55 l -6.55 -14.36 l 1.87 -3.5 l 15.58 -2.53 c 3.17 -0.5 4.9 -4.66 3 -7.27 l -9.2 -12.8 l 1.16 -3.8 l 14.77 -5.52 c 3 -1.13 3.88 -5.53 1.53 -7.72 l -11.56 -10.78 l 0.38 -3.92 l 13.45 -8.32 c 2.72 -1.68 2.73 -6.18 0 -7.87 z";

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
    logo: Scene,
    offset: (f64, f64),
}

impl App {
    fn new() -> Self {
        let (logo, offset) = build_logo();
        App {
            render_state: RenderState::Suspended(None),
            width: CANVAS_WIDTH,
            height: CANVAS_HEIGHT,
            logo,
            offset,
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

/// Build the rough Rust logo into a `Scene` and compute the translation that
/// centers it within the canvas.
fn build_logo() -> (Scene, (f64, f64)) {
    let options = OptionsBuilder::default()
        .stroke(Srgba::from_components((114u8, 87u8, 82u8, 255u8)).into_format())
        .fill(Srgba::from_components((254u8, 246u8, 201u8, 255)).into_format())
        .fill_style(FillStyle::Hachure)
        .fill_weight(96.0 * 0.01)
        .bowing(0.8)
        .build()
        .unwrap();

    let generator = VelloGenerator::new(options);
    let drawing = generator.path::<f32>(RUST_LOGO_SVG_PATH.to_string());

    let mut logo = Scene::new();
    drawing.draw(&mut logo);

    let bbox = PathTransformer::new(RUST_LOGO_SVG_PATH.to_string()).to_box(None);
    let min_x = bbox.min_x.unwrap_or(0.0);
    let min_y = bbox.min_y.unwrap_or(0.0);
    let offset = (
        (CANVAS_WIDTH as f64 - bbox.width()) / 2.0 - min_x,
        (CANVAS_HEIGHT as f64 - bbox.height()) / 2.0 - min_y,
    );

    (logo, offset)
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
                    .with_title("rough_anyrender: rust_logo");
                Arc::new(event_loop.create_window(attr).unwrap())
            }
        };

        let size = window.inner_size();
        self.width = size.width;
        self.height = size.height;

        let mut renderer = VelloHybridWindowRenderer::new();
        renderer.resume(window.clone(), self.width, self.height, || {});
        let _ = renderer.complete_resume();

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
                let logo = &self.logo;
                let offset = self.offset;
                let (width, height) = (self.width, self.height);
                if let RenderState::Active { window, renderer } = &mut self.render_state {
                    renderer.render(|scene| {
                        // Draw the background
                        scene.fill(
                            Fill::NonZero,
                            Affine::IDENTITY,
                            BACKGROUND,
                            None,
                            &Rect::new(0.0, 0.0, width as f64, height as f64),
                        );
                        // Draw the (centered) rough Rust logo
                        scene.append_scene(logo.clone(), Affine::translate(offset));
                    });
                    // Keep requesting redraws so the first frame is presented once the
                    // window is actually shown (a single early frame can be dropped on
                    // some platforms, e.g. macOS, before the window is composited).
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
