use std::sync::Arc;

use anyrender::{PaintScene, WindowRenderer};
use anyrender_vello_hybrid::VelloHybridWindowRenderer;
use kurbo::{Affine, Rect};
use palette::Srgba;
use peniko::{Color, Fill};
use rough_anyrender::{VelloDrawable, VelloGenerator};
use roughr::core::{FillStyle, OptionsBuilder};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const CANVAS_WIDTH: u32 = 800;
const CANVAS_HEIGHT: u32 = 600;
const BACKGROUND: Color = Color::from_rgb8(150, 192, 183);

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
    rectangle: VelloDrawable<f32>,
}

impl App {
    fn new() -> Self {
        App {
            render_state: RenderState::Suspended(None),
            width: CANVAS_WIDTH,
            height: CANVAS_HEIGHT,
            rectangle: build_rectangle(),
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

fn build_rectangle() -> VelloDrawable<f32> {
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
    // Center the rectangle within the canvas: (canvas_size - rect_size) / 2
    generator.rectangle::<f32>(
        (CANVAS_WIDTH as f32 - rect_width) / 2.0,
        (CANVAS_HEIGHT as f32 - rect_height) / 2.0,
        rect_width,
        rect_height,
    )
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
                    .with_title("rough_anyrender: rectangle");
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
                let rectangle = &self.rectangle;
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
                        // Draw the rough rectangle (already in canvas coordinates)
                        rectangle.draw(scene);
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
