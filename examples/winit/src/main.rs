use anyrender::{NullWindowRenderer, PaintScene, WindowRenderer};
use anyrender_skia::SkiaWindowRenderer;
use anyrender_vello::VelloWindowRenderer;
use anyrender_vello_cpu::{PixelsWindowRenderer, SoftbufferWindowRenderer, VelloCpuImageRenderer};
use anyrender_vello_hybrid::VelloHybridWindowRenderer;
use kurbo::{Affine, Circle, Point, Rect, Stroke};
use peniko::{Color, Fill};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

struct App {
    render_state: RenderState,
    width: u32,
    height: u32,
}

type VelloCpuSBWindowRenderer = SoftbufferWindowRenderer<VelloCpuImageRenderer>;
type VelloCpuWindowRenderer = PixelsWindowRenderer<VelloCpuImageRenderer>;

type InitialBackend = SkiaWindowRenderer;
// type InitialBackend = VelloWindowRenderer;
// type InitialBackend = VelloHybridWindowRenderer;
// type InitialBackend = VelloCpuWindowRenderer;
// type InitialBackend = VelloCpuSBWindowRenderer;
// type InitialBackend = NullWindowRenderer;

enum Renderer {
    Gpu(Box<VelloWindowRenderer>),
    Hybrid(Box<VelloHybridWindowRenderer>),
    Cpu(Box<VelloCpuWindowRenderer>),
    CpuSoftbuffer(Box<VelloCpuSBWindowRenderer>),
    Skia(Box<SkiaWindowRenderer>),
    Null(NullWindowRenderer),
}
impl From<VelloWindowRenderer> for Renderer {
    fn from(renderer: VelloWindowRenderer) -> Self {
        Self::Gpu(Box::new(renderer))
    }
}
impl From<VelloHybridWindowRenderer> for Renderer {
    fn from(renderer: VelloHybridWindowRenderer) -> Self {
        Self::Hybrid(Box::new(renderer))
    }
}
impl From<VelloCpuWindowRenderer> for Renderer {
    fn from(renderer: VelloCpuWindowRenderer) -> Self {
        Self::Cpu(Box::new(renderer))
    }
}
impl From<VelloCpuSBWindowRenderer> for Renderer {
    fn from(renderer: VelloCpuSBWindowRenderer) -> Self {
        Self::CpuSoftbuffer(Box::new(renderer))
    }
}
impl From<SkiaWindowRenderer> for Renderer {
    fn from(renderer: SkiaWindowRenderer) -> Self {
        Self::Skia(Box::new(renderer))
    }
}
impl From<NullWindowRenderer> for Renderer {
    fn from(renderer: NullWindowRenderer) -> Self {
        Self::Null(renderer)
    }
}

impl Renderer {
    fn is_active(&self) -> bool {
        match self {
            Renderer::Gpu(r) => r.is_active(),
            Renderer::Hybrid(r) => r.is_active(),
            Renderer::Cpu(r) => r.is_active(),
            Renderer::CpuSoftbuffer(r) => r.is_active(),
            Renderer::Null(r) => r.is_active(),
            Renderer::Skia(r) => r.is_active(),
        }
    }

    fn set_size(&mut self, w: u32, h: u32) {
        match self {
            Renderer::Gpu(r) => r.set_size(w, h),
            Renderer::Hybrid(r) => r.set_size(w, h),
            Renderer::Cpu(r) => r.set_size(w, h),
            Renderer::CpuSoftbuffer(r) => r.set_size(w, h),
            Renderer::Null(r) => r.set_size(w, h),
            Renderer::Skia(r) => r.set_size(w, h),
        }
    }
}

enum RenderState {
    Active {
        window: Arc<Window>,
        renderer: Renderer,
    },
    Suspended(Option<Arc<Window>>),
}

impl App {
    fn request_redraw(&mut self) {
        let window = match &self.render_state {
            RenderState::Active { window, renderer } => {
                if renderer.is_active() {
                    Some(window)
                } else {
                    None
                }
            }
            RenderState::Suspended(_) => None,
        };

        if let Some(window) = window {
            window.request_redraw();
        }
    }

    fn draw_scene<T: PaintScene>(scene: &mut T, color: Color) {
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::WHITE,
            None,
            &Rect::new(0.0, 0.0, 50.0, 50.0),
        );
        scene.stroke(
            &Stroke::new(2.0),
            Affine::IDENTITY,
            Color::BLACK,
            None,
            &Rect::new(5.0, 5.0, 35.0, 35.0),
        );
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            color,
            None,
            &Circle::new(Point::new(20.0, 20.0), 10.0),
        );
    }

    fn set_backend<R: WindowRenderer + Into<Renderer>>(
        &mut self,
        mut renderer: R,
        event_loop: &ActiveEventLoop,
    ) {
        let mut window = match &self.render_state {
            RenderState::Active { window, .. } => Some(window.clone()),
            RenderState::Suspended(cached_window) => cached_window.clone(),
        };
        let window = window.take().unwrap_or_else(|| {
            let attr = Window::default_attributes()
                .with_inner_size(winit::dpi::LogicalSize::new(self.width, self.height))
                .with_resizable(true)
                .with_title("anyrender + winit demo")
                .with_visible(true)
                .with_active(true);
            Arc::new(event_loop.create_window(attr).unwrap())
        });

        renderer
            .resume(window.clone(), self.width, self.height)
            .unwrap();
        let renderer = renderer.into();
        self.render_state = RenderState::Active { window, renderer };
        self.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let RenderState::Active { window, .. } = &self.render_state {
            self.render_state = RenderState::Suspended(Some(window.clone()));
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.set_backend(InitialBackend::new(), event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let RenderState::Active { window, renderer } = &mut self.render_state else {
            return;
        };

        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(physical_size) => {
                self.width = physical_size.width;
                self.height = physical_size.height;
                renderer.set_size(self.width, self.height);
                self.request_redraw();
            }
            WindowEvent::RedrawRequested => match renderer {
                Renderer::Skia(r) => {
                    r.render(|p| App::draw_scene(p, Color::from_rgb8(128, 128, 128)))
                }
                Renderer::Gpu(r) => r.render(|p| App::draw_scene(p, Color::from_rgb8(255, 0, 0))),
                Renderer::Hybrid(r) => r.render(|p| App::draw_scene(p, Color::from_rgb8(0, 0, 0))),
                Renderer::Cpu(r) => r.render(|p| App::draw_scene(p, Color::from_rgb8(0, 255, 0))),
                Renderer::CpuSoftbuffer(r) => {
                    r.render(|p| App::draw_scene(p, Color::from_rgb8(0, 0, 255)))
                }
                Renderer::Null(r) => r.render(|p| App::draw_scene(p, Color::from_rgb8(0, 0, 0))),
            }
            .unwrap(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Space),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match renderer {
                Renderer::Cpu(_) | Renderer::CpuSoftbuffer(_) => {
                    self.set_backend(VelloHybridWindowRenderer::new(), event_loop);
                }
                Renderer::Hybrid(_) => {
                    self.set_backend(VelloWindowRenderer::new(), event_loop);
                }
                Renderer::Gpu(_) => {
                    self.set_backend(SkiaWindowRenderer::new(), event_loop);
                }
                Renderer::Skia(_) => {
                    self.set_backend(NullWindowRenderer::new(), event_loop);
                }
                Renderer::Null(_) => {
                    self.set_backend(VelloCpuWindowRenderer::new(), event_loop);
                }
            },
            _ => {}
        }
    }
}

fn main() {
    let mut app = App {
        render_state: RenderState::Suspended(None),
        width: 800,
        height: 600,
    };

    let event_loop = EventLoop::new().unwrap();
    event_loop
        .run_app(&mut app)
        .expect("Couldn't run event loop");
}
