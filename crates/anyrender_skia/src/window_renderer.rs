use std::sync::Arc;

use anyrender::WindowRenderer;
use debug_timer::debug_timer;
use skia_safe::{Color, Surface};

use crate::scene::{ResourceCache, SkiaScenePainter};

pub(crate) trait SkiaBackend {
    fn set_size(&mut self, width: u32, height: u32);

    fn prepare(&mut self) -> Option<Surface>;

    fn flush(&mut self, surface: Surface);
}

enum RenderState {
    Active(ActiveRenderState),
    Suspended,
}

struct ActiveRenderState {
    backend: Box<dyn SkiaBackend>,
    resource_cache: ResourceCache,
}

pub struct SkiaWindowRenderer {
    render_state: RenderState,
}

impl Default for SkiaWindowRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl SkiaWindowRenderer {
    pub fn new() -> Self {
        Self {
            render_state: RenderState::Suspended,
        }
    }
}

impl SkiaWindowRenderer {}

impl WindowRenderer for SkiaWindowRenderer {
    type ScenePainter<'a>
        = SkiaScenePainter<'a>
    where
        Self: 'a;

    fn resume(&mut self, window: Arc<dyn anyrender::WindowHandle>, width: u32, height: u32) {
        #[cfg(target_os = "macos")]
        let backend = crate::metal::MetalBackend::new(window, width, height);
        #[cfg(not(target_os = "macos"))]
        let backend = crate::opengl::OpenGLBackend::new(window, width, height);

        self.render_state = RenderState::Active(ActiveRenderState {
            backend: Box::new(backend),
            resource_cache: ResourceCache::new(),
        })
    }

    fn suspend(&mut self) {
        self.render_state = RenderState::Suspended;
    }

    fn is_active(&self) -> bool {
        matches!(self.render_state, RenderState::Active(..))
    }

    fn set_size(&mut self, width: u32, height: u32) {
        if let RenderState::Active(state) = &mut self.render_state {
            state.backend.set_size(width, height);
        }
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        debug_timer!(timer, feature = "log_frame_times");

        let mut surface = match state.backend.prepare() {
            Some(it) => it,
            None => return,
        };

        surface.canvas().restore_to_count(1);
        surface.canvas().clear(Color::WHITE);

        draw_fn(&mut SkiaScenePainter {
            inner: surface.canvas(),
            cache: &mut state.resource_cache,
        });
        timer.record_time("cmd");

        state.backend.flush(surface);
        timer.record_time("render");

        timer.print_times("Frame time: ");
    }
}
