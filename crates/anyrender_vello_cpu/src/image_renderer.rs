use crate::VelloCpuScenePainter;
use anyrender::{ImageRenderer, RenderContext as AnyRenderContext};
use debug_timer::debug_timer;
use vello_cpu::{RenderContext, RenderMode, Resources};

pub struct VelloCpuImageRenderer {
    scene: VelloCpuScenePainter,
}

impl AnyRenderContext for VelloCpuImageRenderer {}
impl ImageRenderer for VelloCpuImageRenderer {
    type ScenePainter<'a> = VelloCpuScenePainter;

    fn new(width: u32, height: u32) -> Self {
        Self {
            scene: VelloCpuScenePainter {
                render_ctx: RenderContext::new(width as u16, height as u16),
                resources: Resources::new(),
            },
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.scene.render_ctx = RenderContext::new(width as u16, height as u16);
    }

    fn reset(&mut self) {
        self.scene.render_ctx.reset();
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F, buffer: &mut [u8]) {
        debug_timer!(timer, feature = "log_frame_times");

        draw_fn(&mut self.scene);
        timer.record_time("cmds");

        self.scene.render_ctx.flush();
        timer.record_time("flush");

        self.scene.render_ctx.render_to_buffer(
            &mut self.scene.resources,
            buffer,
            self.scene.render_ctx.width(),
            self.scene.render_ctx.height(),
            RenderMode::OptimizeSpeed,
        );
        timer.record_time("render");

        timer.print_times("vello_cpu: ");
    }

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        buffer: &mut Vec<u8>,
    ) {
        let width = self.scene.render_ctx.width();
        let height = self.scene.render_ctx.height();
        buffer.resize(width as usize * height as usize * 4, 0);
        self.render(draw_fn, buffer);
    }
}
