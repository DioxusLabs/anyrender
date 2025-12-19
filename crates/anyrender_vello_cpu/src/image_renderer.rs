use crate::VelloCpuScenePainter;
use anyrender::ImageRenderer;
use debug_timer::debug_timer;
use vello_cpu::{RenderContext, RenderMode};

pub struct VelloCpuImageRenderer {
    scene: VelloCpuScenePainter,
}

impl ImageRenderer for VelloCpuImageRenderer {
    type ScenePainter<'a> = VelloCpuScenePainter;

    fn new(width: u32, height: u32) -> Self {
        Self {
            scene: VelloCpuScenePainter(RenderContext::new(width as u16, height as u16)),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.scene.0 = RenderContext::new(width as u16, height as u16);
    }

    fn reset(&mut self) {
        self.scene.0.reset();
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F, buffer: &mut [u8]) {
        debug_timer!(timer, feature = "log_frame_times");

        draw_fn(&mut self.scene);
        timer.record_time("cmds");

        self.scene.0.flush();
        timer.record_time("flush");

        self.scene.0.render_to_buffer(
            buffer,
            self.scene.0.width(),
            self.scene.0.height(),
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
        let width = self.scene.0.width();
        let height = self.scene.0.height();
        buffer.resize(width as usize * height as usize * 4, 0);
        self.render(draw_fn, buffer);
    }
}
