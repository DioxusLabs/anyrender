use crate::{VelloCpuRenderContext, VelloCpuScenePainter};
use anyrender::{ImageRenderer, ImageResource, RenderContext, ResourceId};
use debug_timer::debug_timer;
use peniko::ImageData;
use vello_cpu::{RenderContext as VelloCpuRenderCtx, RenderMode};

pub struct VelloCpuImageRenderer {
    ctx: VelloCpuRenderContext,
    render_ctx: VelloCpuRenderCtx,
}

impl RenderContext for VelloCpuImageRenderer {
    fn register_image(&mut self, image: ImageData) -> ImageResource {
        self.ctx.register_image(image)
    }

    fn unregister_resource(&mut self, id: ResourceId) {
        self.ctx.unregister_resource(id)
    }
}

impl ImageRenderer for VelloCpuImageRenderer {
    type ScenePainter<'a> = VelloCpuScenePainter<'a>;

    fn new(width: u32, height: u32) -> Self {
        Self {
            ctx: VelloCpuRenderContext::new(),
            render_ctx: VelloCpuRenderCtx::new(width as u16, height as u16),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.render_ctx = VelloCpuRenderCtx::new(width as u16, height as u16);
    }

    fn reset(&mut self) {
        self.render_ctx.reset();
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F, buffer: &mut [u8]) {
        debug_timer!(timer, feature = "log_frame_times");

        {
            let mut scene = VelloCpuScenePainter::new(&self.ctx, &mut self.render_ctx);
            draw_fn(&mut scene);
        }
        timer.record_time("cmds");

        self.render_ctx.flush();
        timer.record_time("flush");

        self.render_ctx.render_to_buffer(
            buffer,
            self.render_ctx.width(),
            self.render_ctx.height(),
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
        let width = self.render_ctx.width();
        let height = self.render_ctx.height();
        buffer.resize(width as usize * height as usize * 4, 0);
        self.render(draw_fn, buffer);
    }
}
