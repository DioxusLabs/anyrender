use crate::{ImageCacheConfig, VelloCpuScenePainter};
use anyrender::{Backdrop, ImageRenderer, RenderContext as AnyRenderContext};
use debug_timer::debug_timer;
use kurbo::{Point, Rect, Shape as _, Size};
use vello_cpu::{CompositeMode, PixmapMut, RenderContext};

pub struct VelloCpuImageRenderer {
    scene: VelloCpuScenePainter,
}

impl VelloCpuImageRenderer {
    /// Create a renderer with a custom image cache configuration.
    pub fn with_image_cache_config(width: u32, height: u32, config: ImageCacheConfig) -> Self {
        Self {
            scene: VelloCpuScenePainter::with_image_cache_config(
                width as u16,
                height as u16,
                config,
            ),
        }
    }

    /// Drop all cached image conversions.
    pub fn clear_image_cache(&mut self) {
        self.scene.clear_image_cache();
    }
}

impl AnyRenderContext for VelloCpuImageRenderer {}
impl ImageRenderer for VelloCpuImageRenderer {
    type ScenePainter<'a> = VelloCpuScenePainter;

    fn new(width: u32, height: u32) -> Self {
        Self {
            scene: VelloCpuScenePainter::new(width as u16, height as u16),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.scene.render_ctx = RenderContext::new(width as u16, height as u16);
    }

    fn reset(&mut self) {
        self.scene.render_ctx.reset();
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        backdrop: Backdrop,
        draw_fn: F,
        buffer: &mut [u8],
    ) {
        debug_timer!(timer, feature = "log_frame_times");

        if let Backdrop::Clear(color) = backdrop {
            self.scene.render_ctx.set_paint(color);
            self.scene.render_ctx.fill_path(
                &Rect::from_origin_size(
                    Point::ORIGIN,
                    Size::new(
                        self.scene.render_ctx.width() as f64,
                        self.scene.render_ctx.height() as f64,
                    ),
                )
                .to_path(0.1),
            );
        }
        draw_fn(&mut self.scene);
        timer.record_time("cmds");

        self.scene.render_ctx.flush();
        timer.record_time("flush");

        self.scene.render_ctx.render_with(
            PixmapMut::new(
                self.scene.render_ctx.width(),
                self.scene.render_ctx.height(),
                buffer,
            )
            .unwrap(),
            &mut self.scene.resources,
            vello_cpu::RasterizerSettings {
                composite_mode: match backdrop {
                    Backdrop::Preserve => CompositeMode::SrcOver,
                    Backdrop::Clear(_) => CompositeMode::Replace,
                },
                ..Default::default()
            },
        );
        timer.record_time("render");

        self.scene.maintain();
        timer.record_time("maintain");

        timer.print_times("vello_cpu: ");
    }

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        backdrop: Backdrop,
        draw_fn: F,
        buffer: &mut Vec<u8>,
    ) {
        let width = self.scene.render_ctx.width();
        let height = self.scene.render_ctx.height();
        buffer.resize(width as usize * height as usize * 4, 0);
        self.render(backdrop, draw_fn, buffer);
    }
}
