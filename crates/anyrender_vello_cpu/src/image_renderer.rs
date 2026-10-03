use crate::{ImageCacheConfig, VelloCpuScenePainter};
use anyrender::{ImageRenderer, RenderContext as AnyRenderContext};
use debug_timer::debug_timer;
use vello_cpu::{PixmapMut, RenderContext};

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

    /// Enable or disable caching of rasterized glyphs in Vello's glyph atlas.
    ///
    /// Defaults to `false`.
    ///
    /// Note: Vello considers atlas-backed glyph caching experimental.
    pub fn set_glyph_caching(&mut self, enabled: bool) {
        self.scene.set_glyph_caching(enabled);
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

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F, buffer: &mut [u8]) {
        debug_timer!(timer, feature = "log_frame_times");

        draw_fn(&mut self.scene);
        timer.record_time("cmds");

        self.scene.render_ctx.flush();
        timer.record_time("flush");

        self.scene.render_ctx.render(
            PixmapMut::new(
                self.scene.render_ctx.width(),
                self.scene.render_ctx.height(),
                buffer,
            )
            .unwrap(),
            &mut self.scene.resources,
        );
        timer.record_time("render");

        self.scene.maintain();
        timer.record_time("maintain");

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
