use anyrender::ImageRenderer;
use debug_timer::debug_timer;

use crate::TinySkiaScenePainter;

pub struct TinySkiaImageRenderer {
    scene: TinySkiaScenePainter,
}

impl anyrender::RenderContext for TinySkiaImageRenderer {}

impl ImageRenderer for TinySkiaImageRenderer {
    type ScenePainter<'a> = TinySkiaScenePainter;

    fn new(width: u32, height: u32) -> Self {
        Self {
            scene: TinySkiaScenePainter::new(width, height),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.scene.resize(width, height);
    }

    fn reset(&mut self) {
        anyrender::PaintScene::reset(&mut self.scene);
    }

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        buffer: &mut Vec<u8>,
    ) {
        let len = self.scene.width() as usize * self.scene.height() as usize * 4;
        buffer.resize(len, 0);
        self.render(draw_fn, buffer);
    }

    /// Renders into `buffer` as premultiplied RGBA8, like the other CPU backends.
    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F, buffer: &mut [u8]) {
        debug_timer!(timer, feature = "log_frame_times");
        anyrender::PaintScene::reset(&mut self.scene);
        draw_fn(&mut self.scene);
        timer.record_time("cmd");

        let pixmap = self.scene.pixmap();
        let len = pixmap.data().len();
        assert!(
            buffer.len() >= len,
            "buffer too small: {} < {len}",
            buffer.len()
        );
        buffer[..len].copy_from_slice(pixmap.data());
        timer.record_time("copy");
        timer.print_times("tiny-skia: ");
    }
}
