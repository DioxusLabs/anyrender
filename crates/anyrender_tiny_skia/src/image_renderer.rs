use anyrender::{ImageRenderer, PaintScene};
use debug_timer::debug_timer;
use tiny_skia::{Mask, Pixmap};

use crate::TinySkiaScenePainter;

pub struct TinySkiaImageRenderer {
    pub(crate) scene: TinySkiaScenePainter,
}

impl ImageRenderer for TinySkiaImageRenderer {
    type ScenePainter<'a> = TinySkiaScenePainter;

    fn new(width: u32, height: u32) -> Self {
        Self {
            scene: TinySkiaScenePainter::new(width, height),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if !self.scene.layers.is_empty() {
            self.scene.layers[0].pixmap =
                Pixmap::new(width, height).expect("Failed to create pixmap");
            self.scene.layers[0].mask = Mask::new(width, height).expect("Failed to create mask");
        } else {
            self.scene = TinySkiaScenePainter::new(width, height);
        }
    }

    fn reset(&mut self) {
        self.scene.reset();
    }

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        vec: &mut Vec<u8>,
    ) {
        vec.clear();
        vec.reserve(
            (self.scene.layers[0].pixmap.width() * self.scene.layers[0].pixmap.height() * 4)
                as usize,
        );

        let painter = &mut self.scene;

        painter.reset();
        draw_fn(painter);

        // Convert pixmap to RGBA8
        for pixel in self.scene.layers[0].pixmap.pixels() {
            vec.push(pixel.red());
            vec.push(pixel.green());
            vec.push(pixel.blue());
            vec.push(pixel.alpha());
        }
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F, buffer: &mut [u8]) {
        debug_timer!(timer, feature = "log_frame_times");
        let painter = &mut self.scene;
        painter.reset();
        draw_fn(painter);
        timer.record_time("cmd");

        if let Some(layer) = self.scene.layers.first() {
            let pixmap = &layer.pixmap;
            let width = pixmap.width() as usize;
            let height = pixmap.height() as usize;
            let expected_len = width * height * 4;

            assert!(
                buffer.len() >= expected_len,
                "buffer too small: {} < {}",
                buffer.len(),
                expected_len
            );

            let pixels = pixmap.pixels();

            buffer[..expected_len]
                .chunks_exact_mut(4)
                .zip(pixels.iter())
                .for_each(|(chunk, pixel)| {
                    chunk[0] = pixel.red();
                    chunk[1] = pixel.green();
                    chunk[2] = pixel.blue();
                    chunk[3] = pixel.alpha();
                });
        }

        timer.record_time("render");
        timer.print_times("tiny-skia image: ");
    }
}
