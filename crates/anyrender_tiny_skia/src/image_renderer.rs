use anyrender::{ImageRenderer, PaintScene};
use debug_timer::debug_timer;
use tiny_skia::{
    self, FillRule, FilterQuality, GradientStop, LineCap, LineJoin, LinearGradient, Mask, MaskType,
    Paint, Path, PathBuilder, Pattern, Pixmap, RadialGradient, Shader, SpreadMode, Stroke,
    Transform,
};

use crate::TinySkiaScenePainter;

pub struct TinySkiaImageRenderer {
    scene: TinySkiaScenePainter,
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

        let mut painter = &mut self.scene;

        painter.reset();
        draw_fn(&mut painter);

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
        draw_fn(&mut painter);
        timer.record_time("cmd");

        // Convert pixmap to RGBA8
        let expected_len = (self.width * self.height * 4) as usize;
        if buffer.len() >= expected_len {
            for (i, pixel) in self.layers[0].pixmap.pixels().iter().enumerate() {
                let base = i * 4;
                if base + 3 < buffer.len() {
                    buffer[base] = pixel.red();
                    buffer[base + 1] = pixel.green();
                    buffer[base + 2] = pixel.blue();
                    buffer[base + 3] = pixel.alpha();
                }
            }
        }
        timer.record_time("render");
        timer.print_times("tiny-skia image: ");
    }
}
