use anyrender::{Glyph, ImageRenderer, PaintScene};
use anyrender_tiny_skia::TinySkiaImageRenderer;
use kurbo::{Affine, Circle, Rect, RoundedRect};
use peniko::{Blob, Color, Fill, FontData, Mix};
use std::sync::Arc;
use std::time::Instant;

const W: u32 = 1024;
const H: u32 = 768;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn color(&mut self) -> Color {
        Color::from_rgba8(
            (self.f() * 255.) as u8,
            (self.f() * 255.) as u8,
            (self.f() * 255.) as u8,
            200,
        )
    }
    fn rect(&mut self, max: f64) -> Rect {
        let (x, y) = (self.f() * W as f64, self.f() * H as f64);
        Rect::new(x, y, x + 4. + self.f() * max, y + 4. + self.f() * max)
    }
}

fn fills<S: PaintScene>(s: &mut S) {
    let mut r = Rng(1);
    for _ in 0..10_000 {
        let c = r.color();
        s.fill(Fill::NonZero, Affine::IDENTITY, c, None, &r.rect(60.));
    }
    for _ in 0..2_000 {
        let c = r.color();
        let circ = Circle::new((r.f() * W as f64, r.f() * H as f64), 5. + r.f() * 30.);
        s.fill(Fill::NonZero, Affine::IDENTITY, c, None, &circ);
    }
}

fn clips<S: PaintScene>(s: &mut S) {
    let mut r = Rng(2);
    for _ in 0..500 {
        let clip = RoundedRect::from_rect(r.rect(200.), 8.);
        s.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &clip);
        for _ in 0..10 {
            let c = r.color();
            s.fill(Fill::NonZero, Affine::IDENTITY, c, None, &r.rect(150.));
        }
        s.pop_layer();
    }
}

fn nested_clips<S: PaintScene>(s: &mut S) {
    let mut r = Rng(3);
    for _ in 0..50 {
        let base = r.rect(400.);
        for d in 0..8 {
            let clip = RoundedRect::from_rect(base.inset(-(d as f64) * 4.), 6.);
            s.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &clip);
        }
        for _ in 0..10 {
            let c = r.color();
            s.fill(Fill::NonZero, Affine::IDENTITY, c, None, &r.rect(300.));
        }
        for _ in 0..8 {
            s.pop_layer();
        }
    }
}

fn groups<S: PaintScene>(s: &mut S) {
    let mut r = Rng(4);
    for _ in 0..300 {
        let clip = r.rect(300.);
        s.push_layer(
            Fill::NonZero,
            Mix::Normal,
            0.5,
            Affine::IDENTITY,
            &clip,
            None,
            None,
        );
        for _ in 0..10 {
            let c = r.color();
            s.fill(Fill::NonZero, Affine::IDENTITY, c, None, &r.rect(150.));
        }
        s.pop_layer();
    }
}

fn text<S: PaintScene>(s: &mut S, font: &FontData) {
    for line in 0..40 {
        let glyphs = (0..120).map(move |i| Glyph {
            id: 36 + (i * 7 + line) % 50,
            x: 4. + i as f32 * 8.4,
            y: 18. + line as f32 * 19.,
        });
        s.draw_glyphs(
            font,
            16.,
            false,
            &[],
            kurbo::Vec2::ZERO,
            Fill::NonZero,
            Color::BLACK,
            1.,
            Affine::IDENTITY,
            None,
            glyphs,
        );
    }
}

fn bench(
    name: &str,
    iters: u32,
    mut f: impl FnMut(&mut <TinySkiaImageRenderer as ImageRenderer>::ScenePainter<'_>),
) {
    let mut renderer = TinySkiaImageRenderer::new(W, H);
    let mut buf = Vec::new();
    renderer.render_to_vec(|s| f(s), &mut buf);
    let t = Instant::now();
    for _ in 0..iters {
        renderer.render_to_vec(|s| f(s), &mut buf);
    }
    let ms = t.elapsed().as_secs_f64() * 1000. / iters as f64;
    std::fs::write(
        format!("{}/{name}.rgba", std::env::var("OUT").unwrap()),
        &buf,
    )
    .unwrap();
    println!("{name}\t{ms:.2}");
}

fn main() {
    let bytes = std::fs::read("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").unwrap();
    let font = FontData::new(Blob::new(Arc::new(bytes)), 0);
    bench("empty", 50, |_| {});
    bench("fills", 10, fills);
    bench("clips", 10, clips);
    bench("nested_clips", 10, nested_clips);
    bench("groups", 10, groups);
    bench("text", 20, |s| text(s, &font));
}
