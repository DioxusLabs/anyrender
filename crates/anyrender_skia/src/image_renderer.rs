use anyrender::ImageRenderer;
use debug_timer::debug_timer;
use skia_safe::{AlphaType, Color, ColorType, ImageInfo, SurfaceProps, graphics, surfaces};

use crate::{SkiaScenePainter, scene::SkiaSceneCache};

pub struct SkiaImageRenderer {
    image_info: ImageInfo,
    surface_props: SurfaceProps,
    scene_cache: SkiaSceneCache,
}

impl ImageRenderer for SkiaImageRenderer {
    type ScenePainter<'a>
        = SkiaScenePainter<'a>
    where
        Self: 'a;

    fn new(width: u32, height: u32) -> Result<Self, Box<dyn std::error::Error>> {
        graphics::set_font_cache_count_limit(100);
        graphics::set_typeface_cache_count_limit(100);
        graphics::set_resource_cache_total_bytes_limit(10485760);

        Ok(Self {
            image_info: ImageInfo::new(
                (width as i32, height as i32),
                ColorType::RGBA8888,
                AlphaType::Opaque,
                None,
            ),
            surface_props: SurfaceProps::default(),
            scene_cache: SkiaSceneCache::default(),
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.image_info = ImageInfo::new(
            (width as i32, height as i32),
            ColorType::RGBA8888,
            AlphaType::Opaque,
            None,
        );
    }

    fn reset(&mut self) {}

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        buffer: &mut Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug_timer!(timer, feature = "log_frame_times");

        let mut surface = surfaces::wrap_pixels(
            &self.image_info,
            &mut buffer[..],
            None,
            Some(&self.surface_props),
        )
        .ok_or("Invalid surface parameters")?;

        surface.canvas().clear(Color::WHITE);

        draw_fn(&mut SkiaScenePainter {
            inner: surface.canvas(),
            cache: &mut self.scene_cache,
        });
        timer.record_time("render");

        self.scene_cache.next_gen();
        timer.record_time("cache next gen");

        timer.print_times("skia_raster: ");
        Ok(())
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        buffer: &mut [u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug_timer!(timer, feature = "log_frame_times");

        let mut surface = surfaces::wrap_pixels(
            &self.image_info,
            &mut buffer[..],
            None,
            Some(&self.surface_props),
        )
        .ok_or("Invalid surface parameters")?;

        surface.canvas().clear(Color::WHITE);

        draw_fn(&mut SkiaScenePainter {
            inner: surface.canvas(),
            cache: &mut self.scene_cache,
        });
        timer.record_time("render");

        self.scene_cache.next_gen();
        timer.record_time("cache next gen");

        timer.print_times("skia_raster: ");
        Ok(())
    }
}
