use std::{io::Cursor, sync::Arc};

use anyrender::{ImageResource, PaintScene, RenderContext};
use image::ImageReader;
use kurbo::{Affine, Size, Vec2};
use peniko::{Blob, ImageBrush, ImageData, ImageSampler};

const GRAVITY: f64 = 0.5;

#[derive(Debug)]
pub struct Bunny {
    x: f64,
    y: f64,
    speed_x: f64,
    speed_y: f64,
}

impl Bunny {
    pub fn new(canvas: Size) -> Self {
        Self {
            x: fastrand::f64() * canvas.width,
            y: fastrand::f64() * canvas.height,
            speed_x: fastrand::f64() * 10.0,
            speed_y: fastrand::f64() * 10.0,
        }
    }

    pub fn position(&self) -> Vec2 {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }

    pub fn update(&mut self, canvas: Size) {
        // Apply speed to position
        self.x += self.speed_x;
        self.y += self.speed_y;

        // Apply gravity to y speed
        self.speed_y += GRAVITY;

        // Bounce off left wall
        if self.x < 0.0 {
            self.x = 0.0;
            self.speed_x *= -1.0;
        }

        // Bounce off right wall
        if self.x > canvas.width {
            self.x = canvas.width;
            self.speed_x *= -1.0;
        }

        if self.y > canvas.height {
            self.y = canvas.height;
            self.speed_y *= -0.85;
            if fastrand::f64() > 0.5 {
                self.speed_y -= fastrand::f64() * 6.0;
            }
        }

        // Floor y at 0
        if self.y < 0.0 {
            self.y = 0.0;
            self.speed_y = 0.0;
        }
    }
}

pub struct BunnyManager {
    canvas_size: Size,
    bunny_image_data: ImageData,
    bunny_image: Option<ImageBrush<ImageResource>>,
    bunnies: Vec<Bunny>,
}

impl BunnyManager {
    pub fn new(canvas_width: f64, canvas_height: f64) -> Self {
        Self {
            canvas_size: Size {
                width: canvas_width,
                height: canvas_height,
            },
            bunny_image_data: create_bunny_image_data(),
            bunny_image: None,
            bunnies: Vec::new(),
        }
    }

    /// Register the bunny image with the given render context.
    /// Must be called whenever the backend renderer changes.
    pub fn register_image(&mut self, ctx: &mut impl RenderContext) {
        let resource = ctx.register_image(self.bunny_image_data.clone());
        self.bunny_image = Some(ImageBrush {
            image: resource,
            sampler: ImageSampler {
                x_extend: peniko::Extend::Pad,
                y_extend: peniko::Extend::Pad,
                quality: peniko::ImageQuality::Medium,
                alpha: 1.0,
            },
        });
    }

    pub fn add_bunnies(&mut self, count: usize) {
        self.bunnies
            .resize_with(self.bunnies.len() + count, || Bunny::new(self.canvas_size));
    }

    pub fn clear_bunnies(&mut self) {
        self.bunnies.clear();
    }

    pub fn count(&self) -> usize {
        self.bunnies.len()
    }

    pub fn update(&mut self, canvas_width: f64, canvas_height: f64) {
        self.canvas_size.width = canvas_width;
        self.canvas_size.height = canvas_height;
        for bunny in &mut self.bunnies {
            bunny.update(self.canvas_size);
        }
    }

    pub fn draw<S: PaintScene>(&self, scene: &mut S, scale_factor: f64) {
        let Some(bunny_image) = &self.bunny_image else {
            return;
        };
        for bunny in &self.bunnies {
            let pos = bunny.position();
            scene.draw_image(
                bunny_image.clone(),
                Affine::translate(pos).then_scale(scale_factor),
            );
        }
    }
}

fn create_bunny_image_data() -> ImageData {
    static BUNNY_IMAGE_DATA: &[u8] = include_bytes!("./bunny.png");
    let raw_bunny_image =
        ImageReader::with_format(Cursor::new(BUNNY_IMAGE_DATA), image::ImageFormat::Png)
            .decode()
            .unwrap()
            .into_rgba8();
    let width = raw_bunny_image.width();
    let height = raw_bunny_image.height();
    ImageData {
        data: Blob::new(Arc::new(raw_bunny_image.into_vec())),
        format: peniko::ImageFormat::Rgba8,
        alpha_type: peniko::ImageAlphaType::Alpha,
        width,
        height,
    }
}
