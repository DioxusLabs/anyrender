//! A dummy implementation of the AnyRender traits while simply ignores all commands

use crate::{ImageRenderer, PaintScene, WindowHandle, WindowRenderer};
use std::sync::Arc;

#[derive(Copy, Clone, Default)]
pub struct NullWindowRenderer {
    is_active: bool,
}

impl NullWindowRenderer {
    pub fn new() -> Self {
        Self::default()
    }
}

impl WindowRenderer for NullWindowRenderer {
    type ScenePainter<'a>
        = NullScenePainter
    where
        Self: 'a;

    fn resume(&mut self, _window: Arc<dyn WindowHandle>, _width: u32, _height: u32) {
        self.is_active = true;
    }

    fn suspend(&mut self) {
        self.is_active = false
    }

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn set_size(&mut self, _width: u32, _height: u32) {}

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, _draw_fn: F) {}
}

#[derive(Copy, Clone, Default)]
pub struct NullImageRenderer;

impl NullImageRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl ImageRenderer for NullImageRenderer {
    type ScenePainter<'a>
        = NullScenePainter
    where
        Self: 'a;

    fn new(_width: u32, _height: u32) -> Self {
        Self
    }

    fn resize(&mut self, _width: u32, _height: u32) {}

    fn reset(&mut self) {}

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        _draw_fn: F,
        _vec: &mut Vec<u8>,
    ) {
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, _draw_fn: F, _buffer: &mut [u8]) {}
}

#[derive(Copy, Clone, Default)]
pub struct NullScenePainter;

impl NullScenePainter {
    pub fn new() -> Self {
        Self
    }
}

impl PaintScene for NullScenePainter {
    fn reset(&mut self) {}

    fn push_layer(
        &mut self,
        _blend: impl Into<peniko::BlendMode>,
        _alpha: f32,
        _transform: kurbo::Affine,
        _clip: &impl kurbo::Shape,
    ) {
    }

    fn push_clip_layer(&mut self, _clip: &impl kurbo::Shape) {}

    fn pop_layer(&mut self) {}

    fn stroke<'a>(
        &mut self,
        _style: &kurbo::Stroke,
        _transform: kurbo::Affine,
        _brush: impl Into<crate::PaintRef<'a>>,
        _brush_transform: Option<kurbo::Affine>,
        _shape: &impl kurbo::Shape,
    ) {
    }

    fn fill<'a>(
        &mut self,
        _style: peniko::Fill,
        _transform: kurbo::Affine,
        _brush: impl Into<crate::PaintRef<'a>>,
        _brush_transform: Option<kurbo::Affine>,
        _shape: &impl kurbo::Shape,
    ) {
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'s mut self,
        _font: &'a peniko::FontData,
        _font_size: f32,
        _hint: bool,
        _normalized_coords: &'a [crate::NormalizedCoord],
        _style: impl Into<peniko::StyleRef<'a>>,
        _brush: impl Into<crate::PaintRef<'a>>,
        _brush_alpha: f32,
        _transform: kurbo::Affine,
        _glyph_transform: Option<kurbo::Affine>,
        _glyphs: impl Iterator<Item = crate::Glyph>,
    ) {
    }

    fn draw_box_shadow(
        &mut self,
        _transform: kurbo::Affine,
        _rect: kurbo::Rect,
        _brush: peniko::Color,
        _radius: f64,
        _std_dev: f64,
    ) {
    }
}
