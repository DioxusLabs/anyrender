//! A type-erased [`PaintScene`], so one draw function can paint through
//! whichever backend is active when the backend is chosen at run time (a
//! GPU renderer, or a CPU one where there is no usable GPU).
//!
//! [`DynPaintScene`] is the object-safe form of [`PaintScene`], implemented
//! for every painter; [`DynScenePainter`] wraps a `&mut dyn DynPaintScene`
//! and is itself a [`PaintScene`].

use std::{any::Any, sync::Arc};

use kurbo::{Affine, Circle, PathEl, Rect, RoundedRect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrushRef, StyleRef};

use crate::{
    Filter, Glyph, NormalizedCoord, PaintRef, PaintScene, RegisterResourceError, RenderContext,
    ResourceId, Scene,
};

/// A shape passed through [`DynPaintScene`] without allocating for the
/// common cases.
pub enum DynShape<'a> {
    Rect(Rect),
    RoundedRect(RoundedRect),
    Circle(Circle),
    Path(&'a [PathEl]),
    /// Any other shape, flattened to a path.
    Owned(kurbo::BezPath),
}

/// The tolerance shapes without a path of their own are converted with,
/// as the Vello backends use.
const TOLERANCE: f64 = 0.1;

impl<'a> DynShape<'a> {
    pub fn new(shape: &'a impl Shape) -> Self {
        if let Some(rect) = shape.as_rect() {
            Self::Rect(rect)
        } else if let Some(rect) = shape.as_rounded_rect() {
            Self::RoundedRect(rect)
        } else if let Some(circle) = shape.as_circle() {
            Self::Circle(circle)
        } else if let Some(path) = shape.as_path_slice() {
            Self::Path(path)
        } else {
            Self::Owned(shape.to_path(TOLERANCE))
        }
    }
}

/// Calls `$call` with the shape as a concrete `&impl Shape`.
macro_rules! with_shape {
    ($shape:expr, |$s:ident| $call:expr) => {
        match $shape {
            DynShape::Rect($s) => $call,
            DynShape::RoundedRect($s) => $call,
            DynShape::Circle($s) => $call,
            DynShape::Path(path) => {
                let $s = path;
                $call
            }
            DynShape::Owned($s) => $call,
        }
    };
}

/// The object-safe form of [`PaintScene`] (and [`RenderContext`]).
#[allow(clippy::too_many_arguments)]
pub trait DynPaintScene {
    fn dyn_try_register_custom_resource(
        &mut self,
        resource: Box<dyn Any>,
    ) -> Result<ResourceId, RegisterResourceError>;
    fn dyn_unregister_resource(&mut self, resource_id: ResourceId);
    fn dyn_renderer_specific_context(&self) -> Option<Box<dyn Any>>;
    fn dyn_reset(&mut self);
    fn dyn_push_layer(
        &mut self,
        fill: Fill,
        blend: BlendMode,
        alpha: f32,
        transform: Affine,
        clip: &DynShape<'_>,
        filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    );
    fn dyn_push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &DynShape<'_>);
    fn dyn_pop_layer(&mut self);
    fn dyn_stroke(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: PaintRef<'_>,
        brush_transform: Option<Affine>,
        shape: &DynShape<'_>,
    );
    fn dyn_fill(
        &mut self,
        style: Fill,
        transform: Affine,
        brush: PaintRef<'_>,
        brush_transform: Option<Affine>,
        shape: &DynShape<'_>,
    );
    fn dyn_draw_glyphs<'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: StyleRef<'a>,
        brush: PaintRef<'a>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: &[Glyph],
    );
    fn dyn_draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        brush: Color,
        radius: f64,
        std_dev: f64,
    );
    fn dyn_append_scene(&mut self, scene: Scene, scene_transform: Affine);
    fn dyn_draw_image(&mut self, image: ImageBrushRef<'_>, transform: Affine);
}

impl<P: PaintScene> DynPaintScene for P {
    fn dyn_try_register_custom_resource(
        &mut self,
        resource: Box<dyn Any>,
    ) -> Result<ResourceId, RegisterResourceError> {
        self.try_register_custom_resource(resource)
    }

    fn dyn_unregister_resource(&mut self, resource_id: ResourceId) {
        self.unregister_resource(resource_id);
    }

    fn dyn_renderer_specific_context(&self) -> Option<Box<dyn Any>> {
        self.renderer_specific_context()
    }

    fn dyn_reset(&mut self) {
        self.reset();
    }

    fn dyn_push_layer(
        &mut self,
        fill: Fill,
        blend: BlendMode,
        alpha: f32,
        transform: Affine,
        clip: &DynShape<'_>,
        filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    ) {
        with_shape!(clip, |shape| self.push_layer(
            fill,
            blend,
            alpha,
            transform,
            shape,
            filter,
            backdrop_filter
        ));
    }

    fn dyn_push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &DynShape<'_>) {
        with_shape!(clip, |shape| self.push_clip_layer(fill, transform, shape));
    }

    fn dyn_pop_layer(&mut self) {
        self.pop_layer();
    }

    fn dyn_stroke(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: PaintRef<'_>,
        brush_transform: Option<Affine>,
        shape: &DynShape<'_>,
    ) {
        with_shape!(shape, |shape| self.stroke(
            style,
            transform,
            brush,
            brush_transform,
            shape
        ));
    }

    fn dyn_fill(
        &mut self,
        style: Fill,
        transform: Affine,
        brush: PaintRef<'_>,
        brush_transform: Option<Affine>,
        shape: &DynShape<'_>,
    ) {
        with_shape!(shape, |shape| self.fill(
            style,
            transform,
            brush,
            brush_transform,
            shape
        ));
    }

    fn dyn_draw_glyphs<'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: StyleRef<'a>,
        brush: PaintRef<'a>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: &[Glyph],
    ) {
        self.draw_glyphs(
            font,
            font_size,
            hint,
            normalized_coords,
            embolden,
            style,
            brush,
            brush_alpha,
            transform,
            glyph_transform,
            glyphs.iter().copied(),
        );
    }

    fn dyn_draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        brush: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.draw_box_shadow(transform, rect, brush, radius, std_dev);
    }

    fn dyn_append_scene(&mut self, scene: Scene, scene_transform: Affine) {
        self.append_scene(scene, scene_transform);
    }

    fn dyn_draw_image(&mut self, image: ImageBrushRef<'_>, transform: Affine) {
        self.draw_image(image, transform);
    }
}

/// A [`PaintScene`] that paints through a `&mut dyn DynPaintScene`.
pub struct DynScenePainter<'a> {
    inner: &'a mut (dyn DynPaintScene + 'a),
    /// A glyph run's glyphs, gathered for the type-erased call; kept
    /// between runs.
    glyphs: Vec<Glyph>,
}

impl<'a> DynScenePainter<'a> {
    pub fn new(inner: &'a mut (dyn DynPaintScene + 'a)) -> Self {
        Self {
            inner,
            glyphs: Vec::new(),
        }
    }
}

impl RenderContext for DynScenePainter<'_> {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn Any>,
    ) -> Result<ResourceId, RegisterResourceError> {
        self.inner.dyn_try_register_custom_resource(resource)
    }

    fn unregister_resource(&mut self, resource_id: ResourceId) {
        self.inner.dyn_unregister_resource(resource_id);
    }

    fn renderer_specific_context(&self) -> Option<Box<dyn Any>> {
        self.inner.dyn_renderer_specific_context()
    }
}

impl PaintScene for DynScenePainter<'_> {
    fn reset(&mut self) {
        self.inner.dyn_reset();
    }

    fn push_layer(
        &mut self,
        fill: Fill,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    ) {
        self.inner.dyn_push_layer(
            fill,
            blend.into(),
            alpha,
            transform,
            &DynShape::new(clip),
            filter,
            backdrop_filter,
        );
    }

    fn push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &impl Shape) {
        self.inner
            .dyn_push_clip_layer(fill, transform, &DynShape::new(clip));
    }

    fn pop_layer(&mut self) {
        self.inner.dyn_pop_layer();
    }

    fn stroke<'b>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: impl Into<PaintRef<'b>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.inner.dyn_stroke(
            style,
            transform,
            brush.into(),
            brush_transform,
            &DynShape::new(shape),
        );
    }

    fn fill<'b>(
        &mut self,
        style: Fill,
        transform: Affine,
        brush: impl Into<PaintRef<'b>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.inner.dyn_fill(
            style,
            transform,
            brush.into(),
            brush_transform,
            &DynShape::new(shape),
        );
    }

    fn draw_glyphs<'b, 's: 'b>(
        &'s mut self,
        font: &'b FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'b [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'b>>,
        brush: impl Into<PaintRef<'b>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
    ) {
        self.glyphs.clear();
        self.glyphs.extend(glyphs);
        self.inner.dyn_draw_glyphs(
            font,
            font_size,
            hint,
            normalized_coords,
            embolden,
            style.into(),
            brush.into(),
            brush_alpha,
            transform,
            glyph_transform,
            &self.glyphs,
        );
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        brush: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.inner
            .dyn_draw_box_shadow(transform, rect, brush, radius, std_dev);
    }

    fn append_scene(&mut self, scene: Scene, scene_transform: Affine) {
        self.inner.dyn_append_scene(scene, scene_transform);
    }

    fn draw_image(&mut self, image: ImageBrushRef, transform: Affine) {
        self.inner.dyn_draw_image(image, transform);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::{RenderCommand, Scene};
    use kurbo::BezPath;
    use peniko::color::palette::css::RED;

    #[test]
    fn a_dyn_painter_records_what_the_painter_would() {
        let path = BezPath::from_svg("M0 0L10 0L10 10Z").unwrap();
        let paint = |painter: &mut dyn DynPaintScene| {
            let mut painter = DynScenePainter::new(painter);
            painter.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                RED,
                None,
                &Rect::new(0.0, 0.0, 4.0, 4.0),
            );
            painter.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &path);
            painter.stroke(
                &Stroke::new(2.0),
                Affine::IDENTITY,
                RED,
                None,
                &Circle::new((5.0, 5.0), 3.0),
            );
            painter.pop_layer();
        };
        let mut direct = Scene::new();
        direct.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            RED,
            None,
            &Rect::new(0.0, 0.0, 4.0, 4.0),
        );
        direct.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &path);
        direct.stroke(
            &Stroke::new(2.0),
            Affine::IDENTITY,
            RED,
            None,
            &Circle::new((5.0, 5.0), 3.0),
        );
        direct.pop_layer();
        let mut through = Scene::new();
        paint(&mut through);
        assert_eq!(through.commands.len(), direct.commands.len());
        assert!(matches!(
            through.commands[1],
            RenderCommand::PushClipLayer(_)
        ));
        assert_eq!(
            format!("{:?}", through.commands),
            format!("{:?}", direct.commands)
        );
    }
}
