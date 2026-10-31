use std::sync::Arc;

use crate::{
    BoxShadowKind, Filter, Glyph, NonUniformRoundedRect, NormalizedCoord, Paint, PaintRef,
    PaintScene, RenderContext,
};
use kurbo::{Affine, BezPath, PathEl, Point, Rect, RoundedRect, Shape, Stroke, Vec2};
use peniko::{BlendMode, Color, Fill, FontData, Style, StyleRef};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const DEFAULT_TOLERANCE: f64 = 0.1;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RenderCommand<Font = FontData, Brush = Paint> {
    /// Pushes a new layer clipped by the specified shape and composed with previous layers using the specified blend mode.
    /// Every drawing command after this call will be clipped by the shape until the layer is popped.
    /// However, the transforms are not saved or modified by the layer stack.
    PushLayer(LayerCommand),
    /// Pushes a new clip layer clipped by the specified shape.
    /// Every drawing command after this call will be clipped by the shape until the layer is popped.
    /// However, the transforms are not saved or modified by the layer stack.
    PushClipLayer(ClipCommand),
    /// Pops the current layer.
    PopLayer,
    /// Strokes a shape using the specified style and brush.
    Stroke(StrokeCommand<Brush>),
    /// Fills a shape using the specified style and brush.
    Fill(FillCommand<Brush>),
    /// Draws a run of glyphs
    GlyphRun(GlyphRunCommand<Font, Brush>),
    /// Draw a rounded rectangle blurred with a gaussian filter.
    BoxShadow(BoxShadowCommand),
}

impl RenderCommand {
    /// Apply the specific transform to the command
    fn apply_transform(mut self, transform: Affine) -> Self {
        match &mut self {
            RenderCommand::PushLayer(cmd) => cmd.transform = transform * cmd.transform,
            RenderCommand::PushClipLayer(cmd) => cmd.transform = transform * cmd.transform,
            RenderCommand::PopLayer => {}
            RenderCommand::Stroke(cmd) => cmd.transform = transform * cmd.transform,
            RenderCommand::Fill(cmd) => cmd.transform = transform * cmd.transform,
            RenderCommand::GlyphRun(cmd) => cmd.transform = transform * cmd.transform,
            RenderCommand::BoxShadow(cmd) => cmd.transform = transform * cmd.transform,
        };

        self
    }
}

/// Pushes a new layer clipped by the specified shape and composed with previous layers using the specified blend mode.
/// Every drawing command after this call will be clipped by the shape until the layer is popped.
/// However, the transforms are not saved or modified by the layer stack.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LayerCommand {
    #[cfg_attr(feature = "serde", serde(default))]
    pub fill: Fill,
    pub blend: BlendMode,
    pub alpha: f32,
    pub transform: Affine,
    pub clip: RecordedShape,
    pub filter: Option<Arc<Filter>>,
    pub backdrop_filter: Option<Arc<Filter>>,
}

/// Pushes a new clip layer clipped by the specified shape.
/// Every drawing command after this call will be clipped by the shape until the layer is popped.
/// However, the transforms are not saved or modified by the layer stack.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClipCommand {
    #[cfg_attr(feature = "serde", serde(default))]
    pub fill: Fill,
    pub transform: Affine,
    pub clip: RecordedShape,
}

/// Strokes a shape using the specified style and brush.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StrokeCommand<Brush = Paint> {
    pub style: Stroke,
    pub transform: Affine,
    pub brush: Brush, // TODO: review ownership to avoid cloning. Should brushes be a "resource"?
    pub brush_transform: Option<Affine>,
    pub shape: RecordedShape,
}

/// Fills a shape using the specified style and brush.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FillCommand<Brush = Paint> {
    pub fill: Fill,
    pub transform: Affine,
    pub brush: Brush, // TODO: review ownership to avoid cloning. Should brushes be a "resource"?
    pub brush_transform: Option<Affine>,
    pub shape: RecordedShape,
}

/// Draws a run of glyphs
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GlyphRunCommand<Font = FontData, Brush = Paint> {
    pub font_data: Font,
    pub font_size: f32,
    pub hint: bool,
    pub normalized_coords: Vec<NormalizedCoord>,
    #[cfg_attr(feature = "serde", serde(default = "Default::default"))]
    pub embolden: kurbo::Vec2,
    pub style: Style,
    pub brush: Brush,
    pub brush_alpha: f32,
    pub transform: Affine,
    pub glyph_transform: Option<Affine>,
    pub glyphs: Vec<Glyph>,
}

/// Draw a box shadow cast by a box
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BoxShadowCommand {
    pub transform: Affine,
    pub box_shape: NonUniformRoundedRect,
    pub offset: Vec2,
    pub spread: f64,
    pub std_dev: f64,
    pub brush: Color,
    pub kind: BoxShadowKind,
}

/// A shape stored in a recorded command.
///
/// Rects and rounded rects are kept as-is (rather than being converted to paths) so that backends
/// can still use their fast paths for them when the recording is replayed.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum RecordedShape {
    Rect(Rect),
    RoundedRect(RoundedRect),
    Path(#[cfg_attr(feature = "serde", serde(with = "svg_path"))] BezPath),
}

impl RecordedShape {
    /// Record `shape`, converting it to a path unless it is a rect or rounded rect.
    pub fn from_shape(shape: &impl Shape, tolerance: f64) -> Self {
        if let Some(rect) = shape.as_rect() {
            Self::Rect(rect)
        } else if let Some(rounded_rect) = shape.as_rounded_rect() {
            Self::RoundedRect(rounded_rect)
        } else {
            Self::Path(shape.into_path(tolerance))
        }
    }
}

impl From<BezPath> for RecordedShape {
    fn from(path: BezPath) -> Self {
        Self::Path(path)
    }
}

impl From<Rect> for RecordedShape {
    fn from(rect: Rect) -> Self {
        Self::Rect(rect)
    }
}

impl From<RoundedRect> for RecordedShape {
    fn from(rounded_rect: RoundedRect) -> Self {
        Self::RoundedRect(rounded_rect)
    }
}

/// Iterator over the path elements of a [`RecordedShape`].
#[allow(clippy::large_enum_variant, reason = "short-lived iterator")]
pub enum RecordedShapePathIter<'a> {
    Rect(<Rect as Shape>::PathElementsIter<'a>),
    RoundedRect(<RoundedRect as Shape>::PathElementsIter<'a>),
    Path(<BezPath as Shape>::PathElementsIter<'a>),
}

impl Iterator for RecordedShapePathIter<'_> {
    type Item = PathEl;

    fn next(&mut self) -> Option<PathEl> {
        match self {
            Self::Rect(iter) => iter.next(),
            Self::RoundedRect(iter) => iter.next(),
            Self::Path(iter) => iter.next(),
        }
    }
}

macro_rules! delegate {
    ($self:ident, $shape:ident => $expr:expr) => {
        match $self {
            RecordedShape::Rect($shape) => $expr,
            RecordedShape::RoundedRect($shape) => $expr,
            RecordedShape::Path($shape) => $expr,
        }
    };
}

impl Shape for RecordedShape {
    type PathElementsIter<'iter> = RecordedShapePathIter<'iter>;

    fn path_elements(&self, tolerance: f64) -> Self::PathElementsIter<'_> {
        match self {
            Self::Rect(rect) => RecordedShapePathIter::Rect(rect.path_elements(tolerance)),
            Self::RoundedRect(rounded_rect) => {
                RecordedShapePathIter::RoundedRect(rounded_rect.path_elements(tolerance))
            }
            Self::Path(path) => RecordedShapePathIter::Path(path.path_elements(tolerance)),
        }
    }

    fn area(&self) -> f64 {
        delegate!(self, shape => shape.area())
    }

    fn perimeter(&self, accuracy: f64) -> f64 {
        delegate!(self, shape => shape.perimeter(accuracy))
    }

    fn winding(&self, pt: Point) -> i32 {
        delegate!(self, shape => shape.winding(pt))
    }

    fn bounding_box(&self) -> Rect {
        delegate!(self, shape => shape.bounding_box())
    }

    fn to_path(&self, tolerance: f64) -> BezPath {
        delegate!(self, shape => shape.to_path(tolerance))
    }

    fn as_rect(&self) -> Option<Rect> {
        delegate!(self, shape => shape.as_rect())
    }

    fn as_rounded_rect(&self) -> Option<RoundedRect> {
        delegate!(self, shape => shape.as_rounded_rect())
    }

    fn as_path_slice(&self) -> Option<&[PathEl]> {
        delegate!(self, shape => shape.as_path_slice())
    }
}

/// A recording of a Scene or Scene Fragment stored as plain data types that can be stored
/// and passed around.
#[derive(Clone, Debug, PartialEq)]
pub struct Scene {
    pub tolerance: f64,
    pub commands: Vec<RenderCommand>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_TOLERANCE,
            commands: Vec::new(),
        }
    }
}

impl Scene {
    /// Create a new empty
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            commands: Vec::new(),
        }
    }

    fn convert_paint(&mut self, paint_ref: PaintRef<'_>) -> Paint {
        match paint_ref {
            Paint::Solid(color) => Paint::Solid(color),
            Paint::Gradient(gradient) => Paint::Gradient(gradient.clone()),
            Paint::Image(image) => Paint::Image(image.to_owned()),
            // TODO: handle this somehow
            Paint::Resource(id) => Paint::Resource(id),
            Paint::Custom(_) => Paint::Solid(Color::TRANSPARENT),
        }
    }
}

impl RenderContext for Scene {}
impl PaintScene for Scene {
    fn reset(&mut self) {
        self.commands.clear()
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
        let blend = blend.into();
        let clip = RecordedShape::from_shape(clip, self.tolerance);
        let layer = LayerCommand {
            fill,
            blend,
            alpha,
            transform,
            clip,
            filter,
            backdrop_filter,
        };
        self.commands.push(RenderCommand::PushLayer(layer));
    }

    fn push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &impl Shape) {
        let clip = RecordedShape::from_shape(clip, self.tolerance);
        let layer = ClipCommand {
            fill,
            transform,
            clip,
        };
        self.commands.push(RenderCommand::PushClipLayer(layer));
    }

    fn pop_layer(&mut self) {
        self.commands.push(RenderCommand::PopLayer);
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint_ref: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let shape = RecordedShape::from_shape(shape, self.tolerance);
        let brush = self.convert_paint(paint_ref.into());
        let stroke = StrokeCommand {
            style: style.clone(),
            transform,
            brush,
            brush_transform,
            shape,
        };
        self.commands.push(RenderCommand::Stroke(stroke));
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        let shape = RecordedShape::from_shape(shape, self.tolerance);
        let brush = self.convert_paint(paint.into());
        let fill = FillCommand {
            fill: style,
            transform,
            brush,
            brush_transform,
            shape,
        };
        self.commands.push(RenderCommand::Fill(fill));
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'a>>,
        paint_ref: impl Into<PaintRef<'a>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph>,
    ) {
        let brush = self.convert_paint(paint_ref.into());
        let glyph_run = GlyphRunCommand {
            font_data: font.clone(),
            font_size,
            hint,
            normalized_coords: normalized_coords.to_vec(),
            embolden,
            style: style.into().to_owned(),
            brush,
            brush_alpha,
            transform,
            glyph_transform,
            glyphs: glyphs.into_iter().collect(),
        };
        self.commands.push(RenderCommand::GlyphRun(glyph_run));
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        box_shape: &NonUniformRoundedRect,
        offset: Vec2,
        spread: f64,
        std_dev: f64,
        brush: Color,
        kind: BoxShadowKind,
    ) {
        let box_shadow = BoxShadowCommand {
            transform,
            box_shape: *box_shape,
            offset,
            spread,
            std_dev,
            brush,
            kind,
        };
        self.commands.push(RenderCommand::BoxShadow(box_shadow));
    }

    fn append_scene(&mut self, scene: Scene, scene_transform: Affine) {
        self.commands.extend(
            scene
                .commands
                .into_iter()
                .map(|cmd| cmd.apply_transform(scene_transform)),
        );
    }
}

/// Serde helper for serializing `BezPath` as an SVG path string.
#[cfg(feature = "serde")]
mod svg_path {
    use kurbo::BezPath;
    use serde::{self, Deserialize, Deserializer, Serializer};

    use crate::svg_path_parser;

    pub fn serialize<S>(path: &BezPath, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&path.to_svg())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BezPath, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        svg_path_parser::parse_svg_path(&s).map_err(serde::de::Error::custom)
    }
}
