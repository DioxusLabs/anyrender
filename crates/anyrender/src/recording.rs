use crate::{
    Glyph, ImageResource, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext, ResourceId,
};
use kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use peniko::{BlendMode, Brush, Color, Fill, FontData, ImageBrush, ImageData, Style, StyleRef};
use rustc_hash::FxHashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

const DEFAULT_TOLERANCE: f64 = 0.1;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum RenderCommand<Font = FontData, Image = ImageResource> {
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
    Stroke(StrokeCommand<Image>),
    /// Fills a shape using the specified style and brush.
    Fill(FillCommand<Image>),
    /// Draws a run of glyphs
    GlyphRun(GlyphRunCommand<Font, Image>),
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
    pub blend: BlendMode,
    pub alpha: f32,
    pub transform: Affine,
    #[cfg_attr(feature = "serde", serde(with = "svg_path"))]
    pub clip: BezPath, // TODO: more shape options
}

/// Pushes a new clip layer clipped by the specified shape.
/// Every drawing command after this call will be clipped by the shape until the layer is popped.
/// However, the transforms are not saved or modified by the layer stack.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ClipCommand {
    pub transform: Affine,
    #[cfg_attr(feature = "serde", serde(with = "svg_path"))]
    pub clip: BezPath, // TODO: more shape options
}

/// Strokes a shape using the specified style and brush.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StrokeCommand<Image> {
    pub style: Stroke,
    pub transform: Affine,
    pub brush: Brush<ImageBrush<Image>>, // TODO: review ownership to avoid cloning. Should brushes be a "resource"?
    pub brush_transform: Option<Affine>,
    #[cfg_attr(feature = "serde", serde(with = "svg_path"))]
    pub shape: BezPath, // TODO: more shape options
}

/// Fills a shape using the specified style and brush.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FillCommand<Image> {
    pub fill: Fill,
    pub transform: Affine,
    pub brush: Brush<ImageBrush<Image>>, // TODO: review ownership to avoid cloning. Should brushes be a "resource"?
    pub brush_transform: Option<Affine>,
    #[cfg_attr(feature = "serde", serde(with = "svg_path"))]
    pub shape: BezPath, // TODO: more shape options
}

/// Draws a run of glyphs
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GlyphRunCommand<Font = FontData, Image = ImageResource> {
    pub font_data: Font,
    pub font_size: f32,
    pub hint: bool,
    pub normalized_coords: Vec<NormalizedCoord>,
    pub style: Style,
    pub brush: Brush<ImageBrush<Image>>,
    pub brush_alpha: f32,
    pub transform: Affine,
    pub glyph_transform: Option<Affine>,
    pub glyphs: Vec<Glyph>,
}

/// Draw a box shadow around a box
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BoxShadowCommand {
    pub transform: Affine,
    pub rect: Rect,
    pub brush: Color,
    pub radius: f64,
    pub std_dev: f64,
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

    fn convert_paintref(&mut self, paint_ref: PaintRef<'_>) -> Brush<ImageBrush<ImageResource>> {
        match paint_ref {
            Paint::Solid(color) => Brush::Solid(color),
            Paint::Gradient(gradient) => Brush::Gradient(gradient.clone()),
            Paint::Image(image) => Brush::Image(image),
            // TODO: handle this somehow
            Paint::Custom(_) => Brush::Solid(Color::TRANSPARENT),
        }
    }
}

impl PaintScene for Scene {
    fn reset(&mut self) {
        self.commands.clear()
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
    ) {
        let blend = blend.into();
        let clip = clip.into_path(self.tolerance);
        let layer = LayerCommand {
            blend,
            alpha,
            transform,
            clip,
        };
        self.commands.push(RenderCommand::PushLayer(layer));
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        let clip = clip.into_path(self.tolerance);
        let layer = ClipCommand { transform, clip };
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
        let shape = shape.into_path(self.tolerance);
        let brush = self.convert_paintref(paint_ref.into());
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
        let shape = shape.into_path(self.tolerance);
        let brush = self.convert_paintref(paint.into());
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
        style: impl Into<StyleRef<'a>>,
        paint_ref: impl Into<PaintRef<'a>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph>,
    ) {
        let brush = self.convert_paintref(paint_ref.into());
        let glyph_run = GlyphRunCommand {
            font_data: font.clone(),
            font_size,
            hint,
            normalized_coords: normalized_coords.to_vec(),
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
        rect: Rect,
        brush: Color,
        radius: f64,
        std_dev: f64,
    ) {
        let box_shadow = BoxShadowCommand {
            transform,
            rect,
            brush,
            radius,
            std_dev,
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

/// A [`RenderContext`] for use with recording scenes.
///
/// This context assigns [`ResourceId`]s and stores the associated [`ImageData`] in a
/// [`FxHashMap`] so that recorded scenes can later be serialized or replayed through a
/// backend-specific context.
pub struct RecordingRenderContext {
    image_data: FxHashMap<ResourceId, ImageData>,
    next_resource_id: u64,
}

impl RecordingRenderContext {
    pub fn new() -> Self {
        Self {
            image_data: FxHashMap::default(),
            next_resource_id: 0,
        }
    }

    pub fn image_data(&self) -> &FxHashMap<ResourceId, ImageData> {
        &self.image_data
    }
}

impl Default for RecordingRenderContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderContext for RecordingRenderContext {
    fn register_image(&mut self, image: ImageData) -> ImageResource {
        let id = ResourceId(self.next_resource_id);
        self.next_resource_id += 1;
        let width = image.width;
        let height = image.height;
        self.image_data.insert(id, image);
        ImageResource { id, width, height }
    }

    fn unregister_resource(&mut self, id: ResourceId) {
        self.image_data.remove(&id);
    }
}

/// Serde helper for serializing `BezPath` as an SVG path string.
#[cfg(feature = "serde")]
mod svg_path {
    use kurbo::BezPath;
    use serde::{self, Deserialize, Deserializer, Serializer};

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
        BezPath::from_svg(&s).map_err(serde::de::Error::custom)
    }
}
