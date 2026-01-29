use crate::{Glyph, NormalizedCoord, Paint, PaintRef, PaintScene};
use kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use peniko::{BlendMode, BrushRef, Color, Fill, FontData, Gradient, ImageBrush, Style, StyleRef};

const DEFAULT_TOLERANCE: f64 = 0.1;

pub trait Drawable {
    fn draw(&self, scene: &mut impl PaintScene);
}

#[derive(Clone)]
pub enum RenderCommand {
    PushLayer(LayerCommand),
    PushClipLayer(ClipCommand),
    PopLayer,
    Stroke(StrokeCommand),
    Fill(FillCommand),
    GlyphRun(GlyphRunCommand),
    BoxShadow(BoxShadowCommand),
}

#[derive(Clone)]
pub enum RecordedPaint {
    /// Solid color brush.
    Solid(Color),
    /// Gradient brush.
    Gradient(Gradient),
    /// Image brush.
    Image(ImageBrush),
}

#[derive(Clone)]
pub struct LayerCommand {
    pub blend: BlendMode,
    pub alpha: f32,
    pub transform: Affine,
    pub clip: BezPath, // TODO: more shape options
}

#[derive(Clone)]
pub struct ClipCommand {
    pub transform: Affine,
    pub clip: BezPath, // TODO: more shape options
}

#[derive(Clone)]
pub struct StrokeCommand {
    pub style: Stroke,
    pub transform: Affine,
    pub brush: RecordedPaint, // TODO: review ownership to avoid cloning. Should brushes be a "resource"?
    pub brush_transform: Option<Affine>,
    pub shape: BezPath, // TODO: more shape options
}

#[derive(Clone)]
pub struct FillCommand {
    pub fill: Fill,
    pub transform: Affine,
    pub brush: RecordedPaint, // TODO: review ownership to avoid cloning. Should brushes be a "resource"?
    pub brush_transform: Option<Affine>,
    pub shape: BezPath, // TODO: more shape options
}

#[derive(Clone)]
pub struct GlyphRunCommand {
    pub font_data: FontData,
    pub font_index: u32,
    pub font_size: f32,
    pub hint: bool,
    pub normalized_coords: Vec<NormalizedCoord>,
    pub style: Style,
    pub brush: RecordedPaint,
    pub brush_alpha: f32,
    pub transform: Affine,
    pub glyph_transform: Option<Affine>,
    pub glyphs: Vec<Glyph>,
}

#[derive(Clone)]
pub struct BoxShadowCommand {
    pub transform: Affine,
    pub rect: Rect,
    pub brush: Color,
    pub radius: f64,
    pub std_dev: f64,
}

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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            commands: Vec::new(),
        }
    }

    pub fn convert_brushref(&mut self, brush_ref: BrushRef<'_>) -> RecordedPaint {
        match brush_ref {
            BrushRef::Solid(color) => RecordedPaint::Solid(color),
            BrushRef::Gradient(gradient) => RecordedPaint::Gradient(gradient.clone()),
            BrushRef::Image(image) => RecordedPaint::Image(image.to_owned()),
        }
    }

    pub fn convert_paintref(&mut self, paint_ref: PaintRef<'_>) -> RecordedPaint {
        match paint_ref {
            Paint::Solid(color) => RecordedPaint::Solid(color),
            Paint::Gradient(gradient) => RecordedPaint::Gradient(gradient.clone()),
            Paint::Image(image) => RecordedPaint::Image(image.to_owned()),
            // TODO: handle this somehow
            Paint::Custom(_) => RecordedPaint::Solid(Color::TRANSPARENT),
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
        let font_index = font.index;
        let brush = self.convert_paintref(paint_ref.into());
        let glyph_run = GlyphRunCommand {
            font_data: font.clone(),
            font_index,
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
}
