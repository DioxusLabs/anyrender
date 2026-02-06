//! Types that are used within the Anyrender traits

use peniko::{Brush, BrushRef, Color, Gradient, ImageBrush, ImageBrushRef};
use std::{any::Any, sync::Arc};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub type NormalizedCoord = i16;

/// A positioned glyph.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Glyph {
    pub id: u32,
    pub x: f32,
    pub y: f32,
}

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CustomPaint {
    pub source_id: u64,
    pub width: u32,
    pub height: u32,
    pub scale: f64,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Paint<I = ImageBrush, G = Gradient, C = Arc<dyn Any + Send + Sync>> {
    /// Solid color brush.
    Solid(Color),
    /// Gradient brush.
    Gradient(G),
    /// Image brush.
    Image(I),
    /// Custom paint (type erased as each backend will have their own)
    Custom(C),
}

pub type PaintRef<'a> = Paint<ImageBrushRef<'a>, &'a Gradient, &'a (dyn Any + Send + Sync)>;

impl Paint {
    pub fn as_ref(&self) -> PaintRef<'_> {
        match self {
            Paint::Solid(color) => Paint::Solid(*color),
            Paint::Gradient(gradient) => Paint::Gradient(gradient),
            Paint::Image(image) => Paint::Image(image.as_ref()),

            // Custom paints are translated into "invisible" where they are not supported
            Paint::Custom(custom) => Paint::Custom(custom.as_ref()),
        }
    }
}

impl<'a> From<&'a Paint> for PaintRef<'a> {
    fn from(paint: &'a Paint) -> Self {
        paint.as_ref()
    }
}

impl<'a> From<PaintRef<'a>> for BrushRef<'a> {
    fn from(value: PaintRef<'a>) -> Self {
        match value {
            Paint::Solid(color) => Brush::Solid(color),
            Paint::Gradient(gradient) => Brush::Gradient(gradient),
            Paint::Image(image) => Brush::Image(image),

            // Custom paints are translated into "invisible" where they are not supported
            Paint::Custom(_) => Brush::Solid(Color::TRANSPARENT),
        }
    }
}

// #[derive(Clone, Debug)]
// pub enum PaintRef<'a> {
//     /// Solid color brush.
//     Solid(Color),
//     /// Gradient brush.
//     Gradient(&'a Gradient),
//     /// Image brush.
//     Image(ImageBrushRef<'a>),
//     /// Custom paint (type erased as each backend will have their own)
//     Custom(Arc<dyn Any + Send + Sync>),
// }

impl<I, G, C> From<Color> for Paint<I, G, C> {
    fn from(value: Color) -> Self {
        Paint::Solid(value)
    }
}
impl<'a, I, C> From<&'a Gradient> for Paint<I, &'a Gradient, C> {
    fn from(value: &'a Gradient) -> Self {
        Paint::Gradient(value)
    }
}
impl<'a, G, C> From<ImageBrushRef<'a>> for Paint<ImageBrushRef<'a>, G, C> {
    fn from(value: ImageBrushRef<'a>) -> Self {
        Paint::Image(value)
    }
}
impl<I, G> From<Arc<dyn Any + Send + Sync>> for Paint<I, G, Arc<dyn Any + Send + Sync>> {
    fn from(value: Arc<dyn Any + Send + Sync>) -> Self {
        Paint::Custom(value)
    }
}
impl<'a> From<BrushRef<'a>> for PaintRef<'a> {
    fn from(value: BrushRef<'a>) -> Self {
        match value {
            BrushRef::Solid(color) => PaintRef::Solid(color),
            BrushRef::Gradient(gradient) => PaintRef::Gradient(gradient),
            BrushRef::Image(image) => PaintRef::Image(image),
        }
    }
}
