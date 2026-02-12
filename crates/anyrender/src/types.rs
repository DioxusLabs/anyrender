//! Types that are used within the Anyrender traits

use peniko::{Color, Gradient, ImageBrush, ImageData};
use std::{any::Any, sync::Arc};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub type NormalizedCoord = i16;

/// Opaque handle to a registered resource managed by a [`RenderContext`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResourceId(pub u64);

/// A registered image resource that combines a [`ResourceId`] with the image dimensions.
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageResource {
    pub id: ResourceId,
    pub width: u32,
    pub height: u32,
}

/// Renderers implement this trait to handle resource allocation/deallocation separately
/// from scene construction. Resources are registered once and then referenced by
/// [`ResourceId`] during painting.
pub trait RenderContext {
    /// Register an image and upload/convert it into a backend-specific backing resource.
    fn register_image(&mut self, image: ImageData) -> ImageResource;

    /// Unregister a previously registered resource, freeing any backing storage.
    fn unregister_resource(&mut self, id: ResourceId);
}

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
pub enum Paint<I = ImageBrush<ImageResource>, G = Gradient, C = Arc<dyn Any + Send + Sync>> {
    /// Solid color brush.
    Solid(Color),
    /// Gradient brush.
    Gradient(G),
    /// Image brush.
    Image(I),
    /// Custom paint (type erased as each backend will have their own)
    Custom(C),
}

/// A reference-friendly version of [`Paint`] used in [`PaintScene`](crate::PaintScene) method signatures.
///
/// Images are referenced by [`ImageResource`] (containing a [`ResourceId`]) rather than
/// by raw pixel data.
pub type PaintRef<'a> = Paint<ImageBrush<ImageResource>, &'a Gradient, &'a (dyn Any + Send + Sync)>;

impl Paint {
    pub fn as_ref(&self) -> PaintRef<'_> {
        match self {
            Paint::Solid(color) => Paint::Solid(*color),
            Paint::Gradient(gradient) => Paint::Gradient(gradient),
            Paint::Image(image) => Paint::Image(image.clone()),

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
impl<G, C> From<ImageBrush<ImageResource>> for Paint<ImageBrush<ImageResource>, G, C> {
    fn from(value: ImageBrush<ImageResource>) -> Self {
        Paint::Image(value)
    }
}
impl<I, G> From<Arc<dyn Any + Send + Sync>> for Paint<I, G, Arc<dyn Any + Send + Sync>> {
    fn from(value: Arc<dyn Any + Send + Sync>) -> Self {
        Paint::Custom(value)
    }
}
