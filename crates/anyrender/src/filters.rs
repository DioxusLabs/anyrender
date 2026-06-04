// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Filter effects API based on the W3C Filter Effects specification.
//!
//! This module provides a comprehensive filter system supporting both high-level
//! CSS filter functions and low-level SVG filter primitives. The API is designed
//! to follow the W3C Filter Effects Module Level 1 specification.
//!
//! See: <https://drafts.fxtf.org/filter-effects/>

// ## Vello Implementation Status
//
// ### Implemented
//
// **Filter Functions:**
// - `Blur` - Gaussian blur effect
//
// **Filter Primitives (Single Use Only):**
// - `Flood` - Solid color fill
// - `GaussianBlur` - Gaussian blur filter
// - `DropShadow` - Drop shadow effect (compound primitive)
// - `Offset` - Translation/shift (single primitive)

use kurbo::{Affine, Rect, Vec2};
use peniko::color::{AlphaColor, Srgb};
use smallvec::SmallVec;

use self::{
    blur::GaussianBlurFilter,
    color_transformation::ColorMatrix,
    component_transfer::ComponentTransferFilter,
    composite::CompositeOperator,
    convolution::ConvolutionKernel,
    displacement::DisplacementMapFilter,
    lighting::{DiffuseLightingFilter, SpecularLightingFilter},
    morphology::MorphologyFilter,
    shadow::DropShadow,
    turbulence::TurbulenceFilter,
};

/// A directed acyclic graph (DAG) of filter operations.
///
/// The graph represents a pipeline of filter primitives where outputs of some
/// primitives can be used as inputs to others. Each primitive has a unique `FilterId`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Filter {
    /// All filter primitives in the graph, stored in insertion order.
    primitives: SmallVec<[FilterGraphNode; 1]>,
    /// The final output filter ID whose result is the output of this graph.
    output: FilterId,
    /// Accumulated bounds expansion from all primitives in the graph, cached in user space.
    /// This is the axis-aligned bounding box of the expansion region (centered at origin),
    /// which can be transformed to device space when needed.
    expansion_rect: Rect,
    // TODO: Add bounds restricting where the filter applies.
    // Optional bounds restricting where the filter applies.
    // If `None`, the filter applies to the entire filtered element.
    // pub bounds: Option<Rect>,
}

impl Default for Filter {
    fn default() -> Self {
        Self::empty()
    }
}

impl Filter {
    /// Create a new empty filter graph.
    pub fn empty() -> Self {
        Self {
            primitives: SmallVec::new(),
            output: FilterId(0),
            expansion_rect: Rect::ZERO,
        }
    }

    /// Create a filter from a single filter effect.
    ///
    /// Creates a simple filter graph with a single primitive.
    /// Use this for direct access to low-level SVG filter operations.
    pub fn single(primitive: FilterEffect) -> Self {
        let mut graph = Self::empty();
        let filter_id = graph.add(primitive, FilterInputs::NONE);
        graph.set_output(filter_id);
        graph
    }

    /// Create a filter from an iterator of filter effects.
    ///
    /// Creates a filter graph where the effects are applied in order.
    pub fn linear_list(primitives: impl Iterator<Item = FilterEffect>) -> Self {
        let mut graph = Self::empty();
        let mut last_id = None;
        for primitive in primitives {
            let inputs = FilterInputs {
                primary: last_id.map(FilterInput::Result),
                secondary: None,
            };
            let filter_id = graph.add(primitive, inputs);
            graph.set_output(filter_id);
            last_id = Some(filter_id);
        }
        graph
    }

    /// Add a filter primitive with optional inputs.
    ///
    /// Returns a `FilterId` that can be referenced by other primitives.
    /// Automatically updates the accumulated bounds expansion based on the primitive's requirements.
    pub fn add(&mut self, effect: FilterEffect, inputs: FilterInputs) -> FilterId {
        let id = FilterId(self.primitives.len() as u16);

        // Update accumulated expansion by taking the union of rects
        let primitive_rect = effect.expansion_rect();
        self.expansion_rect = self.expansion_rect.union(primitive_rect);

        self.primitives.push(FilterGraphNode { effect, inputs });

        id
    }

    /// The list of nodes in the graph
    pub fn nodes(&self) -> &[FilterGraphNode] {
        &self.primitives
    }

    /// The output filter for the graph.
    pub fn output(&self) -> FilterId {
        self.output
    }

    /// Set the output filter for the graph.
    fn set_output(&mut self, output: FilterId) {
        self.output = output;
    }

    /// Calculate the bounds expansion for this filter in pixel/device space.
    ///
    /// Returns a `Rect` representing how many extra pixels are needed around the
    /// filtered region to correctly compute the filter effect. For example, a blur
    /// filter needs to sample beyond the original bounds to avoid edge artifacts.
    ///
    /// The expansion accounts for the transform (rotation, scale, and shear) to compute
    /// the correct axis-aligned bounding box expansion in device space.
    ///
    /// The returned rect is centered at origin:
    /// - x0: negative left expansion (in pixels)
    /// - y0: negative top expansion (in pixels)
    /// - x1: positive right expansion (in pixels)
    /// - y1: positive bottom expansion (in pixels)
    ///
    /// # Arguments
    /// * `transform` - The transform applied to this filter layer
    pub fn linear_bounds_expansion(&self, transform: &Affine) -> Rect {
        let [a, b, c, d, _e, _f] = transform.as_coeffs();
        let linear_only = Affine::new([a, b, c, d, 0.0, 0.0]);

        self.bounds_expansion(&linear_only)
    }

    /// Get the accumulated bounds expansion for all primitives in this graph.
    ///
    /// This returns the expansion required by all primitives in the graph,
    /// representing the padding needed to render all filter effects correctly.
    ///
    /// The expansion accounts for the transform (rotation, scale, and shear) to compute
    /// the correct axis-aligned bounding box expansion in device space.
    ///
    /// # Arguments
    /// * `transform` - The transform applied to this filter layer
    pub fn bounds_expansion(&self, transform: &Affine) -> Rect {
        // Transform the cached expansion rect to device space
        // transform_rect_bbox computes the axis-aligned bounding box of the transformed rect
        transform.transform_rect_bbox(self.expansion_rect)
    }

    pub fn expansion_rect(&self) -> Rect {
        self.expansion_rect
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FilterGraphNode {
    pub effect: FilterEffect,
    pub inputs: FilterInputs,
}

/// Unique identifier for a filter primitive in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FilterId(pub u16);

/// Input connections for a filter primitive.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FilterInputs {
    /// Primary input ("in" attribute in SVG).
    pub primary: Option<FilterInput>,
    /// Secondary input ("in2" attribute in SVG, for composite/blend operations).
    pub secondary: Option<FilterInput>,
}

impl FilterInputs {
    pub const NONE: Self = Self {
        primary: None,
        secondary: None,
    };
}

impl FilterInputs {
    /// Create filter inputs with a single input.
    ///
    /// Use this for primitives that operate on a single source (blur, color matrix, etc.).
    pub fn single(input: FilterInput) -> Self {
        Self {
            primary: Some(input),
            secondary: None,
        }
    }

    /// Create filter inputs with two inputs (for composite, blend, etc.).
    ///
    /// Use this for primitives that combine two sources (composite, blend, displacement map, etc.).
    pub fn dual(input1: FilterInput, input2: FilterInput) -> Self {
        Self {
            primary: Some(input1),
            secondary: Some(input2),
        }
    }
}

/// A single filter input.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FilterInput {
    /// Input from a source (`SourceGraphic`, `SourceAlpha`, etc.).
    Source(FilterSource),
    /// Input from another filter's result.
    Result(FilterId),
}

/// Filter input sources.
///
/// Defines the various built-in sources that can be used as filter inputs,
/// matching the SVG filter primitive input types. These represent implicit
/// inputs available to any filter primitive without requiring previous operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FilterSource {
    /// The original graphic content being filtered.
    ///
    /// This is the default input - the rendered result of the element
    /// the filter is applied to, including all its fill, stroke, and content.
    SourceGraphic,
    /// Alpha channel only of the original graphic.
    ///
    /// Useful for creating effects based on shape/transparency, such as
    /// shadows that follow the element's outline.
    SourceAlpha,
    /// Background image content behind the filtered element.
    ///
    /// Allows filters to incorporate or blend with content behind the element.
    /// Not always available depending on the rendering context.
    BackgroundImage,
    /// Alpha channel only of the background image.
    ///
    /// The transparency mask of the background content.
    BackgroundAlpha,
    /// The fill paint of the element as an image input.
    ///
    /// For elements with gradient or pattern fills, this provides access
    /// to the fill as a filter input.
    FillPaint,
    /// The stroke paint of the element as an image input.
    ///
    /// For elements with gradient or pattern strokes, this provides access
    /// to the stroke as a filter input.
    StrokePaint,
}

/// Edge mode for filter operations.
///
/// Determines how to extend the input image when filter operations require sampling
/// beyond the original image boundaries. This is particularly important for blur and
/// convolution operations near edges.
///
/// See: <https://drafts.fxtf.org/filter-effects/#element-attrdef-filter-primitive-edgemode>
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EdgeMode {
    /// Extend by duplicating edge pixels (clamp to edge).
    ///
    /// The input image is extended along each border by replicating the color values
    /// at the given edge of the input image. This prevents dark halos around edges.
    Duplicate,
    /// Extend by wrapping to the opposite edge (repeat/tile).
    ///
    /// The input image is extended by taking color values from the opposite edge,
    /// creating a tiling effect.
    Wrap,
    /// Extend by mirroring across the edge.
    ///
    /// The input image is extended by taking color values mirrored across the edge.
    /// This creates seamless continuation at boundaries.
    Mirror,
    /// Extend with transparent black (zeros).
    ///
    /// The input image is extended with pixel values of zero for R, G, B and A.
    /// This is the default and most common mode, creating natural fade-to-transparent edges.
    #[default]
    None,
}

/// Low-level filter primitives for granular control (SVG filter primitives).
///
/// These are the building blocks for complex filter effects, corresponding to SVG
/// filter primitives. They can be combined in a `FilterGraph` to create sophisticated
/// visual effects.
///
/// See: <https://drafts.fxtf.org/filter-effects/#FilterPrimitivesOverview>
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FilterEffect {
    /// Generate a solid color fill.
    ///
    /// Creates a rectangle filled with the specified color, typically used as
    /// input to other filter operations (e.g., for colored shadows).
    Flood(AlphaColor<Srgb>),

    /// Gaussian blur filter.
    ///
    /// Applies a Gaussian blur using the specified standard deviation (σ).
    /// The effective blur range (distance over which pixels are sampled) is
    /// approximately 3 × `std_deviation`, as this captures ~99.7% of the
    /// Gaussian distribution.
    GaussianBlur(GaussianBlurFilter),

    /// Drop shadow effect (compound primitive).
    ///
    /// Creates a drop shadow by blurring the input's alpha channel, offsetting it,
    /// and compositing it with the original. This is a compound operation that
    /// combines multiple primitive operations into one.
    ///
    /// See: <https://drafts.fxtf.org/filter-effects-2/#feDropShadowElement>
    DropShadow(DropShadow),

    /// Matrix-based color transformation.
    ///
    /// Applies a 4x5 matrix transformation to colors, allowing arbitrary
    /// color space transformations, hue shifts, and color adjustments.
    ///
    /// 4x5 color transformation matrix: 4 rows (R,G,B,A) × 5 columns (R,G,B,A,offset).
    /// Each output channel is computed as a linear combination of input channels plus offset.
    ColorMatrix(ColorMatrix),

    /// Geometric offset/translation.
    ///
    /// Shifts the input image by the specified offset. Useful for creating
    /// shadow effects or positioning elements in a filter graph.
    ///
    /// Positive values shift right or down.
    Offset(Vec2),

    /// Composite two inputs using Porter-Duff compositing operations.
    ///
    /// Combines two input images using standard compositing operators
    /// (over, in, out, atop, xor) or custom arithmetic combination.
    Composite(CompositeOperator),

    /// Blend two inputs using blend modes.
    ///
    /// Combines two input images using Photoshop-style blend modes
    /// (multiply, screen, overlay, etc.).
    Blend(BlendMode),

    /// Morphological operations (dilate/erode).
    ///
    /// Expands (dilate) or contracts (erode) the shapes in the input image.
    /// Useful for creating outline effects or cleaning up edges.
    Morphology(MorphologyFilter),
    /// Custom convolution kernel for image processing.
    ///
    /// Applies a custom convolution matrix to the input image, enabling
    /// effects like sharpening, edge detection, embossing, and custom filters.
    ConvolveMatrix(ConvolutionKernel),

    /// Generate Perlin noise/turbulence patterns.
    ///
    /// Creates procedural noise patterns useful for textures, clouds,
    /// marble effects, and other organic-looking randomness.
    Turbulence(TurbulenceFilter),

    /// Displace pixels using a displacement map.
    ///
    /// Uses the color values from a second input to spatially displace pixels
    /// in the primary input, creating warping and distortion effects.
    DisplacementMap(DisplacementMapFilter),

    /// Per-channel component transfer using lookup tables or functions.
    ///
    /// Applies independent transfer functions to each color channel,
    /// enabling color corrections, gamma adjustments, and custom mappings.
    ComponentTransfer(ComponentTransferFilter),

    Image(ExternalImageSource),

    /// Tile the input to fill the filter region.
    ///
    /// Repeats the input image to fill the entire filter primitive subregion,
    /// creating a tiling/repeating pattern.
    Tile,

    /// Diffuse lighting simulation.
    ///
    /// Creates a lighting effect by treating the input's alpha channel as a height map
    /// and calculating diffuse (matte) reflection from a light source.
    DiffuseLighting(DiffuseLightingFilter),

    /// Specular lighting simulation.
    ///
    /// Creates a lighting effect by treating the input's alpha channel as a height map
    /// and calculating specular (shiny) reflection highlights from a light source.
    SpecularLighting(SpecularLightingFilter),
}

// Assert size of FilterEffect.
// This is just for documentation purposes. Feel free to update the value as necessary
#[cfg(not(target_arch = "wasm32"))]
const _: [u8; 128] = [0; std::mem::size_of::<FilterEffect>()];
#[cfg(target_arch = "wasm32")]
const _: [u8; 88] = [0; std::mem::size_of::<FilterEffect>()];

impl FilterEffect {
    /// Gaussian blur effect.
    ///
    /// Applies a Gaussian blur to the input image. Larger radius values
    /// produce more blur. The blur is applied equally in all directions.
    pub fn blur(radius: f32) -> Self {
        Self::GaussianBlur(GaussianBlurFilter {
            std_deviation: radius,
            edge_mode: EdgeMode::None,
        })
    }

    /// Drop shadow effect (compound primitive).
    ///
    /// Creates a drop shadow by blurring the input's alpha channel, offsetting it,
    /// and compositing it with the original. This is a compound operation that
    /// combines multiple primitive operations into one.
    ///
    /// See: <https://drafts.fxtf.org/filter-effects-2/#feDropShadowElement>
    pub fn drop_shadow(dx: f32, dy: f32, std_deviation: f32, color: AlphaColor<Srgb>) -> Self {
        Self::DropShadow(DropShadow {
            dx,
            dy,
            std_deviation,
            color,
            edge_mode: EdgeMode::None,
        })
    }

    /// Construct a CSS opacity() filter effect
    pub fn opacity(amount: f32) -> Self {
        Self::ComponentTransfer(ComponentTransferFilter::opacity(amount))
    }

    /// Construct a CSS invert() filter effect
    pub fn invert(amount: f32) -> Self {
        Self::ComponentTransfer(ComponentTransferFilter::invert(amount))
    }

    /// Construct a CSS brightness() filter effect
    pub fn brightness(amount: f32) -> Self {
        Self::ComponentTransfer(ComponentTransferFilter::brightness(amount))
    }

    /// Construct a CSS contrast() filter effect
    pub fn contrast(amount: f32) -> Self {
        Self::ComponentTransfer(ComponentTransferFilter::contrast(amount))
    }

    /// Construct a CSS hue-rotate() filter effect
    pub fn hue_rotate(angle_radians: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::hue_rotate(angle_radians))
    }

    /// Construct a CSS saturate() filter effect
    pub fn saturate(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::saturate(amount))
    }

    /// Construct a CSS sepia() filter effect
    pub fn sepia(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::sepia(amount))
    }

    /// Construct a CSS grayscale() filter effect
    pub fn grayscale(amount: f32) -> Self {
        Self::ColorMatrix(ColorMatrix::grayscale(amount))
    }

    /// Calculate the bounds expansion as a `Rect` in user space.
    ///
    /// Returns a rectangle centered at the origin representing how much the filter
    /// expands the processing region in each direction. The rect coordinates are:
    /// - x0: negative left expansion
    /// - y0: negative top expansion
    /// - x1: positive right expansion
    /// - y1: positive bottom expansion
    ///
    /// A `Rect::ZERO` means no expansion. This representation allows the expansion
    /// to be correctly transformed (including rotation) using standard rect transforms.
    ///
    /// For example, a blur filter needs additional pixels around the edges (3*sigma).
    /// Most filters that don't sample neighboring pixels return `Rect::ZERO`.
    pub fn expansion_rect(&self) -> Rect {
        match self {
            Self::GaussianBlur(blur) => {
                // Gaussian blur expands uniformly by 3*sigma (covers 99.7% of distribution)
                let radius = (blur.std_deviation * 3.0) as f64;
                Rect::new(-radius, -radius, radius, radius)
            }
            Self::Offset(offset) => {
                // Offset shifts pixels; expand bounds asymmetrically so shifted content isn't cut.
                let dx = offset.x;
                let dy = offset.y;
                Rect::new(dx.min(0.0), dy.min(0.0), dx.max(0.0), dy.max(0.0))
            }
            Self::DropShadow(DropShadow {
                std_deviation,
                dx,
                dy,
                ..
            }) => {
                // Drop shadow = blur + offset + composite with original
                // The expansion rect encompasses both the blur and the offset
                let blur_radius = (*std_deviation * 3.0) as f64;
                let dx = *dx as f64;
                let dy = *dy as f64;

                Rect::new(
                    -(blur_radius + (-dx).max(0.0)),
                    -(blur_radius + (-dy).max(0.0)),
                    blur_radius + dx.max(0.0),
                    blur_radius + dy.max(0.0),
                )
            }
            // Most other filters don't expand bounds
            _ => Rect::ZERO,
        }
    }
}

#[cfg(test)]
mod offset_expansion_tests {
    use super::FilterEffect;
    use kurbo::{Rect, Vec2};

    #[test]
    fn offset_expands_in_direction_of_shift() {
        let p = FilterEffect::Offset(Vec2 { x: 2.5, y: -3.0 });
        assert_eq!(
            p.expansion_rect(),
            Rect::new(0.0, -3.0, 2.5, 0.0),
            "Offset expansion should be asymmetric and include the shift vector"
        );
    }
}

/// Blend modes for combining colors.
///
/// These are blend modes that define how to combine the colors
/// of two layers. Unlike compositing operators which deal with alpha, blend modes
/// focus on color mixing while preserving the compositing behavior.
///
/// See: <https://drafts.fxtf.org/compositing/#blending>
pub type BlendMode = peniko::Mix;

pub mod composite {
    /// Composite operators for combining filter inputs.
    ///
    /// These are the Porter-Duff compositing operators used to combine two images.
    /// Each operator defines how the source (input 1) and destination (input 2)
    /// are combined based on their color and alpha values.
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum CompositeOperator {
        /// Source over destination (standard alpha blending).
        ///
        /// The source is composited over the destination. This is the most common
        /// blending mode where source alpha determines visibility.
        Over,
        /// Source in destination (intersection).
        ///
        /// The source is only visible where the destination is opaque.
        /// Result alpha = `source_alpha` × `dest_alpha`.
        In,
        /// Source out destination (subtract).
        ///
        /// The source is only visible where the destination is transparent.
        /// Useful for masking/cutting out regions.
        Out,
        /// Source atop destination.
        ///
        /// Source is composited over destination, but only where destination is opaque.
        Atop,
        /// Source XOR destination (exclusive or).
        ///
        /// Shows source where destination is transparent and vice versa,
        /// but not where both are opaque.
        Xor,

        Arithmetic(ArithmeticCompositeOperator),
    }

    /// Arithmetic combination with custom coefficients.
    ///
    /// Custom linear combination: result = k1*src*dst + k2*src + k3*dst + k4.
    /// Allows creating custom compositing operations beyond the standard Porter-Duff set.
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ArithmeticCompositeOperator {
        pub k1: f32,
        pub k2: f32,
        pub k3: f32,
        pub k4: f32,
    }
}

mod blur {
    use crate::filters::EdgeMode;

    /// Gaussian blur filter.
    ///
    /// Applies a Gaussian blur using the specified standard deviation (σ).
    /// The effective blur range (distance over which pixels are sampled) is
    /// approximately 3 × `std_deviation`, as this captures ~99.7% of the
    /// Gaussian distribution.
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct GaussianBlurFilter {
        /// Standard deviation for the blur kernel. Larger values create more blur.
        /// Must be non-negative. A value of 0 means no blur.
        ///
        /// This directly corresponds to the σ (sigma) parameter in the Gaussian
        /// function. The visible blur effect extends approximately 3σ in each direction.
        ///
        /// TODO: Per the W3C specification, this should support separate x and y values.
        /// The spec allows `stdDeviation` to be either one number (applied to both axes)
        /// or two numbers (first for x-axis, second for y-axis). Currently only uniform
        /// blur is supported. Consider changing to `(f32, f32)` or a dedicated type.
        pub std_deviation: f32,
        /// Edge mode determining how pixels beyond the input bounds are handled.
        pub edge_mode: EdgeMode,
    }
}

pub mod shadow {
    use super::EdgeMode;
    use peniko::color::{AlphaColor, Srgb};

    /// Drop shadow effect (compound primitive).
    ///
    /// Creates a drop shadow by blurring the input's alpha channel, offsetting it,
    /// and compositing it with the original. This is a compound operation that
    /// combines multiple primitive operations into one.
    ///
    /// See: <https://drafts.fxtf.org/filter-effects-2/#feDropShadowElement>
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct DropShadow {
        pub dx: f32,
        pub dy: f32,
        pub std_deviation: f32,
        pub color: AlphaColor<Srgb>,
        pub edge_mode: EdgeMode,
    }
}

/// Reference an external image as filter input.
///
/// Allows using pre-existing images (from an atlas or resource) as
/// input to filter operations, useful for texturing and overlays.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExternalImageSource {
    pub image_id: u32,
    pub transform: Option<[f32; 6]>,
}

pub mod morphology {

    /// Morphological operations (dilate/erode).
    ///
    /// Expands (dilate) or contracts (erode) the shapes in the input image.
    /// Useful for creating outline effects or cleaning up edges.
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct MorphologyFilter {
        /// Morphological operator determining whether to erode or dilate.
        pub operator: MorphologyOperator,
        /// Operation radius in pixels. Larger values create stronger effects.
        pub radius: f32,
    }

    /// Morphological operators for dilate/erode operations.
    ///
    /// These operators modify the shape of objects by expanding or contracting them.
    /// They work by examining neighborhoods of pixels and applying min/max operations.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum MorphologyOperator {
        /// Erode operation (shrink/thin shapes).
        ///
        /// Makes objects smaller by removing pixels at the edges. Takes the minimum
        /// value in the neighborhood. Useful for removing noise or separating touching objects.
        Erode,
        /// Dilate operation (expand/thicken shapes).
        ///
        /// Makes objects larger by adding pixels at the edges. Takes the maximum
        /// value in the neighborhood. Useful for filling holes or connecting nearby objects.
        Dilate,
    }
}

pub mod turbulence {
    /// Generate Perlin noise/turbulence patterns.
    ///
    /// Creates procedural noise patterns useful for textures, clouds,
    /// marble effects, and other organic-looking randomness.
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct TurbulenceFilter {
        /// Base frequency for noise generation. Higher values create finer detail.
        pub base_frequency: f32,
        /// Number of octaves for fractal noise. More octaves add finer detail.
        pub num_octaves: u32,
        /// Random seed for reproducible noise generation.
        pub seed: u32,
        /// Type of noise: smooth fractal or more chaotic turbulence.
        pub turbulence_type: TurbulenceType,
    }

    /// Types of turbulence noise generation.
    ///
    /// Determines the algorithm used for generating procedural noise patterns.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum TurbulenceType {
        /// Fractal noise (smooth, natural-looking Perlin noise).
        ///
        /// Creates smooth, continuous patterns suitable for natural textures
        /// like clouds, marble, wood grain, or terrain.
        FractalNoise,
        /// Turbulence noise (more chaotic and energetic).
        ///
        /// Creates more chaotic patterns with sharper transitions,
        /// suitable for fire, smoke, or turbulent effects.
        Turbulence,
    }
}

pub mod displacement {

    /// Displace pixels using a displacement map.
    ///
    /// Uses the color values from a second input to spatially displace pixels
    /// in the primary input, creating warping and distortion effects.
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct DisplacementMapFilter {
        /// Scale factor controlling the displacement intensity.
        pub scale: f32,
        /// Color channel from the displacement map used for X-axis displacement.
        pub x_channel: ColorChannel,
        /// Color channel from the displacement map used for Y-axis displacement.
        pub y_channel: ColorChannel,
    }

    /// Color channels for displacement mapping and channel selection.
    ///
    /// Specifies which color channel to use for operations that need to
    /// extract or reference individual channels from an image.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum ColorChannel {
        /// Red color channel (R component).
        Red,
        /// Green color channel (G component).
        Green,
        /// Blue color channel (B component).
        Blue,
        /// Alpha channel (transparency/opacity).
        Alpha,
    }
}

pub mod component_transfer {
    use smallvec::SmallVec;

    /// Per-channel component transfer using lookup tables or functions.
    ///
    /// Applies independent transfer functions to each color channel,
    /// enabling color corrections, gamma adjustments, and custom mappings.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ComponentTransferFilter {
        /// Transfer function applied to the red channel (None = identity).
        pub red_function: TransferFunction,
        /// Transfer function applied to the green channel (None = identity).
        pub green_function: TransferFunction,
        /// Transfer function applied to the blue channel (None = identity).
        pub blue_function: TransferFunction,
        /// Transfer function applied to the alpha channel (None = identity).
        pub alpha_function: TransferFunction,
    }

    impl ComponentTransferFilter {
        /// Component transfer filter for the CSS opacity() filter
        pub fn opacity(amount: f32) -> Self {
            let func = TransferFunction::Table(SmallVec::from([0.0, amount]));
            Self {
                red_function: TransferFunction::Identity,
                green_function: TransferFunction::Identity,
                blue_function: TransferFunction::Identity,
                alpha_function: func,
            }
        }

        /// Component transfer filter for the CSS invert() filter
        pub fn invert(amount: f32) -> Self {
            let func = TransferFunction::Table(SmallVec::from([amount, 1.0 - amount]));
            Self {
                red_function: func.clone(),
                green_function: func.clone(),
                blue_function: func.clone(),
                alpha_function: TransferFunction::Identity,
            }
        }

        /// Component transfer filter for the CSS brightness() filter
        pub fn brightness(amount: f32) -> Self {
            let func = TransferFunction::Linear(LinearTransferFunction {
                slope: amount,
                intercept: 0.0,
            });
            Self {
                red_function: func.clone(),
                green_function: func.clone(),
                blue_function: func.clone(),
                alpha_function: TransferFunction::Identity,
            }
        }

        /// Component transfer filter for the CSS contrast() filter
        pub fn contrast(amount: f32) -> Self {
            let func = TransferFunction::Linear(LinearTransferFunction {
                slope: amount,
                intercept: -(0.5 * amount) + 0.5,
            });
            Self {
                red_function: func.clone(),
                green_function: func.clone(),
                blue_function: func.clone(),
                alpha_function: TransferFunction::Identity,
            }
        }
    }

    /// Transfer functions for component transfer operations.
    ///
    /// These functions map input color channel values to output values,
    /// enabling gamma correction, color grading, and custom color curves.
    /// Input and output values are typically in the range [0, 1].
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum TransferFunction {
        /// Identity function (output = input, no change).
        Identity,

        /// Table lookup with linear interpolation.
        ///
        /// Maps input values using a lookup table with linear interpolation between entries.
        /// Input 0.0 maps to values\[0\], 1.0 maps to values\[n-1\], intermediate values interpolate.
        ///
        /// Lookup table values defining the transfer curve.
        /// More values provide smoother curves. Minimum 2 values required.
        Table(SmallVec<[f32; 2]>),

        /// Discrete step function (posterization).
        ///
        /// Maps input to discrete output values without interpolation, creating step/banding effects.
        /// Each segment gets a constant output value from the table.
        ///
        /// Step values for each discrete output level.
        /// Input range is divided into len(values) segments, each mapping to one value.
        Discrete(Vec<f32>),

        /// Linear function: output = slope × input + intercept.
        Linear(LinearTransferFunction),

        // Gamma correction: output = amplitude × input^exponent + offset.
        Gamma(GammaTransferFunction),
    }

    /// Linear function: output = slope × input + intercept.
    ///
    /// Simple linear transformation of the input value.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct LinearTransferFunction {
        pub slope: f32,
        pub intercept: f32,
    }

    /// Gamma correction: output = amplitude × input^exponent + offset.
    ///
    /// Applies power-law transformation, commonly used for gamma correction and
    /// adjusting midtone brightness without affecting blacks or whites.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct GammaTransferFunction {
        pub amplitude: f32,
        pub exponent: f32,
        pub offset: f32,
    }
}

/// Common color transformation matrices.
///
/// These 4x5 matrices are used with the `ColorMatrix` filter primitive.
/// Each row transforms a color channel: [R, G, B, A, offset].
pub mod color_transformation {

    const LUMA_R: f32 = 0.213;
    const LUMA_G: f32 = 0.715;
    const LUMA_B: f32 = 0.072;

    /// Matrix-based color transformation.
    ///
    /// 4x5 color transformation matrix: 4 rows (R,G,B,A) × 5 columns (R,G,B,A,offset).
    /// Each output channel is computed as a linear combination of input channels plus offset.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ColorMatrix(pub [f32; 20]);

    impl ColorMatrix {
        /// Color matrix filter for the CSS hue-rotate() filter
        pub fn hue_rotate(angle_radians: f32) -> Self {
            let sin = angle_radians.sin();
            let cos = angle_radians.cos();

            Self([
                LUMA_R + cos * (1.0 - LUMA_R) - sin * LUMA_R,
                LUMA_G - cos * LUMA_G - sin * LUMA_G,
                LUMA_B - cos * LUMA_B + sin * (1.0 - LUMA_B),
                0.0,
                0.0,
                LUMA_R - cos * LUMA_R + sin * 0.143,
                LUMA_G + cos * (1.0 - LUMA_G) + sin * 0.140,
                LUMA_B - cos * LUMA_B - sin * 0.283,
                0.0,
                0.0,
                LUMA_R - cos * LUMA_R - sin * (1.0 - LUMA_R),
                LUMA_G - cos * LUMA_G + sin * LUMA_G,
                LUMA_B + cos * (1.0 - LUMA_B) + sin * LUMA_B,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ])
        }

        /// Color matrix filter for the CSS saturate() filter
        pub fn saturate(amount: f32) -> Self {
            Self([
                LUMA_R + amount * (1.0 - LUMA_R),
                LUMA_G - amount * LUMA_G,
                LUMA_B - amount * LUMA_B,
                0.0,
                0.0,
                LUMA_R - amount * LUMA_R,
                LUMA_G + amount * (1.0 - LUMA_G),
                LUMA_B - amount * LUMA_B,
                0.0,
                0.0,
                LUMA_R - amount * LUMA_R,
                LUMA_G - amount * LUMA_G,
                LUMA_B + amount * (1.0 - LUMA_B),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ])
        }

        /// Color matrix filter for the CSS sepia() filter
        /// <https://www.w3.org/TR/filter-effects-1/#sepiaEquivalent>
        pub fn sepia(amount: f32) -> Self {
            Self([
                (0.393 + 0.607 * (1.0 - amount)),
                (0.769 - 0.769 * (1.0 - amount)),
                (0.189 - 0.189 * (1.0 - amount)),
                0.0,
                0.0,
                (0.349 - 0.349 * (1.0 - amount)),
                (0.686 + 0.314 * (1.0 - amount)),
                (0.168 - 0.168 * (1.0 - amount)),
                0.0,
                0.0,
                (0.272 - 0.272 * (1.0 - amount)),
                (0.534 - 0.534 * (1.0 - amount)),
                (0.131 + 0.869 * (1.0 - amount)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ])
        }

        /// Color matrix filter for the CSS grayscale() filter
        /// <https://www.w3.org/TR/filter-effects-1/#grayscaleEquivalent>
        pub fn grayscale(amount: f32) -> Self {
            Self([
                (0.2126 + 0.7874 * (1.0 - amount)),
                (0.7152 - 0.7152 * (1.0 - amount)),
                (0.0722 - 0.0722 * (1.0 - amount)),
                0.0,
                0.0,
                (0.2126 - 0.2126 * (1.0 - amount)),
                (0.7152 + 0.2848 * (1.0 - amount)),
                (0.0722 - 0.0722 * (1.0 - amount)),
                0.0,
                0.0,
                (0.2126 - 0.2126 * (1.0 - amount)),
                (0.7152 - 0.7152 * (1.0 - amount)),
                (0.0722 + 0.9278 * (1.0 - amount)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ])
        }

        /// Identity matrix (no change).
        pub const IDENTITY: Self = Self([
            1.0, 0.0, 0.0, 0.0, 0.0, // Red
            0.0, 1.0, 0.0, 0.0, 0.0, // Green
            0.0, 0.0, 1.0, 0.0, 0.0, // Blue
            0.0, 0.0, 0.0, 1.0, 0.0, // Alpha
        ]);

        /// Extract alpha channel to RGB (for shadow effects).
        pub const ALPHA_TO_BLACK: Self = Self([
            0.0, 0.0, 0.0, 1.0, 0.0, // Red = Alpha
            0.0, 0.0, 0.0, 1.0, 0.0, // Green = Alpha
            0.0, 0.0, 0.0, 1.0, 0.0, // Blue = Alpha
            0.0, 0.0, 0.0, 1.0, 0.0, // Alpha = Alpha
        ]);
    }
}

pub mod lighting {

    /// Diffuse lighting simulation.
    ///
    /// Creates a lighting effect by treating the input's alpha channel as a height map
    /// and calculating diffuse (matte) reflection from a light source.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct DiffuseLightingFilter {
        /// Surface scale factor for converting alpha values to heights.
        pub surface_scale: f32,
        /// Diffuse reflection constant (kd). Controls lighting intensity.
        pub diffuse_constant: f32,
        /// Kernel unit length for gradient calculations in user space.
        pub kernel_unit_length: f32,
        /// Configuration of the light source (point, distant, or spot).
        pub light_source: LightSource,
    }

    /// Specular lighting simulation.
    ///
    /// Creates a lighting effect by treating the input's alpha channel as a height map
    /// and calculating specular (shiny) reflection highlights from a light source.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct SpecularLightingFilter {
        /// Surface scale factor for converting alpha values to heights.
        pub surface_scale: f32,
        /// Specular reflection constant (ks). Controls highlight intensity.
        pub specular_constant: f32,
        /// Specular reflection exponent. Controls highlight sharpness (higher = sharper).
        pub specular_exponent: f32,
        /// Kernel unit length for gradient calculations in user space.
        pub kernel_unit_length: f32,
        /// Configuration of the light source (point, distant, or spot).
        pub light_source: LightSource,
    }

    /// Light source configurations for lighting effects.
    ///
    /// Defines different types of light sources used in diffuse and specular lighting
    /// filter primitives. Each type has different characteristics and use cases.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub enum LightSource {
        /// Distant light source (infinitely far away, like the sun).
        Distant(DistantLightSource),
        /// Point light source at a specific 3D position.
        Point(PointLightSource),
        /// Spot light with position, direction, and cone angle.
        Spot(SpotLightSource),
    }

    /// Distant light source (infinitely far away, like the sun).
    ///
    /// All rays are parallel, creating uniform lighting across the surface.
    /// Direction is specified using spherical coordinates (azimuth and elevation).
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct DistantLightSource {
        pub azimuth: f32,
        pub elevation: f32,
    }

    /// Point light source at a specific 3D position.
    ///
    /// Light radiates uniformly in all directions from a single point.
    /// Intensity decreases with distance. Like a light bulb.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct PointLightSource {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    /// Spot light with position, direction, and cone angle.
    ///
    /// Light emanates from a point in a specific direction with limited spread.
    /// Like a flashlight or stage spotlight with adjustable focus.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct SpotLightSource {
        pub x: f32,
        pub y: f32,
        pub z: f32,
        pub points_at_x: f32,
        pub points_at_y: f32,
        pub points_at_z: f32,
        pub specular_exponent: f32,
        pub limiting_cone_angle: Option<f32>,
    }
}

/// Common convolution kernels.
///
/// These kernels are used with the `ConvolveMatrix` filter primitive
/// for various image processing effects. All provided kernels are 3x3.
pub mod convolution {

    /// Convolution kernel for custom filtering operations.
    ///
    /// Defines a square matrix of weights used for convolution-based image processing.
    /// The kernel is applied to each pixel by multiplying surrounding pixels by the weights,
    /// summing the results, dividing by the divisor, and adding the bias.
    #[derive(Debug, Clone, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ConvolutionKernel {
        /// Kernel size (e.g., 3 for a 3×3 kernel, 5 for 5×5).
        /// The kernel must be square, so this defines both width and height.
        pub size: u32,
        /// Kernel weight values in row-major order.
        /// Length must equal size × size. Center of kernel is typically at (size/2, size/2).
        pub values: Vec<f32>,
        /// Normalization divisor applied to the convolution result.
        /// Common practice is to use the sum of all weights for averaging, or 1.0 otherwise.
        pub divisor: f32,
        /// Bias value added to the result after normalization.
        /// Useful for edge detection or emboss effects to shift the result range.
        pub bias: f32,
        /// Whether to preserve the alpha channel unchanged.
        /// If true, convolution only applies to RGB; if false, it applies to RGBA.
        pub preserve_alpha: bool,
    }

    /// 3x3 Gaussian blur kernel for basic smoothing.
    pub fn gaussian_3x3() -> ConvolutionKernel {
        ConvolutionKernel {
            size: 3,
            values: vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
            divisor: 16.0,
            bias: 0.0,
            preserve_alpha: false,
        }
    }

    /// 3x3 Sharpen kernel to enhance edges and details.
    pub fn sharpen_3x3() -> ConvolutionKernel {
        ConvolutionKernel {
            size: 3,
            values: vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
            divisor: 1.0,
            bias: 0.0,
            preserve_alpha: true,
        }
    }

    /// 3x3 Edge detection kernel (Laplacian operator).
    pub fn edge_detect_3x3() -> ConvolutionKernel {
        ConvolutionKernel {
            size: 3,
            values: vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
            divisor: 1.0,
            bias: 0.0,
            preserve_alpha: true,
        }
    }

    /// 3x3 Emboss kernel for creating a raised/beveled appearance.
    pub fn emboss_3x3() -> ConvolutionKernel {
        ConvolutionKernel {
            size: 3,
            values: vec![-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0],
            divisor: 1.0,
            bias: 0.5,
            preserve_alpha: true,
        }
    }
}
