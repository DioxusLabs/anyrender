//! Filter effects API based on the W3C Filter Effects specification.
//!
//! The types are defined in [`image_filters_types`] and re-exported here, with anyrender's
//! [`FilterImage`] as the content of `feImage` primitives.
//!
//! See: <https://drafts.fxtf.org/filter-effects/>

pub use image_filters_types::*;

/// A directed acyclic graph (DAG) of filter operations. See [`image_filters_types::Filter`].
pub type Filter<Image = FilterImage> = image_filters_types::Filter<Image>;

/// A filter primitive in a [`Filter`]. See [`image_filters_types::FilterGraphNode`].
pub type FilterGraphNode<Image = FilterImage> = image_filters_types::FilterGraphNode<Image>;

/// A filter primitive. See [`image_filters_types::FilterEffect`].
pub type FilterEffect<Image = FilterImage> = image_filters_types::FilterEffect<Image>;

// Assert size of FilterEffect.
// This is just for documentation purposes. Feel free to update the value as necessary
#[cfg(target_pointer_width = "64")]
const _: [u8; 128] = [0; std::mem::size_of::<FilterEffect>()];
// On 32-bit x86, 64-bit fields only have 4-byte alignment.
#[cfg(all(target_pointer_width = "32", target_arch = "x86"))]
const _: [u8; 84] = [0; std::mem::size_of::<FilterEffect>()];
#[cfg(all(target_pointer_width = "32", not(target_arch = "x86")))]
const _: [u8; 88] = [0; std::mem::size_of::<FilterEffect>()];

/// Content rendered by a [`FilterEffect::Image`] primitive (`feImage` in SVG).
#[derive(Debug, Clone, PartialEq)]
pub struct FilterImage {
    /// Drawn in the filter's user space, the space of [`FilterGraphNode::subregion`].
    pub scene: crate::recording::Scene,
}
