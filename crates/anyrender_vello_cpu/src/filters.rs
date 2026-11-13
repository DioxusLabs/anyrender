use std::sync::Arc;

use anyrender::{
    Filter,
    filters::{EdgeMode, FilterEffect, FilterInput, FilterSource},
};
use vello_common::filter_effects::FilterPrimitive;

pub(crate) struct ConvertedFilter {
    pub(crate) filter: vello_common::filter_effects::Filter,
    /// Whether the filter is a flood, whose output covers the whole layer.
    pub(crate) floods: bool,
}

pub(crate) fn convert_filter(filter: Arc<Filter>) -> Option<ConvertedFilter> {
    // Vello CPU only supports single-node filters at the moment, so fold
    // away the primitives that do not change the result first.
    let filter = filter.without_noops();
    // A graph that doesn't fold to one primitive applied to the source
    // graphic can't be run, so the content is drawn unfiltered.
    let [node] = filter.nodes() else {
        return None;
    };
    if !matches!(
        node.inputs.primary,
        None | Some(FilterInput::Source(FilterSource::SourceGraphic))
    ) {
        return None;
    }
    let primitive = convert_filter_effect(&node.effect)?;
    Some(ConvertedFilter {
        floods: matches!(primitive, FilterPrimitive::Flood { .. }),
        filter: vello_common::filter_effects::Filter::from_primitive(primitive),
    })
}

pub(crate) fn convert_filter_effect(effect: &FilterEffect) -> Option<FilterPrimitive> {
    Some(match effect {
        FilterEffect::Flood(color) => FilterPrimitive::Flood { color: *color },
        FilterEffect::GaussianBlur(blur) => FilterPrimitive::GaussianBlur {
            std_deviation_x: blur.std_deviation_x,
            std_deviation_y: blur.std_deviation_y,
            edge_mode: convert_edge_mode(blur.edge_mode),
        },
        // Vello has one standard deviation for both axes.
        FilterEffect::DropShadow(shadow) if shadow.std_deviation_x != shadow.std_deviation_y => {
            return None;
        }
        FilterEffect::DropShadow(shadow) => FilterPrimitive::DropShadow {
            dx: shadow.dx,
            dy: shadow.dy,
            std_deviation: shadow.std_deviation_x,
            color: shadow.color,
            edge_mode: convert_edge_mode(shadow.edge_mode),
        },
        FilterEffect::Offset(offset) => FilterPrimitive::Offset {
            dx: offset.x as f32,
            dy: offset.y as f32,
        },
        FilterEffect::ColorMatrix(_matrix) => return None,
        FilterEffect::ComponentTransfer(_component_transfer_filter) => return None,
        FilterEffect::Blend(_mode) => return None,
        FilterEffect::Composite(_composite_operator) => return None,
        FilterEffect::Morphology(_morphology_filter) => return None,
        FilterEffect::ConvolveMatrix(_convolution_kernel) => return None,
        FilterEffect::Turbulence(_turbulence_filter) => return None,
        FilterEffect::DisplacementMap(_displacement_map_filter) => return None,
        FilterEffect::Image(_external_image_source) => return None,
        FilterEffect::Tile => return None,
        FilterEffect::DiffuseLighting(_diffuse_lighting_filter) => return None,
        FilterEffect::SpecularLighting(_specular_lighting_filter) => return None,
    })
}

fn convert_edge_mode(edge_mode: EdgeMode) -> vello_common::filter_effects::EdgeMode {
    match edge_mode {
        EdgeMode::Duplicate => vello_common::filter_effects::EdgeMode::Duplicate,
        EdgeMode::Wrap => vello_common::filter_effects::EdgeMode::Wrap,
        EdgeMode::Mirror => vello_common::filter_effects::EdgeMode::Mirror,
        EdgeMode::None => vello_common::filter_effects::EdgeMode::None,
    }
}
