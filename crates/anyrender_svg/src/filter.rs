// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conversion of [`usvg::filter::Filter`] chains into an [`anyrender::Filter`] graph.
//!
//! The resulting graph is attached to a layer via [`anyrender::PaintScene::push_layer`].
//! Whether (and how faithfully) the filter is actually applied depends on the backend.
//!
//! Known approximations (accepted silently):
//!
//! - Filter primitive subregions (`x`/`y`/`width`/`height` on a primitive) are ignored.
//!   The overall filter region *is* honoured (it is used as the layer clip).
//! - `color-interpolation-filters` is ignored (backends pick a color space).
//! - `feConvolveMatrix` `targetX`/`targetY`, `edgeMode` and `kernelUnitLength` are ignored.
//! - `feTurbulence` `stitchTiles` is ignored.
//!
//! Filters that cannot be expressed at all (see [`convert_primitive`]) cause the whole
//! conversion to fail, in which case the caller falls back to the error handler and
//! renders the group unfiltered.

use anyrender::filters::{
    Filter, FilterEffect, FilterId, FilterInput, FilterInputs, FilterSource,
    color_transformation::ColorMatrix,
    component_transfer::{
        ComponentTransferFilter, GammaTransferFunction, LinearTransferFunction, TransferFunction,
    },
    composite::{ArithmeticCompositeOperator, CompositeOperator},
    convolution::ConvolutionKernel,
    displacement::{ColorChannel, DisplacementMapFilter},
    lighting::{
        DiffuseLightingFilter, DistantLightSource, LightSource, PointLightSource,
        SpecularLightingFilter, SpotLightSource,
    },
    morphology::{MorphologyFilter, MorphologyOperator},
    turbulence::{TurbulenceFilter, TurbulenceType},
};
use kurbo::Vec2;
use peniko::Color;
use std::collections::HashMap;
use std::sync::Arc;

use crate::util;

/// `SourceAlpha`: RGB channels zeroed, alpha preserved.
const SOURCE_ALPHA_MATRIX: ColorMatrix = ColorMatrix([
    0.0, 0.0, 0.0, 0.0, 0.0, // Red = 0
    0.0, 0.0, 0.0, 0.0, 0.0, // Green = 0
    0.0, 0.0, 0.0, 0.0, 0.0, // Blue = 0
    0.0, 0.0, 0.0, 1.0, 0.0, // Alpha = Alpha
]);

/// `luminanceToAlpha`: alpha computed from RGB luminance, RGB zeroed.
///
/// Coefficients per <https://drafts.fxtf.org/filter-effects/#elementdef-fecolormatrix>.
const LUMINANCE_TO_ALPHA_MATRIX: ColorMatrix = ColorMatrix([
    0.0, 0.0, 0.0, 0.0, 0.0, // Red = 0
    0.0, 0.0, 0.0, 0.0, 0.0, // Green = 0
    0.0, 0.0, 0.0, 0.0, 0.0, // Blue = 0
    0.2125, 0.7154, 0.0721, 0.0, 0.0, // Alpha = luminance(RGB)
]);

/// Convert a chain of usvg filters (an element may have multiple filters applied
/// in sequence) into a single [`anyrender::Filter`] graph.
///
/// Returns `None` if any primitive cannot be expressed with [`FilterEffect`].
pub(crate) fn to_anyrender_filter(filters: &[Arc<usvg::filter::Filter>]) -> Option<Filter> {
    let mut graph = Filter::empty();
    // Output of the previous filter in the chain. It acts as `SourceGraphic`
    // for the next filter.
    let mut chain_input: Option<FilterId> = None;
    for filter in filters {
        chain_input = Some(convert_filter(&mut graph, filter, chain_input)?);
    }
    let output = chain_input?;
    graph.set_output(output);
    Some(graph)
}

/// Convert a single usvg filter, appending its primitives to `graph`.
///
/// Returns the [`FilterId`] of the filter's result.
fn convert_filter(
    graph: &mut Filter,
    filter: &usvg::filter::Filter,
    chain_input: Option<FilterId>,
) -> Option<FilterId> {
    let mut ctx = FilterContext {
        graph,
        chain_input,
        source_alpha: None,
        results: HashMap::new(),
    };

    let mut last: Option<FilterId> = None;
    for primitive in filter.primitives() {
        let id = convert_primitive(&mut ctx, primitive)?;
        if !primitive.result().is_empty() {
            ctx.results.insert(primitive.result().to_string(), id);
        }
        last = Some(id);
    }

    last
}

struct FilterContext<'a> {
    graph: &'a mut Filter,
    /// Result of the previous filter in a filter chain (acts as `SourceGraphic`).
    chain_input: Option<FilterId>,
    /// Memoized `SourceAlpha` node derived from `chain_input`.
    source_alpha: Option<FilterId>,
    /// Map from primitive `result` names to their graph node ids.
    results: HashMap<String, FilterId>,
}

impl FilterContext<'_> {
    fn resolve_input(&mut self, input: &usvg::filter::Input) -> FilterInput {
        match input {
            usvg::filter::Input::SourceGraphic => match self.chain_input {
                Some(id) => FilterInput::Result(id),
                None => FilterInput::Source(FilterSource::SourceGraphic),
            },
            usvg::filter::Input::SourceAlpha => match self.chain_input {
                Some(id) => {
                    let alpha_id = *self.source_alpha.get_or_insert_with(|| {
                        self.graph.add(
                            FilterEffect::ColorMatrix(SOURCE_ALPHA_MATRIX),
                            FilterInputs::single(FilterInput::Result(id)),
                        )
                    });
                    FilterInput::Result(alpha_id)
                }
                None => FilterInput::Source(FilterSource::SourceAlpha),
            },
            // usvg's parser guarantees that references point at an earlier
            // primitive's `result` within the same filter.
            usvg::filter::Input::Reference(name) => match self.results.get(name) {
                Some(id) => FilterInput::Result(*id),
                None => match self.chain_input {
                    Some(id) => FilterInput::Result(id),
                    None => FilterInput::Source(FilterSource::SourceGraphic),
                },
            },
        }
    }
}

/// Convert a single filter primitive, appending it (and any helper nodes) to the graph.
///
/// Returns `None` for primitives that cannot be expressed:
///
/// - `feImage`
/// - `feGaussianBlur`/`feDropShadow` with different X/Y std deviations
/// - `feMorphology` with different X/Y radii
/// - `feTurbulence` with different X/Y base frequencies
/// - `feConvolveMatrix` with a non-square kernel
/// - lighting primitives with a non-white lighting color
fn convert_primitive(
    ctx: &mut FilterContext<'_>,
    primitive: &usvg::filter::Primitive,
) -> Option<FilterId> {
    use usvg::filter::Kind;

    let id = match primitive.kind() {
        Kind::Blend(fe) => {
            let input1 = ctx.resolve_input(fe.input1());
            let input2 = ctx.resolve_input(fe.input2());
            ctx.graph.add(
                FilterEffect::Blend(util::to_mix(fe.mode())),
                FilterInputs::dual(input1, input2),
            )
        }
        Kind::ColorMatrix(fe) => {
            let matrix = to_color_matrix(fe.kind());
            let input = ctx.resolve_input(fe.input());
            ctx.graph.add(
                FilterEffect::ColorMatrix(matrix),
                FilterInputs::single(input),
            )
        }
        Kind::ComponentTransfer(fe) => {
            let filter = ComponentTransferFilter {
                red_function: to_transfer_function(fe.func_r()),
                green_function: to_transfer_function(fe.func_g()),
                blue_function: to_transfer_function(fe.func_b()),
                alpha_function: to_transfer_function(fe.func_a()),
            };
            let input = ctx.resolve_input(fe.input());
            ctx.graph.add(
                FilterEffect::ComponentTransfer(filter),
                FilterInputs::single(input),
            )
        }
        Kind::Composite(fe) => {
            let operator = match fe.operator() {
                usvg::filter::CompositeOperator::Over => CompositeOperator::Over,
                usvg::filter::CompositeOperator::In => CompositeOperator::In,
                usvg::filter::CompositeOperator::Out => CompositeOperator::Out,
                usvg::filter::CompositeOperator::Atop => CompositeOperator::Atop,
                usvg::filter::CompositeOperator::Xor => CompositeOperator::Xor,
                usvg::filter::CompositeOperator::Arithmetic { k1, k2, k3, k4 } => {
                    CompositeOperator::Arithmetic(ArithmeticCompositeOperator { k1, k2, k3, k4 })
                }
            };
            let input1 = ctx.resolve_input(fe.input1());
            let input2 = ctx.resolve_input(fe.input2());
            ctx.graph.add(
                FilterEffect::Composite(operator),
                FilterInputs::dual(input1, input2),
            )
        }
        Kind::ConvolveMatrix(fe) => {
            let matrix = fe.matrix();
            if matrix.columns() != matrix.rows() {
                return None;
            }
            let kernel = ConvolutionKernel {
                size: matrix.columns(),
                values: matrix.data().to_vec(),
                divisor: fe.divisor().get(),
                bias: fe.bias(),
                preserve_alpha: fe.preserve_alpha(),
            };
            let input = ctx.resolve_input(fe.input());
            ctx.graph.add(
                FilterEffect::ConvolveMatrix(kernel),
                FilterInputs::single(input),
            )
        }
        Kind::DiffuseLighting(fe) => {
            if !is_white(fe.lighting_color()) {
                return None;
            }
            let filter = DiffuseLightingFilter {
                surface_scale: fe.surface_scale(),
                diffuse_constant: fe.diffuse_constant(),
                kernel_unit_length: 1.0,
                light_source: to_light_source(fe.light_source()),
            };
            let input = ctx.resolve_input(fe.input());
            ctx.graph.add(
                FilterEffect::DiffuseLighting(filter),
                FilterInputs::single(input),
            )
        }
        Kind::SpecularLighting(fe) => {
            if !is_white(fe.lighting_color()) {
                return None;
            }
            let filter = SpecularLightingFilter {
                surface_scale: fe.surface_scale(),
                specular_constant: fe.specular_constant(),
                specular_exponent: fe.specular_exponent(),
                kernel_unit_length: 1.0,
                light_source: to_light_source(fe.light_source()),
            };
            let input = ctx.resolve_input(fe.input());
            ctx.graph.add(
                FilterEffect::SpecularLighting(filter),
                FilterInputs::single(input),
            )
        }
        Kind::DisplacementMap(fe) => {
            let filter = DisplacementMapFilter {
                scale: fe.scale(),
                x_channel: to_color_channel(fe.x_channel_selector()),
                y_channel: to_color_channel(fe.y_channel_selector()),
            };
            let input1 = ctx.resolve_input(fe.input1());
            let input2 = ctx.resolve_input(fe.input2());
            ctx.graph.add(
                FilterEffect::DisplacementMap(filter),
                FilterInputs::dual(input1, input2),
            )
        }
        Kind::DropShadow(fe) => {
            let std_dev_x = fe.std_dev_x().get();
            let std_dev_y = fe.std_dev_y().get();
            if std_dev_x != std_dev_y {
                return None;
            }
            let color = fe.color();
            let effect = FilterEffect::drop_shadow(
                fe.dx(),
                fe.dy(),
                std_dev_x,
                Color::from_rgba8(color.red, color.green, color.blue, fe.opacity().to_u8()),
            );
            let input = ctx.resolve_input(fe.input());
            ctx.graph.add(effect, FilterInputs::single(input))
        }
        Kind::Flood(fe) => {
            let color = fe.color();
            ctx.graph.add(
                FilterEffect::Flood(Color::from_rgba8(
                    color.red,
                    color.green,
                    color.blue,
                    fe.opacity().to_u8(),
                )),
                FilterInputs::NONE,
            )
        }
        Kind::GaussianBlur(fe) => {
            let std_dev_x = fe.std_dev_x().get();
            let std_dev_y = fe.std_dev_y().get();
            if std_dev_x != std_dev_y {
                return None;
            }
            let input = ctx.resolve_input(fe.input());
            ctx.graph
                .add(FilterEffect::blur(std_dev_x), FilterInputs::single(input))
        }
        // feImage renders an external image or SVG subtree, which cannot be
        // expressed as an `anyrender::FilterEffect`.
        Kind::Image(_) => return None,
        Kind::Merge(fe) => {
            // Express an n-ary merge as a chain of `Over` composites
            // (first input at the bottom).
            let mut acc: Option<FilterInput> = None;
            for input in fe.inputs() {
                let input = ctx.resolve_input(input);
                acc = Some(match acc {
                    None => input,
                    Some(below) => FilterInput::Result(ctx.graph.add(
                        FilterEffect::Composite(CompositeOperator::Over),
                        FilterInputs::dual(input, below),
                    )),
                });
            }
            match acc {
                Some(FilterInput::Result(id)) => id,
                // A single source input: pass it through unchanged.
                Some(input @ FilterInput::Source(_)) => ctx.graph.add(
                    FilterEffect::Offset(Vec2::ZERO),
                    FilterInputs::single(input),
                ),
                // No inputs: transparent black.
                None => ctx
                    .graph
                    .add(FilterEffect::Flood(Color::TRANSPARENT), FilterInputs::NONE),
            }
        }
        Kind::Morphology(fe) => {
            let radius_x = fe.radius_x().get();
            let radius_y = fe.radius_y().get();
            if radius_x != radius_y {
                return None;
            }
            let filter = MorphologyFilter {
                operator: match fe.operator() {
                    usvg::filter::MorphologyOperator::Erode => MorphologyOperator::Erode,
                    usvg::filter::MorphologyOperator::Dilate => MorphologyOperator::Dilate,
                },
                radius: radius_x,
            };
            let input = ctx.resolve_input(fe.input());
            ctx.graph.add(
                FilterEffect::Morphology(filter),
                FilterInputs::single(input),
            )
        }
        Kind::Offset(fe) => {
            let input = ctx.resolve_input(fe.input());
            ctx.graph.add(
                FilterEffect::Offset(Vec2::new(fe.dx() as f64, fe.dy() as f64)),
                FilterInputs::single(input),
            )
        }
        Kind::Tile(fe) => {
            let input = ctx.resolve_input(fe.input());
            ctx.graph
                .add(FilterEffect::Tile, FilterInputs::single(input))
        }
        Kind::Turbulence(fe) => {
            let base_frequency_x = fe.base_frequency_x().get();
            let base_frequency_y = fe.base_frequency_y().get();
            if base_frequency_x != base_frequency_y {
                return None;
            }
            let filter = TurbulenceFilter {
                base_frequency: base_frequency_x,
                num_octaves: fe.num_octaves(),
                seed: fe.seed().max(0) as u32,
                turbulence_type: match fe.kind() {
                    usvg::filter::TurbulenceKind::FractalNoise => TurbulenceType::FractalNoise,
                    usvg::filter::TurbulenceKind::Turbulence => TurbulenceType::Turbulence,
                },
            };
            ctx.graph
                .add(FilterEffect::Turbulence(filter), FilterInputs::NONE)
        }
    };

    Some(id)
}

fn to_color_matrix(kind: &usvg::filter::ColorMatrixKind) -> ColorMatrix {
    match kind {
        usvg::filter::ColorMatrixKind::Matrix(values) => {
            // usvg guarantees exactly 20 values.
            let mut matrix = [0.0; 20];
            matrix.copy_from_slice(values);
            ColorMatrix(matrix)
        }
        usvg::filter::ColorMatrixKind::Saturate(amount) => ColorMatrix::saturate(amount.get()),
        usvg::filter::ColorMatrixKind::HueRotate(degrees) => {
            ColorMatrix::hue_rotate(degrees.to_radians())
        }
        usvg::filter::ColorMatrixKind::LuminanceToAlpha => LUMINANCE_TO_ALPHA_MATRIX,
    }
}

fn to_transfer_function(func: &usvg::filter::TransferFunction) -> TransferFunction {
    match func {
        usvg::filter::TransferFunction::Identity => TransferFunction::Identity,
        usvg::filter::TransferFunction::Table(values) => match values.len() {
            // An empty table is an identity transform, and a single-value
            // table is a constant function.
            // See <https://drafts.fxtf.org/filter-effects/#element-attrdef-fecomponenttransfer-tablevalues>
            0 => TransferFunction::Identity,
            1 => TransferFunction::Table([values[0], values[0]].into_iter().collect()),
            _ => TransferFunction::Table(values.iter().copied().collect()),
        },
        usvg::filter::TransferFunction::Discrete(values) => match values.len() {
            0 => TransferFunction::Identity,
            _ => TransferFunction::Discrete(values.clone()),
        },
        usvg::filter::TransferFunction::Linear { slope, intercept } => {
            TransferFunction::Linear(LinearTransferFunction {
                slope: *slope,
                intercept: *intercept,
            })
        }
        usvg::filter::TransferFunction::Gamma {
            amplitude,
            exponent,
            offset,
        } => TransferFunction::Gamma(GammaTransferFunction {
            amplitude: *amplitude,
            exponent: *exponent,
            offset: *offset,
        }),
    }
}

fn to_color_channel(channel: usvg::filter::ColorChannel) -> ColorChannel {
    match channel {
        usvg::filter::ColorChannel::R => ColorChannel::Red,
        usvg::filter::ColorChannel::G => ColorChannel::Green,
        usvg::filter::ColorChannel::B => ColorChannel::Blue,
        usvg::filter::ColorChannel::A => ColorChannel::Alpha,
    }
}

fn to_light_source(light_source: usvg::filter::LightSource) -> LightSource {
    match light_source {
        usvg::filter::LightSource::DistantLight(light) => {
            LightSource::Distant(DistantLightSource {
                azimuth: light.azimuth,
                elevation: light.elevation,
            })
        }
        usvg::filter::LightSource::PointLight(light) => LightSource::Point(PointLightSource {
            x: light.x,
            y: light.y,
            z: light.z,
        }),
        usvg::filter::LightSource::SpotLight(light) => LightSource::Spot(SpotLightSource {
            x: light.x,
            y: light.y,
            z: light.z,
            points_at_x: light.points_at_x,
            points_at_y: light.points_at_y,
            points_at_z: light.points_at_z,
            specular_exponent: light.specular_exponent.get(),
            limiting_cone_angle: light.limiting_cone_angle,
        }),
    }
}

fn is_white(color: usvg::Color) -> bool {
    color.red == 255 && color.green == 255 && color.blue == 255
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filters_of_first_group(svg: &str) -> Vec<Arc<usvg::filter::Filter>> {
        let tree = usvg::Tree::from_str(svg, &usvg::Options::default()).unwrap();
        let usvg::Node::Group(group) = &tree.root().children()[0] else {
            panic!("expected group");
        };
        group.filters().to_vec()
    }

    fn convert(svg: &str) -> Option<Filter> {
        to_anyrender_filter(&filters_of_first_group(svg))
    }

    #[test]
    fn gaussian_blur() {
        let graph = convert(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <g filter="url(#f)">
                    <filter id="f"><feGaussianBlur stdDeviation="3"/></filter>
                    <rect width="50" height="50" fill="red"/>
                </g>
            </svg>"##,
        )
        .unwrap();
        assert_eq!(graph.nodes().len(), 1);
        assert_eq!(graph.nodes()[0].effect, FilterEffect::blur(3.0));
        assert_eq!(graph.output(), FilterId(0));
    }

    #[test]
    fn drop_shadow() {
        let graph = convert(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <g filter="url(#f)">
                    <filter id="f">
                        <feDropShadow dx="2" dy="4" stdDeviation="1.5" flood-color="black" flood-opacity="0.5"/>
                    </filter>
                    <rect width="50" height="50" fill="red"/>
                </g>
            </svg>"##,
        )
        .unwrap();
        assert_eq!(graph.nodes().len(), 1);
        assert_eq!(
            graph.nodes()[0].effect,
            FilterEffect::drop_shadow(2.0, 4.0, 1.5, Color::from_rgba8(0, 0, 0, 128))
        );
    }

    #[test]
    fn references_resolve_to_named_results() {
        let graph = convert(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <g filter="url(#f)">
                    <filter id="f">
                        <feFlood flood-color="lime" result="fill"/>
                        <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="blurred"/>
                        <feComposite in="fill" in2="blurred" operator="in"/>
                    </filter>
                    <rect width="50" height="50" fill="red"/>
                </g>
            </svg>"##,
        )
        .unwrap();
        assert_eq!(graph.nodes().len(), 3);
        let composite = &graph.nodes()[2];
        assert_eq!(
            composite.effect,
            FilterEffect::Composite(CompositeOperator::In)
        );
        assert_eq!(
            composite.inputs.primary,
            Some(FilterInput::Result(FilterId(0)))
        );
        assert_eq!(
            composite.inputs.secondary,
            Some(FilterInput::Result(FilterId(1)))
        );
        assert_eq!(graph.output(), FilterId(2));
    }

    #[test]
    fn merge_becomes_composite_chain() {
        let graph = convert(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <g filter="url(#f)">
                    <filter id="f">
                        <feOffset dx="5" dy="5" result="off"/>
                        <feMerge>
                            <feMergeNode in="off"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                    <rect width="50" height="50" fill="red"/>
                </g>
            </svg>"##,
        )
        .unwrap();
        // feOffset + one composite for the two merge nodes
        assert_eq!(graph.nodes().len(), 2);
        let composite = &graph.nodes()[1];
        assert_eq!(
            composite.effect,
            FilterEffect::Composite(CompositeOperator::Over)
        );
        // Later merge nodes render on top: SourceGraphic over "off".
        assert_eq!(
            composite.inputs.primary,
            Some(FilterInput::Source(FilterSource::SourceGraphic))
        );
        assert_eq!(
            composite.inputs.secondary,
            Some(FilterInput::Result(FilterId(0)))
        );
    }

    #[test]
    fn anisotropic_blur_is_unsupported() {
        assert!(
            convert(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                    <g filter="url(#f)">
                        <filter id="f"><feGaussianBlur stdDeviation="3 7"/></filter>
                        <rect width="50" height="50" fill="red"/>
                    </g>
                </svg>"##,
            )
            .is_none()
        );
    }

    #[test]
    fn fe_image_is_unsupported() {
        assert!(
            convert(
                r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                    <g filter="url(#f)">
                        <filter id="f"><feImage href="#r"/></filter>
                        <rect id="r" width="50" height="50" fill="red"/>
                    </g>
                </svg>"##,
            )
            .is_none()
        );
    }
}
