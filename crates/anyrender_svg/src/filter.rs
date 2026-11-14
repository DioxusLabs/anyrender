// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conversion of [`usvg::filter::Filter`] chains into an [`anyrender::Filter`] graph.
//!
//! The resulting graph is attached to a layer via [`anyrender::PaintScene::push_layer`].
//! Whether (and how faithfully) the filter is actually applied depends on the backend.
//!
//! Every primitive gets its resolved subregion, except `feOffset`, which (as in resvg)
//! inherits its input's subregion and isn't clipped by its own.
//!
//! Known approximations (accepted silently):
//!
//! - `feConvolveMatrix` `kernelUnitLength` is ignored.
//!
//! Filters that cannot be expressed at all (see [`convert_primitive`]) cause the whole
//! conversion to fail, in which case the caller falls back to the error handler and
//! renders the group unfiltered.

use anyrender::filters::{
    EdgeMode, Filter, FilterColorSpace, FilterEffect, FilterGraphNode, FilterId, FilterInput,
    FilterInputs, FilterSource,
    blur::GaussianBlurFilter,
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
    shadow::DropShadow,
    turbulence::{TurbulenceFilter, TurbulenceType},
};
use kurbo::{Rect, Vec2};
use peniko::Color;
use std::collections::HashMap;
use std::sync::Arc;

use crate::util;

/// Convert a chain of usvg filters (an element may have multiple filters applied
/// in sequence) into a single [`anyrender::Filter`] graph.
///
/// Returns `None` if any primitive cannot be expressed with [`FilterEffect`].
pub(crate) fn to_anyrender_filter(filters: &[Arc<usvg::filter::Filter>]) -> Option<Filter> {
    let mut graph = Filter::empty();
    // Output of the previous filter in the chain. It acts as `SourceGraphic`
    // for the next filter.
    let mut chain_input: Option<FilterId> = None;
    let chained = filters.len() > 1;
    for filter in filters {
        chain_input = Some(convert_filter(&mut graph, filter, chain_input, chained)?);
    }
    let output = chain_input?;
    // Each filter's primitives are clipped to its own region below, so the graph's region is
    // the union of them.
    let region = filters
        .iter()
        .map(|filter| to_rect(filter.rect()))
        .reduce(|a, b| a.union(b));
    graph.set_region(region);
    // `Filter::add` makes each new node the output, but the result can be an
    // earlier node (e.g. a single-input `feMerge` of a named result).
    if graph.output() != output {
        graph.add(
            FilterEffect::Offset(Vec2::ZERO),
            FilterInputs::single(FilterInput::Result(output)),
        );
    }
    Some(graph)
}

/// Convert a single usvg filter, appending its primitives to `graph`.
///
/// Returns the [`FilterId`] of the filter's result.
fn convert_filter(
    graph: &mut Filter,
    filter: &usvg::filter::Filter,
    chain_input: Option<FilterId>,
    chained: bool,
) -> Option<FilterId> {
    let region = to_rect(filter.rect());
    let mut ctx = FilterContext {
        graph,
        chain_input,
        source_alpha: None,
        results: HashMap::new(),
        color_space: FilterColorSpace::default(),
        subregion: None,
    };

    let mut last: Option<FilterId> = None;
    for primitive in filter.primitives() {
        ctx.color_space = match primitive.color_interpolation() {
            usvg::filter::ColorInterpolation::SRGB => FilterColorSpace::Srgb,
            usvg::filter::ColorInterpolation::LinearRGB => FilterColorSpace::LinearRgb,
        };
        ctx.subregion = match primitive.kind() {
            usvg::filter::Kind::Offset(_) => None,
            _ => Some(to_rect(primitive.rect()).intersect(region)),
        };
        let id = convert_primitive(&mut ctx, primitive)?;
        if !primitive.result().is_empty() {
            ctx.results.insert(primitive.result().to_string(), id);
        }
        last = Some(id);
    }

    // In a chain the graph's region is the union of every filter's region, so a final
    // `feOffset` (which has no subregion) needs its own filter's region as an explicit clip.
    if chained
        && let Some(primitive) = filter.primitives().last()
        && matches!(primitive.kind(), usvg::filter::Kind::Offset(_))
    {
        ctx.subregion = Some(region);
        last = Some(ctx.add(
            FilterEffect::Offset(Vec2::ZERO),
            FilterInputs::single(FilterInput::Result(last?)),
        ));
    }

    last
}

fn to_rect(rect: usvg::NonZeroRect) -> Rect {
    Rect::new(
        rect.left() as f64,
        rect.top() as f64,
        rect.right() as f64,
        rect.bottom() as f64,
    )
}

struct FilterContext<'a> {
    graph: &'a mut Filter,
    /// Result of the previous filter in a filter chain (acts as `SourceGraphic`).
    chain_input: Option<FilterId>,
    /// Memoized `SourceAlpha` node derived from `chain_input`.
    source_alpha: Option<FilterId>,
    /// Map from primitive `result` names to their graph node ids.
    results: HashMap<String, FilterId>,
    /// Color space of the primitive being converted.
    color_space: FilterColorSpace,
    /// Subregion of the primitive being converted.
    subregion: Option<Rect>,
}

impl FilterContext<'_> {
    fn add(&mut self, effect: FilterEffect, inputs: FilterInputs) -> FilterId {
        self.graph.add_node(FilterGraphNode {
            effect,
            inputs,
            color_space: self.color_space,
            subregion: self.subregion,
        })
    }

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
                            FilterEffect::ColorMatrix(ColorMatrix::SOURCE_ALPHA),
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
/// Returns `None` for primitives that cannot be expressed, which is only `feImage`.
fn convert_primitive(
    ctx: &mut FilterContext<'_>,
    primitive: &usvg::filter::Primitive,
) -> Option<FilterId> {
    use usvg::filter::Kind;

    let id = match primitive.kind() {
        Kind::Blend(fe) => {
            let input1 = ctx.resolve_input(fe.input1());
            let input2 = ctx.resolve_input(fe.input2());
            ctx.add(
                FilterEffect::Blend(util::to_blend_mode(fe.mode()).mix),
                FilterInputs::dual(input1, input2),
            )
        }
        Kind::ColorMatrix(fe) => {
            let matrix = to_color_matrix(fe.kind());
            let input = ctx.resolve_input(fe.input());
            ctx.add(
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
            ctx.add(
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
                usvg::filter::CompositeOperator::Lighter => CompositeOperator::Lighter,
                usvg::filter::CompositeOperator::Arithmetic { k1, k2, k3, k4 } => {
                    CompositeOperator::Arithmetic(ArithmeticCompositeOperator { k1, k2, k3, k4 })
                }
            };
            let input1 = ctx.resolve_input(fe.input1());
            let input2 = ctx.resolve_input(fe.input2());
            ctx.add(
                FilterEffect::Composite(operator),
                FilterInputs::dual(input1, input2),
            )
        }
        Kind::ConvolveMatrix(fe) => {
            let matrix = fe.matrix();
            let kernel = ConvolutionKernel {
                columns: matrix.columns(),
                rows: matrix.rows(),
                values: matrix.data().to_vec(),
                target_x: matrix.target_x(),
                target_y: matrix.target_y(),
                edge_mode: match fe.edge_mode() {
                    usvg::filter::EdgeMode::None => EdgeMode::None,
                    usvg::filter::EdgeMode::Duplicate => EdgeMode::Duplicate,
                    usvg::filter::EdgeMode::Wrap => EdgeMode::Wrap,
                },
                divisor: fe.divisor().get(),
                bias: fe.bias(),
                preserve_alpha: fe.preserve_alpha(),
            };
            let input = ctx.resolve_input(fe.input());
            ctx.add(
                FilterEffect::ConvolveMatrix(kernel),
                FilterInputs::single(input),
            )
        }
        Kind::DiffuseLighting(fe) => {
            let filter = DiffuseLightingFilter {
                surface_scale: fe.surface_scale(),
                diffuse_constant: fe.diffuse_constant(),
                kernel_unit_length: 1.0,
                light_source: to_light_source(fe.light_source()),
                lighting_color: to_color(fe.lighting_color()),
            };
            let input = ctx.resolve_input(fe.input());
            ctx.add(
                FilterEffect::DiffuseLighting(filter),
                FilterInputs::single(input),
            )
        }
        Kind::SpecularLighting(fe) => {
            let filter = SpecularLightingFilter {
                surface_scale: fe.surface_scale(),
                specular_constant: fe.specular_constant(),
                specular_exponent: fe.specular_exponent(),
                kernel_unit_length: 1.0,
                light_source: to_light_source(fe.light_source()),
                lighting_color: to_color(fe.lighting_color()),
            };
            let input = ctx.resolve_input(fe.input());
            ctx.add(
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
            ctx.add(
                FilterEffect::DisplacementMap(filter),
                FilterInputs::dual(input1, input2),
            )
        }
        Kind::DropShadow(fe) => {
            let color = fe.color();
            let effect = FilterEffect::DropShadow(DropShadow {
                dx: fe.dx(),
                dy: fe.dy(),
                std_deviation_x: fe.std_dev_x().get(),
                std_deviation_y: fe.std_dev_y().get(),
                color: Color::from_rgba8(color.red, color.green, color.blue, fe.opacity().to_u8()),
                edge_mode: EdgeMode::None,
            });
            let input = ctx.resolve_input(fe.input());
            ctx.add(effect, FilterInputs::single(input))
        }
        Kind::Flood(fe) => {
            let color = fe.color();
            ctx.add(
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
            let blur = GaussianBlurFilter {
                std_deviation_x: fe.std_dev_x().get(),
                std_deviation_y: fe.std_dev_y().get(),
                edge_mode: EdgeMode::None,
            };
            let input = ctx.resolve_input(fe.input());
            ctx.add(
                FilterEffect::GaussianBlur(blur),
                FilterInputs::single(input),
            )
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
                    Some(below) => FilterInput::Result(ctx.add(
                        FilterEffect::Composite(CompositeOperator::Over),
                        FilterInputs::dual(input, below),
                    )),
                });
            }
            match acc {
                Some(FilterInput::Result(id)) => id,
                // A single source input: pass it through unchanged.
                Some(input @ FilterInput::Source(_)) => ctx.add(
                    FilterEffect::Offset(Vec2::ZERO),
                    FilterInputs::single(input),
                ),
                // No inputs: transparent black.
                None => ctx.add(FilterEffect::Flood(Color::TRANSPARENT), FilterInputs::NONE),
            }
        }
        Kind::Morphology(fe) => {
            let filter = MorphologyFilter {
                operator: match fe.operator() {
                    usvg::filter::MorphologyOperator::Erode => MorphologyOperator::Erode,
                    usvg::filter::MorphologyOperator::Dilate => MorphologyOperator::Dilate,
                },
                radius_x: fe.radius_x().get(),
                radius_y: fe.radius_y().get(),
            };
            let input = ctx.resolve_input(fe.input());
            ctx.add(
                FilterEffect::Morphology(filter),
                FilterInputs::single(input),
            )
        }
        Kind::Offset(fe) => {
            let input = ctx.resolve_input(fe.input());
            ctx.add(
                FilterEffect::Offset(Vec2::new(fe.dx() as f64, fe.dy() as f64)),
                FilterInputs::single(input),
            )
        }
        Kind::Tile(fe) => {
            let input = ctx.resolve_input(fe.input());
            ctx.add(FilterEffect::Tile, FilterInputs::single(input))
        }
        Kind::Turbulence(fe) => {
            let filter = TurbulenceFilter {
                base_frequency_x: fe.base_frequency_x().get(),
                base_frequency_y: fe.base_frequency_y().get(),
                stitch_tiles: fe.stitch_tiles(),
                num_octaves: fe.num_octaves(),
                seed: fe.seed() as f32,
                turbulence_type: match fe.kind() {
                    usvg::filter::TurbulenceKind::FractalNoise => TurbulenceType::FractalNoise,
                    usvg::filter::TurbulenceKind::Turbulence => TurbulenceType::Turbulence,
                },
            };
            ctx.add(FilterEffect::Turbulence(filter), FilterInputs::NONE)
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
        usvg::filter::ColorMatrixKind::LuminanceToAlpha => ColorMatrix::LUMINANCE_TO_ALPHA,
    }
}

fn to_transfer_function(func: &usvg::filter::TransferFunction) -> TransferFunction {
    match func {
        usvg::filter::TransferFunction::Identity => TransferFunction::Identity,
        usvg::filter::TransferFunction::Table(values) => match values.len() {
            // An empty table is an identity transform, and a single-value
            // table is a constant function.
            // See <https://www.w3.org/TR/filter-effects-1/#element-attrdef-fecomponenttransfer-tablevalues>
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

fn to_color(color: usvg::Color) -> Color {
    Color::from_rgb8(color.red, color.green, color.blue)
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
    fn output_is_final_result_when_it_is_an_earlier_node() {
        let graph = convert(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <g filter="url(#f)">
                    <filter id="f">
                        <feGaussianBlur stdDeviation="2" result="blur"/>
                        <feOffset dx="4" dy="4"/>
                        <feMerge><feMergeNode in="blur"/></feMerge>
                    </filter>
                    <rect width="50" height="50" fill="red"/>
                </g>
            </svg>"##,
        )
        .unwrap();
        let output = &graph.nodes()[graph.output().0 as usize];
        assert_eq!(output.effect, FilterEffect::Offset(Vec2::ZERO));
        assert_eq!(
            output.inputs,
            FilterInputs::single(FilterInput::Result(FilterId(0)))
        );
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

    fn single_node(primitive: &str) -> anyrender::filters::FilterGraphNode {
        let graph = convert(&format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <g filter="url(#f)">
                    <filter id="f">{primitive}</filter>
                    <rect width="50" height="50" fill="red"/>
                </g>
            </svg>"##
        ))
        .unwrap();
        assert_eq!(graph.nodes().len(), 1);
        graph.nodes()[0].clone()
    }

    #[test]
    fn anisotropic_blur() {
        let node = single_node(r#"<feGaussianBlur stdDeviation="3 7"/>"#);
        let FilterEffect::GaussianBlur(blur) = node.effect else {
            panic!("expected blur, got {:?}", node.effect);
        };
        assert_eq!((blur.std_deviation_x, blur.std_deviation_y), (3.0, 7.0));
    }

    #[test]
    fn color_interpolation_filters() {
        let default = single_node(r#"<feGaussianBlur stdDeviation="3"/>"#);
        assert_eq!(default.color_space, FilterColorSpace::LinearRgb);
        let srgb =
            single_node(r#"<feGaussianBlur stdDeviation="3" color-interpolation-filters="sRGB"/>"#);
        assert_eq!(srgb.color_space, FilterColorSpace::Srgb);
    }

    #[test]
    fn rectangular_convolve_matrix() {
        let node = single_node(
            r#"<feConvolveMatrix order="3 2" kernelMatrix="1 2 3 4 5 6" targetX="2" targetY="0" edgeMode="wrap"/>"#,
        );
        let FilterEffect::ConvolveMatrix(kernel) = node.effect else {
            panic!("expected convolve, got {:?}", node.effect);
        };
        assert_eq!((kernel.columns, kernel.rows), (3, 2));
        assert_eq!((kernel.target_x, kernel.target_y), (2, 0));
        assert_eq!(kernel.edge_mode, anyrender::filters::EdgeMode::Wrap);
        assert_eq!(kernel.values.len(), 6);
    }

    #[test]
    fn lighting_color() {
        let node = single_node(
            r#"<feDiffuseLighting lighting-color="rgb(10, 20, 30)"><feDistantLight/></feDiffuseLighting>"#,
        );
        let FilterEffect::DiffuseLighting(lighting) = node.effect else {
            panic!("expected diffuse lighting, got {:?}", node.effect);
        };
        assert_eq!(lighting.lighting_color, Color::from_rgb8(10, 20, 30));
    }

    #[test]
    fn turbulence_frequency_pair_and_stitch_tiles() {
        let node = single_node(r#"<feTurbulence baseFrequency="0.05 0.2" stitchTiles="stitch"/>"#);
        let FilterEffect::Turbulence(turbulence) = node.effect else {
            panic!("expected turbulence, got {:?}", node.effect);
        };
        assert_eq!(
            (turbulence.base_frequency_x, turbulence.base_frequency_y),
            (0.05, 0.2)
        );
        assert!(turbulence.stitch_tiles);
    }

    /// usvg computes the filter region in f32.
    fn rounded(rect: Option<Rect>) -> Option<Rect> {
        rect.map(|r| Rect::new(r.x0.round(), r.y0.round(), r.x1.round(), r.y1.round()))
    }

    #[test]
    fn primitive_subregions() {
        let flood = single_node(r#"<feFlood x="10" y="20" width="30" height="5"/>"#);
        assert_eq!(flood.subregion, Some(Rect::new(10.0, 20.0, 40.0, 25.0)));

        // Without x/y/width/height, the subregion is the filter region: by default the
        // 50x50 bounding box grown by 10% on each side.
        let blur = single_node(r#"<feGaussianBlur stdDeviation="3"/>"#);
        assert_eq!(
            rounded(blur.subregion),
            Some(Rect::new(-5.0, -5.0, 55.0, 55.0))
        );

        let offset = single_node(r#"<feOffset x="10" dx="5"/>"#);
        assert_eq!(offset.subregion, None);
    }

    #[test]
    fn filter_region() {
        let graph = convert(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <g filter="url(#f)">
                    <filter id="f" filterUnits="userSpaceOnUse" x="0" y="0" width="20" height="30">
                        <feFlood x="10" y="10" width="40" height="40" flood-color="green"/>
                    </filter>
                    <rect width="50" height="50" fill="red"/>
                </g>
            </svg>"##,
        )
        .unwrap();
        assert_eq!(graph.region(), Some(Rect::new(0.0, 0.0, 20.0, 30.0)));
        // Subregions are clipped to the filter region.
        assert_eq!(
            graph.nodes()[0].subregion,
            Some(Rect::new(10.0, 10.0, 20.0, 30.0))
        );

        // Without x/y/width/height, the region is the bounding box grown by 10% on each side.
        let graph = convert(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <g filter="url(#f)">
                    <filter id="f"><feGaussianBlur stdDeviation="3"/></filter>
                    <rect width="50" height="50" fill="red"/>
                </g>
            </svg>"##,
        )
        .unwrap();
        assert_eq!(
            rounded(graph.region()),
            Some(Rect::new(-5.0, -5.0, 55.0, 55.0))
        );
    }

    #[test]
    fn chained_offset_is_clipped_to_its_filter_region() {
        let graph = convert(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
                <g filter="url(#small) url(#large)">
                    <filter id="small" filterUnits="userSpaceOnUse" x="40" y="40" width="120" height="120">
                        <feOffset/>
                    </filter>
                    <filter id="large" filterUnits="userSpaceOnUse" x="0" y="0" width="200" height="200">
                        <feOffset/>
                    </filter>
                    <rect width="200" height="200" fill="red"/>
                </g>
            </svg>"##,
        )
        .unwrap();
        assert_eq!(graph.region(), Some(Rect::new(0.0, 0.0, 200.0, 200.0)));
        let subregions: Vec<_> = graph.nodes().iter().map(|n| n.subregion).collect();
        assert_eq!(
            subregions,
            [
                None,
                Some(Rect::new(40.0, 40.0, 160.0, 160.0)),
                None,
                Some(Rect::new(0.0, 0.0, 200.0, 200.0)),
            ]
        );
    }

    #[test]
    fn tile_takes_its_input_subregion() {
        let graph = convert(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
                <g filter="url(#f)">
                    <filter id="f">
                        <feFlood x="0" y="0" width="10" height="10" flood-color="green"/>
                        <feTile/>
                    </filter>
                    <rect width="50" height="50" fill="red"/>
                </g>
            </svg>"##,
        )
        .unwrap();
        let [flood, tile] = graph.nodes() else {
            panic!("expected two nodes, got {:?}", graph.nodes());
        };
        assert_eq!(flood.subregion, Some(Rect::new(0.0, 0.0, 10.0, 10.0)));
        assert_eq!(tile.effect, FilterEffect::Tile);
        assert_eq!(tile.inputs.primary, Some(FilterInput::Result(FilterId(0))));
        assert_eq!(
            rounded(tile.subregion),
            Some(Rect::new(-5.0, -5.0, 55.0, 55.0))
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
