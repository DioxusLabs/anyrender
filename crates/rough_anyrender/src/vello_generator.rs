use std::fmt::Display;
use std::ops::MulAssign;

use anyrender::{Paint, PaintScene};
use euclid::default::Point2D;
use euclid::Trig;
use kurbo::{Affine, BezPath, Cap, Join, PathEl, Point, Stroke};
use num_traits::{Float, FromPrimitive};
use palette::rgb::Rgba;
use palette::Srgba;
use peniko::{Color, Fill};
use roughr::core::{Drawable, OpSet, OpSetType, OpType, Options};
use roughr::generator::Generator;
use roughr::PathSegment;

#[derive(Default)]
pub struct VelloGenerator {
    gen: Generator,
    options: Option<Options>,
}

#[derive(Clone)]
pub struct VelloOpset<F: Float + Trig> {
    pub op_set_type: OpSetType,
    pub ops: BezPath,
    pub size: Option<Point2D<F>>,
    pub path: Option<String>,
}

pub trait ToVelloOpset<F: Float + Trig> {
    fn to_vello_opset(self) -> VelloOpset<F>;
}

impl<F: Float + Trig + FromPrimitive> ToVelloOpset<F> for OpSet<F> {
    fn to_vello_opset(self) -> VelloOpset<F> {
        VelloOpset {
            op_set_type: self.op_set_type.clone(),
            size: self.size,
            path: self.path.clone(),
            ops: opset_to_shape(&self),
        }
    }
}

pub struct VelloDrawable<F: Float + Trig> {
    pub shape: String,
    pub options: Options,
    pub sets: Vec<VelloOpset<F>>,
}

pub trait ToVelloDrawable<F: Float + Trig> {
    fn to_vello_drawable(self) -> VelloDrawable<F>;
}

impl<F: Float + Trig + FromPrimitive> ToVelloDrawable<F> for Drawable<F> {
    fn to_vello_drawable(self) -> VelloDrawable<F> {
        VelloDrawable {
            shape: self.shape,
            options: self.options,
            sets: self.sets.into_iter().map(|s| s.to_vello_opset()).collect(),
        }
    }
}

impl VelloGenerator {
    pub fn new(options: Options) -> Self {
        VelloGenerator {
            gen: Generator::default(),
            options: Some(options),
        }
    }
}

impl<F: Float + Trig> VelloDrawable<F> {
    pub fn draw(&self, scene: &mut impl PaintScene) {
        for set in self.sets.iter() {
            match set.op_set_type {
                OpSetType::Path => {
                    // Convert stroke options to Vello stroke
                    let mut stroke = Stroke::new(self.options.stroke_width.unwrap_or(1.0) as f64);

                    // Set dash pattern if available
                    if let Some(ref dash_pattern) = self.options.stroke_line_dash {
                        stroke = stroke.with_dashes(
                            self.options.stroke_line_dash_offset.unwrap_or(0.0),
                            dash_pattern,
                        );
                    }

                    // Set line caps and joins
                    stroke = stroke
                        .with_caps(convert_line_cap_from_roughr_to_vello(self.options.line_cap));
                    stroke = stroke.with_join(convert_line_join_from_roughr_to_vello(
                        self.options.line_join,
                    ));

                    // Convert stroke color
                    let stroke_color = self
                        .options
                        .stroke
                        .unwrap_or_else(|| Srgba::new(0.0, 0.0, 0.0, 1.0));
                    let stroke_brush = convert_color_to_paint(stroke_color);

                    scene.stroke(&stroke, Affine::IDENTITY, &stroke_brush, None, &set.ops);
                }
                OpSetType::FillPath => {
                    let fill_rule = match self.shape.as_str() {
                        "curve" | "polygon" | "path" => Fill::EvenOdd,
                        _ => Fill::NonZero,
                    };

                    let fill_color = self.options.fill.unwrap_or(Rgba::new(0.0, 0.0, 0.0, 1.0));
                    let fill_brush = convert_rgba_to_paint(fill_color);

                    scene.fill(fill_rule, Affine::IDENTITY, &fill_brush, None, &set.ops);
                }
                OpSetType::FillSketch => {
                    let mut fweight = self.options.fill_weight.unwrap_or_default();
                    if fweight < 0.0 {
                        fweight = self.options.stroke_width.unwrap_or(1.0) / 2.0;
                    }

                    let mut stroke = Stroke::new(fweight as f64);

                    // Set dash pattern if available
                    if let Some(ref dash_pattern) = self.options.fill_line_dash {
                        stroke = stroke.with_dashes(
                            self.options.fill_line_dash_offset.unwrap_or(0.0),
                            dash_pattern,
                        );
                    }

                    stroke = stroke
                        .with_caps(convert_line_cap_from_roughr_to_vello(self.options.line_cap));
                    stroke = stroke.with_join(convert_line_join_from_roughr_to_vello(
                        self.options.line_join,
                    ));

                    let fill_color = self
                        .options
                        .fill
                        .unwrap_or_else(|| Rgba::new(0.0, 0.0, 0.0, 1.0));
                    let fill_brush = convert_rgba_to_paint(fill_color);

                    scene.stroke(&stroke, Affine::IDENTITY, &fill_brush, None, &set.ops);
                }
            }
        }
    }
}

fn opset_to_shape<F: Trig + Float + FromPrimitive>(op_set: &OpSet<F>) -> BezPath {
    let mut path: BezPath = BezPath::new();
    for item in op_set.ops.iter() {
        match item.op {
            OpType::Move => path.extend([PathEl::MoveTo(Point::new(
                item.data[0].to_f64().unwrap(),
                item.data[1].to_f64().unwrap(),
            ))]),
            OpType::BCurveTo => path.extend([PathEl::CurveTo(
                Point::new(
                    item.data[0].to_f64().unwrap(),
                    item.data[1].to_f64().unwrap(),
                ),
                Point::new(
                    item.data[2].to_f64().unwrap(),
                    item.data[3].to_f64().unwrap(),
                ),
                Point::new(
                    item.data[4].to_f64().unwrap(),
                    item.data[5].to_f64().unwrap(),
                ),
            )]),
            OpType::LineTo => {
                path.extend([PathEl::LineTo(Point::new(
                    item.data[0].to_f64().unwrap(),
                    item.data[1].to_f64().unwrap(),
                ))]);
            }
        }
    }
    path
}

impl VelloGenerator {
    pub fn line<F: Trig + Float + FromPrimitive>(
        &self,
        x1: F,
        y1: F,
        x2: F,
        y2: F,
    ) -> VelloDrawable<F> {
        let drawable = self.gen.line(x1, y1, x2, y2, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn rectangle<F: Trig + Float + FromPrimitive>(
        &self,
        x: F,
        y: F,
        width: F,
        height: F,
    ) -> VelloDrawable<F> {
        let drawable = self.gen.rectangle(x, y, width, height, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn ellipse<F: Trig + Float + FromPrimitive>(
        &self,
        x: F,
        y: F,
        width: F,
        height: F,
    ) -> VelloDrawable<F> {
        let drawable = self.gen.ellipse(x, y, width, height, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn circle<F: Trig + Float + FromPrimitive>(
        &self,
        x: F,
        y: F,
        diameter: F,
    ) -> VelloDrawable<F> {
        let drawable = self.gen.circle(x, y, diameter, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn linear_path<F: Trig + Float + FromPrimitive>(
        &self,
        points: &[Point2D<F>],
        close: bool,
    ) -> VelloDrawable<F> {
        let drawable = self.gen.linear_path(points, close, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn polygon<F: Trig + Float + FromPrimitive + MulAssign + Display>(
        &self,
        points: &[Point2D<F>],
    ) -> VelloDrawable<F> {
        let drawable = self.gen.polygon(points, &self.options);
        drawable.to_vello_drawable()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn arc<F: Trig + Float + FromPrimitive>(
        &self,
        x: F,
        y: F,
        width: F,
        height: F,
        start: F,
        stop: F,
        closed: bool,
    ) -> VelloDrawable<F> {
        let drawable = self
            .gen
            .arc(x, y, width, height, start, stop, closed, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn bezier_quadratic<F: Trig + Float + FromPrimitive + MulAssign + Display>(
        &self,
        start: Point2D<F>,
        cp: Point2D<F>,
        end: Point2D<F>,
    ) -> VelloDrawable<F> {
        let drawable = self.gen.bezier_quadratic(start, cp, end, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn bezier_cubic<F: Trig + Float + FromPrimitive + MulAssign + Display>(
        &self,
        start: Point2D<F>,
        cp1: Point2D<F>,
        cp2: Point2D<F>,
        end: Point2D<F>,
    ) -> VelloDrawable<F> {
        let drawable = self.gen.bezier_cubic(start, cp1, cp2, end, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn curve<F: Trig + Float + FromPrimitive + MulAssign + Display>(
        &self,
        points: &[Point2D<F>],
    ) -> VelloDrawable<F> {
        let drawable = self.gen.curve(points, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn path<F: Trig + Float + FromPrimitive + MulAssign + Display>(
        &self,
        svg_path: String,
    ) -> VelloDrawable<F> {
        let drawable = self.gen.path(svg_path, &self.options);
        drawable.to_vello_drawable()
    }

    pub fn bez_path<F: Trig + Float + FromPrimitive + MulAssign + Display>(
        &self,
        bezier_path: BezPath,
    ) -> VelloDrawable<F> {
        let segments = bezpath_to_svg_segments(&bezier_path);
        self.gen
            .path_from_segments(segments, &self.options)
            .to_vello_drawable()
    }
}

fn convert_line_cap_from_roughr_to_vello(roughr_line_cap: Option<roughr::core::LineCap>) -> Cap {
    match roughr_line_cap {
        Some(roughr::core::LineCap::Butt) => Cap::Butt,
        Some(roughr::core::LineCap::Round) => Cap::Round,
        Some(roughr::core::LineCap::Square) => Cap::Square,
        None => Cap::Butt,
    }
}

fn convert_line_join_from_roughr_to_vello(
    roughr_line_join: Option<roughr::core::LineJoin>,
) -> Join {
    match roughr_line_join {
        Some(roughr::core::LineJoin::Miter { limit: _ }) => Join::Miter, // Kurbo doesn't store limit in enum
        Some(roughr::core::LineJoin::Round) => Join::Round,
        Some(roughr::core::LineJoin::Bevel) => Join::Bevel,
        None => Join::Miter,
    }
}

fn convert_color_to_paint(color: Srgba) -> Paint {
    let (r, g, b, a) = color.into_components();
    Paint::Solid(Color::new([r, g, b, a]))
}

fn convert_rgba_to_paint(color: Rgba) -> Paint {
    let (r, g, b, a) = color.into_components();
    Paint::Solid(Color::new([r, g, b, a]))
}

pub fn bezpath_to_svg_segments(path: &BezPath) -> Vec<PathSegment> {
    let mut segments = Vec::new();

    for elem in path.elements() {
        match elem {
            kurbo::PathEl::MoveTo(p) => {
                segments.push(PathSegment::MoveTo {
                    abs: true,
                    x: p.x,
                    y: p.y,
                });
            }
            kurbo::PathEl::LineTo(p) => {
                segments.push(PathSegment::LineTo {
                    abs: true,
                    x: p.x,
                    y: p.y,
                });
            }
            kurbo::PathEl::QuadTo(p1, p2) => {
                segments.push(PathSegment::Quadratic {
                    abs: true,
                    x1: p1.x,
                    y1: p1.y,
                    x: p2.x,
                    y: p2.y,
                });
            }
            kurbo::PathEl::CurveTo(p1, p2, p3) => {
                segments.push(PathSegment::CurveTo {
                    abs: true,
                    x1: p1.x,
                    y1: p1.y,
                    x2: p2.x,
                    y2: p2.y,
                    x: p3.x,
                    y: p3.y,
                });
            }
            kurbo::PathEl::ClosePath => {
                segments.push(PathSegment::ClosePath { abs: true });
            }
        }
    }

    segments
}
