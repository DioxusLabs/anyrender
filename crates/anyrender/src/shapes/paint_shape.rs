use std::borrow::Cow;

use kurbo::{BezPath, Circle, Ellipse, Line, PathEl, Point, Rect, RoundedRect, Shape};

use crate::NonUniformRoundedRect;

/// A shape passed to the drawing methods of [`PaintScene`](crate::PaintScene).
///
/// Shapes are borrowed, so that passing one doesn't copy it. Backends match on the variant to pick
/// a fast path (for example a native rect or rounded rect), and can fall back to the
/// [`Shape`] implementation (or [`PaintShape::into_bez_path`]) to draw anything else as a path.
///
/// A [`BezPath`] passed by value is moved into the [`PaintShape::Path`] variant without copying
/// its elements. Other shapes can be converted with [`PaintShape::from_shape`].
#[derive(Clone, Debug, PartialEq)]
pub enum PaintShape<'a> {
    Rect(&'a Rect),
    RoundedRect(&'a RoundedRect),
    NonUniformRoundedRect(&'a NonUniformRoundedRect),
    Circle(&'a Circle),
    Ellipse(&'a Ellipse),
    Arc(&'a kurbo::Arc),
    Line(&'a Line),
    Path(Cow<'a, [PathEl]>),
}

/// Calls `$expr` with `$shape` bound to a reference to the shape in each variant.
macro_rules! with_shape {
    ($paint_shape:expr, $shape:ident => $expr:expr) => {
        match $paint_shape {
            PaintShape::Rect($shape) => $expr,
            PaintShape::RoundedRect($shape) => $expr,
            PaintShape::NonUniformRoundedRect($shape) => $expr,
            PaintShape::Circle($shape) => $expr,
            PaintShape::Ellipse($shape) => $expr,
            PaintShape::Arc($shape) => $expr,
            PaintShape::Line($shape) => $expr,
            PaintShape::Path(path) => {
                let $shape = &&**path;
                $expr
            }
        }
    };
}

impl<'a> PaintShape<'a> {
    /// Convert any [`Shape`] to a `PaintShape`.
    ///
    /// The path elements are borrowed if the shape has them as a slice (see
    /// [`Shape::as_path_slice`]), and otherwise are collected (with the given `tolerance`) into a
    /// new path. Prefer converting concrete shapes with `From`, which keeps rects, circles etc as
    /// they are.
    pub fn from_shape(shape: &'a impl Shape, tolerance: f64) -> Self {
        match shape.as_path_slice() {
            Some(elements) => PaintShape::Path(Cow::Borrowed(elements)),
            None => PaintShape::Path(Cow::Owned(shape.path_elements(tolerance).collect())),
        }
    }

    /// Reborrow the shape, without copying an owned path.
    pub fn as_ref(&self) -> PaintShape<'_> {
        match self {
            PaintShape::Rect(rect) => PaintShape::Rect(rect),
            PaintShape::RoundedRect(rect) => PaintShape::RoundedRect(rect),
            PaintShape::NonUniformRoundedRect(rect) => PaintShape::NonUniformRoundedRect(rect),
            PaintShape::Circle(circle) => PaintShape::Circle(circle),
            PaintShape::Ellipse(ellipse) => PaintShape::Ellipse(ellipse),
            PaintShape::Arc(arc) => PaintShape::Arc(arc),
            PaintShape::Line(line) => PaintShape::Line(line),
            PaintShape::Path(path) => PaintShape::Path(Cow::Borrowed(path)),
        }
    }

    /// Convert the shape to a [`BezPath`]. An owned path is moved into the `BezPath` without
    /// copying it.
    pub fn into_bez_path(self, tolerance: f64) -> BezPath {
        match self {
            PaintShape::Path(path) => BezPath::from_vec(path.into_owned()),
            shape => shape.to_path(tolerance),
        }
    }
}

impl<'a> From<&'a PaintShape<'_>> for PaintShape<'a> {
    fn from(shape: &'a PaintShape<'_>) -> Self {
        shape.as_ref()
    }
}

macro_rules! impl_from_ref {
    ($($variant:ident($ty:ty)),* $(,)?) => {
        $(
            impl<'a> From<&'a $ty> for PaintShape<'a> {
                fn from(shape: &'a $ty) -> Self {
                    PaintShape::$variant(shape)
                }
            }
        )*
    };
}

impl_from_ref!(
    Rect(Rect),
    RoundedRect(RoundedRect),
    NonUniformRoundedRect(NonUniformRoundedRect),
    Circle(Circle),
    Ellipse(Ellipse),
    Arc(kurbo::Arc),
    Line(Line),
);

impl<'a> From<&'a BezPath> for PaintShape<'a> {
    fn from(path: &'a BezPath) -> Self {
        PaintShape::Path(Cow::Borrowed(path.elements()))
    }
}

impl<'a> From<&'a [PathEl]> for PaintShape<'a> {
    fn from(elements: &'a [PathEl]) -> Self {
        PaintShape::Path(Cow::Borrowed(elements))
    }
}

impl From<BezPath> for PaintShape<'_> {
    fn from(path: BezPath) -> Self {
        PaintShape::Path(Cow::Owned(path.into_elements()))
    }
}

impl From<Vec<PathEl>> for PaintShape<'_> {
    fn from(elements: Vec<PathEl>) -> Self {
        PaintShape::Path(Cow::Owned(elements))
    }
}

/// The [`Shape::PathElementsIter`] of a [`PaintShape`].
pub enum PaintShapePathIter<'a> {
    Rect(<Rect as Shape>::PathElementsIter<'a>),
    RoundedRect(<RoundedRect as Shape>::PathElementsIter<'a>),
    NonUniformRoundedRect(<NonUniformRoundedRect as Shape>::PathElementsIter<'a>),
    Circle(<Circle as Shape>::PathElementsIter<'a>),
    Ellipse(<Ellipse as Shape>::PathElementsIter<'a>),
    Arc(<kurbo::Arc as Shape>::PathElementsIter<'a>),
    Line(<Line as Shape>::PathElementsIter<'a>),
    Path(<&'a [PathEl] as Shape>::PathElementsIter<'a>),
}

impl Iterator for PaintShapePathIter<'_> {
    type Item = PathEl;

    fn next(&mut self) -> Option<PathEl> {
        match self {
            PaintShapePathIter::Rect(iter) => iter.next(),
            PaintShapePathIter::RoundedRect(iter) => iter.next(),
            PaintShapePathIter::NonUniformRoundedRect(iter) => iter.next(),
            PaintShapePathIter::Circle(iter) => iter.next(),
            PaintShapePathIter::Ellipse(iter) => iter.next(),
            PaintShapePathIter::Arc(iter) => iter.next(),
            PaintShapePathIter::Line(iter) => iter.next(),
            PaintShapePathIter::Path(iter) => iter.next(),
        }
    }
}

impl Shape for PaintShape<'_> {
    type PathElementsIter<'iter>
        = PaintShapePathIter<'iter>
    where
        Self: 'iter;

    fn path_elements(&self, tolerance: f64) -> PaintShapePathIter<'_> {
        match self {
            PaintShape::Rect(rect) => PaintShapePathIter::Rect(rect.path_elements(tolerance)),
            PaintShape::RoundedRect(rect) => {
                PaintShapePathIter::RoundedRect(rect.path_elements(tolerance))
            }
            PaintShape::NonUniformRoundedRect(rect) => {
                PaintShapePathIter::NonUniformRoundedRect(rect.path_elements(tolerance))
            }
            PaintShape::Circle(circle) => {
                PaintShapePathIter::Circle(circle.path_elements(tolerance))
            }
            PaintShape::Ellipse(ellipse) => {
                PaintShapePathIter::Ellipse(ellipse.path_elements(tolerance))
            }
            PaintShape::Arc(arc) => PaintShapePathIter::Arc(arc.path_elements(tolerance)),
            PaintShape::Line(line) => PaintShapePathIter::Line(line.path_elements(tolerance)),
            PaintShape::Path(path) => PaintShapePathIter::Path((&**path).path_elements(tolerance)),
        }
    }

    fn to_path(&self, tolerance: f64) -> BezPath {
        with_shape!(self, shape => shape.to_path(tolerance))
    }

    fn area(&self) -> f64 {
        with_shape!(self, shape => shape.area())
    }

    fn perimeter(&self, accuracy: f64) -> f64 {
        with_shape!(self, shape => shape.perimeter(accuracy))
    }

    fn winding(&self, pt: Point) -> i32 {
        with_shape!(self, shape => shape.winding(pt))
    }

    fn bounding_box(&self) -> Rect {
        with_shape!(self, shape => shape.bounding_box())
    }

    fn as_line(&self) -> Option<Line> {
        with_shape!(self, shape => shape.as_line())
    }

    fn as_rect(&self) -> Option<Rect> {
        with_shape!(self, shape => shape.as_rect())
    }

    fn as_rounded_rect(&self) -> Option<RoundedRect> {
        with_shape!(self, shape => shape.as_rounded_rect())
    }

    fn as_circle(&self) -> Option<Circle> {
        with_shape!(self, shape => shape.as_circle())
    }

    fn as_path_slice(&self) -> Option<&[PathEl]> {
        match self {
            PaintShape::Path(path) => Some(path),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_path_is_not_copied() {
        let mut path = BezPath::new();
        path.move_to((0.0, 0.0));
        path.line_to((10.0, 10.0));
        let ptr = path.elements().as_ptr();

        let shape = PaintShape::from(path);
        assert!(matches!(shape, PaintShape::Path(Cow::Owned(_))));
        assert_eq!(shape.as_path_slice().unwrap().as_ptr(), ptr);
        assert_eq!(shape.into_bez_path(0.1).elements().as_ptr(), ptr);
    }

    #[test]
    fn shape_impl_matches_inner_shape() {
        let rect = Rect::new(0.0, 0.0, 10.0, 20.0);
        let shape = PaintShape::from(&rect);
        assert_eq!(shape.as_rect(), Some(rect));
        assert_eq!(shape.area(), rect.area());
        assert_eq!(shape.to_path(0.1), rect.to_path(0.1));
        assert_eq!(
            shape.path_elements(0.1).collect::<Vec<_>>(),
            rect.path_elements(0.1).collect::<Vec<_>>()
        );

        let circle = Circle::new((5.0, 5.0), 5.0);
        let shape = PaintShape::from(&circle);
        assert_eq!(shape.as_circle(), Some(circle));
        assert_eq!(shape.as_rect(), None);
        assert_eq!(shape.bounding_box(), circle.bounding_box());
    }

    #[test]
    fn size() {
        assert!(std::mem::size_of::<PaintShape<'_>>() <= 32);
    }
}
