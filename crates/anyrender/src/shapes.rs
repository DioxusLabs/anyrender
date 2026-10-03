//! Additional shapes that backends can recognise and draw efficiently

use kurbo::{
    Arc, ArcAppendIter, Ellipse, PathEl, Point, Rect, RoundedRect, RoundedRectRadii, Shape, Vec2,
};
use std::f64::consts::{FRAC_PI_2, PI};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// The radii of each corner of a [`NonUniformRoundedRect`].
///
/// Each corner is an ellipse with independent horizontal (`x`) and vertical (`y`) radii.
/// A corner is sharp if either of its radii is zero (or negative).
#[derive(Clone, Copy, Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NonUniformRoundedRectRadii {
    pub top_left: Vec2,
    pub top_right: Vec2,
    pub bottom_right: Vec2,
    pub bottom_left: Vec2,
}

impl NonUniformRoundedRectRadii {
    fn corners(&self) -> [Vec2; 4] {
        [
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        ]
    }
}

fn is_sharp(radii: Vec2) -> bool {
    radii.x <= 0.0 || radii.y <= 0.0
}

/// A rectangle whose corners are each rounded by an ellipse with their own radii (like a CSS box
/// with `border-radius`).
///
/// [`Shape::as_rect`] and [`Shape::as_rounded_rect`] return a [`Rect`] or [`RoundedRect`] when the
/// shape can be represented as one, which allows backends to use faster code paths. The radii are
/// used as-is: callers are responsible for scaling down radii that would overlap.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NonUniformRoundedRect {
    pub rect: Rect,
    pub radii: NonUniformRoundedRectRadii,
}

impl NonUniformRoundedRect {
    /// Create a new `NonUniformRoundedRect`. The rect is normalized to have non-negative width and
    /// height.
    pub fn new(rect: Rect, radii: NonUniformRoundedRectRadii) -> Self {
        Self {
            rect: rect.abs(),
            radii,
        }
    }

    /// Grow the shape by `spread` on every side (or shrink it, if `spread` is negative), adjusting
    /// its corner radii as CSS does for the spread of a `box-shadow`.
    ///
    /// Sharp corners stay sharp. Shrinking subtracts the spread from each radius (flooring at
    /// zero). Growing adds the spread to each radius, except that a radius `r` smaller than the
    /// spread grows by `spread * (1 + (r / spread - 1)^3)`, so that small radii stay small. The
    /// resulting radii are then scaled down with [`Self::scale_overlapping_radii`].
    pub fn spread(&self, spread: f64) -> Self {
        let Rect { x0, y0, x1, y1 } = self.rect;
        let grow = |min: f64, max: f64| {
            if max - min + 2.0 * spread >= 0.0 {
                (min - spread, max + spread)
            } else {
                let mid = (min + max) / 2.0;
                (mid, mid)
            }
        };
        let (x0, x1) = grow(x0, x1);
        let (y0, y1) = grow(y0, y1);
        let adjust = |r: f64| {
            if spread < 0.0 {
                (r + spread).max(0.0)
            } else if r < spread {
                r + spread * (1.0 + (r / spread - 1.0).powi(3))
            } else {
                r + spread
            }
        };
        let [top_left, top_right, bottom_right, bottom_left] = self.radii.corners().map(|r| {
            if is_sharp(r) {
                Vec2::ZERO
            } else {
                Vec2::new(adjust(r.x), adjust(r.y))
            }
        });
        Self {
            rect: Rect::new(x0, y0, x1, y1),
            radii: NonUniformRoundedRectRadii {
                top_left,
                top_right,
                bottom_right,
                bottom_left,
            },
        }
        .scale_overlapping_radii()
    }

    /// Scale all of the corner radii down by the same factor so that the radii of adjacent corners
    /// don't add up to more than the length of the side between them, as CSS does.
    pub fn scale_overlapping_radii(&self) -> Self {
        let [tl, tr, br, bl] = self
            .radii
            .corners()
            .map(|r| if is_sharp(r) { Vec2::ZERO } else { r });
        let (width, height) = (self.rect.width().abs(), self.rect.height().abs());
        let factor = [
            (width, tl.x + tr.x),
            (width, bl.x + br.x),
            (height, tl.y + bl.y),
            (height, tr.y + br.y),
        ]
        .into_iter()
        .filter(|&(_, sum)| sum > 0.0)
        .map(|(len, sum)| len / sum)
        .fold(1.0, f64::min);
        Self {
            rect: self.rect,
            radii: NonUniformRoundedRectRadii {
                top_left: tl * factor,
                top_right: tr * factor,
                bottom_right: br * factor,
                bottom_left: bl * factor,
            },
        }
    }

    /// The average of the corners' radii (counting sharp corners as zero), for backends that only
    /// support a single radius.
    pub fn average_radius(&self) -> f64 {
        self.radii
            .corners()
            .into_iter()
            .filter(|r| !is_sharp(*r))
            .map(|r| r.x + r.y)
            .sum::<f64>()
            / 8.0
    }

    /// For each corner, in clockwise order (in a y-down coordinate space): the corner point, the
    /// corner's radii, the center of the corner's ellipse, and the angle at which its arc starts.
    fn corner_geometry(&self) -> [(Point, Vec2, Point, f64); 4] {
        let Rect { x0, y0, x1, y1 } = self.rect;
        let r = &self.radii;
        [
            (
                Point::new(x0, y0),
                r.top_left,
                Point::new(x0 + r.top_left.x, y0 + r.top_left.y),
                PI,
            ),
            (
                Point::new(x1, y0),
                r.top_right,
                Point::new(x1 - r.top_right.x, y0 + r.top_right.y),
                -FRAC_PI_2,
            ),
            (
                Point::new(x1, y1),
                r.bottom_right,
                Point::new(x1 - r.bottom_right.x, y1 - r.bottom_right.y),
                0.0,
            ),
            (
                Point::new(x0, y1),
                r.bottom_left,
                Point::new(x0 + r.bottom_left.x, y1 - r.bottom_left.y),
                FRAC_PI_2,
            ),
        ]
    }
}

impl From<Rect> for NonUniformRoundedRect {
    fn from(rect: Rect) -> Self {
        Self::new(rect, NonUniformRoundedRectRadii::default())
    }
}

impl From<RoundedRect> for NonUniformRoundedRect {
    fn from(rounded_rect: RoundedRect) -> Self {
        let radii = rounded_rect.radii();
        let corner = |r: f64| Vec2::new(r, r);
        Self::new(
            rounded_rect.rect(),
            NonUniformRoundedRectRadii {
                top_left: corner(radii.top_left),
                top_right: corner(radii.top_right),
                bottom_right: corner(radii.bottom_right),
                bottom_left: corner(radii.bottom_left),
            },
        )
    }
}

/// Path elements of a [`NonUniformRoundedRect`].
pub struct NonUniformRoundedRectPathIter {
    corners: [(Point, Vec2, Point, f64); 4],
    tolerance: f64,
    /// Index of the next corner to start (4 = close path, 5 = done).
    next_corner: usize,
    arc: Option<ArcAppendIter>,
}

impl Iterator for NonUniformRoundedRectPathIter {
    type Item = PathEl;

    fn next(&mut self) -> Option<PathEl> {
        if let Some(arc) = &mut self.arc {
            if let Some(el) = arc.next() {
                return Some(el);
            }
            self.arc = None;
        }

        let idx = self.next_corner;
        self.next_corner = (idx + 1).min(5);
        let Some(&(corner, radii, center, start_angle)) = self.corners.get(idx) else {
            return (idx == 4).then_some(PathEl::ClosePath);
        };
        let start = if is_sharp(radii) {
            corner
        } else {
            let arc = Arc::new(center, radii, start_angle, FRAC_PI_2, 0.0);
            self.arc = Some(arc.append_iter(self.tolerance));
            // The (exact) direction of `start_angle` from the center, for each corner.
            const START_DIRECTIONS: [(f64, f64); 4] =
                [(-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0)];
            let (dx, dy) = START_DIRECTIONS[idx];
            center + Vec2::new(radii.x * dx, radii.y * dy)
        };
        Some(if idx == 0 {
            PathEl::MoveTo(start)
        } else {
            PathEl::LineTo(start)
        })
    }
}

impl Shape for NonUniformRoundedRect {
    type PathElementsIter<'iter> = NonUniformRoundedRectPathIter;

    fn path_elements(&self, tolerance: f64) -> Self::PathElementsIter<'_> {
        NonUniformRoundedRectPathIter {
            corners: self.corner_geometry(),
            tolerance,
            next_corner: 0,
            arc: None,
        }
    }

    fn area(&self) -> f64 {
        let corner_area: f64 = self
            .radii
            .corners()
            .into_iter()
            .filter(|r| !is_sharp(*r))
            .map(|r| r.x * r.y * (1.0 - PI / 4.0))
            .sum();
        self.rect.area() - corner_area
    }

    fn perimeter(&self, accuracy: f64) -> f64 {
        // Start with the perimeter of the rect, then replace the straight edges surrounding each
        // rounded corner with a quarter of the corner's ellipse.
        let corners: f64 = self
            .radii
            .corners()
            .into_iter()
            .filter(|r| !is_sharp(*r))
            .map(|r| {
                let quarter_ellipse = if r.x == r.y {
                    FRAC_PI_2 * r.x
                } else {
                    Ellipse::new(Point::ORIGIN, r, 0.0).perimeter(accuracy) / 4.0
                };
                quarter_ellipse - r.x - r.y
            })
            .sum();
        self.rect.abs().perimeter(accuracy) + corners
    }

    fn winding(&self, pt: Point) -> i32 {
        if !self.rect.abs().contains(pt) {
            return 0;
        }
        let outside_corner =
            self.corner_geometry()
                .into_iter()
                .any(|(corner, radii, center, _)| {
                    if is_sharp(radii) {
                        return false;
                    }
                    let in_corner_box = (pt.x - center.x) * (corner.x - center.x) > 0.0
                        && (pt.y - center.y) * (corner.y - center.y) > 0.0;
                    let d = pt - center;
                    in_corner_box && (d.x / radii.x).powi(2) + (d.y / radii.y).powi(2) > 1.0
                });
        if outside_corner { 0 } else { 1 }
    }

    fn bounding_box(&self) -> Rect {
        self.rect.abs()
    }

    fn as_rect(&self) -> Option<Rect> {
        self.radii
            .corners()
            .into_iter()
            .all(is_sharp)
            .then_some(self.rect)
    }

    fn as_rounded_rect(&self) -> Option<RoundedRect> {
        let corners = self.radii.corners();
        // `RoundedRect` clamps radii to half the shortest side, which would change the shape.
        let max_radius = self.rect.width().abs().min(self.rect.height().abs()) / 2.0;
        if !corners
            .iter()
            .all(|r| is_sharp(*r) || (r.x == r.y && r.x <= max_radius))
        {
            return None;
        }
        let [tl, tr, br, bl] = corners.map(|r| if is_sharp(r) { 0.0 } else { r.x });
        Some(RoundedRect::from_rect(
            self.rect,
            RoundedRectRadii::new(tl, tr, br, bl),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn radii(
        tl: (f64, f64),
        tr: (f64, f64),
        br: (f64, f64),
        bl: (f64, f64),
    ) -> NonUniformRoundedRectRadii {
        NonUniformRoundedRectRadii {
            top_left: tl.into(),
            top_right: tr.into(),
            bottom_right: br.into(),
            bottom_left: bl.into(),
        }
    }

    const RECT: Rect = Rect::new(10.0, 20.0, 110.0, 80.0);

    #[test]
    fn new_normalizes_rect() {
        let flipped = Rect::new(RECT.x1, RECT.y1, RECT.x0, RECT.y0);
        let shape = NonUniformRoundedRect::new(flipped, NonUniformRoundedRectRadii::default());
        assert_eq!(shape.rect, RECT);
    }

    #[test]
    fn large_circular_radii_are_not_a_rounded_rect() {
        // RECT is 60 tall, so a 40 radius would be clamped by `RoundedRect`.
        let shape = NonUniformRoundedRect::new(
            RECT,
            radii((40.0, 40.0), (0.0, 0.0), (0.0, 0.0), (20.0, 20.0)),
        );
        assert_eq!(shape.as_rounded_rect(), None);
    }

    #[test]
    fn arc_start_points_are_exact() {
        let shape = NonUniformRoundedRect::new(
            RECT,
            radii((5.0, 3.0), (6.0, 2.0), (10.0, 10.0), (1.0, 8.0)),
        );
        let starts: Vec<Point> = shape
            .path_elements(0.1)
            .filter_map(|el| match el {
                PathEl::MoveTo(p) | PathEl::LineTo(p) => Some(p),
                _ => None,
            })
            .collect();
        assert_eq!(
            starts,
            [
                Point::new(RECT.x0, RECT.y0 + 3.0),
                Point::new(RECT.x1 - 6.0, RECT.y0),
                Point::new(RECT.x1, RECT.y1 - 10.0),
                Point::new(RECT.x0 + 1.0, RECT.y1),
            ]
        );
    }

    #[test]
    fn spread_grows_radii_like_css() {
        let shape = NonUniformRoundedRect::new(
            RECT,
            radii((10.0, 10.0), (2.0, 2.0), (0.0, 0.0), (0.0, 6.0)),
        );
        let spread = shape.spread(4.0);
        assert_eq!(spread.rect, RECT.inflate(4.0, 4.0));
        assert_eq!(spread.radii.top_left, Vec2::new(14.0, 14.0));
        // 2 + 4 * (1 + (2 / 4 - 1)^3)
        assert_eq!(spread.radii.top_right, Vec2::new(5.5, 5.5));
        assert_eq!(spread.radii.bottom_right, Vec2::ZERO);
        assert_eq!(spread.radii.bottom_left, Vec2::ZERO);
    }

    #[test]
    fn negative_spread_shrinks_radii() {
        let shape = NonUniformRoundedRect::new(
            RECT,
            radii((10.0, 10.0), (3.0, 3.0), (0.0, 0.0), (0.0, 0.0)),
        );
        let spread = shape.spread(-4.0);
        assert_eq!(spread.rect, RECT.inflate(-4.0, -4.0));
        assert_eq!(spread.radii.top_left, Vec2::new(6.0, 6.0));
        assert_eq!(spread.radii.top_right, Vec2::ZERO);

        // Shrinking past the size of the rect collapses it to its center.
        let collapsed = shape.spread(-40.0);
        assert_eq!(
            collapsed.rect,
            Rect::new(10.0, 50.0, 110.0, 50.0).inflate(-40.0, 0.0)
        );
        assert_eq!(collapsed.area(), 0.0);
    }

    #[test]
    fn overlapping_radii_are_scaled_down() {
        // The left side is 60 tall, but its radii add up to 120.
        let shape = NonUniformRoundedRect::new(
            RECT,
            radii((10.0, 80.0), (0.0, 0.0), (0.0, 0.0), (10.0, 40.0)),
        )
        .scale_overlapping_radii();
        assert_eq!(shape.radii.top_left, Vec2::new(5.0, 40.0));
        assert_eq!(shape.radii.bottom_left, Vec2::new(5.0, 20.0));
    }

    #[test]
    fn path_matches_shape() {
        for radii in [
            radii((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
            radii((0.0, 0.0), (5.0, 5.0), (0.0, 0.0), (10.0, 4.0)),
            radii((5.0, 3.0), (0.0, 0.0), (10.0, 10.0), (0.0, 0.0)),
            radii((5.0, 3.0), (6.0, 2.0), (10.0, 10.0), (1.0, 8.0)),
        ] {
            let shape = NonUniformRoundedRect::new(RECT, radii);
            let path = shape.to_path(1e-6);
            assert!((path.area() - shape.area()).abs() < 1e-3, "{radii:?}");
            let path_perimeter = shape.to_path(1e-9).perimeter(1e-9);
            assert!(
                (path_perimeter - shape.perimeter(1e-6)).abs() < 1e-6,
                "{radii:?}"
            );
            for x in (5..=115).step_by(2) {
                for y in (15..=85).step_by(2) {
                    let pt = Point::new(x as f64 + 0.25, y as f64 + 0.25);
                    assert_eq!(path.winding(pt), shape.winding(pt), "{radii:?} {pt:?}");
                }
            }
            let bbox = path.bounding_box();
            for (a, b) in [
                (bbox.x0, RECT.x0),
                (bbox.y0, RECT.y0),
                (bbox.x1, RECT.x1),
                (bbox.y1, RECT.y1),
            ] {
                assert!((a - b).abs() < 1e-9, "{radii:?} {bbox:?}");
            }
        }
    }

    #[test]
    fn sharp_corners_are_a_rect() {
        let shape =
            NonUniformRoundedRect::new(RECT, radii((0.0, 0.0), (5.0, 0.0), (0.0, 3.0), (0.0, 0.0)));
        assert_eq!(shape.as_rect(), Some(RECT));
        assert_eq!(shape.area(), RECT.area());
    }

    #[test]
    fn circular_corners_are_a_rounded_rect() {
        let shape = NonUniformRoundedRect::new(
            RECT,
            radii((5.0, 5.0), (0.0, 0.0), (10.0, 10.0), (2.0, 0.0)),
        );
        assert_eq!(shape.as_rect(), None);
        assert_eq!(
            shape.as_rounded_rect(),
            Some(RoundedRect::from_rect(
                RECT,
                RoundedRectRadii::new(5.0, 0.0, 10.0, 0.0)
            ))
        );
    }

    #[test]
    fn elliptical_corners_are_a_path() {
        let shape = NonUniformRoundedRect::new(
            RECT,
            radii((10.0, 5.0), (8.0, 8.0), (20.0, 10.0), (0.0, 0.0)),
        );
        assert_eq!(shape.as_rect(), None);
        assert_eq!(shape.as_rounded_rect(), None);

        let path = shape.to_path(0.01);
        assert!(
            (path.area() - shape.area()).abs() < 0.5,
            "{} vs {}",
            path.area(),
            shape.area()
        );
        let bbox = path.bounding_box();
        for (a, b) in [
            (bbox.x0, RECT.x0),
            (bbox.y0, RECT.y0),
            (bbox.x1, RECT.x1),
            (bbox.y1, RECT.y1),
        ] {
            assert!((a - b).abs() < 1e-9, "{bbox:?}");
        }
        assert_eq!(shape.winding(Point::new(60.0, 50.0)), 1);
        assert_eq!(shape.winding(Point::new(11.0, 21.0)), 0);
        assert_eq!(shape.winding(Point::new(11.0, 79.0)), 1);
    }
}
