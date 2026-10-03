//! Box shadows, as drawn by [`PaintScene::draw_box_shadow`].

use crate::{NonUniformRoundedRect, PaintScene};
use kurbo::{Affine, BezPath, Shape, Vec2};
use peniko::{BlendMode, Color, Compose, Fill, Mix};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Where a box shadow is painted relative to its box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BoxShadowKind {
    /// A shadow cast outside of the box (a CSS outer shadow, cast by the border box).
    Outset {
        /// Only paint the shadow outside of the box, as CSS specifies.
        ///
        /// If the box will be painted opaquely on top of the shadow, setting this to `false`
        /// avoids clipping the shadow, and the faint seam that can appear where the shadow's and
        /// the box's anti-aliased edges meet.
        clip_to_box: bool,
    },
    /// A shadow cast inside of the box (a CSS `inset` shadow, cast by the padding box). It is only
    /// painted inside of the box.
    Inset,
}

const TOLERANCE: f64 = 0.1;

/// The geometry of a box shadow, computed from the arguments of [`PaintScene::draw_box_shadow`].
///
/// This is intended for use by backends.
#[derive(Clone, Debug)]
pub struct BoxShadowGeometry {
    /// The shape of the shadow before it is blurred: the box spread by the shadow's spread (or
    /// for an inset shadow, the box shrunk by the spread, which is the unshadowed "hole"),
    /// translated by the shadow's offset.
    pub shadow: NonUniformRoundedRect,
    /// The area to paint, which contains every pixel that the shadow can affect, but no pixels
    /// that the shadow must not be painted over. Uses the non-zero fill rule.
    pub area: BezPath,
    /// The box that the shadow is cast by.
    pub box_shape: NonUniformRoundedRect,
    pub std_dev: f64,
    pub kind: BoxShadowKind,
}

impl BoxShadowGeometry {
    pub fn new(
        box_shape: &NonUniformRoundedRect,
        offset: Vec2,
        spread: f64,
        std_dev: f64,
        kind: BoxShadowKind,
    ) -> Self {
        let std_dev = std_dev.max(0.0);
        let mut shadow = match kind {
            BoxShadowKind::Outset { .. } => box_shape.spread(spread),
            BoxShadowKind::Inset => box_shape.spread(-spread),
        };
        shadow.rect = shadow.rect + offset;
        let area = match kind {
            BoxShadowKind::Outset { clip_to_box } => {
                // Beyond 3 standard deviations the blurred shadow is effectively transparent.
                let extent = shadow.rect.inflate(3.0 * std_dev, 3.0 * std_dev);
                if clip_to_box && extent.overlaps(box_shape.rect) {
                    // The box, reversed so that it is cut out of the extent.
                    let mut area = extent.union(box_shape.rect).to_path(TOLERANCE);
                    area.extend(box_shape.to_path(TOLERANCE).reverse_subpaths());
                    area
                } else {
                    extent.to_path(TOLERANCE)
                }
            }
            BoxShadowKind::Inset => box_shape.to_path(TOLERANCE),
        };
        Self {
            shadow,
            area,
            box_shape: *box_shape,
            std_dev,
            kind,
        }
    }

    pub fn is_inset(&self) -> bool {
        self.kind == BoxShadowKind::Inset
    }

    /// Draw a blurred inset shadow using compositing layers, for backends that can't paint the
    /// inverse of a blurred rounded rect: fill the box with `brush`, then cut out the blurred hole
    /// with [`Compose::DestOut`], drawn as an unclipped outset shadow.
    pub fn draw_inset_with_layers(
        &self,
        scene: &mut impl PaintScene,
        transform: Affine,
        brush: Color,
    ) {
        // An isolated layer (not a clip, which may not be isolated) so that the `DestOut` layer
        // only cuts the hole out of the shadow, not out of what is underneath it.
        scene.push_layer(
            Fill::NonZero,
            Mix::Normal,
            1.0,
            transform,
            &self.box_shape,
            None,
            None,
        );
        scene.fill(Fill::NonZero, transform, brush, None, &self.box_shape);
        scene.push_layer(
            Fill::NonZero,
            BlendMode::new(Mix::Normal, Compose::DestOut),
            1.0,
            transform,
            &self.box_shape,
            None,
            None,
        );
        scene.draw_box_shadow(
            transform,
            &self.shadow,
            Vec2::ZERO,
            0.0,
            self.std_dev,
            Color::BLACK,
            BoxShadowKind::Outset { clip_to_box: false },
        );
        scene.pop_layer();
        scene.pop_layer();
    }

    /// Whether the shadow must be clipped to [`Self::area`].
    pub fn needs_clip(&self) -> bool {
        !matches!(self.kind, BoxShadowKind::Outset { clip_to_box: false })
    }

    /// The path to fill (inside a clip of [`Self::area`] if [`Self::needs_clip`]) to draw the
    /// shadow when it isn't blurred. Uses the non-zero fill rule.
    pub fn unblurred_path(&self) -> BezPath {
        if self.is_inset() {
            // Everything except the hole.
            let mut path = self
                .box_shape
                .rect
                .union(self.shadow.rect)
                .to_path(TOLERANCE);
            path.extend(self.shadow.to_path(TOLERANCE).reverse_subpaths());
            path
        } else {
            self.shadow.to_path(TOLERANCE)
        }
    }

    /// Draw the shadow when it isn't blurred (when `std_dev` is zero) with a fill and a clip,
    /// keeping the radii of each corner.
    ///
    /// Backends that have a cheaper non-isolated clip than [`PaintScene::push_clip_layer`] can
    /// use [`Self::unblurred_path`] instead: there is only one fill inside of the clip.
    pub fn draw_unblurred(&self, scene: &mut impl PaintScene, transform: Affine, brush: Color) {
        let clip = self.needs_clip();
        if clip {
            scene.push_clip_layer(Fill::NonZero, transform, &self.area);
        }
        scene.fill(
            Fill::NonZero,
            transform,
            brush,
            None,
            &self.unblurred_path(),
        );
        if clip {
            scene.pop_layer();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NonUniformRoundedRectRadii;
    use kurbo::{Point, Rect};

    fn box_shape() -> NonUniformRoundedRect {
        let r = Vec2::new(10.0, 10.0);
        NonUniformRoundedRect::new(
            Rect::new(0.0, 0.0, 100.0, 50.0),
            NonUniformRoundedRectRadii {
                top_left: r,
                top_right: r,
                bottom_right: r,
                bottom_left: r,
            },
        )
    }

    #[test]
    fn outset_area_excludes_box() {
        let kind = BoxShadowKind::Outset { clip_to_box: true };
        let geometry = BoxShadowGeometry::new(&box_shape(), Vec2::new(5.0, 5.0), 2.0, 4.0, kind);
        assert_eq!(geometry.shadow.rect, Rect::new(3.0, 3.0, 107.0, 57.0));
        assert_eq!(geometry.area.winding(Point::new(50.0, 25.0)), 0);
        // Inside the box's bounds, but outside its rounded corner.
        assert_ne!(geometry.area.winding(Point::new(1.0, 1.0)), 0);
        assert_ne!(geometry.area.winding(Point::new(115.0, 60.0)), 0);
        assert_eq!(geometry.area.winding(Point::new(125.0, 60.0)), 0);

        let kind = BoxShadowKind::Outset { clip_to_box: false };
        let geometry = BoxShadowGeometry::new(&box_shape(), Vec2::new(5.0, 5.0), 2.0, 4.0, kind);
        assert_ne!(geometry.area.winding(Point::new(50.0, 25.0)), 0);
    }

    #[test]
    fn inset_hole_is_shrunk_by_spread() {
        let geometry = BoxShadowGeometry::new(
            &box_shape(),
            Vec2::new(5.0, 0.0),
            2.0,
            4.0,
            BoxShadowKind::Inset,
        );
        assert_eq!(geometry.shadow.rect, Rect::new(7.0, 2.0, 103.0, 48.0));
        assert_eq!(geometry.shadow.radii.top_left, Vec2::new(8.0, 8.0));
        assert_eq!(geometry.area.winding(Point::new(1.0, 1.0)), 0);
        assert_ne!(geometry.area.winding(Point::new(50.0, 25.0)), 0);
    }
}
