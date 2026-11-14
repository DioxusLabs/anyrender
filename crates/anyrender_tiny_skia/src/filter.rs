//! Filters, evaluated on tiny-skia pixmaps by [`crate::image_filters`].

use anyrender::{Filter, PaintScene};
use kurbo::Affine;
use tiny_skia::{IntRect, Pixmap};

pub(crate) use crate::image_filters::blur;

/// Applies `filter` to the `region` of `pixmap` (pixels outside the region become transparent).
/// `transform` maps user space into the pixel space of `pixmap`.
pub(crate) fn apply(filter: &Filter, transform: Affine, region: IntRect, pixmap: &mut Pixmap) {
    crate::image_filters::apply(filter, transform, region, pixmap, |image, local, w, h| {
        let mut painter = crate::TinySkiaScenePainter::new(w, h);
        painter.append_scene(image.scene.clone(), local);
        Some(painter.pixmap().clone())
    });
}
