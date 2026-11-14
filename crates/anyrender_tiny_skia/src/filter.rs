//! Filters, evaluated on tiny-skia pixmaps by [`image_filters`].

use anyrender::Filter;
use kurbo::Affine;
use tiny_skia::{IntRect, Pixmap};

pub(crate) use image_filters::blur;

/// Applies `filter` to the `region` of `pixmap` (pixels outside the region become transparent).
/// `transform` maps user space into the pixel space of `pixmap`.
pub(crate) fn apply(filter: &Filter, transform: Affine, region: IntRect, pixmap: &mut Pixmap) {
    // An `ExternalImageSource` can't be resolved to an image, so feImage passes its input
    // through.
    image_filters::apply(filter, transform, region, pixmap, |_, _, _, _| None);
}
