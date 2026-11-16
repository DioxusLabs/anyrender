use kurbo::Affine;

pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

pub(crate) enum LayerKind {
    Layer,
    Clip,
    /// A filter layer whose input and output are clipped to its filter region.
    FilterRegionLayer,
}

/// Whether a glyph run drawn with the given transforms can use Vello's glyph atlas cache.
///
/// Vello only supports caching glyphs that are drawn upright with a uniform scale. It checks
/// this itself when inserting a glyph into the atlas, but not when looking one up, so a skewed
/// (e.g. synthetic italic) or rotated run would otherwise be drawn using upright cached glyphs.
pub(crate) fn supports_glyph_caching(transform: Affine, glyph_transform: Option<Affine>) -> bool {
    let [a, b, c, d, _, _] = (transform * glyph_transform.unwrap_or_default()).as_coeffs();
    b == 0.0 && c == 0.0 && a == d && a > 0.0
}
