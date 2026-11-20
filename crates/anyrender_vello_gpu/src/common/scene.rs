pub(crate) const DEFAULT_TOLERANCE: f64 = 0.1;

pub(crate) enum LayerKind {
    Layer,
    Clip,
    /// A filter layer whose input and output are clipped to its filter region.
    FilterRegionLayer,
}
