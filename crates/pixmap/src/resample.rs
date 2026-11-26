use alloc::vec;
use alloc::vec::Vec;

use crate::{BYTES_PER_PIXEL, Pixmap, PixmapMut, PixmapRef};

impl Pixmap {
    /// Resamples the pixmap to `width × height` using `filter`, returning a new tightly packed
    /// [`Pixmap`].
    ///
    /// If `self` is empty and the target dimensions are non-zero, the returned pixmap is filled
    /// with transparent black (`[0, 0, 0, 0]`).
    pub fn resize(&self, width: u16, height: u16, filter: ResampleFilter) -> Self {
        self.as_ref().resize(width, height, filter)
    }

    /// Resamples the pixmap into `dst` using `filter`, leaving any inter-row stride padding on
    /// `dst` untouched.
    pub fn resize_into(&self, dst: PixmapMut<'_>, filter: ResampleFilter) {
        self.as_ref().resize_into(dst, filter);
    }
}

impl PixmapRef<'_> {
    /// Resamples the pixmap view to `width × height` using `filter`, returning a new tightly
    /// packed [`Pixmap`].
    ///
    /// If `self` is empty and the target dimensions are non-zero, the returned pixmap is filled
    /// with transparent black (`[0, 0, 0, 0]`).
    pub fn resize(&self, width: u16, height: u16, filter: ResampleFilter) -> Pixmap {
        resize(*self, width, height, filter)
    }

    /// Resamples the pixmap view into `dst` using `filter`, leaving any inter-row stride padding
    /// on `dst` untouched.
    pub fn resize_into(&self, mut dst: PixmapMut<'_>, filter: ResampleFilter) {
        resize_into(*self, &mut dst, filter);
    }
}

impl PixmapMut<'_> {
    /// Resamples the pixmap view to `width × height` using `filter`, returning a new tightly
    /// packed [`Pixmap`].
    ///
    /// If `self` is empty and the target dimensions are non-zero, the returned pixmap is filled
    /// with transparent black (`[0, 0, 0, 0]`).
    pub fn resize(&self, width: u16, height: u16, filter: ResampleFilter) -> Pixmap {
        self.as_ref().resize(width, height, filter)
    }

    /// Resamples the pixmap view into `dst` using `filter`, leaving any inter-row stride padding
    /// on `dst` untouched.
    pub fn resize_into(&self, dst: PixmapMut<'_>, filter: ResampleFilter) {
        self.as_ref().resize_into(dst, filter);
    }
}

/// Fixed-point fractional bit width for precomputed convolution weights (`1 << 14 = 16384`).
const WEIGHT_BITS: u32 = 14;
const WEIGHT_ONE: i32 = 1 << WEIGHT_BITS;

/// Extra fractional bits retained in the intermediate `[i16; 4]` horizontal-pass buffer.
const INTER_BITS: u32 = 4;
const H_SHIFT: u32 = WEIGHT_BITS - INTER_BITS; // 10
const H_ROUND: i32 = 1 << (H_SHIFT - 1); // 512

const V_SHIFT: u32 = WEIGHT_BITS + INTER_BITS; // 18
const V_ROUND: i32 = 1 << (V_SHIFT - 1); // 131072

const SINGLE_SHIFT: u32 = WEIGHT_BITS; // 14
const SINGLE_ROUND: i32 = 1 << (SINGLE_SHIFT - 1); // 8192

/// Target memory budget (in bytes) for the streaming horizontal row cache in 2D resampling.
const RING_BUDGET_BYTES: usize = 512 * 1024;

/// Resampling filter used when scaling a pixmap.
///
/// Use one of the built-in presets ([`NEAREST`](Self::NEAREST), [`BOX`](Self::BOX),
/// [`BILINEAR`](Self::BILINEAR), [`CATMULL_ROM`](Self::CATMULL_ROM),
/// [`MITCHELL`](Self::MITCHELL), [`LANCZOS3`](Self::LANCZOS3)), or construct a custom
/// separable filter with [`ResampleFilter::custom`].
///
/// All convolution filters run through the same separable 2-pass fixed-point engine with
/// precomputed per-axis tap tables, scaling kernel support automatically when downscaling to
/// avoid aliasing. Filters that produce negative tap weights automatically clamp output RGB
/// channels to `0..=alpha` to preserve valid premultiplied RGBA8 pixels.
#[derive(Debug, Clone, Copy)]
pub struct ResampleFilter {
    kind: FilterKind,
}

#[derive(Debug, Clone, Copy)]
enum FilterKind {
    Nearest,
    Box,
    Kernel { radius: f64, eval: fn(f64) -> f64 },
}

impl Default for ResampleFilter {
    #[inline]
    fn default() -> Self {
        Self::BILINEAR
    }
}

impl ResampleFilter {
    /// Nearest-neighbor sampling (1-tap, no interpolation). Fastest; preserves hard pixel edges.
    pub const NEAREST: Self = Self {
        kind: FilterKind::Nearest,
    };

    /// Exact pixel-area (box) average. Clean downscaling with no ringing; reduces to nearest
    /// pixel coverage when upscaling at integer factors.
    pub const BOX: Self = Self {
        kind: FilterKind::Box,
    };

    /// Linear (triangle) filter, with support scaled when downscaling to prevent aliasing.
    pub const BILINEAR: Self = Self::from_kernel(1.0, bilinear_kernel);

    /// Catmull-Rom bicubic filter (`B = 0, C = 0.5`). Sharp with slight overshoot on high-contrast
    /// edges.
    pub const CATMULL_ROM: Self = Self::from_kernel(2.0, catmull_rom_kernel);

    /// Mitchell-Netravali bicubic filter (`B = 1/3, C = 1/3`). Smoother bicubic with less ringing
    /// than [`ResampleFilter::CATMULL_ROM`].
    pub const MITCHELL: Self = Self::from_kernel(2.0, mitchell_kernel);

    /// 3-lobe Lanczos windowed sinc filter. Sharpest downscaling for photographic content, with
    /// stronger ringing near sharp edges.
    pub const LANCZOS3: Self = Self::from_kernel(3.0, lanczos3_kernel);

    #[inline]
    const fn from_kernel(radius: f64, eval: fn(f64) -> f64) -> Self {
        Self {
            kind: FilterKind::Kernel { radius, eval },
        }
    }

    /// Creates a custom separable resampling filter from a 1D convolution kernel function.
    ///
    /// - `radius` is the positive support radius in source pixels (scaled by `src / dst` when
    ///   downscaling).
    /// - `eval` evaluates the kernel at distance `x` in `[-radius, radius]`. Tap weights are
    ///   normalized to sum to 1 per output pixel, and premultiplied RGB clamping is enabled
    ///   automatically whenever any tap weight is negative.
    ///
    /// # Panics
    ///
    /// Panics if `radius` is not finite or not strictly positive.
    pub const fn custom(radius: f64, eval: fn(f64) -> f64) -> Self {
        assert!(
            radius > 0.0 && radius < f64::INFINITY,
            "resample filter radius must be finite and positive"
        );
        Self::from_kernel(radius, eval)
    }
}

#[inline]
fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn bilinear_kernel(x: f64) -> f64 {
    let ax = abs_f64(x);
    if ax < 1.0 { 1.0 - ax } else { 0.0 }
}

fn catmull_rom_kernel(x: f64) -> f64 {
    let ax = abs_f64(x);
    if ax < 1.0 {
        (1.5 * ax - 2.5) * ax * ax + 1.0
    } else if ax < 2.0 {
        ((-0.5 * ax + 2.5) * ax - 4.0) * ax + 2.0
    } else {
        0.0
    }
}

fn mitchell_kernel(x: f64) -> f64 {
    // Mitchell-Netravali with B = 1/3, C = 1/3.
    let ax = abs_f64(x);
    if ax < 1.0 {
        ((7.0 * ax - 12.0) * ax * ax + 16.0 / 3.0) * (1.0 / 6.0)
    } else if ax < 2.0 {
        (((-(7.0 / 3.0) * ax + 12.0) * ax - 20.0) * ax + 32.0 / 3.0) * (1.0 / 6.0)
    } else {
        0.0
    }
}

fn lanczos3_kernel(x: f64) -> f64 {
    let ax = abs_f64(x);
    if ax < 1e-9 {
        1.0
    } else if ax < 3.0 {
        sinc_pi(ax) * sinc_pi(ax * (1.0 / 3.0))
    } else {
        0.0
    }
}

#[inline]
fn floor_f64(x: f64) -> i64 {
    let i = x as i64;
    if (i as f64) > x { i - 1 } else { i }
}

#[inline]
fn ceil_f64(x: f64) -> i64 {
    let i = x as i64;
    if (i as f64) < x { i + 1 } else { i }
}

#[inline]
fn round_i32(x: f64) -> i32 {
    if x >= 0.0 {
        (x + 0.5) as i32
    } else {
        (x - 0.5) as i32
    }
}

/// Accurate `#![no_std]` approximation of `sin(pi * x)` for `x >= 0.0` (max error < 1e-9).
#[inline]
fn sin_pi(x: f64) -> f64 {
    let k = (x * 0.5) as i64;
    let mut u = x - 2.0 * (k as f64);
    if u < 0.0 {
        u += 2.0;
    }
    let sign = if u >= 1.0 {
        u -= 1.0;
        -1.0
    } else {
        1.0
    };
    if u > 0.5 {
        u = 1.0 - u;
    }
    let t = core::f64::consts::PI * u;
    let t2 = t * t;
    let poly = 1.0
        - (t2 / 6.0)
            * (1.0
                - (t2 / 20.0)
                    * (1.0
                        - (t2 / 42.0)
                            * (1.0 - (t2 / 72.0) * (1.0 - (t2 / 110.0) * (1.0 - t2 / 156.0)))));
    sign * t * poly
}

#[inline]
fn sinc_pi(x: f64) -> f64 {
    let px = core::f64::consts::PI * x;
    sin_pi(x) / px
}

#[inline]
fn nearest_index(dst_idx: usize, src_len: usize, dst_len: usize) -> usize {
    let idx = ((2 * dst_idx as u64 + 1) * src_len as u64) / (2 * dst_len as u64);
    (idx as usize).min(src_len - 1)
}

#[derive(Clone, Copy)]
struct TapSpan {
    start: u16,
    len: u16,
    offset: u32,
}

struct TapTable {
    spans: Vec<TapSpan>,
    weights: Vec<i16>,
    max_taps: usize,
    has_negative: bool,
}

impl TapTable {
    fn build(src_len: u16, dst_len: u16, filter: ResampleFilter) -> Self {
        match filter.kind {
            FilterKind::Nearest => unreachable!("nearest filter uses the direct fast path"),
            FilterKind::Box => Self::build_box(src_len, dst_len),
            FilterKind::Kernel { radius, eval } => {
                Self::build_kernel(src_len, dst_len, radius, eval)
            }
        }
    }

    fn build_box(src_len: u16, dst_len: u16) -> Self {
        debug_assert!(src_len > 0 && dst_len > 0);
        let src_n = src_len as usize;
        let dst_n = dst_len as usize;
        let ratio = src_n as f64 / dst_n as f64;
        let est_taps = ceil_f64(ratio) as usize + 2;

        let mut spans = Vec::with_capacity(dst_n);
        let mut weights = Vec::with_capacity(dst_n.saturating_mul(est_taps));
        let mut raw = Vec::with_capacity(est_taps);
        let mut max_taps = 1usize;
        let mut has_negative = false;

        for d in 0..dst_n {
            raw.clear();
            // Exact pixel-area overlap between output pixel [left, right] and source pixels.
            let left = d as f64 * ratio;
            let right = (d + 1) as f64 * ratio;
            let lo = floor_f64(left).clamp(0, (src_n - 1) as i64) as usize;
            let hi = (ceil_f64(right) - 1).clamp(lo as i64, (src_n - 1) as i64) as usize;
            for s in lo..=hi {
                let s0 = s as f64;
                let s1 = (s + 1) as f64;
                let overlap = (right.min(s1) - left.max(s0)).max(0.0);
                raw.push(overlap);
            }
            quantize_and_push_taps(
                d,
                src_n,
                dst_n,
                lo,
                &raw,
                &mut spans,
                &mut weights,
                &mut max_taps,
                &mut has_negative,
            );
        }

        Self {
            spans,
            weights,
            max_taps,
            has_negative,
        }
    }

    fn build_kernel(src_len: u16, dst_len: u16, radius: f64, eval: fn(f64) -> f64) -> Self {
        debug_assert!(src_len > 0 && dst_len > 0);
        let src_n = src_len as usize;
        let dst_n = dst_len as usize;
        let ratio = src_n as f64 / dst_n as f64;
        let scale = if ratio > 1.0 { ratio } else { 1.0 };
        let support = radius * scale;
        let inv_scale = 1.0 / scale;

        let est_taps = ceil_f64(2.0 * support) as usize + 2;
        let mut spans = Vec::with_capacity(dst_n);
        let mut weights = Vec::with_capacity(dst_n.saturating_mul(est_taps));
        let mut raw = Vec::with_capacity(est_taps);
        let mut max_taps = 1usize;
        let mut has_negative = false;

        for d in 0..dst_n {
            raw.clear();
            let center = (d as f64 + 0.5) * ratio - 0.5;
            let lo = ceil_f64(center - support).clamp(0, (src_n - 1) as i64) as usize;
            let hi = floor_f64(center + support).clamp(0, (src_n - 1) as i64) as usize;
            let first = if lo <= hi {
                for s in lo..=hi {
                    let dist = (s as f64 - center) * inv_scale;
                    raw.push(eval(dist));
                }
                lo
            } else {
                let fallback = nearest_index(d, src_n, dst_n);
                raw.push(1.0);
                fallback
            };
            quantize_and_push_taps(
                d,
                src_n,
                dst_n,
                first,
                &raw,
                &mut spans,
                &mut weights,
                &mut max_taps,
                &mut has_negative,
            );
        }

        Self {
            spans,
            weights,
            max_taps,
            has_negative,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_and_push_taps(
    d: usize,
    src_n: usize,
    dst_n: usize,
    first: usize,
    raw: &[f64],
    spans: &mut Vec<TapSpan>,
    weights: &mut Vec<i16>,
    max_taps: &mut usize,
    has_negative: &mut bool,
) {
    let mut sum = 0.0f64;
    let mut max_abs = 0.0f64;
    for &w in raw {
        sum += w;
        let aw = abs_f64(w);
        if aw > max_abs {
            max_abs = aw;
        }
    }

    let offset = weights.len() as u32;
    let abs_sum = abs_f64(sum);
    let max_safe_weight = (i16::MAX - 1) as f64;
    if !(abs_sum > 1e-12 && sum.is_finite() && max_abs.is_finite())
        || max_abs * (WEIGHT_ONE as f64) > max_safe_weight * abs_sum
    {
        let fallback = nearest_index(d, src_n, dst_n);
        weights.push(WEIGHT_ONE as i16);
        spans.push(TapSpan {
            start: fallback as u16,
            len: 1,
            offset,
        });
        return;
    }

    let norm = (WEIGHT_ONE as f64) / sum;
    let mut err = 0.0f64;
    let mut q_sum = 0i32;
    let mut best_rel = 0usize;
    let mut best_val = i32::MIN;
    let tap_count = raw.len();
    for (i, &w) in raw.iter().enumerate() {
        // Distribute quantization error across taps so heavy downscales stay uniform
        // and `|WEIGHT_ONE - q_sum| <= 1`.
        let scaled = w * norm + err;
        let q = round_i32(scaled);
        err = scaled - (q as f64);
        q_sum += q;
        if q > best_val {
            best_val = q;
            best_rel = i;
        }
        weights.push(q as i16);
    }

    // Adjust the dominant tap so quantized weights sum to WEIGHT_ONE exactly, preserving
    // constant-color inputs without rounding drift.
    let diff = WEIGHT_ONE - q_sum;
    let base = offset as usize;
    weights[base + best_rel] = (i32::from(weights[base + best_rel]) + diff) as i16;

    // Trim zero-weight edge taps to avoid unnecessary MACs.
    let mut trim_start = 0usize;
    let mut trim_end = tap_count;
    while trim_end - trim_start > 1 && weights[base + trim_start] == 0 {
        trim_start += 1;
    }
    while trim_end - trim_start > 1 && weights[base + trim_end - 1] == 0 {
        trim_end -= 1;
    }

    if trim_start > 0 {
        let keep = trim_end - trim_start;
        weights.copy_within(base + trim_start..base + trim_end, base);
        weights.truncate(base + keep);
    } else if trim_end < tap_count {
        weights.truncate(base + trim_end);
    }

    let kept = trim_end - trim_start;
    for &q in &weights[base..base + kept] {
        if q < 0 {
            *has_negative = true;
            break;
        }
    }
    if kept > *max_taps {
        *max_taps = kept;
    }
    spans.push(TapSpan {
        start: (first + trim_start) as u16,
        len: kept as u16,
        offset,
    });
}

#[inline(always)]
fn finalize_pixel<const CLAMP_PREMUL: bool, const SHIFT: u32, const ROUND: i32>(
    r: i32,
    g: i32,
    b: i32,
    a: i32,
) -> [u8; 4] {
    if CLAMP_PREMUL {
        let a = ((a + ROUND) >> SHIFT).clamp(0, 255);
        let r = ((r + ROUND) >> SHIFT).clamp(0, a);
        let g = ((g + ROUND) >> SHIFT).clamp(0, a);
        let b = ((b + ROUND) >> SHIFT).clamp(0, a);
        [r as u8, g as u8, b as u8, a as u8]
    } else {
        let a = ((a + ROUND) >> SHIFT) as u8;
        let r = ((r + ROUND) >> SHIFT) as u8;
        let g = ((g + ROUND) >> SHIFT) as u8;
        let b = ((b + ROUND) >> SHIFT) as u8;
        [r, g, b, a]
    }
}

fn resample_nearest(src: PixmapRef<'_>, dst: &mut PixmapMut<'_>) {
    let src_w = src.width() as usize;
    let src_h = src.height() as usize;
    let dst_w = dst.width() as usize;
    let dst_h = dst.height() as usize;

    if src_w == dst_w {
        for (dy, dst_row) in dst.rows_mut().enumerate() {
            let sy = nearest_index(dy, src_h, dst_h);
            dst_row.copy_from_slice(src.row(sy as u16));
        }
        return;
    }

    let x_map: Vec<usize> = (0..dst_w)
        .map(|dx| nearest_index(dx, src_w, dst_w))
        .collect();

    for (dy, dst_row) in dst.rows_mut().enumerate() {
        let sy = nearest_index(dy, src_h, dst_h);
        let (src_px, src_rem) = src.row(sy as u16).as_chunks::<BYTES_PER_PIXEL>();
        let (dst_px, dst_rem) = dst_row.as_chunks_mut::<BYTES_PER_PIXEL>();
        debug_assert!(src_rem.is_empty() && dst_rem.is_empty());
        for (d, &sx) in dst_px.iter_mut().zip(&x_map) {
            *d = src_px[sx];
        }
    }
}

/// Resamples a single source row horizontally into a Q4 `[i16; 4]` intermediate buffer.
fn resample_row_h_inter(src_row: &[u8], out: &mut [[i16; 4]], h_table: &TapTable) {
    let (src_px, src_rem) = src_row.as_chunks::<BYTES_PER_PIXEL>();
    debug_assert!(src_rem.is_empty());
    debug_assert_eq!(out.len(), h_table.spans.len());

    for (dst_px, span) in out.iter_mut().zip(&h_table.spans) {
        let start = span.start as usize;
        let len = span.len as usize;
        let offset = span.offset as usize;
        let weights = &h_table.weights[offset..offset + len];
        let taps = &src_px[start..start + len];
        let mut r = 0i32;
        let mut g = 0i32;
        let mut b = 0i32;
        let mut a = 0i32;
        for (&px, &w) in taps.iter().zip(weights) {
            let w = i32::from(w);
            r += i32::from(px[0]) * w;
            g += i32::from(px[1]) * w;
            b += i32::from(px[2]) * w;
            a += i32::from(px[3]) * w;
        }
        *dst_px = [
            ((r + H_ROUND) >> H_SHIFT) as i16,
            ((g + H_ROUND) >> H_SHIFT) as i16,
            ((b + H_ROUND) >> H_SHIFT) as i16,
            ((a + H_ROUND) >> H_SHIFT) as i16,
        ];
    }
}

/// 1D horizontal-only resampling (when `src.height() == dst.height()`).
fn resample_h_only<const CLAMP_PREMUL: bool>(
    src: PixmapRef<'_>,
    dst: &mut PixmapMut<'_>,
    h_table: &TapTable,
) {
    for (dst_row, src_row) in dst.rows_mut().zip(src.rows()) {
        let (src_px, src_rem) = src_row.as_chunks::<BYTES_PER_PIXEL>();
        let (dst_px, dst_rem) = dst_row.as_chunks_mut::<BYTES_PER_PIXEL>();
        debug_assert!(src_rem.is_empty() && dst_rem.is_empty());
        for (d, span) in dst_px.iter_mut().zip(&h_table.spans) {
            let start = span.start as usize;
            let len = span.len as usize;
            let offset = span.offset as usize;
            let weights = &h_table.weights[offset..offset + len];
            let taps = &src_px[start..start + len];
            let mut r = 0i32;
            let mut g = 0i32;
            let mut b = 0i32;
            let mut a = 0i32;
            for (&px, &w) in taps.iter().zip(weights) {
                let w = i32::from(w);
                r += i32::from(px[0]) * w;
                g += i32::from(px[1]) * w;
                b += i32::from(px[2]) * w;
                a += i32::from(px[3]) * w;
            }
            *d = finalize_pixel::<CLAMP_PREMUL, SINGLE_SHIFT, SINGLE_ROUND>(r, g, b, a);
        }
    }
}

/// 1D vertical-only resampling (when `src.width() == dst.width()`).
fn resample_v_only<const CLAMP_PREMUL: bool>(
    src: PixmapRef<'_>,
    dst: &mut PixmapMut<'_>,
    v_table: &TapTable,
) {
    let width = dst.width() as usize;
    let mut acc = vec![[0i32; 4]; width];

    for (dst_row, span) in dst.rows_mut().zip(&v_table.spans) {
        acc.fill([0; 4]);
        let start = span.start as usize;
        let len = span.len as usize;
        let offset = span.offset as usize;
        let weights = &v_table.weights[offset..offset + len];

        for (i, &w) in weights.iter().enumerate() {
            let w = i32::from(w);
            let (src_px, src_rem) = src.row((start + i) as u16).as_chunks::<BYTES_PER_PIXEL>();
            debug_assert!(src_rem.is_empty());
            for (acc_px, &px) in acc.iter_mut().zip(src_px) {
                acc_px[0] += i32::from(px[0]) * w;
                acc_px[1] += i32::from(px[1]) * w;
                acc_px[2] += i32::from(px[2]) * w;
                acc_px[3] += i32::from(px[3]) * w;
            }
        }

        let (dst_px, dst_rem) = dst_row.as_chunks_mut::<BYTES_PER_PIXEL>();
        debug_assert!(dst_rem.is_empty());
        for (d, &[r, g, b, a]) in dst_px.iter_mut().zip(&acc) {
            *d = finalize_pixel::<CLAMP_PREMUL, SINGLE_SHIFT, SINGLE_ROUND>(r, g, b, a);
        }
    }
}

/// Separable 2-pass resampling with a bounded direct-mapped ring cache of Q4 `[i16; 4]` rows.
fn resample_2d<const CLAMP_PREMUL: bool>(
    src: PixmapRef<'_>,
    dst: &mut PixmapMut<'_>,
    h_table: &TapTable,
    v_table: &TapTable,
) {
    let dst_w = dst.width() as usize;
    let row_bytes_inter = dst_w
        .saturating_mul(core::mem::size_of::<[i16; 4]>())
        .max(1);
    let budget_rows = (RING_BUDGET_BYTES / row_bytes_inter).clamp(2, 128);
    let ring_len = v_table.max_taps.clamp(1, budget_rows);

    let mut ring = vec![[0i16; 4]; ring_len * dst_w];
    let mut cached_y = vec![usize::MAX; ring_len];
    let mut acc = vec![[0i32; 4]; dst_w];

    for (dst_row, span) in dst.rows_mut().zip(&v_table.spans) {
        acc.fill([0; 4]);
        let start = span.start as usize;
        let len = span.len as usize;
        let offset = span.offset as usize;
        let weights = &v_table.weights[offset..offset + len];

        for (i, &w) in weights.iter().enumerate() {
            let sy = start + i;
            let slot = sy % ring_len;
            let row_start = slot * dst_w;
            let h_row = &mut ring[row_start..row_start + dst_w];
            if cached_y[slot] != sy {
                resample_row_h_inter(src.row(sy as u16), h_row, h_table);
                cached_y[slot] = sy;
            }

            let w = i32::from(w);
            for (acc_px, &px) in acc.iter_mut().zip(h_row.iter()) {
                acc_px[0] += i32::from(px[0]) * w;
                acc_px[1] += i32::from(px[1]) * w;
                acc_px[2] += i32::from(px[2]) * w;
                acc_px[3] += i32::from(px[3]) * w;
            }
        }

        let (dst_px, dst_rem) = dst_row.as_chunks_mut::<BYTES_PER_PIXEL>();
        debug_assert!(dst_rem.is_empty());
        for (d, &[r, g, b, a]) in dst_px.iter_mut().zip(&acc) {
            *d = finalize_pixel::<CLAMP_PREMUL, V_SHIFT, V_ROUND>(r, g, b, a);
        }
    }
}

/// Resamples `src` into `dst` using `filter`, preserving any inter-row stride padding on `dst`.
fn resize_into(src: PixmapRef<'_>, dst: &mut PixmapMut<'_>, filter: ResampleFilter) {
    let dst_w = dst.width();
    let dst_h = dst.height();
    if dst_w == 0 || dst_h == 0 {
        return;
    }

    let src_w = src.width();
    let src_h = src.height();
    if src_w == 0 || src_h == 0 {
        dst.clear();
        return;
    }

    if src_w == dst_w && src_h == dst_h {
        dst.copy_from(src, 0, 0);
        return;
    }

    if matches!(filter.kind, FilterKind::Nearest) {
        resample_nearest(src, dst);
        return;
    }

    if src_w == dst_w {
        let v_table = TapTable::build(src_h, dst_h, filter);
        if v_table.has_negative {
            resample_v_only::<true>(src, dst, &v_table);
        } else {
            resample_v_only::<false>(src, dst, &v_table);
        }
        return;
    }

    if src_h == dst_h {
        let h_table = TapTable::build(src_w, dst_w, filter);
        if h_table.has_negative {
            resample_h_only::<true>(src, dst, &h_table);
        } else {
            resample_h_only::<false>(src, dst, &h_table);
        }
        return;
    }

    let h_table = TapTable::build(src_w, dst_w, filter);
    let v_table = TapTable::build(src_h, dst_h, filter);
    if h_table.has_negative || v_table.has_negative {
        resample_2d::<true>(src, dst, &h_table, &v_table);
    } else {
        resample_2d::<false>(src, dst, &h_table, &v_table);
    }
}

/// Resamples `src` into a new tightly packed [`Pixmap`] of size `width × height` using `filter`.
fn resize(src: PixmapRef<'_>, width: u16, height: u16, filter: ResampleFilter) -> Pixmap {
    let mut out = Pixmap::new(width, height);
    resize_into(src, &mut out.as_mut(), filter);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const ALL_FILTERS: [ResampleFilter; 6] = [
        ResampleFilter::NEAREST,
        ResampleFilter::BOX,
        ResampleFilter::BILINEAR,
        ResampleFilter::CATMULL_ROM,
        ResampleFilter::MITCHELL,
        ResampleFilter::LANCZOS3,
    ];

    #[test]
    fn sin_pi_matches_known_exact_values() {
        let cases = [
            (0.0, 0.0),
            (1.0 / 6.0, 0.5),
            (0.25, core::f64::consts::FRAC_1_SQRT_2),
            (0.5, 1.0),
            (1.0, 0.0),
            (1.5, -1.0),
            (2.0, 0.0),
            (2.5, 1.0),
            (3.0, 0.0),
        ];
        for (x, expected) in cases {
            let got = sin_pi(x);
            assert!(
                (got - expected).abs() < 1e-8,
                "sin_pi({x}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn solid_color_is_preserved_exactly_across_all_filters_and_scales() {
        let color = [47, 113, 180, 200];
        let mut src = Pixmap::new(17, 13);
        src.fill(color);

        for filter in ALL_FILTERS {
            for &(dw, dh) in &[(17, 13), (34, 13), (17, 26), (5, 7), (41, 29), (1, 1)] {
                let out = src.resize(dw, dh, filter);
                assert_eq!((out.width(), out.height()), (dw, dh));
                for y in 0..dh {
                    for x in 0..dw {
                        assert_eq!(
                            out.pixel(x, y),
                            color,
                            "solid color drift for {filter:?} at ({x},{y}) -> {dw}x{dh}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nearest_neighbor_upscale_and_downscale() {
        let src = Pixmap::from_parts(
            vec![
                10, 0, 0, 255, 20, 0, 0, 255, // row 0
                30, 0, 0, 255, 40, 0, 0, 255, // row 1
            ],
            2,
            2,
        );

        let up = src.resize(4, 4, ResampleFilter::NEAREST);
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(up.pixel(x, y), src.pixel(x / 2, y / 2));
            }
        }

        // 1D vertical-only nearest path (same width).
        let up_v = src.resize(2, 4, ResampleFilter::NEAREST);
        for y in 0..4 {
            for x in 0..2 {
                assert_eq!(up_v.pixel(x, y), src.pixel(x, y / 2));
            }
        }

        // Downscale 4x4 back to 2x2.
        let down = up.resize(2, 2, ResampleFilter::NEAREST);
        assert_eq!(down, src);
    }

    #[test]
    fn box_filter_computes_exact_area_averages() {
        // 4x2 image where each 2x2 block has known average.
        let mut src = Pixmap::new(4, 2);
        // Left 2x2: [0, 40, 80, 120] -> average 60
        src.set_pixel(0, 0, [0, 0, 0, 255]);
        src.set_pixel(1, 0, [40, 0, 0, 255]);
        src.set_pixel(0, 1, [80, 0, 0, 255]);
        src.set_pixel(1, 1, [120, 0, 0, 255]);
        // Right 2x2: [100, 100, 200, 200] -> average 150
        src.set_pixel(2, 0, [100, 20, 40, 200]);
        src.set_pixel(3, 0, [100, 20, 40, 200]);
        src.set_pixel(2, 1, [200, 60, 80, 200]);
        src.set_pixel(3, 1, [200, 60, 80, 200]);

        let down = src.resize(2, 1, ResampleFilter::BOX);
        assert_eq!(down.pixel(0, 0), [60, 0, 0, 255]);
        assert_eq!(down.pixel(1, 0), [150, 40, 60, 200]);

        // Non-integer downscale 3x1 -> 2x1: left gets 2/3*0 + 1/3*90 = 30; right gets 1/3*90 + 2/3*180 = 150.
        let src3 = Pixmap::from_parts(vec![0, 0, 0, 255, 90, 0, 0, 255, 180, 0, 0, 255], 3, 1);
        let down3 = src3.resize(2, 1, ResampleFilter::BOX);
        assert_eq!(down3.pixel(0, 0), [30, 0, 0, 255]);
        assert_eq!(down3.pixel(1, 0), [150, 0, 0, 255]);

        // Heavy downscale (500x1 -> 1x1) with error-diffused weights stays accurate even when
        // 16384 is not divisible by the tap count.
        let mut wide = Pixmap::new(500, 1);
        for x in 0..500u16 {
            let v = if x < 250 { 0 } else { 200 };
            wide.set_pixel(x, 0, [v, 0, 0, 255]);
        }
        let collapsed = wide.resize(1, 1, ResampleFilter::BOX);
        assert_eq!(collapsed.pixel(0, 0), [100, 0, 0, 255]);
    }

    #[test]
    fn bilinear_interpolates_linear_ramp_accurately() {
        // 2x1: [0, 0, 0, 255] and [200, 100, 40, 255] -> 4x1 centers at -0.25, 0.25, 0.75, 1.25.
        let src = Pixmap::from_parts(vec![0, 0, 0, 255, 200, 100, 40, 255], 2, 1);
        let up = src.resize(4, 1, ResampleFilter::BILINEAR);
        assert_eq!(up.pixel(0, 0), [0, 0, 0, 255]);
        assert_eq!(up.pixel(1, 0), [50, 25, 10, 255]);
        assert_eq!(up.pixel(2, 0), [150, 75, 30, 255]);
        assert_eq!(up.pixel(3, 0), [200, 100, 40, 255]);

        // 2D bilinear on 2x2 corners -> 4x4 interior center average.
        let mut src2 = Pixmap::new(2, 2);
        src2.set_pixel(0, 0, [0, 0, 0, 255]);
        src2.set_pixel(1, 0, [200, 0, 0, 255]);
        src2.set_pixel(0, 1, [0, 200, 0, 255]);
        src2.set_pixel(1, 1, [200, 200, 0, 255]);
        let up2 = src2.resize(4, 4, ResampleFilter::BILINEAR);
        assert_eq!(up2.pixel(0, 0), [0, 0, 0, 255]);
        assert_eq!(up2.pixel(3, 3), [200, 200, 0, 255]);
        assert_eq!(up2.pixel(1, 1), [50, 50, 0, 255]);
        assert_eq!(up2.pixel(2, 1), [150, 50, 0, 255]);
        assert_eq!(up2.pixel(1, 2), [50, 150, 0, 255]);
        assert_eq!(up2.pixel(2, 2), [150, 150, 0, 255]);
    }

    #[test]
    fn negative_lobe_filters_preserve_valid_premultiplied_invariants_and_match_f64_reference() {
        // High-contrast step edge with alpha transitions to trigger maximum overshoot/undershoot.
        let mut src = Pixmap::new(12, 10);
        for y in 0..10 {
            for x in 0..12 {
                let px = if x < 6 && y < 5 {
                    [255, 255, 255, 255]
                } else if x >= 6 && y < 5 {
                    [0, 0, 0, 0]
                } else if x < 6 {
                    [128, 0, 64, 128]
                } else {
                    [0, 200, 0, 200]
                };
                src.set_pixel(x, y, px);
            }
        }

        for filter in ALL_FILTERS {
            for &(dw, dh) in &[(25, 19), (5, 4), (12, 5), (5, 10)] {
                let out = src.resize(dw, dh, filter);
                for y in 0..dh {
                    for x in 0..dw {
                        let [r, g, b, a] = out.pixel(x, y);
                        assert!(
                            r <= a && g <= a && b <= a,
                            "invalid premul pixel {:?} for {filter:?} at ({x},{y}) -> {dw}x{dh}",
                            [r, g, b, a]
                        );
                    }
                }
            }
        }

        // Verify CatmullRom, Mitchell, and Lanczos3 closely match an f64 reference on 2D resize.
        for filter in [
            ResampleFilter::CATMULL_ROM,
            ResampleFilter::MITCHELL,
            ResampleFilter::LANCZOS3,
        ] {
            let got = src.resize(9, 7, filter);
            let expected = reference_resize_f64(src.as_ref(), 9, 7, filter);
            for y in 0..7 {
                for x in 0..9 {
                    let g = got.pixel(x, y);
                    let e = expected.pixel(x, y);
                    for c in 0..4 {
                        assert!(
                            g[c].abs_diff(e[c]) <= 1,
                            "{filter:?} mismatch at ({x},{y}) ch {c}: got {g:?}, expected {e:?}"
                        );
                    }
                }
            }
        }
    }

    fn reference_resize_f64(
        src: PixmapRef<'_>,
        dst_w: u16,
        dst_h: u16,
        filter: ResampleFilter,
    ) -> Pixmap {
        let FilterKind::Kernel { radius, eval } = filter.kind else {
            unreachable!("reference_resize_f64 is only for kernel filters");
        };
        let sw = src.width() as usize;
        let sh = src.height() as usize;
        let dw = dst_w as usize;
        let dh = dst_h as usize;

        let build_f64 = |src_n: usize, dst_n: usize| -> Vec<Vec<(usize, f64)>> {
            let ratio = src_n as f64 / dst_n as f64;
            let scale = if ratio > 1.0 { ratio } else { 1.0 };
            let support = radius * scale;
            (0..dst_n)
                .map(|d| {
                    let center = (d as f64 + 0.5) * ratio - 0.5;
                    let lo = ceil_f64(center - support).clamp(0, (src_n - 1) as i64) as usize;
                    let hi = floor_f64(center + support).clamp(0, (src_n - 1) as i64) as usize;
                    let mut taps: Vec<(usize, f64)> = (lo..=hi)
                        .map(|s| (s, eval((s as f64 - center) / scale)))
                        .collect();
                    let sum: f64 = taps.iter().map(|&(_, w)| w).sum();
                    for (_, w) in &mut taps {
                        *w /= sum;
                    }
                    taps
                })
                .collect()
        };

        let h_taps = build_f64(sw, dw);
        let v_taps = build_f64(sh, dh);
        let mut inter = vec![[0.0f64; 4]; sh * dw];
        for sy in 0..sh {
            for (dx, taps) in h_taps.iter().enumerate() {
                let mut acc = [0.0f64; 4];
                for &(sx, w) in taps {
                    let px = src.pixel(sx as u16, sy as u16);
                    for c in 0..4 {
                        acc[c] += f64::from(px[c]) * w;
                    }
                }
                inter[sy * dw + dx] = acc;
            }
        }

        let mut out = Pixmap::new(dst_w, dst_h);
        for (dy, taps) in v_taps.iter().enumerate() {
            for dx in 0..dw {
                let mut acc = [0.0f64; 4];
                for &(sy, w) in taps {
                    let px = inter[sy * dw + dx];
                    for c in 0..4 {
                        acc[c] += px[c] * w;
                    }
                }
                let a = round_i32(acc[3]).clamp(0, 255);
                let r = round_i32(acc[0]).clamp(0, a);
                let g = round_i32(acc[1]).clamp(0, a);
                let b = round_i32(acc[2]).clamp(0, a);
                out.set_pixel(dx as u16, dy as u16, [r as u8, g as u8, b as u8, a as u8]);
            }
        }
        out
    }

    #[test]
    fn resize_into_preserves_stride_padding_and_supports_subregions() {
        let mut src = Pixmap::new(8, 6);
        for y in 0..6 {
            for x in 0..8 {
                src.set_pixel(x, y, [(x * 30) as u8, (y * 40) as u8, 50, 255]);
            }
        }

        let mut buf = vec![0xAAu8; 3 * 28 + 16];
        let mut dst = PixmapMut::from_parts_with_stride(&mut buf, 4, 4, 28);
        src.subregion(1, 1, 6, 4)
            .resize_into(dst.reborrow(), ResampleFilter::LANCZOS3);

        let expected = src
            .subregion(1, 1, 6, 4)
            .to_owned()
            .resize(4, 4, ResampleFilter::LANCZOS3);
        assert_eq!(dst.as_ref(), expected.as_ref());
        // Stride padding bytes between rows must remain untouched.
        assert_eq!(&dst.raw_data()[16..28], &[0xAA; 12]);
        assert_eq!(&dst.raw_data()[44..56], &[0xAA; 12]);
        assert_eq!(&dst.raw_data()[72..84], &[0xAA; 12]);
    }

    #[test]
    fn empty_dimensions_and_identity_copy_behave_cleanly() {
        let mut src = Pixmap::new(4, 3);
        src.fill([10, 20, 30, 255]);

        for filter in ALL_FILTERS {
            assert_eq!(src.resize(4, 3, filter), src);
            assert_eq!(src.resize(0, 5, filter), Pixmap::new(0, 5));
            assert_eq!(src.resize(5, 0, filter), Pixmap::new(5, 0));

            // Empty source into non-empty destination clears destination.
            let mut dst = Pixmap::new(3, 2);
            dst.fill([99, 99, 99, 255]);
            Pixmap::new(0, 4).resize_into(dst.as_mut(), filter);
            assert_eq!(dst, Pixmap::new(3, 2));
        }
    }

    #[test]
    fn custom_filter_supports_arbitrary_kernels_and_detects_negative_lobes() {
        // Hermite (smoothstep cubic) filter: positive-only, radius 1.0.
        fn hermite(x: f64) -> f64 {
            let ax = if x < 0.0 { -x } else { x };
            if ax < 1.0 {
                (2.0 * ax - 3.0) * ax * ax + 1.0
            } else {
                0.0
            }
        }

        // Lanczos2 filter: negative lobes, radius 2.0.
        fn lanczos2(x: f64) -> f64 {
            let ax = if x < 0.0 { -x } else { x };
            if ax < 1e-9 {
                1.0
            } else if ax < 2.0 {
                sinc_pi(ax) * sinc_pi(ax * 0.5)
            } else {
                0.0
            }
        }

        let mut src = Pixmap::new(8, 6);
        for y in 0..6 {
            for x in 0..8 {
                let px = if x < 4 {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 0]
                };
                src.set_pixel(x, y, px);
            }
        }

        let f_hermite = ResampleFilter::custom(1.0, hermite);
        let f_lanczos2 = ResampleFilter::custom(2.0, lanczos2);

        // Positive-only custom kernel matches f64 reference and skips negative clamping.
        assert!(!TapTable::build(8, 13, f_hermite).has_negative);
        assert!(TapTable::build(8, 13, f_lanczos2).has_negative);

        for filter in [f_hermite, f_lanczos2] {
            let got = src.resize(13, 9, filter);
            let expected = reference_resize_f64(src.as_ref(), 13, 9, filter);
            for y in 0..9 {
                for x in 0..13 {
                    let g = got.pixel(x, y);
                    let e = expected.pixel(x, y);
                    assert!(g[0] <= g[3] && g[1] <= g[3] && g[2] <= g[3]);
                    for c in 0..4 {
                        assert!(g[c].abs_diff(e[c]) <= 1);
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "radius must be finite and positive")]
    fn custom_filter_rejects_non_positive_radius() {
        let _ = ResampleFilter::custom(0.0, bilinear_kernel);
    }

    #[test]
    #[should_panic(expected = "radius must be finite and positive")]
    fn custom_filter_rejects_nan_radius() {
        let _ = ResampleFilter::custom(f64::NAN, bilinear_kernel);
    }
}
