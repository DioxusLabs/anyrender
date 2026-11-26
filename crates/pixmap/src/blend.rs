use crate::alpha::mul_div_255;
use crate::blit::clip_blit;
use crate::{BYTES_PER_PIXEL, Pixmap, PixmapMut, PixmapRef};

/// Defines the color mixing function for a [`BlendMode`].
///
/// Corresponds to the separable and non-separable blend modes in
/// [W3C Compositing and Blending Level 1](https://www.w3.org/TR/compositing-1/).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
#[repr(u8)]
pub enum Mix {
    /// Selects the source color without mixing with the backdrop.
    #[default]
    Normal = 0,
    /// Multiplies the backdrop and source colors.
    Multiply = 1,
    /// Multiplies the complements of backdrop and source, then complements the result.
    Screen = 2,
    /// Multiplies or screens depending on the backdrop color.
    Overlay = 3,
    /// Selects the darker of the backdrop and source colors per channel.
    Darken = 4,
    /// Selects the lighter of the backdrop and source colors per channel.
    Lighten = 5,
    /// Brightens the backdrop to reflect the source color.
    ColorDodge = 6,
    /// Darkens the backdrop to reflect the source color.
    ColorBurn = 7,
    /// Multiplies or screens depending on the source color.
    HardLight = 8,
    /// Darkens or lightens depending on the source color (softer than [`HardLight`](Self::HardLight)).
    SoftLight = 9,
    /// Subtracts the darker of the two colors from the lighter per channel.
    Difference = 10,
    /// Similar to [`Difference`](Self::Difference) with lower contrast (`cb + cs - 2 * cb * cs`).
    Exclusion = 11,
    /// Hue of the source with the saturation and luminosity of the backdrop.
    Hue = 12,
    /// Saturation of the source with the hue and luminosity of the backdrop.
    Saturation = 13,
    /// Hue and saturation of the source with the luminosity of the backdrop.
    Color = 14,
    /// Luminosity of the source with the hue and saturation of the backdrop.
    Luminosity = 15,
}

/// Defines the Porter-Duff alpha composition function for a [`BlendMode`].
///
/// Corresponds to the Porter-Duff operators in
/// [W3C Compositing and Blending Level 1](https://www.w3.org/TR/compositing-1/).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
#[repr(u8)]
pub enum Compose {
    /// Clears the destination to transparent black (`0`).
    Clear = 0,
    /// Replaces the destination with the source (`src`).
    Copy = 1,
    /// Leaves the destination unchanged (`dst`).
    Dest = 2,
    /// Places the source over the destination (`src + (1 - sa) * dst`).
    #[default]
    SrcOver = 3,
    /// Places the destination over the source (`(1 - da) * src + dst`).
    DestOver = 4,
    /// Keeps the part of the source overlapping the destination (`da * src`).
    SrcIn = 5,
    /// Keeps the part of the destination overlapping the source (`sa * dst`).
    DestIn = 6,
    /// Keeps the part of the source outside the destination (`(1 - da) * src`).
    SrcOut = 7,
    /// Keeps the part of the destination outside the source (`(1 - sa) * dst`).
    DestOut = 8,
    /// Places overlapping source over destination and keeps destination elsewhere (`da * src + (1 - sa) * dst`).
    SrcAtop = 9,
    /// Places overlapping destination over source and keeps source elsewhere (`(1 - da) * src + sa * dst`).
    DestAtop = 10,
    /// Combines non-overlapping regions of source and destination (`(1 - da) * src + (1 - sa) * dst`).
    Xor = 11,
    /// Adds source and destination channels with clamping (`min(1, src + dst)`).
    Plus = 12,
    /// Cross-fade additive composition (`min(1, src + dst)`).
    PlusLighter = 13,
}

/// A blend mode combining a color [`Mix`] function and a Porter-Duff [`Compose`] operator.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlendMode {
    /// The color mixing function.
    pub mix: Mix,
    /// The Porter-Duff composition function.
    pub compose: Compose,
}

impl BlendMode {
    /// [`Mix::Normal`] with [`Compose::Clear`].
    pub const CLEAR: Self = Self::new(Mix::Normal, Compose::Clear);
    /// [`Mix::Normal`] with [`Compose::Copy`].
    pub const COPY: Self = Self::new(Mix::Normal, Compose::Copy);
    /// [`Mix::Normal`] with [`Compose::Dest`].
    pub const DEST: Self = Self::new(Mix::Normal, Compose::Dest);
    /// [`Mix::Normal`] with [`Compose::SrcOver`] (the default blend mode).
    pub const SRC_OVER: Self = Self::new(Mix::Normal, Compose::SrcOver);
    /// [`Mix::Normal`] with [`Compose::DestOver`].
    pub const DEST_OVER: Self = Self::new(Mix::Normal, Compose::DestOver);
    /// [`Mix::Normal`] with [`Compose::SrcIn`].
    pub const SRC_IN: Self = Self::new(Mix::Normal, Compose::SrcIn);
    /// [`Mix::Normal`] with [`Compose::DestIn`].
    pub const DEST_IN: Self = Self::new(Mix::Normal, Compose::DestIn);
    /// [`Mix::Normal`] with [`Compose::SrcOut`].
    pub const SRC_OUT: Self = Self::new(Mix::Normal, Compose::SrcOut);
    /// [`Mix::Normal`] with [`Compose::DestOut`].
    pub const DEST_OUT: Self = Self::new(Mix::Normal, Compose::DestOut);
    /// [`Mix::Normal`] with [`Compose::SrcAtop`].
    pub const SRC_ATOP: Self = Self::new(Mix::Normal, Compose::SrcAtop);
    /// [`Mix::Normal`] with [`Compose::DestAtop`].
    pub const DEST_ATOP: Self = Self::new(Mix::Normal, Compose::DestAtop);
    /// [`Mix::Normal`] with [`Compose::Xor`].
    pub const XOR: Self = Self::new(Mix::Normal, Compose::Xor);
    /// [`Mix::Normal`] with [`Compose::Plus`].
    pub const PLUS: Self = Self::new(Mix::Normal, Compose::Plus);
    /// [`Mix::Normal`] with [`Compose::PlusLighter`].
    pub const PLUS_LIGHTER: Self = Self::new(Mix::Normal, Compose::PlusLighter);

    /// [`Mix::Normal`] with [`Compose::SrcOver`].
    pub const NORMAL: Self = Self::new(Mix::Normal, Compose::SrcOver);
    /// [`Mix::Multiply`] with [`Compose::SrcOver`].
    pub const MULTIPLY: Self = Self::new(Mix::Multiply, Compose::SrcOver);
    /// [`Mix::Screen`] with [`Compose::SrcOver`].
    pub const SCREEN: Self = Self::new(Mix::Screen, Compose::SrcOver);
    /// [`Mix::Overlay`] with [`Compose::SrcOver`].
    pub const OVERLAY: Self = Self::new(Mix::Overlay, Compose::SrcOver);
    /// [`Mix::Darken`] with [`Compose::SrcOver`].
    pub const DARKEN: Self = Self::new(Mix::Darken, Compose::SrcOver);
    /// [`Mix::Lighten`] with [`Compose::SrcOver`].
    pub const LIGHTEN: Self = Self::new(Mix::Lighten, Compose::SrcOver);
    /// [`Mix::ColorDodge`] with [`Compose::SrcOver`].
    pub const COLOR_DODGE: Self = Self::new(Mix::ColorDodge, Compose::SrcOver);
    /// [`Mix::ColorBurn`] with [`Compose::SrcOver`].
    pub const COLOR_BURN: Self = Self::new(Mix::ColorBurn, Compose::SrcOver);
    /// [`Mix::HardLight`] with [`Compose::SrcOver`].
    pub const HARD_LIGHT: Self = Self::new(Mix::HardLight, Compose::SrcOver);
    /// [`Mix::SoftLight`] with [`Compose::SrcOver`].
    pub const SOFT_LIGHT: Self = Self::new(Mix::SoftLight, Compose::SrcOver);
    /// [`Mix::Difference`] with [`Compose::SrcOver`].
    pub const DIFFERENCE: Self = Self::new(Mix::Difference, Compose::SrcOver);
    /// [`Mix::Exclusion`] with [`Compose::SrcOver`].
    pub const EXCLUSION: Self = Self::new(Mix::Exclusion, Compose::SrcOver);
    /// [`Mix::Hue`] with [`Compose::SrcOver`].
    pub const HUE: Self = Self::new(Mix::Hue, Compose::SrcOver);
    /// [`Mix::Saturation`] with [`Compose::SrcOver`].
    pub const SATURATION: Self = Self::new(Mix::Saturation, Compose::SrcOver);
    /// [`Mix::Color`] with [`Compose::SrcOver`].
    pub const COLOR: Self = Self::new(Mix::Color, Compose::SrcOver);
    /// [`Mix::Luminosity`] with [`Compose::SrcOver`].
    pub const LUMINOSITY: Self = Self::new(Mix::Luminosity, Compose::SrcOver);

    /// Creates a new blend mode from color mixing and composition functions.
    #[must_use]
    pub const fn new(mix: Mix, compose: Compose) -> Self {
        Self { mix, compose }
    }

    /// Returns `true` if a transparent source pixel (`[0, 0, 0, 0]`) can modify the destination.
    #[must_use]
    pub const fn is_destructive(self) -> bool {
        matches!(
            self.compose,
            Compose::Clear
                | Compose::Copy
                | Compose::SrcIn
                | Compose::DestIn
                | Compose::SrcOut
                | Compose::DestAtop
        )
    }

    /// Blends and composites a single premultiplied RGBA8 `src` pixel onto `dst`.
    #[inline]
    #[must_use]
    pub fn blend_pixel(self, src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
        match self.compose {
            Compose::Clear => return [0, 0, 0, 0],
            Compose::Dest => return dst,
            _ => {}
        }
        if self.mix == Mix::Normal {
            compose_pixel_u8(self.compose, src, dst)
        } else {
            let ms = mix_src_pixel(self.mix, src, dst);
            compose_pixel_u8(self.compose, ms, dst)
        }
    }
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::SRC_OVER
    }
}

/// Composites premultiplied RGBA8 `src` bytes onto `dst` bytes in place using `blend_mode` and
/// uniform `opacity` (`0..=255`).
///
/// # Panics
///
/// Panics if `dst.len() != src.len()` or if `dst.len()` is not a multiple of
/// [`BYTES_PER_PIXEL`] (4).
pub fn composite_rgba8(dst: &mut [u8], src: &[u8], blend_mode: BlendMode, opacity: u8) {
    assert_eq!(
        dst.len(),
        src.len(),
        "destination slice length ({}) must match source slice length ({})",
        dst.len(),
        src.len()
    );
    let (dst_pixels, dst_rem) = dst.as_chunks_mut::<BYTES_PER_PIXEL>();
    let (src_pixels, src_rem) = src.as_chunks::<BYTES_PER_PIXEL>();
    assert!(
        dst_rem.is_empty() && src_rem.is_empty(),
        "RGBA8 slice length ({}) must be a multiple of {BYTES_PER_PIXEL}",
        dst.len()
    );

    if blend_mode.compose == Compose::Dest {
        return;
    }
    if blend_mode.compose == Compose::Clear {
        dst.fill(0);
        return;
    }
    if opacity == 0 {
        if blend_mode.is_destructive() {
            dst.fill(0);
        }
        return;
    }
    if blend_mode.mix == Mix::Normal && blend_mode.compose == Compose::Copy {
        dst.copy_from_slice(src);
        if opacity < 255 {
            crate::multiply_alpha_rgba8(dst, opacity);
        }
        return;
    }

    if blend_mode.mix == Mix::Normal {
        if opacity == 255 {
            composite_slice_normal_opaque(blend_mode.compose, dst_pixels, src_pixels);
        } else {
            composite_slice_normal(blend_mode.compose, dst_pixels, src_pixels, |s| {
                scale_pixel(s, opacity)
            });
        }
    } else if opacity == 255 {
        composite_slice_mixed(
            blend_mode.mix,
            blend_mode.compose,
            dst_pixels,
            src_pixels,
            |s| s,
        );
    } else {
        composite_slice_mixed(
            blend_mode.mix,
            blend_mode.compose,
            dst_pixels,
            src_pixels,
            |s| scale_pixel(s, opacity),
        );
    }
}

/// Composites a solid premultiplied RGBA8 `color` onto every pixel of `dst` in place using
/// `blend_mode`.
///
/// # Panics
///
/// Panics if `dst.len()` is not a multiple of [`BYTES_PER_PIXEL`] (4).
pub fn composite_color_rgba8(dst: &mut [u8], color: [u8; 4], blend_mode: BlendMode) {
    assert!(
        dst.as_chunks::<BYTES_PER_PIXEL>().1.is_empty(),
        "RGBA8 slice length ({}) must be a multiple of {BYTES_PER_PIXEL}",
        dst.len()
    );
    if blend_mode.compose == Compose::Dest {
        return;
    }
    if blend_mode.compose == Compose::Clear {
        dst.fill(0);
        return;
    }
    if color == [0, 0, 0, 0] {
        if blend_mode.is_destructive() {
            dst.fill(0);
        }
        return;
    }
    if blend_mode.mix == Mix::Normal && blend_mode.compose == Compose::Copy {
        crate::util::fill_slice(dst, color);
        return;
    }

    let (dst_pixels, _) = dst.as_chunks_mut::<BYTES_PER_PIXEL>();
    if blend_mode.mix == Mix::Normal {
        composite_color_normal(blend_mode.compose, dst_pixels, color);
    } else {
        composite_color_mixed(blend_mode.mix, blend_mode.compose, dst_pixels, color);
    }
}

impl Pixmap {
    /// Composites `src` onto `self` with `src`'s top-left corner placed at `(x, y)`, using
    /// `blend_mode` and uniform `opacity` (`0..=255`).
    ///
    /// Any part of `src` falling outside `self` is clipped; if there is no overlap, `self` is
    /// left unchanged.
    pub fn composite(
        &mut self,
        src: PixmapRef<'_>,
        x: i32,
        y: i32,
        blend_mode: BlendMode,
        opacity: u8,
    ) {
        self.as_mut().composite(src, x, y, blend_mode, opacity);
    }

    /// Composites a solid premultiplied RGBA8 `color` onto every pixel of `self` using
    /// `blend_mode`.
    pub fn composite_color(&mut self, color: [u8; 4], blend_mode: BlendMode) {
        composite_color_rgba8(self.data_mut(), color, blend_mode);
    }
}

impl PixmapMut<'_> {
    /// Composites `src` onto `self` with `src`'s top-left corner placed at `(x, y)`, using
    /// `blend_mode` and uniform `opacity` (`0..=255`), leaving any inter-row stride padding
    /// untouched.
    ///
    /// Any part of `src` falling outside `self` is clipped; if there is no overlap, `self` is
    /// left unchanged.
    pub fn composite(
        &mut self,
        src: PixmapRef<'_>,
        x: i32,
        y: i32,
        blend_mode: BlendMode,
        opacity: u8,
    ) {
        let Some(rect) = clip_blit(self.width(), self.height(), src.width(), src.height(), x, y)
        else {
            return;
        };
        let src_region = src.subregion(rect.src_x, rect.src_y, rect.width, rect.height);
        let mut dst_region = self.subregion_mut(rect.dst_x, rect.dst_y, rect.width, rect.height);
        if let (Some(dst_data), Some(src_data)) = (dst_region.data_mut(), src_region.data()) {
            composite_rgba8(dst_data, src_data, blend_mode, opacity);
        } else {
            for (dst_row, src_row) in dst_region.rows_mut().zip(src_region.rows()) {
                composite_rgba8(dst_row, src_row, blend_mode, opacity);
            }
        }
    }

    /// Composites a solid premultiplied RGBA8 `color` onto every active pixel of `self` using
    /// `blend_mode`, leaving any inter-row stride padding untouched.
    pub fn composite_color(&mut self, color: [u8; 4], blend_mode: BlendMode) {
        if let Some(data) = self.data_mut() {
            composite_color_rgba8(data, color, blend_mode);
        } else {
            for row in self.rows_mut() {
                composite_color_rgba8(row, color, blend_mode);
            }
        }
    }
}

/// Stack-allocated tile size for combining non-`Normal` [`Mix`] with non-`SrcOver` [`Compose`]
/// in two hoisted, branch-free passes without monomorphizing all `Mix × Compose` pairs.
const MIX_TILE_LEN: usize = 64;

/// Reciprocal lookup table `1.0 / a` for `a in 1..=255` (`0.0` for `a == 0`), avoiding per-pixel
/// floating-point division when unpremultiplying channels in `f32` blend modes.
const RECIP_ALPHA_F32: [f32; 256] = {
    let mut table = [0.0f32; 256];
    let mut a = 1usize;
    while a < 256 {
        table[a] = 1.0 / (a as f32);
        a += 1;
    }
    table
};

#[inline]
fn scale_pixel(px: [u8; 4], opacity: u8) -> [u8; 4] {
    [
        mul_div_255(px[0], opacity),
        mul_div_255(px[1], opacity),
        mul_div_255(px[2], opacity),
        mul_div_255(px[3], opacity),
    ]
}

/// Exact rounded `v / 255` clamped to `255` for `u16` sums of channel products.
#[inline]
pub(crate) fn div_255_round_u16(v: u16) -> u8 {
    let v = v.min(255 * 255) + 128;
    ((v + (v >> 8)) >> 8) as u8
}

#[inline]
fn combine_u8(src: [u8; 4], dst: [u8; 4], fa: u8, fb: u8) -> [u8; 4] {
    let fa = u16::from(fa);
    let fb = u16::from(fb);
    [
        div_255_round_u16((u16::from(src[0]) * fa).saturating_add(u16::from(dst[0]) * fb)),
        div_255_round_u16((u16::from(src[1]) * fa).saturating_add(u16::from(dst[1]) * fb)),
        div_255_round_u16((u16::from(src[2]) * fa).saturating_add(u16::from(dst[2]) * fb)),
        div_255_round_u16((u16::from(src[3]) * fa).saturating_add(u16::from(dst[3]) * fb)),
    ]
}

#[inline]
fn compose_src_over_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let inv_sa = 255 - src[3];
    [
        src[0].saturating_add(mul_div_255(dst[0], inv_sa)),
        src[1].saturating_add(mul_div_255(dst[1], inv_sa)),
        src[2].saturating_add(mul_div_255(dst[2], inv_sa)),
        src[3].saturating_add(mul_div_255(dst[3], inv_sa)),
    ]
}

#[inline]
fn compose_pixel_u8(compose: Compose, src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    match compose {
        Compose::Clear => [0, 0, 0, 0],
        Compose::Copy => src,
        Compose::Dest => dst,
        Compose::SrcOver => compose_src_over_u8(src, dst),
        Compose::DestOver => {
            let inv_da = 255 - dst[3];
            [
                dst[0].saturating_add(mul_div_255(src[0], inv_da)),
                dst[1].saturating_add(mul_div_255(src[1], inv_da)),
                dst[2].saturating_add(mul_div_255(src[2], inv_da)),
                dst[3].saturating_add(mul_div_255(src[3], inv_da)),
            ]
        }
        Compose::SrcIn => scale_pixel(src, dst[3]),
        Compose::DestIn => scale_pixel(dst, src[3]),
        Compose::SrcOut => scale_pixel(src, 255 - dst[3]),
        Compose::DestOut => scale_pixel(dst, 255 - src[3]),
        Compose::SrcAtop => combine_u8(src, dst, dst[3], 255 - src[3]),
        Compose::DestAtop => combine_u8(src, dst, 255 - dst[3], src[3]),
        Compose::Xor => combine_u8(src, dst, 255 - dst[3], 255 - src[3]),
        Compose::Plus | Compose::PlusLighter => [
            src[0].saturating_add(dst[0]),
            src[1].saturating_add(dst[1]),
            src[2].saturating_add(dst[2]),
            src[3].saturating_add(dst[3]),
        ],
    }
}

#[inline(never)]
fn composite_slice_normal_opaque(compose: Compose, dst: &mut [[u8; 4]], src: &[[u8; 4]]) {
    composite_slice_normal(compose, dst, src, |s| s);
}

fn composite_slice_normal(
    compose: Compose,
    dst: &mut [[u8; 4]],
    src: &[[u8; 4]],
    map_src: impl Fn([u8; 4]) -> [u8; 4],
) {
    macro_rules! apply_slice {
        ($op:expr) => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = compose_pixel_u8($op, map_src(s), *d);
            }
        };
    }
    match compose {
        Compose::Clear => dst.fill([0, 0, 0, 0]),
        Compose::Copy => {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = map_src(s);
            }
        }
        Compose::Dest => {}
        Compose::SrcOver => apply_slice!(Compose::SrcOver),
        Compose::DestOver => apply_slice!(Compose::DestOver),
        Compose::SrcIn => apply_slice!(Compose::SrcIn),
        Compose::DestIn => apply_slice!(Compose::DestIn),
        Compose::SrcOut => apply_slice!(Compose::SrcOut),
        Compose::DestOut => apply_slice!(Compose::DestOut),
        Compose::SrcAtop => apply_slice!(Compose::SrcAtop),
        Compose::DestAtop => apply_slice!(Compose::DestAtop),
        Compose::Xor => apply_slice!(Compose::Xor),
        Compose::Plus | Compose::PlusLighter => apply_slice!(Compose::Plus),
    }
}

fn composite_color_normal(compose: Compose, dst: &mut [[u8; 4]], color: [u8; 4]) {
    macro_rules! apply_color {
        ($op:expr) => {
            for d in dst {
                *d = compose_pixel_u8($op, color, *d);
            }
        };
    }
    if color[3] == 255 {
        match compose {
            Compose::SrcOver => {
                dst.fill(color);
                return;
            }
            Compose::DestIn => return,
            Compose::DestOut => {
                dst.fill([0, 0, 0, 0]);
                return;
            }
            Compose::SrcAtop => {
                apply_color!(Compose::SrcIn);
                return;
            }
            Compose::Xor => {
                apply_color!(Compose::SrcOut);
                return;
            }
            _ => {}
        }
    }
    match compose {
        Compose::Clear => dst.fill([0, 0, 0, 0]),
        Compose::Copy => dst.fill(color),
        Compose::Dest => {}
        Compose::SrcOver => apply_color!(Compose::SrcOver),
        Compose::DestOver => apply_color!(Compose::DestOver),
        Compose::SrcIn => apply_color!(Compose::SrcIn),
        Compose::DestIn => apply_color!(Compose::DestIn),
        Compose::SrcOut => apply_color!(Compose::SrcOut),
        Compose::DestOut => apply_color!(Compose::DestOut),
        Compose::SrcAtop => apply_color!(Compose::SrcAtop),
        Compose::DestAtop => apply_color!(Compose::DestAtop),
        Compose::Xor => apply_color!(Compose::Xor),
        Compose::Plus | Compose::PlusLighter => apply_color!(Compose::Plus),
    }
}

// --- Direct premultiplied `u16` modified-source formulas for separable `Mix` modes ---
//
// Each function computes the W3C modified premultiplied source
// `ms = (1 - ab) * cs + as * ab * B(Cb, Cs)` directly in `u16` without unpremultiplying or
// floating-point math.

#[inline]
fn mix_multiply_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let inv_da = 255 - dst[3];
    [
        mul_div_255(src[0], inv_da.saturating_add(dst[0])),
        mul_div_255(src[1], inv_da.saturating_add(dst[1])),
        mul_div_255(src[2], inv_da.saturating_add(dst[2])),
        src[3],
    ]
}

#[inline]
fn mix_screen_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let sa = src[3];
    [
        src[0].saturating_add(mul_div_255(sa.saturating_sub(src[0]), dst[0])),
        src[1].saturating_add(mul_div_255(sa.saturating_sub(src[1]), dst[1])),
        src[2].saturating_add(mul_div_255(sa.saturating_sub(src[2]), dst[2])),
        sa,
    ]
}

#[inline]
fn mix_darken_ch(cs: u8, sa: u8, cb: u8, da: u8) -> u8 {
    let inv_da = u16::from(255 - da);
    let lhs = u16::from(sa) * u16::from(cb);
    let rhs = u16::from(da) * u16::from(cs);
    div_255_round_u16(inv_da * u16::from(cs) + lhs.min(rhs))
}

#[inline]
fn mix_darken_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let (sa, da) = (src[3], dst[3]);
    [
        mix_darken_ch(src[0], sa, dst[0], da),
        mix_darken_ch(src[1], sa, dst[1], da),
        mix_darken_ch(src[2], sa, dst[2], da),
        sa,
    ]
}

#[inline]
fn mix_lighten_ch(cs: u8, sa: u8, cb: u8, da: u8) -> u8 {
    let inv_da = u16::from(255 - da);
    let lhs = u16::from(sa) * u16::from(cb);
    let rhs = u16::from(da) * u16::from(cs);
    div_255_round_u16((inv_da * u16::from(cs)).saturating_add(lhs.max(rhs)))
}

#[inline]
fn mix_lighten_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let (sa, da) = (src[3], dst[3]);
    [
        mix_lighten_ch(src[0], sa, dst[0], da),
        mix_lighten_ch(src[1], sa, dst[1], da),
        mix_lighten_ch(src[2], sa, dst[2], da),
        sa,
    ]
}

#[inline]
fn mix_difference_ch(cs: u8, sa: u8, cb: u8, da: u8) -> u8 {
    let inv_da = u16::from(255 - da);
    let lhs = u16::from(sa) * u16::from(cb);
    let rhs = u16::from(da) * u16::from(cs);
    div_255_round_u16((inv_da * u16::from(cs)).saturating_add(lhs.abs_diff(rhs)))
}

#[inline]
fn mix_difference_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let (sa, da) = (src[3], dst[3]);
    [
        mix_difference_ch(src[0], sa, dst[0], da),
        mix_difference_ch(src[1], sa, dst[1], da),
        mix_difference_ch(src[2], sa, dst[2], da),
        sa,
    ]
}

#[inline]
fn mix_exclusion_ch(cs: u8, sa: u8, cb: u8) -> u8 {
    let term1 = u16::from(cs) * u16::from(255 - cb);
    let term2 = u16::from(sa.saturating_sub(cs)) * u16::from(cb);
    div_255_round_u16(term1.saturating_add(term2))
}

#[inline]
fn mix_exclusion_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let sa = src[3];
    [
        mix_exclusion_ch(src[0], sa, dst[0]),
        mix_exclusion_ch(src[1], sa, dst[1]),
        mix_exclusion_ch(src[2], sa, dst[2]),
        sa,
    ]
}

#[inline]
fn mix_hard_light_ch(cs: u8, sa: u8, cb: u8, da: u8) -> u8 {
    let cs16 = u16::from(cs);
    let sa16 = u16::from(sa);
    let cb16 = u16::from(cb);
    let da16 = u16::from(da);
    let inv_da_cs = u16::from(255 - da) * cs16;
    let blended = if cs16 * 2 <= sa16 {
        (2 * cs16).saturating_mul(cb16)
    } else {
        let inv_s = u16::from(sa.saturating_sub(cs));
        let inv_b = u16::from(da.saturating_sub(cb));
        (sa16 * da16).saturating_sub((2 * inv_s).saturating_mul(inv_b))
    };
    div_255_round_u16(inv_da_cs.saturating_add(blended))
}

#[inline]
fn mix_hard_light_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let (sa, da) = (src[3], dst[3]);
    [
        mix_hard_light_ch(src[0], sa, dst[0], da),
        mix_hard_light_ch(src[1], sa, dst[1], da),
        mix_hard_light_ch(src[2], sa, dst[2], da),
        sa,
    ]
}

#[inline]
fn mix_overlay_ch(cs: u8, sa: u8, cb: u8, da: u8) -> u8 {
    let cs16 = u16::from(cs);
    let sa16 = u16::from(sa);
    let cb16 = u16::from(cb);
    let da16 = u16::from(da);
    let inv_da_cs = u16::from(255 - da) * cs16;
    let blended = if cb16 * 2 <= da16 {
        (2 * cb16).saturating_mul(cs16)
    } else {
        let inv_s = u16::from(sa.saturating_sub(cs));
        let inv_b = u16::from(da.saturating_sub(cb));
        (sa16 * da16).saturating_sub((2 * inv_s).saturating_mul(inv_b))
    };
    div_255_round_u16(inv_da_cs.saturating_add(blended))
}

#[inline]
fn mix_overlay_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    let (sa, da) = (src[3], dst[3]);
    [
        mix_overlay_ch(src[0], sa, dst[0], da),
        mix_overlay_ch(src[1], sa, dst[1], da),
        mix_overlay_ch(src[2], sa, dst[2], da),
        sa,
    ]
}

/// Square root on `[0.0, 1.0]` without `std`/`libm`, accurate to `< 1e-6` across `[0.0, 1.0]`.
#[inline]
pub(crate) fn sqrt_unit(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // Minimax bit-level initial guess + two Newton-Raphson iterations (< 4e-7 max error).
    let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1fbd_1df5);
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    y
}

/// Unpremultiplies a channel in byte space, preserving exact `1.0` when `c >= a` so reciprocal
/// multiplication cannot round just below `1.0` and miss discontinuous W3C blend endpoints
/// (such as Color Burn's `cb == 1.0` and Color Dodge's `cs == 1.0`).
#[inline]
fn unpremul_ch_f32(c: u8, a: u8, inv_a: f32) -> f32 {
    if c >= a { 1.0 } else { f32::from(c) * inv_a }
}

#[inline]
fn mix_src_f32(
    src: [u8; 4],
    dst: [u8; 4],
    mix_fn: impl FnOnce([f32; 3], [f32; 3]) -> [f32; 3],
) -> [u8; 4] {
    // When either pixel is transparent, W3C blending reduces to `Mix::Normal` (`ms == src`).
    if src[3] == 0 || dst[3] == 0 {
        return src;
    }

    let inv_sa = RECIP_ALPHA_F32[usize::from(src[3])];
    let inv_ba = RECIP_ALPHA_F32[usize::from(dst[3])];

    let cs = [
        unpremul_ch_f32(src[0], src[3], inv_sa),
        unpremul_ch_f32(src[1], src[3], inv_sa),
        unpremul_ch_f32(src[2], src[3], inv_sa),
    ];
    let cb = [
        unpremul_ch_f32(dst[0], dst[3], inv_ba),
        unpremul_ch_f32(dst[1], dst[3], inv_ba),
        unpremul_ch_f32(dst[2], dst[3], inv_ba),
    ];

    let blended = mix_fn(cs, cb);

    // W3C Compositing and Blending Level 1:
    // Cs_out = (1 - ab) * Cs + ab * B(Cb, Cs), premultiplied directly by `sa` in `0..=255`.
    const INV_255: f32 = 1.0 / 255.0;
    let ba = f32::from(dst[3]) * INV_255;
    let inv_ba_1 = 1.0 - ba;
    let sa = f32::from(src[3]);
    [
        ((inv_ba_1 * cs[0] + ba * blended[0]).clamp(0.0, 1.0) * sa + 0.5) as u8,
        ((inv_ba_1 * cs[1] + ba * blended[1]).clamp(0.0, 1.0) * sa + 0.5) as u8,
        ((inv_ba_1 * cs[2] + ba * blended[2]).clamp(0.0, 1.0) * sa + 0.5) as u8,
        src[3],
    ]
}

#[inline]
fn mix_color_dodge_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    mix_src_f32(src, dst, |cs, cb| {
        [
            color_dodge_ch(cs[0], cb[0]),
            color_dodge_ch(cs[1], cb[1]),
            color_dodge_ch(cs[2], cb[2]),
        ]
    })
}

#[inline]
fn mix_color_burn_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    mix_src_f32(src, dst, |cs, cb| {
        [
            color_burn_ch(cs[0], cb[0]),
            color_burn_ch(cs[1], cb[1]),
            color_burn_ch(cs[2], cb[2]),
        ]
    })
}

#[inline]
fn mix_soft_light_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    mix_src_f32(src, dst, |cs, cb| {
        [
            soft_light_ch(cs[0], cb[0]),
            soft_light_ch(cs[1], cb[1]),
            soft_light_ch(cs[2], cb[2]),
        ]
    })
}

#[inline]
fn mix_hue_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    mix_src_f32(src, dst, |cs, cb| set_lum(set_sat(cs, sat(cb)), lum(cb)))
}

#[inline]
fn mix_saturation_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    mix_src_f32(src, dst, |cs, cb| set_lum(set_sat(cb, sat(cs)), lum(cb)))
}

#[inline]
fn mix_color_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    mix_src_f32(src, dst, |cs, cb| set_lum(cs, lum(cb)))
}

#[inline]
fn mix_luminosity_u8(src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    mix_src_f32(src, dst, |cs, cb| set_lum(cb, lum(cs)))
}

#[inline]
fn mix_src_pixel(mix: Mix, src: [u8; 4], dst: [u8; 4]) -> [u8; 4] {
    match mix {
        Mix::Normal => src,
        Mix::Multiply => mix_multiply_u8(src, dst),
        Mix::Screen => mix_screen_u8(src, dst),
        Mix::Overlay => mix_overlay_u8(src, dst),
        Mix::Darken => mix_darken_u8(src, dst),
        Mix::Lighten => mix_lighten_u8(src, dst),
        Mix::ColorDodge => mix_color_dodge_u8(src, dst),
        Mix::ColorBurn => mix_color_burn_u8(src, dst),
        Mix::HardLight => mix_hard_light_u8(src, dst),
        Mix::SoftLight => mix_soft_light_u8(src, dst),
        Mix::Difference => mix_difference_u8(src, dst),
        Mix::Exclusion => mix_exclusion_u8(src, dst),
        Mix::Hue => mix_hue_u8(src, dst),
        Mix::Saturation => mix_saturation_u8(src, dst),
        Mix::Color => mix_color_u8(src, dst),
        Mix::Luminosity => mix_luminosity_u8(src, dst),
    }
}

/// Mixes each source pixel with its backdrop, then applies `map_mixed` (the uniform opacity) and
/// composes the result.
///
/// Opacity is applied after mixing because blend functions take unpremultiplied colours: scaling
/// the 8-bit source first would quantise the colour they see, which the discontinuous and
/// non-separable modes amplify.
fn composite_slice_mixed_with(
    compose: Compose,
    dst: &mut [[u8; 4]],
    src: &[[u8; 4]],
    map_mixed: impl Fn([u8; 4]) -> [u8; 4],
    mix_fn: impl Fn([u8; 4], [u8; 4]) -> [u8; 4],
) {
    if compose == Compose::SrcOver {
        for (d, &s) in dst.iter_mut().zip(src) {
            let ms = map_mixed(mix_fn(s, *d));
            *d = compose_src_over_u8(ms, *d);
        }
    } else {
        let mut ms_buf = [[0u8; 4]; MIX_TILE_LEN];
        for (dst_chunk, src_chunk) in dst.chunks_mut(MIX_TILE_LEN).zip(src.chunks(MIX_TILE_LEN)) {
            let len = dst_chunk.len();
            for ((ms, &s), &d) in ms_buf[..len].iter_mut().zip(src_chunk).zip(&*dst_chunk) {
                *ms = map_mixed(mix_fn(s, d));
            }
            composite_slice_normal_opaque(compose, dst_chunk, &ms_buf[..len]);
        }
    }
}

fn composite_slice_mixed(
    mix: Mix,
    compose: Compose,
    dst: &mut [[u8; 4]],
    src: &[[u8; 4]],
    map_src: impl Fn([u8; 4]) -> [u8; 4],
) {
    macro_rules! dispatch {
        ($f:ident) => {
            composite_slice_mixed_with(compose, dst, src, map_src, $f)
        };
    }
    match mix {
        Mix::Normal => composite_slice_normal(compose, dst, src, map_src),
        Mix::Multiply => dispatch!(mix_multiply_u8),
        Mix::Screen => dispatch!(mix_screen_u8),
        Mix::Overlay => dispatch!(mix_overlay_u8),
        Mix::Darken => dispatch!(mix_darken_u8),
        Mix::Lighten => dispatch!(mix_lighten_u8),
        Mix::ColorDodge => dispatch!(mix_color_dodge_u8),
        Mix::ColorBurn => dispatch!(mix_color_burn_u8),
        Mix::HardLight => dispatch!(mix_hard_light_u8),
        Mix::SoftLight => dispatch!(mix_soft_light_u8),
        Mix::Difference => dispatch!(mix_difference_u8),
        Mix::Exclusion => dispatch!(mix_exclusion_u8),
        Mix::Hue => dispatch!(mix_hue_u8),
        Mix::Saturation => dispatch!(mix_saturation_u8),
        Mix::Color => dispatch!(mix_color_u8),
        Mix::Luminosity => dispatch!(mix_luminosity_u8),
    }
}

fn composite_color_mixed_with(
    compose: Compose,
    dst: &mut [[u8; 4]],
    color: [u8; 4],
    mix_fn: impl Fn([u8; 4], [u8; 4]) -> [u8; 4],
) {
    if compose == Compose::SrcOver {
        for d in dst {
            let ms = mix_fn(color, *d);
            *d = compose_src_over_u8(ms, *d);
        }
    } else {
        let mut ms_buf = [[0u8; 4]; MIX_TILE_LEN];
        for dst_chunk in dst.chunks_mut(MIX_TILE_LEN) {
            let len = dst_chunk.len();
            for (ms, &d) in ms_buf[..len].iter_mut().zip(&*dst_chunk) {
                *ms = mix_fn(color, d);
            }
            composite_slice_normal_opaque(compose, dst_chunk, &ms_buf[..len]);
        }
    }
}

fn composite_color_mixed(mix: Mix, compose: Compose, dst: &mut [[u8; 4]], color: [u8; 4]) {
    macro_rules! dispatch {
        ($f:ident) => {
            composite_color_mixed_with(compose, dst, color, $f)
        };
    }
    match mix {
        Mix::Normal => composite_color_normal(compose, dst, color),
        Mix::Multiply => dispatch!(mix_multiply_u8),
        Mix::Screen => dispatch!(mix_screen_u8),
        Mix::Overlay => dispatch!(mix_overlay_u8),
        Mix::Darken => dispatch!(mix_darken_u8),
        Mix::Lighten => dispatch!(mix_lighten_u8),
        Mix::ColorDodge => dispatch!(mix_color_dodge_u8),
        Mix::ColorBurn => dispatch!(mix_color_burn_u8),
        Mix::HardLight => dispatch!(mix_hard_light_u8),
        Mix::SoftLight => dispatch!(mix_soft_light_u8),
        Mix::Difference => dispatch!(mix_difference_u8),
        Mix::Exclusion => dispatch!(mix_exclusion_u8),
        Mix::Hue => dispatch!(mix_hue_u8),
        Mix::Saturation => dispatch!(mix_saturation_u8),
        Mix::Color => dispatch!(mix_color_u8),
        Mix::Luminosity => dispatch!(mix_luminosity_u8),
    }
}

#[inline]
fn soft_light_ch(cs: f32, cb: f32) -> f32 {
    if cs <= 0.5 {
        cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
    } else {
        let d = if cb <= 0.25 {
            ((16.0 * cb - 12.0) * cb + 4.0) * cb
        } else {
            sqrt_unit(cb)
        };
        cb + (2.0 * cs - 1.0) * (d - cb)
    }
}

#[inline]
fn color_dodge_ch(cs: f32, cb: f32) -> f32 {
    if cb == 0.0 {
        0.0
    } else if cs == 1.0 {
        1.0
    } else {
        (cb / (1.0 - cs)).min(1.0)
    }
}

#[inline]
fn color_burn_ch(cs: f32, cb: f32) -> f32 {
    if cb == 1.0 {
        1.0
    } else if cs == 0.0 {
        0.0
    } else {
        1.0 - ((1.0 - cb) / cs).min(1.0)
    }
}

#[inline]
fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

#[inline]
fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

#[inline]
fn clip_color(mut c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    if n < 0.0 && l > n {
        let scale = l / (l - n);
        c[0] = l + (c[0] - l) * scale;
        c[1] = l + (c[1] - l) * scale;
        c[2] = l + (c[2] - l) * scale;
    }
    if x > 1.0 && x > l {
        let scale = (1.0 - l) / (x - l);
        c[0] = l + (c[0] - l) * scale;
        c[1] = l + (c[1] - l) * scale;
        c[2] = l + (c[2] - l) * scale;
    }
    c
}

#[inline]
fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d])
}

#[inline]
fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let mn = c[0].min(c[1]).min(c[2]);
    let mx = c[0].max(c[1]).max(c[2]);
    let cur = mx - mn;
    if cur == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        let scale = s / cur;
        [
            (c[0] - mn) * scale,
            (c[1] - mn) * scale,
            (c[2] - mn) * scale,
        ]
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::{Pixmap, PixmapMut, PixmapRef};

    fn assert_pixel_near(actual: [u8; 4], expected: [u8; 4], tol: u8) {
        for i in 0..4 {
            assert!(
                actual[i].abs_diff(expected[i]) <= tol,
                "channel {i}: actual {actual:?} vs expected {expected:?} (tol {tol})"
            );
        }
    }

    #[test]
    fn div_255_round_u16_is_exact_for_all_valid_products_and_clamps() {
        for v in 0..=(255u16 * 255) {
            let expected = ((u32::from(v) + 127) / 255) as u8;
            assert_eq!(div_255_round_u16(v), expected, "mismatch at v={v}");
        }
        for v in (255u16 * 255 + 1)..=u16::MAX {
            assert_eq!(div_255_round_u16(v), 255, "clamp mismatch at v={v}");
        }
    }

    #[test]
    fn sqrt_unit_is_accurate_across_unit_interval() {
        assert_eq!(sqrt_unit(0.0), 0.0);
        assert_eq!(sqrt_unit(-0.5), 0.0);
        for i in 1..=1000 {
            let x = i as f32 / 1000.0;
            let y = sqrt_unit(x);
            assert!(
                (y * y - x).abs() < 1e-6,
                "sqrt_unit({x}) = {y}, y^2 = {}",
                y * y
            );
        }
    }

    #[test]
    fn blend_mode_constants_and_destructive_flags() {
        assert_eq!(BlendMode::default(), BlendMode::SRC_OVER);
        assert_eq!(BlendMode::NORMAL, BlendMode::SRC_OVER);
        assert_eq!(
            BlendMode::MULTIPLY,
            BlendMode::new(Mix::Multiply, Compose::SrcOver)
        );
        assert_eq!(
            BlendMode::SCREEN,
            BlendMode::new(Mix::Screen, Compose::SrcOver)
        );
        assert_eq!(BlendMode::XOR, BlendMode::new(Mix::Normal, Compose::Xor));

        let all_compose = [
            Compose::Clear,
            Compose::Copy,
            Compose::Dest,
            Compose::SrcOver,
            Compose::DestOver,
            Compose::SrcIn,
            Compose::DestIn,
            Compose::SrcOut,
            Compose::DestOut,
            Compose::SrcAtop,
            Compose::DestAtop,
            Compose::Xor,
            Compose::Plus,
            Compose::PlusLighter,
        ];
        let dst = [100, 150, 200, 220];
        for compose in all_compose {
            let mode = BlendMode::new(Mix::Normal, compose);
            let out = mode.blend_pixel([0, 0, 0, 0], dst);
            if mode.is_destructive() {
                assert_eq!(out, [0, 0, 0, 0], "{compose:?} should clear dst");
            } else {
                assert_eq!(out, dst, "{compose:?} should preserve dst");
            }
        }
    }

    #[test]
    fn porter_duff_compose_modes() {
        // Semitransparent premultiplied pixels:
        // src: straight (255, 0, 0) at alpha 128 -> premul [128, 0, 0, 128]
        // dst: straight (0, 0, 255) at alpha 128 -> premul [0, 0, 128, 128]
        let src = [128, 0, 0, 128];
        let dst = [0, 0, 128, 128];

        assert_eq!(BlendMode::CLEAR.blend_pixel(src, dst), [0, 0, 0, 0]);
        assert_eq!(BlendMode::COPY.blend_pixel(src, dst), src);
        assert_eq!(BlendMode::DEST.blend_pixel(src, dst), dst);
        // SrcOver: src + (1 - 128/255)*dst ≈ [128, 0, 64, 192]
        assert_pixel_near(
            BlendMode::SRC_OVER.blend_pixel(src, dst),
            [128, 0, 64, 192],
            1,
        );
        // DestOver: (1 - 128/255)*src + dst ≈ [64, 0, 128, 192]
        assert_pixel_near(
            BlendMode::DEST_OVER.blend_pixel(src, dst),
            [64, 0, 128, 192],
            1,
        );
        // SrcIn: da * src ≈ [64, 0, 0, 64]
        assert_pixel_near(BlendMode::SRC_IN.blend_pixel(src, dst), [64, 0, 0, 64], 1);
        // DestIn: sa * dst ≈ [0, 0, 64, 64]
        assert_pixel_near(BlendMode::DEST_IN.blend_pixel(src, dst), [0, 0, 64, 64], 1);
        // SrcOut: (1 - da) * src ≈ [64, 0, 0, 64]
        assert_pixel_near(BlendMode::SRC_OUT.blend_pixel(src, dst), [64, 0, 0, 64], 1);
        // DestOut: (1 - sa) * dst ≈ [0, 0, 64, 64]
        assert_pixel_near(BlendMode::DEST_OUT.blend_pixel(src, dst), [0, 0, 64, 64], 1);
        // SrcAtop: da * src + (1 - sa) * dst ≈ [64, 0, 64, 128]
        assert_pixel_near(
            BlendMode::SRC_ATOP.blend_pixel(src, dst),
            [64, 0, 64, 128],
            1,
        );
        // DestAtop: (1 - da) * src + sa * dst ≈ [64, 0, 64, 128]
        assert_pixel_near(
            BlendMode::DEST_ATOP.blend_pixel(src, dst),
            [64, 0, 64, 128],
            1,
        );
        // Xor: (1 - da) * src + (1 - sa) * dst ≈ [64, 0, 64, 128]
        assert_pixel_near(BlendMode::XOR.blend_pixel(src, dst), [64, 0, 64, 128], 1);
        // Plus / PlusLighter: saturating sum
        assert_eq!(BlendMode::PLUS.blend_pixel(src, dst), [128, 0, 128, 255]);
        assert_eq!(
            BlendMode::PLUS_LIGHTER.blend_pixel([200, 100, 50, 200], [100, 200, 50, 200]),
            [255, 255, 100, 255]
        );
    }

    #[test]
    fn w3c_separable_mix_modes_opaque() {
        // Opaque source (0.8, 0.4, 0.2) over opaque backdrop (0.25, 0.5, 0.75).
        let src = [204, 102, 51, 255];
        let dst = [64, 128, 191, 255];

        // Multiply: [0.2, 0.2, 0.15] -> [51, 51, 38]
        assert_pixel_near(
            BlendMode::MULTIPLY.blend_pixel(src, dst),
            [51, 51, 38, 255],
            1,
        );
        // Screen: [0.85, 0.70, 0.80] -> [217, 179, 204]
        assert_pixel_near(
            BlendMode::SCREEN.blend_pixel(src, dst),
            [217, 179, 204, 255],
            1,
        );
        // Overlay: cb <= 0.5 -> 2*cb*cs else 1 - 2*(1-cb)*(1-cs)
        // r: 2*0.25*0.8 = 0.40 (102), g: 2*0.5*0.4 = 0.40 (102), b: 1 - 2*0.25*0.8 = 0.60 (153)
        assert_pixel_near(
            BlendMode::OVERLAY.blend_pixel(src, dst),
            [102, 102, 153, 255],
            1,
        );
        // Darken / Lighten
        assert_eq!(BlendMode::DARKEN.blend_pixel(src, dst), [64, 102, 51, 255]);
        assert_eq!(
            BlendMode::LIGHTEN.blend_pixel(src, dst),
            [204, 128, 191, 255]
        );
        // ColorDodge: r: 0.25/0.2 = 1.0 (255), g: 0.5/0.6 = 0.833 (212), b: 0.75/0.8 = 0.9375 (239)
        assert_pixel_near(
            BlendMode::COLOR_DODGE.blend_pixel(src, dst),
            [255, 213, 239, 255],
            1,
        );
        // ColorDodge edge cases: cb == 0 -> 0 even if cs == 1; cb > 0 && cs == 1 -> 1
        assert_eq!(
            BlendMode::COLOR_DODGE.blend_pixel([255, 255, 0, 255], [0, 10, 0, 255]),
            [0, 255, 0, 255]
        );
        // ColorBurn: r: 1 - 0.75/0.8 = 0.0625 (16), g: 1 - min(1, 0.5/0.4) = 0, b: 1 - min(1, 0.25/0.2) = 0
        assert_pixel_near(
            BlendMode::COLOR_BURN.blend_pixel(src, dst),
            [16, 0, 0, 255],
            1,
        );
        // ColorBurn edge cases: cb == 1 -> 1 even if cs == 0; cb < 1 && cs == 0 -> 0
        assert_eq!(
            BlendMode::COLOR_BURN.blend_pixel([0, 0, 255, 255], [255, 200, 255, 255]),
            [255, 0, 255, 255]
        );
        // HardLight: cs <= 0.5 -> 2*cb*cs else 1 - 2*(1-cb)*(1-cs)
        // r (cs=0.8): 1 - 2*0.75*0.2 = 0.70 (179), g (cs=0.4): 2*0.5*0.4 = 0.40 (102), b (cs=0.2): 2*0.75*0.2 = 0.30 (77)
        assert_pixel_near(
            BlendMode::HARD_LIGHT.blend_pixel(src, dst),
            [179, 102, 77, 255],
            1,
        );
        // SoftLight:
        // r (cs=0.8 > 0.5, cb=0.25 <= 0.25): d = 0.5, 0.25 + 0.6*(0.5 - 0.25) = 0.40 (102)
        // g (cs=0.4 <= 0.5, cb=0.5): 0.5 - 0.2*0.25 = 0.45 (115)
        // b (cs=0.2 <= 0.5, cb=0.75): 0.75 - 0.6*0.75*0.25 = 0.6375 (163)
        assert_pixel_near(
            BlendMode::SOFT_LIGHT.blend_pixel(src, dst),
            [102, 115, 163, 255],
            1,
        );
        // SoftLight with cb > 0.25 and cs > 0.5 (exercises sqrt_unit branch):
        // cb = 0.64 (163), cs = 1.0 (255) -> d = sqrt(0.64) = 0.8 -> 0.8 * 255 = 204
        assert_pixel_near(
            BlendMode::SOFT_LIGHT.blend_pixel([255, 255, 255, 255], [163, 163, 163, 255]),
            [204, 204, 204, 255],
            1,
        );
        // Difference: |cb - cs| = [140, 26, 140]
        assert_pixel_near(
            BlendMode::DIFFERENCE.blend_pixel(src, dst),
            [140, 26, 140, 255],
            1,
        );
        // Exclusion: cb + cs - 2*cb*cs = [0.65, 0.50, 0.65] -> [166, 128, 166]
        assert_pixel_near(
            BlendMode::EXCLUSION.blend_pixel(src, dst),
            [166, 128, 166, 255],
            1,
        );
    }

    #[test]
    fn w3c_hsl_mix_modes() {
        let red = [255, 0, 0, 255];
        let blue = [0, 0, 255, 255];
        let mid_gray = [128, 128, 128, 255];

        // Luminosity of red over gray -> gray with red's luminance (0.3 * 255 ≈ 77).
        assert_pixel_near(
            BlendMode::LUMINOSITY.blend_pixel(red, mid_gray),
            [77, 77, 77, 255],
            1,
        );
        // Color of gray over red -> desaturated gray with red's luminance (0.3 * 255 ≈ 77).
        assert_pixel_near(
            BlendMode::COLOR.blend_pixel(mid_gray, red),
            [77, 77, 77, 255],
            1,
        );
        // Saturation of gray (sat=0) over red -> gray with red's luminance (77).
        assert_pixel_near(
            BlendMode::SATURATION.blend_pixel(mid_gray, red),
            [77, 77, 77, 255],
            1,
        );
        // Hue of red over gray (sat=0) -> gray unchanged.
        assert_pixel_near(BlendMode::HUE.blend_pixel(red, mid_gray), mid_gray, 1);

        // Hue of red over blue: red's hue (r max, g=b=0), blue's sat (1.0), blue's lum (0.11).
        // set_sat([1,0,0], 1) = [1,0,0]; set_lum([1,0,0], 0.11) -> [0.81, -0.19, -0.19] -> clip_color:
        // l = 0.11, n = -0.19, scale = 0.11 / 0.30 = 0.3667 -> r = 0.11 + 0.7*0.3667 = 0.3667 (94), g=b=0.
        assert_pixel_near(BlendMode::HUE.blend_pixel(red, blue), [94, 0, 0, 255], 1);
    }

    #[test]
    fn mix_combined_with_semitransparency_and_non_src_over_compose() {
        // Semitransparent multiply over semitransparent backdrop:
        // src = straight [255, 0, 0] @ 0.5 -> premul [128, 0, 0, 128]
        // dst = straight [255, 255, 0] @ 0.5 -> premul [128, 128, 0, 128]
        // B(cb, cs) = [1, 0, 0]; Cs_out = 0.5*[1,0,0] + 0.5*[1,0,0] = [1,0,0]; ms = [0.5, 0, 0, 0.5]
        // SrcOver: ms + 0.5*md = [0.75, 0.25, 0, 0.75] -> [191, 64, 0, 191]
        let src = [128, 0, 0, 128];
        let dst = [128, 128, 0, 128];
        assert_pixel_near(
            BlendMode::MULTIPLY.blend_pixel(src, dst),
            [191, 64, 0, 191],
            1,
        );

        // Multiply + SrcAtop: ba * ms + (1 - sa) * md = 0.5*[0.5,0,0,0.5] + 0.5*[0.5,0.5,0,0.5] = [0.5,0.25,0,0.5]
        let mul_atop = BlendMode::new(Mix::Multiply, Compose::SrcAtop);
        assert_pixel_near(mul_atop.blend_pixel(src, dst), [128, 64, 0, 128], 1);

        // Mix + Clear / Dest ignore mix and follow Compose.
        assert_eq!(
            BlendMode::new(Mix::Color, Compose::Clear).blend_pixel(src, dst),
            [0, 0, 0, 0]
        );
        assert_eq!(
            BlendMode::new(Mix::Hue, Compose::Dest).blend_pixel(src, dst),
            dst
        );

        // Transparent backdrop with non-Normal Mix reduces to plain Compose.
        assert_eq!(BlendMode::MULTIPLY.blend_pixel(src, [0, 0, 0, 0]), src);
        assert_eq!(BlendMode::MULTIPLY.blend_pixel([0, 0, 0, 0], dst), dst);
    }

    #[test]
    fn composite_clipping_opacity_and_strided_padding() {
        let mut src = Pixmap::new(2, 2);
        src.fill([200, 0, 0, 200]);

        // Interior composite at 50% opacity (opacity = 128).
        let mut dst = Pixmap::new(4, 4);
        let bg = [0, 0, 255, 255];
        dst.fill(bg);
        dst.composite(src.as_ref(), 1, 1, BlendMode::SRC_OVER, 128);

        let expected = BlendMode::SRC_OVER.blend_pixel(scale_pixel([200, 0, 0, 200], 128), bg);
        for y in 0..4 {
            for x in 0..4 {
                if (1..3).contains(&x) && (1..3).contains(&y) {
                    assert_eq!(dst.pixel(x, y), expected);
                } else {
                    assert_eq!(dst.pixel(x, y), bg);
                }
            }
        }

        // Opacity 0: non-destructive leaves dst untouched; destructive clears the clipped rect.
        let before = dst.clone();
        dst.composite(src.as_ref(), 0, 0, BlendMode::SRC_OVER, 0);
        assert_eq!(dst, before);

        dst.composite(src.as_ref(), -1, -1, BlendMode::COPY, 0);
        assert_eq!(dst.pixel(0, 0), [0, 0, 0, 0]);
        assert_eq!(dst.pixel(1, 0), bg);

        // Out-of-bounds and empty are no-ops even for Clear.
        let before = dst.clone();
        dst.composite(src.as_ref(), 4, 0, BlendMode::CLEAR, 255);
        dst.composite(src.as_ref(), i32::MIN, i32::MAX, BlendMode::CLEAR, 255);
        dst.composite(Pixmap::new(0, 2).as_ref(), 0, 0, BlendMode::CLEAR, 255);
        assert_eq!(dst, before);

        // Strided destination and source preserve padding bytes.
        let pad = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut dst_buf = vec![0u8; 2 * 16];
        for row in 0..2 {
            for x in 0..3 {
                let off = row * 16 + x * 4;
                dst_buf[off..off + 4].copy_from_slice(&[0, 100, 200, 255]);
            }
            dst_buf[row * 16 + 12..row * 16 + 16].copy_from_slice(&pad);
        }
        let mut src_buf = vec![0u8; 2 * 16];
        for row in 0..2 {
            for x in 0..2 {
                let off = row * 16 + x * 4;
                src_buf[off..off + 4].copy_from_slice(&[128, 64, 0, 128]);
            }
            src_buf[row * 16 + 8..row * 16 + 16].fill(0x55);
        }
        {
            let src_view = PixmapRef::from_parts_with_stride(&src_buf, 2, 2, 16);
            let mut dst_view = PixmapMut::from_parts_with_stride(&mut dst_buf, 3, 2, 16);
            dst_view.composite(src_view, 1, 0, BlendMode::MULTIPLY, 255);
            let blended = BlendMode::MULTIPLY.blend_pixel([128, 64, 0, 128], [0, 100, 200, 255]);
            assert_eq!(dst_view.pixel(0, 0), [0, 100, 200, 255]);
            assert_eq!(dst_view.pixel(1, 0), blended);
            assert_eq!(dst_view.pixel(2, 1), blended);

            dst_view.composite_color([0, 0, 0, 128], BlendMode::DEST_IN);
            assert_eq!(
                dst_view.pixel(0, 0),
                BlendMode::DEST_IN.blend_pixel([0, 0, 0, 128], [0, 100, 200, 255])
            );
        }
        for row in 0..2 {
            assert_eq!(&dst_buf[row * 16 + 12..row * 16 + 16], &pad);
        }
    }

    #[test]
    fn composite_color_on_pixmap_and_subregion() {
        let mut pixmap = Pixmap::new(3, 2);
        pixmap.fill([200, 100, 50, 255]);
        pixmap
            .subregion_mut(1, 0, 2, 1)
            .composite_color([0, 128, 0, 128], BlendMode::SRC_OVER);

        let expected = BlendMode::SRC_OVER.blend_pixel([0, 128, 0, 128], [200, 100, 50, 255]);
        assert_eq!(pixmap.pixel(0, 0), [200, 100, 50, 255]);
        assert_eq!(pixmap.pixel(1, 0), expected);
        assert_eq!(pixmap.pixel(2, 0), expected);
        assert_eq!(pixmap.pixel(1, 1), [200, 100, 50, 255]);

        pixmap.composite_color([10, 20, 30, 255], BlendMode::COPY);
        assert_eq!(pixmap.pixel(0, 0), [10, 20, 30, 255]);
        assert_eq!(pixmap.pixel(2, 1), [10, 20, 30, 255]);
    }

    #[test]
    #[should_panic(expected = "must match source slice length")]
    fn composite_rgba8_panics_on_mismatched_lengths() {
        let mut dst = [0u8; 8];
        composite_rgba8(&mut dst, &[0u8; 4], BlendMode::DEST, 255);
    }

    #[test]
    #[should_panic(expected = "multiple of 4")]
    fn composite_rgba8_panics_on_non_multiple_of_four() {
        let mut dst = [0u8; 5];
        composite_rgba8(&mut dst, &[0u8; 5], BlendMode::DEST, 255);
    }

    #[test]
    #[should_panic(expected = "multiple of 4")]
    fn composite_color_rgba8_panics_on_non_multiple_of_four() {
        let mut dst = [0u8; 3];
        composite_color_rgba8(&mut dst, [0; 4], BlendMode::DEST);
    }

    fn reference_w3c_blend_pixel(
        mix: Mix,
        compose: Compose,
        src: [u8; 4],
        dst: [u8; 4],
    ) -> [u8; 4] {
        let sa = f32::from(src[3]) / 255.0;
        let ba = f32::from(dst[3]) / 255.0;
        let sr = f32::from(src[0]) / 255.0;
        let sg = f32::from(src[1]) / 255.0;
        let sb = f32::from(src[2]) / 255.0;
        let br = f32::from(dst[0]) / 255.0;
        let bg = f32::from(dst[1]) / 255.0;
        let bb = f32::from(dst[2]) / 255.0;

        let cs = if sa == 0.0 {
            [0.0; 3]
        } else {
            [(sr / sa).min(1.0), (sg / sa).min(1.0), (sb / sa).min(1.0)]
        };
        let cb = if ba == 0.0 {
            [0.0; 3]
        } else {
            [(br / ba).min(1.0), (bg / ba).min(1.0), (bb / ba).min(1.0)]
        };

        let screen = |s: f32, b: f32| b + s - s * b;
        let hard_light = |s: f32, b: f32| {
            if s <= 0.5 {
                b * (2.0 * s)
            } else {
                screen(2.0 * s - 1.0, b)
            }
        };
        let b_rgb = match mix {
            Mix::Normal => cs,
            Mix::Multiply => [cs[0] * cb[0], cs[1] * cb[1], cs[2] * cb[2]],
            Mix::Screen => [
                screen(cs[0], cb[0]),
                screen(cs[1], cb[1]),
                screen(cs[2], cb[2]),
            ],
            Mix::Overlay => [
                hard_light(cb[0], cs[0]),
                hard_light(cb[1], cs[1]),
                hard_light(cb[2], cs[2]),
            ],
            Mix::Darken => [cs[0].min(cb[0]), cs[1].min(cb[1]), cs[2].min(cb[2])],
            Mix::Lighten => [cs[0].max(cb[0]), cs[1].max(cb[1]), cs[2].max(cb[2])],
            Mix::ColorDodge => [
                color_dodge_ch(cs[0], cb[0]),
                color_dodge_ch(cs[1], cb[1]),
                color_dodge_ch(cs[2], cb[2]),
            ],
            Mix::ColorBurn => [
                color_burn_ch(cs[0], cb[0]),
                color_burn_ch(cs[1], cb[1]),
                color_burn_ch(cs[2], cb[2]),
            ],
            Mix::HardLight => [
                hard_light(cs[0], cb[0]),
                hard_light(cs[1], cb[1]),
                hard_light(cs[2], cb[2]),
            ],
            Mix::SoftLight => [
                soft_light_ch(cs[0], cb[0]),
                soft_light_ch(cs[1], cb[1]),
                soft_light_ch(cs[2], cb[2]),
            ],
            Mix::Difference => [
                (cb[0] - cs[0]).abs(),
                (cb[1] - cs[1]).abs(),
                (cb[2] - cs[2]).abs(),
            ],
            Mix::Exclusion => [
                cb[0] + cs[0] - 2.0 * cb[0] * cs[0],
                cb[1] + cs[1] - 2.0 * cb[1] * cs[1],
                cb[2] + cs[2] - 2.0 * cb[2] * cs[2],
            ],
            Mix::Hue => set_lum(set_sat(cs, sat(cb)), lum(cb)),
            Mix::Saturation => set_lum(set_sat(cb, sat(cs)), lum(cb)),
            Mix::Color => set_lum(cs, lum(cb)),
            Mix::Luminosity => set_lum(cb, lum(cs)),
        };

        let ms = if sa == 0.0 || ba == 0.0 {
            [sr, sg, sb, sa]
        } else {
            let inv_ba = 1.0 - ba;
            [
                ((inv_ba * cs[0] + ba * b_rgb[0]) * sa).clamp(0.0, 1.0),
                ((inv_ba * cs[1] + ba * b_rgb[1]) * sa).clamp(0.0, 1.0),
                ((inv_ba * cs[2] + ba * b_rgb[2]) * sa).clamp(0.0, 1.0),
                sa,
            ]
        };
        let md = [br, bg, bb, ba];
        let (fa, fb) = match compose {
            Compose::Clear => (0.0, 0.0),
            Compose::Copy => (1.0, 0.0),
            Compose::Dest => (0.0, 1.0),
            Compose::SrcOver => (1.0, 1.0 - sa),
            Compose::DestOver => (1.0 - ba, 1.0),
            Compose::SrcIn => (ba, 0.0),
            Compose::DestIn => (0.0, sa),
            Compose::SrcOut => (1.0 - ba, 0.0),
            Compose::DestOut => (0.0, 1.0 - sa),
            Compose::SrcAtop => (ba, 1.0 - sa),
            Compose::DestAtop => (1.0 - ba, sa),
            Compose::Xor => (1.0 - ba, 1.0 - sa),
            Compose::Plus | Compose::PlusLighter => (1.0, 1.0),
        };
        let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        [
            to_u8(fa * ms[0] + fb * md[0]),
            to_u8(fa * ms[1] + fb * md[1]),
            to_u8(fa * ms[2] + fb * md[2]),
            to_u8(fa * ms[3] + fb * md[3]),
        ]
    }

    #[test]
    fn all_mix_and_compose_paths_match_w3c_reference_across_tiles() {
        let all_mix = [
            Mix::Normal,
            Mix::Multiply,
            Mix::Screen,
            Mix::Overlay,
            Mix::Darken,
            Mix::Lighten,
            Mix::ColorDodge,
            Mix::ColorBurn,
            Mix::HardLight,
            Mix::SoftLight,
            Mix::Difference,
            Mix::Exclusion,
            Mix::Hue,
            Mix::Saturation,
            Mix::Color,
            Mix::Luminosity,
        ];
        let all_compose = [
            Compose::Clear,
            Compose::Copy,
            Compose::Dest,
            Compose::SrcOver,
            Compose::DestOver,
            Compose::SrcIn,
            Compose::DestIn,
            Compose::SrcOut,
            Compose::DestOut,
            Compose::SrcAtop,
            Compose::DestAtop,
            Compose::Xor,
            Compose::Plus,
            Compose::PlusLighter,
        ];
        let samples: &[[u8; 4]] = &[
            [0, 0, 0, 0],
            [0, 0, 0, 255],
            [255, 255, 255, 255],
            [200, 100, 50, 255],
            [40, 180, 240, 255],
            [128, 64, 32, 128],
            [10, 90, 150, 180],
            [200, 20, 180, 200],
            [1, 0, 1, 1],
        ];
        // Build an 81-pixel buffer (> MIX_TILE_LEN = 64) covering every (src, dst) pair.
        let mut src_bytes = vec![0u8; samples.len() * samples.len() * 4];
        let mut dst_init = vec![0u8; samples.len() * samples.len() * 4];
        for (i, &s) in samples.iter().enumerate() {
            for (j, &d) in samples.iter().enumerate() {
                let off = (i * samples.len() + j) * 4;
                src_bytes[off..off + 4].copy_from_slice(&s);
                dst_init[off..off + 4].copy_from_slice(&d);
            }
        }

        for mix in all_mix {
            for compose in all_compose {
                let mode = BlendMode::new(mix, compose);
                let mut dst_buf = dst_init.clone();
                composite_rgba8(&mut dst_buf, &src_bytes, mode, 255);

                for (idx, (d_out, (&s, &d_in))) in dst_buf
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(
                        src_bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .zip(dst_init.as_chunks::<4>().0),
                    )
                    .enumerate()
                {
                    let px_out = mode.blend_pixel(s, d_in);
                    assert_eq!(
                        *d_out, px_out,
                        "slice vs pixel mismatch for {mode:?} at {idx}"
                    );
                    let expected = reference_w3c_blend_pixel(mix, compose, s, d_in);
                    assert_pixel_near(*d_out, expected, 1);
                }

                // Also verify `composite_color_rgba8` (both opaque and semitransparent colors).
                for &color in &[[0, 0, 0, 0], [200, 100, 50, 255], [128, 64, 32, 128]] {
                    let mut color_buf = dst_init.clone();
                    composite_color_rgba8(&mut color_buf, color, mode);
                    for (d_out, &d_in) in color_buf
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(dst_init.as_chunks::<4>().0)
                    {
                        assert_eq!(
                            *d_out,
                            mode.blend_pixel(color, d_in),
                            "composite_color mismatch for {mode:?} color={color:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn color_burn_and_color_dodge_endpoints_hold_for_every_nonzero_alpha() {
        for a in 1u8..=255 {
            // Color Burn: white backdrop (`cb == 1.0`, premul `c == a`) stays white even under
            // opaque black source (`cs == 0.0`), producing `[a, a, a, 255]`.
            assert_eq!(
                BlendMode::COLOR_BURN.blend_pixel([0, 0, 0, 255], [a, a, a, a]),
                [a, a, a, 255],
                "ColorBurn white-backdrop endpoint failed at alpha={a}"
            );
            assert_eq!(
                BlendMode::COLOR_BURN.blend_pixel([0, 0, 0, 255], [a, 0, 0, a]),
                [a, 0, 0, 255],
                "ColorBurn single-channel white-backdrop endpoint failed at alpha={a}"
            );
            let semi_burn = BlendMode::COLOR_BURN.blend_pixel([0, 0, 0, 128], [a, a, a, a]);
            let semi_burn_ref = reference_w3c_blend_pixel(
                Mix::ColorBurn,
                Compose::SrcOver,
                [0, 0, 0, 128],
                [a, a, a, a],
            );
            assert_pixel_near(semi_burn, semi_burn_ref, 1);

            // Color Dodge: white source (`cs == 1.0`, premul `c == a`) over non-black backdrop
            // produces 1.0, while black backdrop (`cb == 0.0`) stays black.
            assert_eq!(
                BlendMode::COLOR_DODGE.blend_pixel([a, a, a, a], [0, 0, 0, 255]),
                [0, 0, 0, 255],
                "ColorDodge black-backdrop endpoint failed at alpha={a}"
            );
            let dodge = BlendMode::COLOR_DODGE.blend_pixel([a, a, a, a], [1, 1, 1, 255]);
            let dodge_ref = reference_w3c_blend_pixel(
                Mix::ColorDodge,
                Compose::SrcOver,
                [a, a, a, a],
                [1, 1, 1, 255],
            );
            assert_pixel_near(dodge, dodge_ref, 1);
        }
    }

    #[test]
    fn opacity_scales_the_mixed_source_not_the_mix_input() {
        // W3C blending is defined on unpremultiplied colours, so a uniform opacity must not change
        // the colour fed to the blend function: only the mixed result's coverage scales.
        let mixes = [
            Mix::Multiply,
            Mix::Screen,
            Mix::Overlay,
            Mix::Darken,
            Mix::Lighten,
            Mix::ColorDodge,
            Mix::ColorBurn,
            Mix::HardLight,
            Mix::SoftLight,
            Mix::Difference,
            Mix::Exclusion,
            Mix::Hue,
            Mix::Saturation,
            Mix::Color,
            Mix::Luminosity,
        ];
        let mut pairs = vec![
            // Colour Dodge / Colour Burn: sources one step from the `cs == 1` / `cs == 0` endpoint.
            ([254, 254, 254, 255], [1, 1, 1, 255]),
            ([1, 1, 1, 255], [254, 254, 254, 255]),
            // Hue: a nearly grey source whose hue is lost when its channels are quantised.
            ([116, 115, 115, 255], [146, 200, 13, 255]),
            ([254, 254, 254, 255], [253, 254, 1, 254]),
        ];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut premul_pixel = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let [r, g, b, a, ..] = state.to_le_bytes();
            let mut px = [r, g, b, a];
            crate::premultiply_rgba8(&mut px);
            px
        };
        for _ in 0..2000 {
            pairs.push((premul_pixel(), premul_pixel()));
        }

        for mix in mixes {
            for opacity in [1u8, 64, 128, 200, 254] {
                for &(src, dst) in &pairs {
                    let mixed = reference_w3c_blend_pixel(mix, Compose::Copy, src, dst);
                    let o = f32::from(opacity) / 255.0;
                    let keep = 1.0 - f32::from(mixed[3]) * o / 255.0;
                    let expected: [u8; 4] = core::array::from_fn(|i| {
                        (f32::from(mixed[i]) * o + f32::from(dst[i]) * keep + 0.5) as u8
                    });

                    let mut actual = dst;
                    composite_rgba8(
                        &mut actual,
                        &src,
                        BlendMode::new(mix, Compose::SrcOver),
                        opacity,
                    );
                    for i in 0..4 {
                        assert!(
                            actual[i].abs_diff(expected[i]) <= 2,
                            "{mix:?} opacity={opacity} src={src:?} dst={dst:?}: \
                             got {actual:?}, expected {expected:?}"
                        );
                    }
                }
            }
        }
    }
}
