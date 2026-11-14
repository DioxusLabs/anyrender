// Copyright 2018 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Pixel kernels of each filter primitive on RGBA8 images, ported from resvg
//! (`crates/resvg/src/filter`).

use rgb::RGBA8;

pub mod box_blur;
pub mod color_matrix;
pub mod component_transfer;
pub mod composite;
pub mod convolve_matrix;
pub mod displacement_map;
pub mod iir_blur;
pub mod lighting;
pub mod morphology;
pub mod turbulence;

/// An image reference.
///
/// Image pixels should be stored in RGBA order.
///
/// Some filters will require premultiplied channels, some not.
/// See specific filter documentation for details.
#[derive(Clone, Copy)]
pub struct ImageRef<'a> {
    data: &'a [RGBA8],
    width: u32,
    height: u32,
}

impl<'a> ImageRef<'a> {
    /// Creates a new image reference.
    ///
    /// Doesn't clone the provided data.
    #[inline]
    pub fn new(width: u32, height: u32, data: &'a [RGBA8]) -> Self {
        ImageRef {
            data,
            width,
            height,
        }
    }

    #[inline]
    fn alpha_at(&self, x: u32, y: u32) -> i16 {
        self.data[(self.width * y + x) as usize].a as i16
    }
}

/// A mutable `ImageRef` variant.
pub struct ImageRefMut<'a> {
    data: &'a mut [RGBA8],
    width: u32,
    height: u32,
}

impl<'a> ImageRefMut<'a> {
    /// Creates a new mutable image reference.
    ///
    /// Doesn't clone the provided data.
    #[inline]
    pub fn new(width: u32, height: u32, data: &'a mut [RGBA8]) -> Self {
        ImageRefMut {
            data,
            width,
            height,
        }
    }

    #[inline]
    fn pixel_at(&self, x: u32, y: u32) -> RGBA8 {
        self.data[(self.width * y + x) as usize]
    }

    #[inline]
    fn pixel_at_mut(&mut self, x: u32, y: u32) -> &mut RGBA8 {
        &mut self.data[(self.width * y + x) as usize]
    }
}

/// Multiplies provided pixels alpha.
pub fn multiply_alpha(data: &mut [RGBA8]) {
    for p in data {
        let a = p.a as f32 / 255.0;
        p.b = (p.b as f32 * a + 0.5) as u8;
        p.g = (p.g as f32 * a + 0.5) as u8;
        p.r = (p.r as f32 * a + 0.5) as u8;
    }
}

/// Demultiplies provided pixels alpha.
pub fn demultiply_alpha(data: &mut [RGBA8]) {
    for p in data {
        let a = p.a as f32 / 255.0;
        p.b = (p.b as f32 / a + 0.5) as u8;
        p.g = (p.g as f32 / a + 0.5) as u8;
        p.r = (p.r as f32 / a + 0.5) as u8;
    }
}

/// Precomputed sRGB to LinearRGB table.
///
/// Since we are storing the result in `u8`, there is no need to compute those
/// values each time. Mainly because it's very expensive.
///
/// ```text
/// if (C_srgb <= 0.04045)
///     C_lin = C_srgb / 12.92;
///  else
///     C_lin = pow((C_srgb + 0.055) / 1.055, 2.4);
/// ```
///
/// Thanks to librsvg for the idea.
#[rustfmt::skip]
const SRGB_TO_LINEAR_RGB_TABLE: &[u8; 256] = &[
    0,   0,   0,   0,   0,   0,  0,    1,   1,   1,   1,   1,   1,   1,   1,   1,
    1,   1,   2,   2,   2,   2,  2,    2,   2,   2,   3,   3,   3,   3,   3,   3,
    4,   4,   4,   4,   4,   5,  5,    5,   5,   6,   6,   6,   6,   7,   7,   7,
    8,   8,   8,   8,   9,   9,  9,   10,  10,  10,  11,  11,  12,  12,  12,  13,
    13,  13,  14,  14,  15,  15,  16,  16,  17,  17,  17,  18,  18,  19,  19,  20,
    20,  21,  22,  22,  23,  23,  24,  24,  25,  25,  26,  27,  27,  28,  29,  29,
    30,  30,  31,  32,  32,  33,  34,  35,  35,  36,  37,  37,  38,  39,  40,  41,
    41,  42,  43,  44,  45,  45,  46,  47,  48,  49,  50,  51,  51,  52,  53,  54,
    55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,
    71,  72,  73,  74,  76,  77,  78,  79,  80,  81,  82,  84,  85,  86,  87,  88,
    90,  91,  92,  93,  95,  96,  97,  99, 100, 101, 103, 104, 105, 107, 108, 109,
    111, 112, 114, 115, 116, 118, 119, 121, 122, 124, 125, 127, 128, 130, 131, 133,
    134, 136, 138, 139, 141, 142, 144, 146, 147, 149, 151, 152, 154, 156, 157, 159,
    161, 163, 164, 166, 168, 170, 171, 173, 175, 177, 179, 181, 183, 184, 186, 188,
    190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220,
    222, 224, 226, 229, 231, 233, 235, 237, 239, 242, 244, 246, 248, 250, 253, 255,
];

/// Precomputed LinearRGB to sRGB table.
///
/// Since we are storing the result in `u8`, there is no need to compute those
/// values each time. Mainly because it's very expensive.
///
/// ```text
/// if (C_lin <= 0.0031308)
///     C_srgb = C_lin * 12.92;
/// else
///     C_srgb = 1.055 * pow(C_lin, 1.0 / 2.4) - 0.055;
/// ```
///
/// Thanks to librsvg for the idea.
#[rustfmt::skip]
const LINEAR_RGB_TO_SRGB_TABLE: &[u8; 256] = &[
    0,  13,  22,  28,  34,  38,  42,  46,  50,  53,  56,  59,  61,  64,  66,  69,
    71,  73,  75,  77,  79,  81,  83,  85,  86,  88,  90,  92,  93,  95,  96,  98,
    99, 101, 102, 104, 105, 106, 108, 109, 110, 112, 113, 114, 115, 117, 118, 119,
    120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
    137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 148, 149, 150, 151,
    152, 153, 154, 155, 155, 156, 157, 158, 159, 159, 160, 161, 162, 163, 163, 164,
    165, 166, 167, 167, 168, 169, 170, 170, 171, 172, 173, 173, 174, 175, 175, 176,
    177, 178, 178, 179, 180, 180, 181, 182, 182, 183, 184, 185, 185, 186, 187, 187,
    188, 189, 189, 190, 190, 191, 192, 192, 193, 194, 194, 195, 196, 196, 197, 197,
    198, 199, 199, 200, 200, 201, 202, 202, 203, 203, 204, 205, 205, 206, 206, 207,
    208, 208, 209, 209, 210, 210, 211, 212, 212, 213, 213, 214, 214, 215, 215, 216,
    216, 217, 218, 218, 219, 219, 220, 220, 221, 221, 222, 222, 223, 223, 224, 224,
    225, 226, 226, 227, 227, 228, 228, 229, 229, 230, 230, 231, 231, 232, 232, 233,
    233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238, 238, 239, 239, 240, 240,
    241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 246, 247, 247, 248,
    248, 249, 249, 250, 250, 251, 251, 251, 252, 252, 253, 253, 254, 254, 255, 255,
];

/// Converts input pixel from sRGB into LinearRGB.
///
/// Provided pixels should have an **unpremultiplied alpha**.
///
/// RGB channels order of the input image doesn't matter, but alpha channel must be the last one.
pub fn into_linear_rgb(data: &mut [RGBA8]) {
    for p in data {
        p.r = SRGB_TO_LINEAR_RGB_TABLE[p.r as usize];
        p.g = SRGB_TO_LINEAR_RGB_TABLE[p.g as usize];
        p.b = SRGB_TO_LINEAR_RGB_TABLE[p.b as usize];
    }
}

/// Converts input pixel from LinearRGB into sRGB.
///
/// Provided pixels should have an **unpremultiplied alpha**.
///
/// RGB channels order of the input image doesn't matter, but alpha channel must be the last one.
pub fn from_linear_rgb(data: &mut [RGBA8]) {
    for p in data {
        p.r = LINEAR_RGB_TO_SRGB_TABLE[p.r as usize];
        p.g = LINEAR_RGB_TO_SRGB_TABLE[p.g as usize];
        p.b = LINEAR_RGB_TO_SRGB_TABLE[p.b as usize];
    }
}

// TODO: https://github.com/rust-lang/rust/issues/44095
#[inline]
fn f32_bound(min: f32, val: f32, max: f32) -> f32 {
    debug_assert!(min.is_finite());
    debug_assert!(max.is_finite());

    if val > max {
        max
    } else if val >= min {
        val
    } else {
        // Catches `val < min` as well as a NaN `val`.
        min
    }
}
