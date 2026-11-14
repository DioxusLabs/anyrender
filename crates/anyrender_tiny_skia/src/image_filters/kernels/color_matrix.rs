// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use super::{ImageRefMut, f32_bound};
use image_filters_types::color_transformation::ColorMatrix;
use rgb::RGBA8;

/// Applies a color matrix filter.
///
/// `saturate`, `hueRotate` and `luminanceToAlpha` are expressed as a [`ColorMatrix`] by anyrender,
/// so only resvg's `matrix` case is needed.
///
/// Input image pixels should have an **unpremultiplied alpha**.
pub fn apply(matrix: &ColorMatrix, src: ImageRefMut) {
    let m = &matrix.0;
    for pixel in src.data {
        let (r, g, b, a) = to_normalized_components(*pixel);

        let new_r = r * m[0] + g * m[1] + b * m[2] + a * m[3] + m[4];
        let new_g = r * m[5] + g * m[6] + b * m[7] + a * m[8] + m[9];
        let new_b = r * m[10] + g * m[11] + b * m[12] + a * m[13] + m[14];
        let new_a = r * m[15] + g * m[16] + b * m[17] + a * m[18] + m[19];

        pixel.r = from_normalized(new_r);
        pixel.g = from_normalized(new_g);
        pixel.b = from_normalized(new_b);
        pixel.a = from_normalized(new_a);
    }
}

#[inline]
fn to_normalized_components(pixel: RGBA8) -> (f32, f32, f32, f32) {
    (
        pixel.r as f32 / 255.0,
        pixel.g as f32 / 255.0,
        pixel.b as f32 / 255.0,
        pixel.a as f32 / 255.0,
    )
}

#[inline]
fn from_normalized(c: f32) -> u8 {
    (f32_bound(0.0, c, 1.0) * 255.0) as u8
}
