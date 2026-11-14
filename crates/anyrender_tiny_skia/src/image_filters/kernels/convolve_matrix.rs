// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use image_filters_types::{EdgeMode, convolution::ConvolutionKernel};
use rgb::RGBA8;

use super::{ImageRefMut, f32_bound};

/// Applies a convolve matrix.
///
/// Input image pixels should have a **premultiplied alpha** when `preserve_alpha=false`.
pub fn apply(kernel: &ConvolutionKernel, src: ImageRefMut) {
    let (columns, rows) = (kernel.columns as i32, kernel.rows as i32);
    if columns == 0
        || rows == 0
        || kernel.values.len() != (columns * rows) as usize
        || kernel.target_x >= kernel.columns
        || kernel.target_y >= kernel.rows
    {
        return;
    }
    let (target_x, target_y) = (kernel.target_x as i32, kernel.target_y as i32);
    let (width, height) = (src.width as i32, src.height as i32);
    // As in SVG, a zero divisor falls back to the sum of the kernel, or 1 if that is zero too.
    let divisor = match kernel.divisor {
        0.0 => match kernel.values.iter().sum::<f32>() {
            0.0 => 1.0,
            sum => sum,
        },
        divisor => divisor,
    };

    let width_max = src.width as i32 - 1;
    let height_max = src.height as i32 - 1;

    let mut buf = vec![RGBA8::default(); src.data.len()];
    let mut buf = ImageRefMut::new(src.width, src.height, &mut buf);
    let mut x = 0;
    let mut y = 0;
    for in_p in src.data.iter() {
        let mut new_r = 0.0;
        let mut new_g = 0.0;
        let mut new_b = 0.0;
        let mut new_a = 0.0;
        for oy in 0..rows {
            for ox in 0..columns {
                let mut tx = x as i32 - target_x + ox;
                let mut ty = y as i32 - target_y + oy;

                match kernel.edge_mode {
                    EdgeMode::None => {
                        if tx < 0 || tx > width_max || ty < 0 || ty > height_max {
                            continue;
                        }
                    }
                    EdgeMode::Duplicate => {
                        tx = tx.clamp(0, width_max);
                        ty = ty.clamp(0, height_max);
                    }
                    EdgeMode::Wrap => {
                        tx = tx.rem_euclid(width);
                        ty = ty.rem_euclid(height);
                    }
                    EdgeMode::Mirror => {
                        let mirror = |v: i32, len: i32| {
                            let v = v.rem_euclid(2 * len);
                            if v < len { v } else { 2 * len - 1 - v }
                        };
                        tx = mirror(tx, width);
                        ty = mirror(ty, height);
                    }
                }

                let k = kernel.values[((rows - oy - 1) * columns + (columns - ox - 1)) as usize];

                let p = src.pixel_at(tx as u32, ty as u32);
                new_r += (p.r as f32) / 255.0 * k;
                new_g += (p.g as f32) / 255.0 * k;
                new_b += (p.b as f32) / 255.0 * k;

                if !kernel.preserve_alpha {
                    new_a += (p.a as f32) / 255.0 * k;
                }
            }
        }

        if kernel.preserve_alpha {
            new_a = in_p.a as f32 / 255.0;
        } else {
            new_a = new_a / divisor + kernel.bias;
        }

        let bounded_new_a = f32_bound(0.0, new_a, 1.0);

        let calc = |x| {
            let x = x / divisor + kernel.bias * new_a;

            let x = if kernel.preserve_alpha {
                f32_bound(0.0, x, 1.0) * bounded_new_a
            } else {
                f32_bound(0.0, x, bounded_new_a)
            };

            (x * 255.0 + 0.5) as u8
        };

        let out_p = buf.pixel_at_mut(x, y);
        out_p.r = calc(new_r);
        out_p.g = calc(new_g);
        out_p.b = calc(new_b);
        out_p.a = (bounded_new_a * 255.0 + 0.5) as u8;

        x += 1;
        if x == src.width {
            x = 0;
            y += 1;
        }
    }

    src.data.copy_from_slice(buf.data);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convolve_row(edge_mode: EdgeMode) -> Vec<(u8, u8)> {
        let mut pixels = [10, 20, 30].map(|r| RGBA8::new(r, 0, 0, 255));
        let kernel = ConvolutionKernel {
            columns: 2,
            rows: 1,
            values: vec![1.0, 0.0],
            target_x: 0,
            target_y: 0,
            edge_mode,
            divisor: 1.0,
            bias: 0.0,
            preserve_alpha: false,
        };
        apply(&kernel, ImageRefMut::new(3, 1, &mut pixels));
        pixels.iter().map(|p| (p.r, p.a)).collect()
    }

    #[test]
    fn non_square_kernel_with_target_and_edge_modes() {
        assert_eq!(
            convolve_row(EdgeMode::Wrap),
            [(20, 255), (30, 255), (10, 255)]
        );
        assert_eq!(
            convolve_row(EdgeMode::Duplicate),
            [(20, 255), (30, 255), (30, 255)]
        );
        assert_eq!(convolve_row(EdgeMode::None), [(20, 255), (30, 255), (0, 0)]);
        assert_eq!(
            convolve_row(EdgeMode::Mirror),
            [(20, 255), (30, 255), (30, 255)]
        );
    }
}
