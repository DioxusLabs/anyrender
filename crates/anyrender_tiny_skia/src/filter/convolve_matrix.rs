// Copyright 2020 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use anyrender::filters::convolution::ConvolutionKernel;
use rgb::RGBA8;

use super::{ImageRefMut, f32_bound};

/// Applies a convolve matrix, with the target at the kernel center and `edgeMode="duplicate"`
/// (the SVG defaults).
///
/// Input image pixels should have a **premultiplied alpha** when `preserve_alpha=false`.
pub fn apply(kernel: &ConvolutionKernel, src: ImageRefMut) {
    let size = kernel.size as i32;
    if size == 0 || kernel.values.len() != (size * size) as usize {
        return;
    }
    let target = size / 2;
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
        for oy in 0..size {
            for ox in 0..size {
                let tx = (x as i32 - target + ox).clamp(0, width_max);
                let ty = (y as i32 - target + oy).clamp(0, height_max);

                let k = kernel.values[((size - oy - 1) * size + (size - ox - 1)) as usize];

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
