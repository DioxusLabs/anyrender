//! Raw-kernel benchmark comparing `pixmap` imaging operations against `vello_cpu`'s
//! low-level `U8Kernel` (`vello_cpu::fine`) / `vello_common::pixmap` SIMD routines
//! and `tiny-skia`'s `RasterPipeline` (compile-time SIMD), without scene/tile-scheduling
//! overhead.
//!
//! Run on x86 or ARM/macOS with:
//!
//! ```sh
//! cargo bench -p pixmap --bench vs_vello_cpu
//! ```
//!
//! To compare `pixmap` auto-vectorization and `tiny-skia` compile-time SIMD under host
//! ISA (e.g. AVX2 or NEON) as well:
//!
//! ```sh
//! RUSTFLAGS="-C target-cpu=native" cargo bench -p pixmap --bench vs_vello_cpu
//! ```
//!
//! # Methodological notes
//!
//! - **Alpha conventions**: `premultiply_*` takes straight-alpha RGBA8; all compositing,
//!   blending, resampling, `multiply_alpha`, and `unpremultiply` benchmarks take valid
//!   premultiplied RGBA8 (`r, g, b <= a`).
//! - **`multiply_alpha`**: `pixmap` uses exact rounded `mul_div_255`; `vello_common`'s
//!   `Pixmap::multiply_alpha` uses truncating integer division.
//! - **Porter-Duff & W3C blends**: `vello_cpu` routes default `SrcOver` through the
//!   dedicated `U8Kernel::alpha_composite_buffer` / `alpha_composite_solid` fast paths and
//!   all other modes through `U8Kernel::blend` (whose `blend::mix` and `ComposeExt::compose`
//!   helpers are `#[inline(never)]` per 8-pixel chunk). `vello_cpu` maps `PlusLighter` to
//!   `Plus`. `tiny-skia` has no public per-stage API so its benchmarks call `PixmapMut::draw_pixmap`
//!   / `fill_rect` (which compiles and runs its `RasterPipeline` directly over the rect;
//!   `tiny-skia` has no `PlusLighter` variant).
//! - **Resampling**: `pixmap` scales separable filter support when downscaling to avoid
//!   aliasing, whereas `vello_cpu` and `tiny-skia` always sample a fixed 2x2 (bilinear)
//!   or 4x4 (bicubic, Mitchell-Netravali B=1/3, C=1/3) grid. `vello_cpu_nopack` omits the
//!   tile-to-row-major `pack` step.
//! - **Unpaired baselines**: `swap_rb`, `flip_horizontal`, `flip_vertical`, `rotate_180`,
//!   and `transpose` have no direct `vello_cpu` or `tiny-skia` counterpart and are reported
//!   as `pixmap` baselines only.

use std::hint::black_box;
use std::iter;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pixmap::{
    BlendMode as PmBlendMode, Compose as PmCompose, Mix as PmMix, Pixmap as AnyPixmap,
    ResampleFilter, composite_color_rgba8, composite_rgba8, multiply_alpha_rgba8,
    premultiply_rgba8, swap_rb_rgba8, unpremultiply_rgba8,
};
use tiny_skia::{
    BlendMode as TsBlendMode, Color as TsColor, ColorU8 as TsColorU8,
    FilterQuality as TsFilterQuality, IntSize as TsIntSize, Paint as TsPaint, Pixmap as TsPixmap,
    PixmapMut as TsPixmapMut, PixmapPaint as TsPixmapPaint, PixmapRef as TsPixmapRef,
    Rect as TsRect, Transform as TsTransform,
};
use vello_common::encode::{EncodeExt, EncodedImage, EncodedPaint};
use vello_common::fearless_simd::{Level, Simd, dispatch, u8x32};
use vello_common::geometry::RectU16;
use vello_common::kurbo::Affine;
use vello_common::paint::{Image, ImageSource};
use vello_common::peniko::{
    BlendMode as VelloBlendMode, Compose as VelloCompose, Extend, ImageQuality, ImageSampler,
    Mix as VelloMix,
};
use vello_common::pixmap::{
    Pixmap as VelloPixmap, PixmapMut as VelloPixmapMut,
    premultiply_rgba8 as vello_premultiply_rgba8, unpremultiply_rgba8 as vello_unpremultiply_rgba8,
};
use vello_common::tile::Tile;
use vello_cpu::fine::{CompositeType, FineKernel, U8Kernel};
use vello_cpu::region::Region;

const W: u16 = 256;
const H: u16 = 256;
const PIXELS: usize = W as usize * H as usize;

const LARGE_W: u16 = 1024;
const LARGE_H: u16 = 1024;
const LARGE_PIXELS: usize = LARGE_W as usize * LARGE_H as usize;

const SOLID_COLOR: [u8; 4] = [180, 90, 45, 200];
// Straight-alpha equivalent of `SOLID_COLOR` (premultiplies to ~[180, 90, 45, 200]).
const SOLID_COLOR_STRAIGHT: [u8; 4] = [230, 115, 57, 200];
const OPACITY: u8 = 180;
const OPACITY_F32: f32 = OPACITY as f32 / 255.0;

struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Generates deterministic straight-alpha RGBA8 pixels with mixed alpha (~1/8 opaque, ~1/8 zero,
/// ~3/4 semi-transparent).
fn make_straight_rgba8(pixels: usize, seed: u64, all_opaque: bool) -> Vec<u8> {
    let mut rng = XorShift64::new(seed);
    let mut data = Vec::with_capacity(pixels * 4);
    for _ in 0..pixels {
        let v = rng.next_u64();
        let r = (v & 0xff) as u8;
        let g = ((v >> 8) & 0xff) as u8;
        let b = ((v >> 16) & 0xff) as u8;
        let a = if all_opaque {
            255
        } else {
            match (v >> 24) & 0x7 {
                0 => 0,
                1 => 255,
                _ => (((v >> 32) & 0xff) as u8).clamp(1, 254),
            }
        };
        data.extend_from_slice(&[r, g, b, a]);
    }
    data
}

/// Generates deterministic valid premultiplied RGBA8 pixels (`r, g, b <= a`).
fn make_premul_rgba8(pixels: usize, seed: u64) -> Vec<u8> {
    let mut data = make_straight_rgba8(pixels, seed, false);
    premultiply_rgba8(&mut data);
    data
}

fn make_vello_pixmap(width: u16, height: u16, premul_bytes: &[u8]) -> VelloPixmap {
    let mut pm = VelloPixmap::new(width, height);
    pm.data_as_u8_slice_mut().copy_from_slice(premul_bytes);
    pm
}

fn encode_vello_image(
    src: &Arc<VelloPixmap>,
    quality: ImageQuality,
    transform: Affine,
) -> EncodedImage {
    let image = Image {
        image: ImageSource::Pixmap(Arc::clone(src)),
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality,
            alpha: 1.0,
        },
    };
    let mut paints = Vec::new();
    image
        .encode_into(&mut paints, transform, None)
        .expect("benchmark image transform must be invertible");
    match paints.into_iter().next().unwrap() {
        EncodedPaint::Image(img) => img,
        _ => unreachable!(),
    }
}

/// Times an in-place mutating operation while restoring the input buffer before every iteration
/// outside the timed region so data-dependent branches and pixel values do not decay across runs.
fn bench_in_place<F: FnMut(&mut [u8])>(b: &mut Bencher<'_>, initial: &[u8], mut op: F) {
    let mut buf = initial.to_vec();
    b.iter_custom(|iters| {
        let mut total = Duration::ZERO;
        for _ in 0..iters {
            buf.copy_from_slice(initial);
            let start = Instant::now();
            op(black_box(&mut buf));
            total += start.elapsed();
            black_box(&buf);
        }
        total
    });
}

/// Draws `src` over the whole of `dst` (same size) through `tiny-skia`'s raster pipeline.
fn ts_draw(
    dst: &mut [u8],
    w: u16,
    h: u16,
    src: TsPixmapRef<'_>,
    blend_mode: TsBlendMode,
    opacity: f32,
) {
    let paint = TsPixmapPaint {
        opacity,
        blend_mode,
        quality: TsFilterQuality::Nearest,
    };
    let mut ts_dst = TsPixmapMut::from_bytes(dst, u32::from(w), u32::from(h)).unwrap();
    ts_dst.draw_pixmap(0, 0, src, &paint, TsTransform::identity(), None);
}

/// Resamples `src` into the whole of `dst` with `transform` through `tiny-skia`'s raster
/// pipeline (`Source` blend, so the destination is overwritten).
fn ts_resample(
    dst: &mut [u8],
    w: u16,
    h: u16,
    src: TsPixmapRef<'_>,
    quality: TsFilterQuality,
    transform: TsTransform,
) {
    let paint = TsPixmapPaint {
        opacity: 1.0,
        blend_mode: TsBlendMode::Source,
        quality,
    };
    let mut ts_dst = TsPixmapMut::from_bytes(dst, u32::from(w), u32::from(h)).unwrap();
    ts_dst.draw_pixmap(0, 0, src, &paint, transform, None);
}

/// Runs `U8Kernel::alpha_composite_buffer` inside the detected SIMD level.
fn vello_alpha_composite_buffer(level: Level, dst: &mut [u8], src: &[u8], alphas: Option<&[u8]>) {
    dispatch!(level, simd => {
        U8Kernel::alpha_composite_buffer(simd, dst, src, alphas);
    });
}

/// Runs `U8Kernel::blend` over a flat premultiplied RGBA8 source slice inside the detected SIMD
/// level.
fn vello_blend_buffer(
    level: Level,
    dst: &mut [u8],
    src: &[u8],
    blend_mode: VelloBlendMode,
    alphas: Option<&[u8]>,
) {
    #[inline(always)]
    fn inner<S: Simd>(
        simd: S,
        dst: &mut [u8],
        src: &[u8],
        blend_mode: VelloBlendMode,
        alphas: Option<&[u8]>,
    ) {
        simd.vectorize(|| {
            let src_iter = src
                .as_chunks::<32>()
                .0
                .iter()
                .map(|el| u8x32::from_slice(simd, el));
            U8Kernel::blend(simd, dst, 0, 0, src_iter, blend_mode, alphas, None);
        });
    }

    dispatch!(level, simd => inner(simd, dst, src, blend_mode, alphas));
}

/// Runs `U8Kernel::blend` for a solid premultiplied color inside the detected SIMD level.
fn vello_blend_solid(level: Level, dst: &mut [u8], color: [u8; 4], blend_mode: VelloBlendMode) {
    #[inline(always)]
    fn inner<S: Simd>(simd: S, dst: &mut [u8], color: [u8; 4], blend_mode: VelloBlendMode) {
        simd.vectorize(|| {
            let c = <u8x32<S> as CompositeType<u8, S>>::from_color(simd, color);
            U8Kernel::blend(simd, dst, 0, 0, iter::repeat(c), blend_mode, None, None);
        });
    }

    dispatch!(level, simd => inner(simd, dst, color, blend_mode));
}

/// Runs `vello_cpu`'s raw image painter strip-by-strip across a full `dst_w × dst_h` image.
/// When `pack_output` is true, also packs each 4-row strip into row-major RGBA8 output via
/// `U8Kernel::pack`.
#[derive(Clone, Copy)]
enum VelloImageKind {
    PlainNN,
    GeneralNN,
    PlainBilinear,
    HighBicubic,
}

#[allow(clippy::too_many_arguments)]
fn vello_sample_image(
    level: Level,
    kind: VelloImageKind,
    encoded: &EncodedImage,
    src_pixmap: &VelloPixmap,
    strip_buf: &mut [u8],
    dst_bytes: &mut [u8],
    dst_w: u16,
    dst_h: u16,
    pack_output: bool,
) {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn inner<S: Simd>(
        simd: S,
        kind: VelloImageKind,
        encoded: &EncodedImage,
        src_pixmap: &VelloPixmap,
        strip_buf: &mut [u8],
        dst_bytes: &mut [u8],
        dst_w: u16,
        dst_h: u16,
        pack_output: bool,
    ) {
        simd.vectorize(|| {
            let mut dst_pm = VelloPixmapMut::new(dst_w, dst_h, dst_bytes).unwrap();
            let tile_h = Tile::HEIGHT;
            for y in (0..dst_h).step_by(tile_h as usize) {
                let start_y = f64::from(y);
                match kind {
                    VelloImageKind::PlainNN => {
                        let painter = U8Kernel::plain_nn_image_painter(
                            simd, encoded, src_pixmap, 0.0, start_y,
                        );
                        U8Kernel::apply_painter(simd, strip_buf, painter);
                    }
                    VelloImageKind::GeneralNN => {
                        let painter =
                            U8Kernel::nn_image_painter(simd, encoded, src_pixmap, 0.0, start_y);
                        U8Kernel::apply_painter(simd, strip_buf, painter);
                    }
                    VelloImageKind::PlainBilinear => {
                        let painter = U8Kernel::plain_medium_quality_image_painter(
                            simd, encoded, src_pixmap, 0.0, start_y,
                        );
                        U8Kernel::apply_painter(simd, strip_buf, painter);
                    }
                    VelloImageKind::HighBicubic => {
                        let painter = U8Kernel::high_quality_image_painter(
                            simd, encoded, src_pixmap, 0.0, start_y,
                        );
                        U8Kernel::apply_painter(simd, strip_buf, painter);
                    }
                }
                if pack_output {
                    let mut region = Region::new(
                        &mut dst_pm,
                        RectU16::new(0, y, dst_w, (y + tile_h).min(dst_h)),
                    );
                    U8Kernel::pack(simd, strip_buf, usize::from(dst_w), &mut region);
                }
            }
        });
    }

    dispatch!(level, simd => inner(
        simd,
        kind,
        encoded,
        src_pixmap,
        strip_buf,
        dst_bytes,
        dst_w,
        dst_h,
        pack_output,
    ));
}

fn bench_alpha(c: &mut Criterion) {
    let mut group = c.benchmark_group("alpha");

    for &(label, w, h, pixels) in &[
        ("256x256", W, H, PIXELS),
        ("1024x1024", LARGE_W, LARGE_H, LARGE_PIXELS),
    ] {
        group.throughput(Throughput::Elements(pixels as u64));

        let straight_mixed = make_straight_rgba8(pixels, 0x1111_2222_3333_4444, false);
        let straight_opaque = make_straight_rgba8(pixels, 0x5555_6666_7777_8888, true);
        let premul = make_premul_rgba8(pixels, 0x9999_AAAA_BBBB_CCCC);

        group.bench_function(
            BenchmarkId::new(format!("premultiply_mixed/{label}"), "pixmap"),
            |b| {
                bench_in_place(b, &straight_mixed, premultiply_rgba8);
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("premultiply_mixed/{label}"), "vello_cpu"),
            |b| {
                bench_in_place(b, &straight_mixed, |buf| {
                    black_box(vello_premultiply_rgba8(buf));
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("premultiply_mixed/{label}"), "tiny_skia"),
            |b| {
                bench_in_place(b, &straight_mixed, |buf| {
                    for px in buf.chunks_exact_mut(4) {
                        let p = TsColorU8::from_rgba(px[0], px[1], px[2], px[3]).premultiply();
                        px[0] = p.red();
                        px[1] = p.green();
                        px[2] = p.blue();
                        px[3] = p.alpha();
                    }
                });
            },
        );

        group.bench_function(
            BenchmarkId::new(format!("premultiply_opaque/{label}"), "pixmap"),
            |b| {
                bench_in_place(b, &straight_opaque, premultiply_rgba8);
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("premultiply_opaque/{label}"), "vello_cpu"),
            |b| {
                bench_in_place(b, &straight_opaque, |buf| {
                    black_box(vello_premultiply_rgba8(buf));
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("premultiply_opaque/{label}"), "tiny_skia"),
            |b| {
                bench_in_place(b, &straight_opaque, |buf| {
                    for px in buf.chunks_exact_mut(4) {
                        let p = TsColorU8::from_rgba(px[0], px[1], px[2], px[3]).premultiply();
                        px[0] = p.red();
                        px[1] = p.green();
                        px[2] = p.blue();
                        px[3] = p.alpha();
                    }
                });
            },
        );

        group.bench_function(
            BenchmarkId::new(format!("unpremultiply/{label}"), "pixmap"),
            |b| {
                bench_in_place(b, &premul, unpremultiply_rgba8);
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("unpremultiply/{label}"), "vello_cpu"),
            |b| {
                bench_in_place(b, &premul, |buf| {
                    black_box(vello_unpremultiply_rgba8(buf));
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("unpremultiply/{label}"), "tiny_skia"),
            |b| {
                let size = TsIntSize::from_wh(u32::from(w), u32::from(h)).unwrap();
                let mut buf = Some(premul.clone());
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut v = buf.take().unwrap();
                        v.copy_from_slice(&premul);
                        let pm = TsPixmap::from_vec(v, size).unwrap();
                        let start = Instant::now();
                        let out = black_box(pm).take_demultiplied();
                        total += start.elapsed();
                        black_box(&out);
                        buf = Some(out);
                    }
                    total
                });
            },
        );

        group.bench_function(
            BenchmarkId::new(format!("multiply_alpha/{label}"), "pixmap"),
            |b| {
                bench_in_place(b, &premul, |buf| {
                    multiply_alpha_rgba8(buf, black_box(OPACITY));
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("multiply_alpha/{label}"), "vello_cpu"),
            |b| {
                let mut pm = make_vello_pixmap(w, h, &premul);
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        pm.data_as_u8_slice_mut().copy_from_slice(&premul);
                        let start = Instant::now();
                        pm.multiply_alpha(black_box(OPACITY));
                        total += start.elapsed();
                        black_box(pm.data_as_u8_slice());
                    }
                    total
                });
            },
        );

        // Full-scan opacity check (all-opaque input = worst case, scans every pixel).
        let pm_opaque = AnyPixmap::from_parts(straight_opaque.clone(), w, h);
        let mut vello_opaque = make_vello_pixmap(w, h, &straight_opaque);
        group.bench_function(
            BenchmarkId::new(format!("is_opaque/{label}"), "pixmap"),
            |b| {
                b.iter(|| black_box(black_box(&pm_opaque).is_opaque()));
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("is_opaque/{label}"), "vello_cpu"),
            |b| {
                b.iter(|| {
                    vello_opaque.recompute_may_have_transparency();
                    black_box(vello_opaque.may_have_transparency());
                });
            },
        );

        if w == W {
            let mut buf = premul.clone();
            group.bench_function(
                BenchmarkId::new(format!("swap_rb/{label}"), "pixmap"),
                |b| {
                    b.iter(|| {
                        swap_rb_rgba8(black_box(&mut buf));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_solid(c: &mut Criterion) {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut group = c.benchmark_group("solid");
    group.throughput(Throughput::Elements(PIXELS as u64));

    let dst_orig = make_premul_rgba8(PIXELS, 0xAAAA_BBBB_CCCC_DDDD);

    // Solid fill (opaque overwrite).
    let mut fill_pm = AnyPixmap::from_parts(dst_orig.clone(), W, H);
    group.bench_function(BenchmarkId::new("fill", "pixmap"), |b| {
        b.iter(|| {
            fill_pm.fill(black_box(SOLID_COLOR));
            black_box(fill_pm.data());
        });
    });
    let mut fill_buf_vello = dst_orig.clone();
    group.bench_function(BenchmarkId::new("fill", "vello_cpu"), |b| {
        b.iter(|| {
            dispatch!(level, simd => {
                U8Kernel::copy_solid(simd, black_box(&mut fill_buf_vello), black_box(SOLID_COLOR));
            });
            black_box(&fill_buf_vello);
        });
    });
    let mut fill_buf_ts = dst_orig.clone();
    group.bench_function(BenchmarkId::new("fill", "tiny_skia"), |b| {
        let mut ts_pm =
            TsPixmapMut::from_bytes(&mut fill_buf_ts, u32::from(W), u32::from(H)).unwrap();
        let ts_color = TsColor::from_rgba8(
            SOLID_COLOR_STRAIGHT[0],
            SOLID_COLOR_STRAIGHT[1],
            SOLID_COLOR_STRAIGHT[2],
            SOLID_COLOR_STRAIGHT[3],
        );
        b.iter(|| {
            ts_pm.fill(black_box(ts_color));
            black_box(ts_pm.data_mut());
        });
    });

    // Solid SrcOver (uses vello_cpu's dedicated `alpha_composite_solid` fast path).
    group.bench_function(BenchmarkId::new("src_over", "pixmap"), |b| {
        bench_in_place(b, &dst_orig, |dst| {
            composite_color_rgba8(dst, black_box(SOLID_COLOR), PmBlendMode::SRC_OVER);
        });
    });
    group.bench_function(BenchmarkId::new("src_over", "vello_cpu"), |b| {
        bench_in_place(b, &dst_orig, |dst| {
            dispatch!(level, simd => {
                U8Kernel::alpha_composite_solid(simd, dst, black_box(SOLID_COLOR), None);
            });
        });
    });
    let full_rect = TsRect::from_xywh(0.0, 0.0, f32::from(W), f32::from(H)).unwrap();
    group.bench_function(BenchmarkId::new("src_over", "tiny_skia"), |b| {
        let mut paint = TsPaint::default();
        paint.set_color_rgba8(
            SOLID_COLOR_STRAIGHT[0],
            SOLID_COLOR_STRAIGHT[1],
            SOLID_COLOR_STRAIGHT[2],
            SOLID_COLOR_STRAIGHT[3],
        );
        paint.blend_mode = TsBlendMode::SourceOver;
        paint.anti_alias = false;
        bench_in_place(b, &dst_orig, |dst| {
            let mut ts_pm = TsPixmapMut::from_bytes(dst, u32::from(W), u32::from(H)).unwrap();
            ts_pm.fill_rect(full_rect, black_box(&paint), TsTransform::identity(), None);
        });
    });

    // Solid color with non-Normal mix modes.
    let solid_mixes: &[(&str, PmBlendMode, VelloBlendMode, TsBlendMode)] = &[
        (
            "multiply",
            PmBlendMode::MULTIPLY,
            VelloBlendMode::new(VelloMix::Multiply, VelloCompose::SrcOver),
            TsBlendMode::Multiply,
        ),
        (
            "screen",
            PmBlendMode::SCREEN,
            VelloBlendMode::new(VelloMix::Screen, VelloCompose::SrcOver),
            TsBlendMode::Screen,
        ),
        (
            "soft_light",
            PmBlendMode::SOFT_LIGHT,
            VelloBlendMode::new(VelloMix::SoftLight, VelloCompose::SrcOver),
            TsBlendMode::SoftLight,
        ),
    ];

    for &(name, pm_mode, vello_mode, ts_mode) in solid_mixes {
        group.bench_function(BenchmarkId::new(name, "pixmap"), |b| {
            bench_in_place(b, &dst_orig, |dst| {
                composite_color_rgba8(dst, black_box(SOLID_COLOR), pm_mode);
            });
        });
        group.bench_function(BenchmarkId::new(name, "vello_cpu"), |b| {
            bench_in_place(b, &dst_orig, |dst| {
                vello_blend_solid(level, dst, black_box(SOLID_COLOR), vello_mode);
            });
        });
        group.bench_function(BenchmarkId::new(name, "tiny_skia"), |b| {
            let mut paint = TsPaint::default();
            paint.set_color_rgba8(
                SOLID_COLOR_STRAIGHT[0],
                SOLID_COLOR_STRAIGHT[1],
                SOLID_COLOR_STRAIGHT[2],
                SOLID_COLOR_STRAIGHT[3],
            );
            paint.blend_mode = ts_mode;
            paint.anti_alias = false;
            bench_in_place(b, &dst_orig, |dst| {
                let mut ts_pm = TsPixmapMut::from_bytes(dst, u32::from(W), u32::from(H)).unwrap();
                ts_pm.fill_rect(full_rect, black_box(&paint), TsTransform::identity(), None);
            });
        });
    }

    group.finish();
}

fn bench_compose(c: &mut Criterion) {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut group = c.benchmark_group("compose");
    group.throughput(Throughput::Elements(PIXELS as u64));

    let dst_orig = make_premul_rgba8(PIXELS, 0x1020_3040_5060_7080);
    let src = make_premul_rgba8(PIXELS, 0x0102_0304_0506_0708);
    let uniform_alphas = vec![OPACITY; PIXELS];

    let ts_src = TsPixmapRef::from_bytes(&src, u32::from(W), u32::from(H)).unwrap();

    // SrcOver fast path (what vello_cpu actually uses for default BlendMode::default()).
    group.bench_function(BenchmarkId::new("SrcOver", "pixmap"), |b| {
        bench_in_place(b, &dst_orig, |dst| {
            composite_rgba8(dst, black_box(&src), PmBlendMode::SRC_OVER, 255);
        });
    });
    group.bench_function(BenchmarkId::new("SrcOver", "vello_cpu"), |b| {
        bench_in_place(b, &dst_orig, |dst| {
            vello_alpha_composite_buffer(level, dst, black_box(&src), None);
        });
    });
    group.bench_function(BenchmarkId::new("SrcOver", "tiny_skia"), |b| {
        let paint = TsPixmapPaint {
            opacity: 1.0,
            blend_mode: TsBlendMode::SourceOver,
            quality: TsFilterQuality::Nearest,
        };
        bench_in_place(b, &dst_orig, |dst| {
            let mut ts_dst = TsPixmapMut::from_bytes(dst, u32::from(W), u32::from(H)).unwrap();
            ts_dst.draw_pixmap(
                0,
                0,
                black_box(ts_src),
                &paint,
                TsTransform::identity(),
                None,
            );
        });
    });

    // SrcOver with uniform opacity < 255.
    group.bench_function(BenchmarkId::new("SrcOver_opacity", "pixmap"), |b| {
        bench_in_place(b, &dst_orig, |dst| {
            composite_rgba8(
                dst,
                black_box(&src),
                PmBlendMode::SRC_OVER,
                black_box(OPACITY),
            );
        });
    });
    group.bench_function(BenchmarkId::new("SrcOver_opacity", "vello_cpu"), |b| {
        bench_in_place(b, &dst_orig, |dst| {
            vello_alpha_composite_buffer(
                level,
                dst,
                black_box(&src),
                Some(black_box(&uniform_alphas)),
            );
        });
    });
    group.bench_function(BenchmarkId::new("SrcOver_opacity", "tiny_skia"), |b| {
        let paint = TsPixmapPaint {
            opacity: OPACITY_F32,
            blend_mode: TsBlendMode::SourceOver,
            quality: TsFilterQuality::Nearest,
        };
        bench_in_place(b, &dst_orig, |dst| {
            let mut ts_dst = TsPixmapMut::from_bytes(dst, u32::from(W), u32::from(H)).unwrap();
            ts_dst.draw_pixmap(
                0,
                0,
                black_box(ts_src),
                black_box(&paint),
                TsTransform::identity(),
                None,
            );
        });
    });

    // All remaining Porter-Duff Compose modes (vello_cpu dispatches these through `U8Kernel::blend`).
    let modes: &[(&str, PmCompose, VelloCompose, Option<TsBlendMode>)] = &[
        (
            "Clear",
            PmCompose::Clear,
            VelloCompose::Clear,
            Some(TsBlendMode::Clear),
        ),
        (
            "Copy",
            PmCompose::Copy,
            VelloCompose::Copy,
            Some(TsBlendMode::Source),
        ),
        (
            "Dest",
            PmCompose::Dest,
            VelloCompose::Dest,
            Some(TsBlendMode::Destination),
        ),
        (
            "SrcOver_blend",
            PmCompose::SrcOver,
            VelloCompose::SrcOver,
            None,
        ),
        (
            "DestOver",
            PmCompose::DestOver,
            VelloCompose::DestOver,
            Some(TsBlendMode::DestinationOver),
        ),
        (
            "SrcIn",
            PmCompose::SrcIn,
            VelloCompose::SrcIn,
            Some(TsBlendMode::SourceIn),
        ),
        (
            "DestIn",
            PmCompose::DestIn,
            VelloCompose::DestIn,
            Some(TsBlendMode::DestinationIn),
        ),
        (
            "SrcOut",
            PmCompose::SrcOut,
            VelloCompose::SrcOut,
            Some(TsBlendMode::SourceOut),
        ),
        (
            "DestOut",
            PmCompose::DestOut,
            VelloCompose::DestOut,
            Some(TsBlendMode::DestinationOut),
        ),
        (
            "SrcAtop",
            PmCompose::SrcAtop,
            VelloCompose::SrcAtop,
            Some(TsBlendMode::SourceAtop),
        ),
        (
            "DestAtop",
            PmCompose::DestAtop,
            VelloCompose::DestAtop,
            Some(TsBlendMode::DestinationAtop),
        ),
        (
            "Xor",
            PmCompose::Xor,
            VelloCompose::Xor,
            Some(TsBlendMode::Xor),
        ),
        (
            "Plus",
            PmCompose::Plus,
            VelloCompose::Plus,
            Some(TsBlendMode::Plus),
        ),
        (
            "PlusLighter",
            PmCompose::PlusLighter,
            VelloCompose::PlusLighter,
            None,
        ),
    ];

    for &(name, pm_compose, vello_compose, ts_mode) in modes {
        let pm_mode = PmBlendMode::new(PmMix::Normal, pm_compose);
        let vello_mode = VelloBlendMode::new(VelloMix::Normal, vello_compose);
        // Clear, Copy, and Dest do not depend on prior destination contents, so time directly
        // without per-iteration reset (especially important for Dest which is a no-op).
        let stateless_dst = matches!(name, "Clear" | "Copy" | "Dest");

        if name != "SrcOver_blend" {
            if stateless_dst {
                let mut dst = dst_orig.clone();
                group.bench_function(BenchmarkId::new(name, "pixmap"), |b| {
                    b.iter(|| {
                        composite_rgba8(black_box(&mut dst), black_box(&src), pm_mode, 255);
                        black_box(&dst);
                    });
                });
            } else {
                group.bench_function(BenchmarkId::new(name, "pixmap"), |b| {
                    bench_in_place(b, &dst_orig, |dst| {
                        composite_rgba8(dst, black_box(&src), pm_mode, 255);
                    });
                });
            }
        }
        if stateless_dst {
            let mut dst = dst_orig.clone();
            group.bench_function(BenchmarkId::new(name, "vello_cpu"), |b| {
                b.iter(|| {
                    vello_blend_buffer(
                        level,
                        black_box(&mut dst),
                        black_box(&src),
                        vello_mode,
                        None,
                    );
                    black_box(&dst);
                });
            });
        } else {
            group.bench_function(BenchmarkId::new(name, "vello_cpu"), |b| {
                bench_in_place(b, &dst_orig, |dst| {
                    vello_blend_buffer(level, dst, black_box(&src), vello_mode, None);
                });
            });
        }
        if let Some(ts_blend) = ts_mode {
            let paint = TsPixmapPaint {
                opacity: 1.0,
                blend_mode: ts_blend,
                quality: TsFilterQuality::Nearest,
            };
            if stateless_dst {
                let mut dst = dst_orig.clone();
                group.bench_function(BenchmarkId::new(name, "tiny_skia"), |b| {
                    let mut ts_dst =
                        TsPixmapMut::from_bytes(&mut dst, u32::from(W), u32::from(H)).unwrap();
                    b.iter(|| {
                        ts_dst.draw_pixmap(
                            0,
                            0,
                            black_box(ts_src),
                            &paint,
                            TsTransform::identity(),
                            None,
                        );
                        black_box(ts_dst.data_mut());
                    });
                });
            } else {
                group.bench_function(BenchmarkId::new(name, "tiny_skia"), |b| {
                    bench_in_place(b, &dst_orig, |dst| {
                        let mut ts_dst =
                            TsPixmapMut::from_bytes(dst, u32::from(W), u32::from(H)).unwrap();
                        ts_dst.draw_pixmap(
                            0,
                            0,
                            black_box(ts_src),
                            &paint,
                            TsTransform::identity(),
                            None,
                        );
                    });
                });
            }
        }
    }

    // Large (1024x1024) memory-bandwidth spot check for SrcOver.
    group.throughput(Throughput::Elements(LARGE_PIXELS as u64));
    let large_dst = make_premul_rgba8(LARGE_PIXELS, 0x1020_3040_5060_7080);
    let large_src = make_premul_rgba8(LARGE_PIXELS, 0x0102_0304_0506_0708);
    let ts_large_src =
        TsPixmapRef::from_bytes(&large_src, u32::from(LARGE_W), u32::from(LARGE_H)).unwrap();
    group.bench_function(BenchmarkId::new("SrcOver_1024x1024", "pixmap"), |b| {
        bench_in_place(b, &large_dst, |dst| {
            composite_rgba8(dst, black_box(&large_src), PmBlendMode::SRC_OVER, 255);
        });
    });
    group.bench_function(BenchmarkId::new("SrcOver_1024x1024", "vello_cpu"), |b| {
        bench_in_place(b, &large_dst, |dst| {
            vello_alpha_composite_buffer(level, dst, black_box(&large_src), None);
        });
    });
    group.bench_function(BenchmarkId::new("SrcOver_1024x1024", "tiny_skia"), |b| {
        let paint = TsPixmapPaint {
            opacity: 1.0,
            blend_mode: TsBlendMode::SourceOver,
            quality: TsFilterQuality::Nearest,
        };
        bench_in_place(b, &large_dst, |dst| {
            let mut ts_dst =
                TsPixmapMut::from_bytes(dst, u32::from(LARGE_W), u32::from(LARGE_H)).unwrap();
            ts_dst.draw_pixmap(
                0,
                0,
                black_box(ts_large_src),
                &paint,
                TsTransform::identity(),
                None,
            );
        });
    });

    group.finish();
}

fn bench_mix(c: &mut Criterion) {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut group = c.benchmark_group("mix");
    group.throughput(Throughput::Elements(PIXELS as u64));

    let dst_orig = make_premul_rgba8(PIXELS, 0x2233_4455_6677_8899);
    let src = make_premul_rgba8(PIXELS, 0x9988_7766_5544_3322);
    let uniform_alphas = vec![OPACITY; PIXELS];

    let ts_src = TsPixmapRef::from_bytes(&src, u32::from(W), u32::from(H)).unwrap();

    #[rustfmt::skip]
    let modes: &[(&str, PmMix, VelloMix, TsBlendMode)] = &[
        // Integer u8 fast path in both pixmap and vello_cpu
        ("Multiply", PmMix::Multiply, VelloMix::Multiply, TsBlendMode::Multiply),
        ("Screen", PmMix::Screen, VelloMix::Screen, TsBlendMode::Screen),
        ("Overlay", PmMix::Overlay, VelloMix::Overlay, TsBlendMode::Overlay),
        ("Darken", PmMix::Darken, VelloMix::Darken, TsBlendMode::Darken),
        ("Lighten", PmMix::Lighten, VelloMix::Lighten, TsBlendMode::Lighten),
        ("HardLight", PmMix::HardLight, VelloMix::HardLight, TsBlendMode::HardLight),
        ("Difference", PmMix::Difference, VelloMix::Difference, TsBlendMode::Difference),
        ("Exclusion", PmMix::Exclusion, VelloMix::Exclusion, TsBlendMode::Exclusion),
        // Float f32 path in both pixmap and vello_cpu
        ("ColorDodge", PmMix::ColorDodge, VelloMix::ColorDodge, TsBlendMode::ColorDodge),
        ("ColorBurn", PmMix::ColorBurn, VelloMix::ColorBurn, TsBlendMode::ColorBurn),
        ("SoftLight", PmMix::SoftLight, VelloMix::SoftLight, TsBlendMode::SoftLight),
        ("Hue", PmMix::Hue, VelloMix::Hue, TsBlendMode::Hue),
        ("Saturation", PmMix::Saturation, VelloMix::Saturation, TsBlendMode::Saturation),
        ("Color", PmMix::Color, VelloMix::Color, TsBlendMode::Color),
        ("Luminosity", PmMix::Luminosity, VelloMix::Luminosity, TsBlendMode::Luminosity),
    ];

    for &(name, pm_mix, vello_mix, ts_blend) in modes {
        group.bench_function(BenchmarkId::new(name, "tiny_skia"), |b| {
            bench_in_place(b, &dst_orig, |dst| {
                ts_draw(dst, W, H, black_box(ts_src), ts_blend, 1.0);
            });
        });
        let pm_mode = PmBlendMode::new(pm_mix, PmCompose::SrcOver);
        let vello_mode = VelloBlendMode::new(vello_mix, VelloCompose::SrcOver);

        group.bench_function(BenchmarkId::new(name, "pixmap"), |b| {
            bench_in_place(b, &dst_orig, |dst| {
                composite_rgba8(dst, black_box(&src), pm_mode, 255);
            });
        });
        group.bench_function(BenchmarkId::new(name, "vello_cpu"), |b| {
            bench_in_place(b, &dst_orig, |dst| {
                vello_blend_buffer(level, dst, black_box(&src), vello_mode, None);
            });
        });
    }

    // Mix + uniform opacity < 255.
    let pm_mul = PmBlendMode::MULTIPLY;
    let vello_mul = VelloBlendMode::new(VelloMix::Multiply, VelloCompose::SrcOver);
    group.bench_function(BenchmarkId::new("Multiply_opacity", "pixmap"), |b| {
        bench_in_place(b, &dst_orig, |dst| {
            composite_rgba8(dst, black_box(&src), pm_mul, black_box(OPACITY));
        });
    });
    group.bench_function(BenchmarkId::new("Multiply_opacity", "vello_cpu"), |b| {
        bench_in_place(b, &dst_orig, |dst| {
            vello_blend_buffer(
                level,
                dst,
                black_box(&src),
                vello_mul,
                Some(black_box(&uniform_alphas)),
            );
        });
    });

    group.bench_function(BenchmarkId::new("Multiply_opacity", "tiny_skia"), |b| {
        bench_in_place(b, &dst_orig, |dst| {
            ts_draw(
                dst,
                W,
                H,
                black_box(ts_src),
                TsBlendMode::Multiply,
                black_box(OPACITY_F32),
            );
        });
    });

    // Large (1024x1024) memory-bandwidth spot check for Multiply.
    group.throughput(Throughput::Elements(LARGE_PIXELS as u64));
    let large_dst = make_premul_rgba8(LARGE_PIXELS, 0x2233_4455_6677_8899);
    let large_src = make_premul_rgba8(LARGE_PIXELS, 0x9988_7766_5544_3322);
    group.bench_function(BenchmarkId::new("Multiply_1024x1024", "pixmap"), |b| {
        bench_in_place(b, &large_dst, |dst| {
            composite_rgba8(dst, black_box(&large_src), pm_mul, 255);
        });
    });
    group.bench_function(BenchmarkId::new("Multiply_1024x1024", "tiny_skia"), |b| {
        let ts_large_src =
            TsPixmapRef::from_bytes(&large_src, u32::from(LARGE_W), u32::from(LARGE_H)).unwrap();
        bench_in_place(b, &large_dst, |dst| {
            ts_draw(
                dst,
                LARGE_W,
                LARGE_H,
                black_box(ts_large_src),
                TsBlendMode::Multiply,
                1.0,
            );
        });
    });
    group.bench_function(BenchmarkId::new("Multiply_1024x1024", "vello_cpu"), |b| {
        bench_in_place(b, &large_dst, |dst| {
            vello_blend_buffer(level, dst, black_box(&large_src), vello_mul, None);
        });
    });

    group.finish();
}

fn bench_resample(c: &mut Criterion) {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut group = c.benchmark_group("resample");

    // 2x upscale (256x256 -> 512x512) and 2x downscale (512x512 -> 256x256).
    // Note: on downscaling, `pixmap` widens its separable filter support for anti-aliasing,
    // whereas `vello_cpu` always samples a fixed 2x2 (Medium) or 4x4 (High) grid.
    for &(dir, src_w, src_h, dst_w, dst_h) in &[
        ("upscale_256_to_512", 256u16, 256u16, 512u16, 512u16),
        ("downscale_512_to_256", 512u16, 512u16, 256u16, 256u16),
    ] {
        let dst_pixels = usize::from(dst_w) * usize::from(dst_h);
        group.throughput(Throughput::Elements(dst_pixels as u64));

        let src_bytes = make_premul_rgba8(
            usize::from(src_w) * usize::from(src_h),
            0xABCD_EF01_2345_6789,
        );
        let pm_src = AnyPixmap::from_parts(src_bytes.clone(), src_w, src_h);
        let mut pm_dst = AnyPixmap::new(dst_w, dst_h);

        let vello_src = Arc::new(make_vello_pixmap(src_w, src_h, &src_bytes));
        let scale_x = f64::from(dst_w) / f64::from(src_w);
        let scale_y = f64::from(dst_h) / f64::from(src_h);
        let transform = Affine::scale_non_uniform(scale_x, scale_y);

        let mut strip_buf = vec![0u8; usize::from(dst_w) * usize::from(Tile::HEIGHT) * 4];
        let mut vello_dst = vec![0u8; dst_pixels * 4];

        let ts_src =
            TsPixmapRef::from_bytes(&src_bytes, u32::from(src_w), u32::from(src_h)).unwrap();
        let ts_transform = TsTransform::from_scale(scale_x as f32, scale_y as f32);
        let mut ts_dst = vec![0u8; dst_pixels * 4];
        for (name, quality) in [
            ("nearest", TsFilterQuality::Nearest),
            ("bilinear", TsFilterQuality::Bilinear),
            ("bicubic", TsFilterQuality::Bicubic),
        ] {
            group.bench_function(
                BenchmarkId::new(format!("{dir}/{name}"), "tiny_skia"),
                |b| {
                    b.iter(|| {
                        ts_resample(
                            &mut ts_dst,
                            dst_w,
                            dst_h,
                            black_box(ts_src),
                            quality,
                            ts_transform,
                        );
                        black_box(&ts_dst);
                    });
                },
            );
        }

        // Nearest neighbor
        group.bench_function(BenchmarkId::new(format!("{dir}/nearest"), "pixmap"), |b| {
            b.iter(|| {
                pm_src.resize_into(black_box(pm_dst.as_mut()), ResampleFilter::NEAREST);
                black_box(pm_dst.data());
            });
        });
        let enc_nn = encode_vello_image(&vello_src, ImageQuality::Low, transform);
        group.bench_function(
            BenchmarkId::new(format!("{dir}/nearest"), "vello_cpu"),
            |b| {
                b.iter(|| {
                    vello_sample_image(
                        level,
                        VelloImageKind::PlainNN,
                        &enc_nn,
                        &vello_src,
                        &mut strip_buf,
                        &mut vello_dst,
                        dst_w,
                        dst_h,
                        true,
                    );
                    black_box(&vello_dst);
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("{dir}/nearest"), "vello_cpu_nopack"),
            |b| {
                b.iter(|| {
                    vello_sample_image(
                        level,
                        VelloImageKind::PlainNN,
                        &enc_nn,
                        &vello_src,
                        &mut strip_buf,
                        &mut vello_dst,
                        dst_w,
                        dst_h,
                        false,
                    );
                    black_box(&strip_buf);
                });
            },
        );

        // Bilinear
        group.bench_function(BenchmarkId::new(format!("{dir}/bilinear"), "pixmap"), |b| {
            b.iter(|| {
                pm_src.resize_into(black_box(pm_dst.as_mut()), ResampleFilter::BILINEAR);
                black_box(pm_dst.data());
            });
        });
        let enc_bilin = encode_vello_image(&vello_src, ImageQuality::Medium, transform);
        group.bench_function(
            BenchmarkId::new(format!("{dir}/bilinear"), "vello_cpu"),
            |b| {
                b.iter(|| {
                    vello_sample_image(
                        level,
                        VelloImageKind::PlainBilinear,
                        &enc_bilin,
                        &vello_src,
                        &mut strip_buf,
                        &mut vello_dst,
                        dst_w,
                        dst_h,
                        true,
                    );
                    black_box(&vello_dst);
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("{dir}/bilinear"), "vello_cpu_nopack"),
            |b| {
                b.iter(|| {
                    vello_sample_image(
                        level,
                        VelloImageKind::PlainBilinear,
                        &enc_bilin,
                        &vello_src,
                        &mut strip_buf,
                        &mut vello_dst,
                        dst_w,
                        dst_h,
                        false,
                    );
                    black_box(&strip_buf);
                });
            },
        );

        // Bicubic (Mitchell-Netravali B=1/3, C=1/3 matches vello_cpu's ImageQuality::High)
        group.bench_function(
            BenchmarkId::new(format!("{dir}/bicubic"), "pixmap_mitchell"),
            |b| {
                b.iter(|| {
                    pm_src.resize_into(black_box(pm_dst.as_mut()), ResampleFilter::MITCHELL);
                    black_box(pm_dst.data());
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("{dir}/bicubic"), "pixmap_catmull_rom"),
            |b| {
                b.iter(|| {
                    pm_src.resize_into(black_box(pm_dst.as_mut()), ResampleFilter::CATMULL_ROM);
                    black_box(pm_dst.data());
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("{dir}/bicubic"), "pixmap_lanczos3"),
            |b| {
                b.iter(|| {
                    pm_src.resize_into(black_box(pm_dst.as_mut()), ResampleFilter::LANCZOS3);
                    black_box(pm_dst.data());
                });
            },
        );
        let enc_high = encode_vello_image(&vello_src, ImageQuality::High, transform);
        group.bench_function(
            BenchmarkId::new(format!("{dir}/bicubic"), "vello_cpu"),
            |b| {
                b.iter(|| {
                    vello_sample_image(
                        level,
                        VelloImageKind::HighBicubic,
                        &enc_high,
                        &vello_src,
                        &mut strip_buf,
                        &mut vello_dst,
                        dst_w,
                        dst_h,
                        true,
                    );
                    black_box(&vello_dst);
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(format!("{dir}/bicubic"), "vello_cpu_nopack"),
            |b| {
                b.iter(|| {
                    vello_sample_image(
                        level,
                        VelloImageKind::HighBicubic,
                        &enc_high,
                        &vello_src,
                        &mut strip_buf,
                        &mut vello_dst,
                        dst_w,
                        dst_h,
                        false,
                    );
                    black_box(&strip_buf);
                });
            },
        );
    }

    group.finish();
}

fn bench_blit_and_transform(c: &mut Criterion) {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut group = c.benchmark_group("blit_transform");
    group.throughput(Throughput::Elements(PIXELS as u64));

    let src_bytes = make_premul_rgba8(PIXELS, 0xCAFE_BABE_DEAD_BEEF);
    let pm_src = AnyPixmap::from_parts(src_bytes.clone(), W, H);
    let mut pm_dst = AnyPixmap::new(W, H);

    let vello_src = Arc::new(make_vello_pixmap(W, H, &src_bytes));
    let mut strip_buf = vec![0u8; usize::from(W) * usize::from(Tile::HEIGHT) * 4];
    let mut vello_dst = vec![0u8; PIXELS * 4];

    // Full blit vs vello_cpu identity NN painter + pack.
    group.bench_function(BenchmarkId::new("blit_copy_from", "pixmap"), |b| {
        b.iter(|| {
            pm_dst.copy_from(black_box(pm_src.as_ref()), 0, 0);
            black_box(pm_dst.data());
        });
    });
    let enc_ident = encode_vello_image(&vello_src, ImageQuality::Low, Affine::IDENTITY);
    group.bench_function(BenchmarkId::new("blit_copy_from", "vello_cpu_nn"), |b| {
        b.iter(|| {
            vello_sample_image(
                level,
                VelloImageKind::PlainNN,
                &enc_ident,
                &vello_src,
                &mut strip_buf,
                &mut vello_dst,
                W,
                H,
                true,
            );
            black_box(&vello_dst);
        });
    });

    let ts_src = TsPixmapRef::from_bytes(&src_bytes, u32::from(W), u32::from(H)).unwrap();
    let mut ts_dst = vec![0u8; PIXELS * 4];
    group.bench_function(BenchmarkId::new("blit_copy_from", "tiny_skia_nn"), |b| {
        b.iter(|| {
            ts_resample(
                &mut ts_dst,
                W,
                H,
                black_box(ts_src),
                TsFilterQuality::Nearest,
                TsTransform::identity(),
            );
            black_box(&ts_dst);
        });
    });
    group.bench_function(BenchmarkId::new("rotate_90", "tiny_skia_nn"), |b| {
        let rot90 = TsTransform::from_row(0.0, 1.0, -1.0, 0.0, f32::from(H), 0.0);
        b.iter(|| {
            ts_resample(
                &mut ts_dst,
                W,
                H,
                black_box(ts_src),
                TsFilterQuality::Nearest,
                rot90,
            );
            black_box(&ts_dst);
        });
    });

    // 90-degree clockwise rotation vs vello_cpu rotated NN painter + pack.
    group.bench_function(BenchmarkId::new("rotate_90", "pixmap"), |b| {
        b.iter(|| black_box(black_box(&pm_src).rotated_90()));
    });
    let rot90 =
        Affine::translate((f64::from(H), 0.0)) * Affine::rotate(std::f64::consts::FRAC_PI_2);
    let enc_rot90 = encode_vello_image(&vello_src, ImageQuality::Low, rot90);
    group.bench_function(BenchmarkId::new("rotate_90", "vello_cpu_nn"), |b| {
        b.iter(|| {
            vello_sample_image(
                level,
                VelloImageKind::GeneralNN,
                &enc_rot90,
                &vello_src,
                &mut strip_buf,
                &mut vello_dst,
                W,
                H,
                true,
            );
            black_box(&vello_dst);
        });
    });

    // Structural pixmap-only transforms (baselines).
    let mut pm_inplace = pm_src.clone();
    group.bench_function(BenchmarkId::new("flip_horizontal", "pixmap"), |b| {
        b.iter(|| {
            black_box(&mut pm_inplace).flip_horizontal();
        });
    });
    group.bench_function(BenchmarkId::new("flip_vertical", "pixmap"), |b| {
        b.iter(|| {
            black_box(&mut pm_inplace).flip_vertical();
        });
    });
    group.bench_function(BenchmarkId::new("rotate_180", "pixmap"), |b| {
        b.iter(|| {
            black_box(&mut pm_inplace).rotate_180();
        });
    });
    group.bench_function(BenchmarkId::new("transpose", "pixmap"), |b| {
        b.iter(|| black_box(black_box(&pm_src).transposed()));
    });

    group.finish();
}

fn configure_criterion() -> Criterion {
    let level = Level::try_detect().unwrap_or(Level::baseline());
    eprintln!(
        "pixmap vs vello_cpu vs tiny-skia benchmark | arch={} os={} | vello_cpu SIMD level={level:?}",
        std::env::consts::ARCH,
        std::env::consts::OS,
    );
    Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets =
        bench_alpha,
        bench_solid,
        bench_compose,
        bench_mix,
        bench_resample,
        bench_blit_and_transform,
}
criterion_main!(benches);
