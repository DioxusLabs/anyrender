// Copyright 2018 the Resvg Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU implementation of SVG/CSS filter graphs ([`image_filters_types::Filter`]): [`apply`]
//! evaluates a whole graph on a tiny-skia [`Pixmap`](tiny_skia::Pixmap), using the pixel kernels
//! of each filter primitive in [`kernels`]. Ported from resvg (`crates/resvg/src/filter`).

mod graph;
pub mod kernels;

pub use graph::{apply, blur};
