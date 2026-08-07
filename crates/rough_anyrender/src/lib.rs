// This crate is entirely safe
#![forbid(unsafe_code)]
// Ensures that `pub` means published in the public API.
// This property is useful for reasoning about breaking API changes.
#![deny(unreachable_pub)]

//!
//! This crate is an adapter crate between [roughr](https://github.com/orhanbalci/rough-rs/tree/main/roughr) and
//! [anyrender](https://github.com/dioxuslabs/anyrender). It converts roughr drawing
//! primitives into calls on AnyRender's `PaintScene`, so hand-sketched shapes can be
//! rendered with any AnyRender backend. For more detailed information you can check the
//! roughr crate.
//!
//! ## 📦 Cargo.toml
//!
//! ```toml
//! [dependencies]
//! rough_anyrender = "0.1"
//! # Plus any AnyRender backend, e.g.:
//! anyrender_vello_hybrid = "0.8"
//! ```
//!
//! ## 🔧 Example
//!
//! ### Rust Logo
//!
//! ```ignore
//! use anyrender::{PaintScene, Scene};
//! use palette::Srgba;
//! use rough_anyrender::AnyRenderGenerator;
//! use roughr::core::{FillStyle, OptionsBuilder};
//!
//! let options = OptionsBuilder::default()
//!     .stroke(Srgba::from_components((114u8, 87u8, 82u8, 255u8)).into_format())
//!     .fill(Srgba::from_components((254u8, 246u8, 201u8, 255)).into_format())
//!     .fill_style(FillStyle::Hachure)
//!     .fill_weight(1.0)
//!     .bowing(0.8)
//!     .build()
//!     .unwrap();
//!
//! let generator = AnyRenderGenerator::new(options);
//! let rust_logo_svg_path = "..."; // SVG path data for the Rust logo
//! let rust_logo_drawing = generator.path::<f32>(rust_logo_svg_path.to_string());
//!
//! // `draw` accepts any `anyrender::PaintScene`. Here we record into a `anyrender::Scene`, but you
//! // can also draw straight into the scene painter handed to a `WindowRenderer::render`
//! // closure (see the examples).
//! let mut scene = Scene::new();
//! rust_logo_drawing.draw(&mut scene);
//! ```
//!
//! ### 🖨️ Output Rust Logo
//! ![rust_logo](https://raw.githubusercontent.com/orhanbalci/rough-rs/main/rough_vello/assets/rust_logo.png)
//!
//! ## Filler Implementation Status
//! - [x] Hachure
//! - [x] Zigzag
//! - [x] Cross-Hatch
//! - [x] Dots
//! - [x] Dashed
//! - [x] Zigzag-Line
//!
//! ## 🔭 Examples
//!
//! Runnable [winit](https://docs.rs/winit) examples live in the `examples/` directory
//! (`rectangle`, `rust_logo` and `animate`). Run one with, for example:
//!
//! ```sh
//! cargo run --example rust_logo
//! ```
//!
//! ## 🔌 Integration
//!
//! Because drawing targets AnyRender's `PaintScene`, `rough_anyrender` works with any
//! AnyRender backend, including `anyrender_vello`, `anyrender_vello_hybrid`,
//! `anyrender_vello_cpu` and `anyrender_skia`. The bundled examples use
//! `anyrender_vello_hybrid` together with `winit`.

mod generator;
pub use generator::*;
