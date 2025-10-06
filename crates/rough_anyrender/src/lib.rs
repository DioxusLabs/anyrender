// This crate is entirely safe
#![forbid(unsafe_code)]
// Ensures that `pub` means published in the public API.
// This property is useful for reasoning about breaking API changes.
#![deny(unreachable_pub)]

//!
//! This crate is an adapter crate between [roughr](https://github.com/orhanbalci/rough-rs/main/roughr) and
//! [vello](https://github.com/linebender/vello) crates. Converts from roughr drawing
//! primitives to vello's Scene types. Also has convenience traits for drawing onto vello scenes. For more detailed
//! information you can check roughr crate.
//!
//! Below examples are output of [rough_vello](https://github.com/orhanbalci/rough-rs/tree/main/rough_vello) adapter.
//!
//! ## 📦 Cargo.toml
//!
//! ```toml
//! [dependencies]
//! rough_vello = "0.1"
//! ```
//!
//! ## 🔧 Example
//!
//! ### Rust Logo
//!
//! ```ignore
//! use rough_vello::VelloGenerator;
//! use vello::Scene;
//! use palette::Srgba;
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
//! let generator = VelloGenerator::new(options);
//! let rust_logo_svg_path = "..."; // SVG path data for the Rust logo
//! let rust_logo_drawing = generator.path::<f32>(rust_logo_svg_path);
//!
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
//! For more examples have a look at the
//! [examples](https://github.com/orhanbalci/rough-rs/tree/main/rough_vello/examples) folder.
//!
//! ## 🔌 Integration
//!
//! ### Bevy Integration
//!
//! For Bevy game engine integration, you can use [bevy_vello](https://github.com/linebender/bevy_vello) which provides a Bevy plugin for vello. This allows you to render `rough_vello` drawings directly in your Bevy applications by converting the vello Scene to Bevy-compatible rendering.
//!
//! ```toml
//! [dependencies]
//! rough_vello = "0.1"
//! bevy_vello = "0.1"  # Check latest version
//! bevy = "0.14"       # Or latest compatible version
//! ```

pub mod vello_generator;
pub use vello_generator::*;
