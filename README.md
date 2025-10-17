# AnyRender

**A Rust 2D drawing abstraction.**

[![Linebender Zulip, #kurbo channel](https://img.shields.io/badge/Linebender-grey?logo=Zulip)](https://xi.zulipchat.com)
[![dependency status](https://deps.rs/repo/github/dioxuslabs/anyrender/status.svg)](https://deps.rs/repo/github/dioxuslabs/anyrender)
[![Apache 2.0 or MIT license.](https://img.shields.io/badge/license-Apache--2.0_OR_MIT-blue.svg)](#license)
[![Crates.io](https://img.shields.io/crates/v/anyrender.svg)](https://crates.io/crates/anyrender)
[![Docs](https://docs.rs/anyrender/badge.svg)](https://docs.rs/anyrender)

AnyRender is a 2D drawing abstaction that allows applications/frameworks to support many rendering backends through a unified API.

Discussion of AnyRender development happens in the Linebender Zulip at <https://xi.zulipchat.com/>.

## Crates

### `anyrender`

The core [anyrender](https://docs.rs/anyrender) crate is a lightweight type/trait-only crate that defines three abstractions:

- **The [PaintScene](https://docs.rs/anyrender/latest/anyrender/trait.PaintScene.html) trait accepts drawing commands.**
  Applications and libraries draw by pushing commands into a `PaintScene`. Backends generally execute those commands to
  produce an output (although they may do other things like store them for later use).
- **The [WindowRenderer](https://docs.rs/anyrender/latest/anyrender/trait.WindowRenderer.html) trait abstracts over types that can render to a window**
- **The [ImageRenderer](https://docs.rs/anyrender/latest/anyrender/trait.ImageRenderer.html) trait abstracts over types that can render to a `Vec<u8>` image buffer**

### Backends

Currently existing backends are:

- [anyrender_vello](https://docs.rs/anyrender_vello) which draws using [vello](https://docs.rs/vello)
- [anyrender_vello_cpu](https://docs.rs/anyrender_vello_cpu) which draws using [vello_cpu](https://docs.rs/vello_cpu)
- [anyrender_vello_hybrid](https://docs.rs/anyrender_vello_hybrid) which draws using [vello_hybrid](https://docs.rs/vello_hybrid)

Contributions for other backends (Skia, FemtoVG, etc) would be very welcome.

### Content renderers

These crates sit on top of the the AnyRender abstraction, and allow you render content through it:

- [anyrender_svg](https://docs.rs/anyrender_svg) allows you to render SVGs with AnyRender. [usvg](https://docs.rs/usvg) is used to parse the SVGs.
- [polymorpher](https://github.com/Aiving/polymorpher) implements Material Design 3 shape morphing, and can be used with AnyRender by enabling the `kurbo` feature.

### Utility crates

- [wgpu_context](https://docs.rs/wgpu_context) is a utility for managing `Device`s and other WGPU types
- [pixels_window_renderer](https://docs.rs/pixels_window_renderer) implements an AnyRender `WindowRenderer` for any AnyRenderer `ImageRenderer` using the [pixels](https://docs.rs/pixels) crate.
- [softbuffer_window_renderer](https://docs.rs/softbuffer_window_renderer) implements an AnyRender `WindowRenderer` for any AnyRenderer `ImageRenderer` using the [softbuffer](https://docs.rs/softbuffer) crate.


## Minimum supported Rust Version (MSRV)

This version of AnyRender has been verified to compile with **Rust 1.86** and later.

Future versions of AnyRender might increase the Rust version requirement.
It will not be treated as a breaking change and as such can even happen with small patch releases.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Contributions are welcome by pull request. The [Rust code of conduct] applies.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.

[color]: https://crates.io/crates/color
[kurbo]: https://crates.io/crates/kurbo
[Rust Code of Conduct]: https://www.rust-lang.org/policies/code-of-conduct

