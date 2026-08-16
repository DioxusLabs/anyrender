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

- [anyrender_vello_hybrid](https://docs.rs/anyrender_vello_hybrid) which draws using [vello_hybrid](https://docs.rs/vello_hybrid)
- [anyrender_vello_cpu](https://docs.rs/anyrender_vello_cpu) which draws using [vello_cpu](https://docs.rs/vello_cpu)
- [anyrender_vello](https://docs.rs/anyrender_vello) which draws using [vello](https://docs.rs/vello)
- [anyrender_skia](https://crates.io/crates/anyrender_skia) which draws using Skia (via the [skia-safe](https://github.com/rust-skia/rust-skia) crate)

Contributions for other backends (tiny-skia, femtovg, etc) would be very welcome.

### Content renderers

These crates sit on top of the the AnyRender abstraction, and allow you render content through it:

- [anyrender_svg](https://docs.rs/anyrender_svg) allows you to render SVGs with AnyRender. [usvg](https://docs.rs/usvg) is used to parse the SVGs.
- [blitz-paint](https://docs.rs/blitz-paint) can be used to HTML/CSS (and markdown) that has been parsed, styled, and layouted by [blitz-dom](https://docs.rs/blitz-dom) using AnyRender.
- [polymorpher](https://github.com/Aiving/polymorpher) implements Material Design 3 shape morphing, and can be used with AnyRender by enabling the `kurbo` feature.

### Utility crates

- [wgpu_context](https://docs.rs/wgpu_context) is a utility for managing `Device`s and other WGPU types
- [pixels_window_renderer](https://docs.rs/pixels_window_renderer) implements an AnyRender `WindowRenderer` for any AnyRenderer `ImageRenderer` using the [pixels](https://docs.rs/pixels) crate.
- [softbuffer_window_renderer](https://docs.rs/softbuffer_window_renderer) implements an AnyRender `WindowRenderer` for any AnyRenderer `ImageRenderer` using the [softbuffer](https://docs.rs/softbuffer) crate.


## Version compatibility

AnyRender crates are released together, so a given `anyrender` core release lines up with a
specific set of backend, content-renderer and utility crate versions, as well as the upstream
dependency versions they were built against. Use the same release line across all AnyRender
crates in your project.

<table>
  <thead>
    <tr><th>AnyRender</th><th>0.6</th><th>0.7</th><th>0.8</th><th>0.9</th><th>0.10</th><th>0.11</th><th>0.12</th></tr>
  </thead>
  <tbody>
    <tr><td><code>kurbo</code></td><td>0.12</td><td>0.13</td><td>0.13</td><td>0.13</td><td>0.13</td><td>0.13</td><td>0.13</td></tr>
    <tr><td><code>peniko</code></td><td>0.5</td><td>0.6</td><td>0.6</td><td>0.6</td><td>0.6</td><td>0.6</td><td>0.6</td></tr>
    <tr><th colspan="8" align=left>WGPU</th></tr>
    <tr><td><code>wgpu</code></td><td>26</td><td>27</td><td>28</td><td>28</td><td>29</td><td>29</td><td>29</td></tr>
    <tr><td><code>wgpu_context</code></td><td>0.1</td><td>0.2</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.6–0.7</td><td>0.8</td></tr>
    <tr><th colspan="8" align=left>Vello</th></tr>
    <tr><td><code>anyrender_vello</code></td><td>0.6</td><td>0.7</td><td>0.8</td><td>0.9</td><td>0.10</td><td>0.11–0.12</td><td>0.13</td></tr>
    <tr><td><code>vello</code></td><td>0.6</td><td>0.7 <sup><a href="#fn-vello-git">1</a></sup></td><td>0.8</td><td>0.8</td><td>0.9</td><td>0.9</td><td>0.9</td></tr>
    <tr><th colspan="8" align=left>Vello Hybrid</th></tr>
    <tr><td><code>vello_hybrid</code></td><td>0.0.4</td><td>0.0.6</td><td>0.0.7</td><td>0.0.7</td><td>0.0.8</td><td>0.0.9</td><td>0.0.9</td></tr>
    <tr><td><code>anyrender_vello_hybrid</code></td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.7–0.8</td><td>0.9</td></tr>
    <tr><th colspan="8" align=left>Vello CPU</th></tr>
    <tr><td><code>vello_cpu</code></td><td>0.0.4</td><td>0.0.6</td><td>0.0.7</td><td>0.0.7</td><td>0.0.8</td><td>0.0.9</td><td>0.0.9</td></tr>
    <tr><td><code>anyrender_vello_cpu</code></td><td>0.8</td><td>0.9</td><td>0.10</td><td>0.11</td><td>0.12</td><td>0.14</td><td>0.15</td></tr>
    <tr><th colspan="8" align=left>Skia</th></tr>
    <tr><td><code>skia-safe</code></td><td>0.89</td><td>0.91</td><td>0.93</td><td>0.93</td><td>0.93 <sup><a href="#fn-skia-097">2</a></sup></td><td>0.97</td><td>0.97</td></tr>
    <tr><td><code>anyrender_skia</code></td><td>0.1</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td><td>0.9</td><td>0.10</td></tr>
    <tr><th colspan="8" align=left>Content Libs</th></tr>
    <tr><td><code>anyrender_svg</code></td><td>0.6</td><td>0.8</td><td>0.9</td><td>0.10</td><td>0.11</td><td>0.12</td><td>0.13</td></tr>
    <tr><td><code>anyrender_serialize</code></td><td>—</td><td>—</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.5</td><td>0.6</td></tr>
    <tr><th colspan="8" align=left>CPU <code>WindowRenderer</code>s</th></tr>
    <tr><td><code>pixels_window_renderer</code></td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td></tr>
    <tr><td><code>softbuffer_window_renderer</code></td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td><td>0.5</td><td>0.6</td><td>0.7</td></tr>
  </tbody>
</table>

1. <a id="fn-vello-git"></a>The 0.7 release line depended on a git revision of Vello, equivalent to Vello 0.7 and sparse strips 0.0.6.
2. <a id="fn-skia-097"></a>Within the 0.10 line, `anyrender_skia` 0.8.x upgraded `skia-safe` from 0.93 to 0.97.

## Minimum supported Rust Version (MSRV)

This version of AnyRender has been verified to compile with **Rust 1.88** and later.

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

[kurbo]: https://crates.io/crates/kurbo
[Rust Code of Conduct]: https://www.rust-lang.org/policies/code-of-conduct

