# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Support for SVG masks. Backends without luminance mask support render luminance masks as alpha masks.
- Support for SVG filter effects, translated into AnyRender `Filter` graphs. How faithfully they are applied depends on the backend.
- Filters set the `Filter::region` from the `<filter>` element's region, and primitive subregions are clipped to it.

### Fixed

- Keep the `<image>` placement for nested groups and raster images inside embedded SVG images.
- Draw the red placeholder for unsupported elements at the element's position, including the transform passed to `render_svg_tree`.
- Interpolate gradient stops unpremultiplied, as SVG requires (and resvg does). A transparent white stop no longer fades through black.

### Fixed

- Apply `clip-path`s whose content is transformed, nested in groups, made of several shapes or text, or itself clipped. Previously only a clip whose first child was a plain path was applied, and the clip path's own `transform` (including `objectBoundingBox` units) was ignored.

## [0.15.0] - 2026-10-03

### Added

- Support for SVG gradient `spreadMethod` values `pad`, `reflect` and `repeat` (#86).
- Support for SVG `image-rendering` on raster images (#87).
- Support for SVG `clip-rule`, including even-odd clipping (#105).

### Changed

- Updated to AnyRender 0.14.

### Fixed

- Correct radial gradient focal/center mapping and focal radius (#84).
- Correct fallback group clipping bounds and preserve antialiased pixels at layer edges (#89).

## [0.14.0] - 2026-08-16

### Changed

- Upgraded `usvg` to 0.48.1 (#78).

## [0.13.0] - 2026-07-23

### Changed

- Version bump for the AnyRender 0.12 release (#71).

## [0.12.0] - 2026-06-04

### Added

- Support for filter effects (#64).

## [0.11.0] - 2026-05-15

### Changed

- Version bump for the AnyRender 0.10 release ([`21d7c7d`](https://github.com/dioxuslabs/anyrender/commit/21d7c7d)).

## [0.10.0] - 2026-05-10

### Changed

- Version bump for the AnyRender 0.9 release ([`1389e05`](https://github.com/dioxuslabs/anyrender/commit/1389e05)).

## [0.9.1] - 2026-03-25

### Changed

- Removed the `thiserror` dependency ([`44a5798`](https://github.com/dioxuslabs/anyrender/commit/44a5798)).

## [0.9.0] - 2026-03-25

### Changed

- Version bump for the AnyRender 0.8 release ([`0adf06b`](https://github.com/dioxuslabs/anyrender/commit/0adf06b)).

## [0.8.0] - 2026-01-15

### Changed

- Upgraded to kurbo 0.13, peniko 0.6, wgpu 27, vello 0.7 and vello_cpu/hybrid 0.0.6 (#37).

## [0.6.3] - 2025-10-12

### Fixed

- Fixed the docs.rs build ([`1f2835a`](https://github.com/dioxuslabs/anyrender/commit/1f2835a)).

## [0.6.2] - 2025-10-09

### Changed

- Made the `text` feature enabled by default ([`ba7bf95`](https://github.com/dioxuslabs/anyrender/commit/ba7bf95)).

## [0.6.1] - 2025-10-09

### Changed

- Made the `text` feature optional ([`5a04e45`](https://github.com/dioxuslabs/anyrender/commit/5a04e45)).

## [0.6.0] - 2025-10-06

### Added

- Owned version of `Paint`; the `Scene` API now consistently uses `PaintRef` ([`d7e08b8`](https://github.com/dioxuslabs/anyrender/commit/d7e08b8)).

---

Versions prior to 0.6.0 predate this repository (the crate was previously
developed in the dioxus-native repository) and are not documented here.
