# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.13.0] - 2026-07-23

### Added

- Configurable composite alpha mode and base color via `VelloRendererOptions`, with builder-pattern methods (#71).

### Changed

- Marked `VelloRendererOptions` as `#[non_exhaustive]` (#71).

## [0.12.0] - 2026-06-23

### Changed

- Bumped `vello` and related backend dependencies ([`e465127`](https://github.com/dioxuslabs/anyrender/commit/e465127)).

## [0.11.0] - 2026-06-04

### Added

- Support for filter effects (#64).

## [0.10.1] - 2026-05-15

### Fixed

- Don't automatically apply a shift to emboldened glyphs ([`0da6d5d`](https://github.com/dioxuslabs/anyrender/commit/0da6d5d)).

## [0.10.0] - 2026-05-15

### Changed

- Upgraded to WGPU v29, Vello 0.9 and Sparse Strips 0.0.8 (#62).

## [0.9.0] - 2026-05-10

### Added

- Render `Context` and resource (e.g. wgpu `Texture`) registration (#58).
- wasm support ([`1f400d9`](https://github.com/dioxuslabs/anyrender/commit/1f400d9)).

### Changed

- Made `WindowRenderer::resume` async (with a callback) for wasm-friendly wgpu initialization (#59).

## [0.8.0] - 2026-03-25

### Fixed

- Handle transient WGPU `SurfaceError`s without panicking (#46).

## [0.7.1] - 2026-02-02

### Fixed

- Fixed a `Timeout` crash when getting the surface texture (via `wgpu_context`) (#38).

## [0.7.0] - 2026-01-15

### Changed

- Upgraded to kurbo 0.13, peniko 0.6, wgpu 27, vello 0.7 and vello_cpu/hybrid 0.0.6 (#37).

## [0.6.2] - 2025-12-27

### Fixed

- Buffer size fix ([`d747934`](https://github.com/dioxuslabs/anyrender/commit/d747934)).

## [0.6.1] - 2025-10-30

### Added

- `push_clip_layer` support (#30).

## [0.6.0] - 2025-10-06

### Added

- Owned version of `Paint`; the `Scene` API now consistently uses `PaintRef` ([`d7e08b8`](https://github.com/dioxuslabs/anyrender/commit/d7e08b8)).
- Re-export `DeviceHandle` ([`eb74b2b`](https://github.com/dioxuslabs/anyrender/commit/eb74b2b)).

### Changed

- CPU renderers are now generic (#1).
- Simplified the `VelloScenePainter` API ([`328a56b`](https://github.com/dioxuslabs/anyrender/commit/328a56b)).

---

Versions prior to 0.6.0 predate this repository (the crate was previously
developed in the dioxus-native repository) and are not documented here.
