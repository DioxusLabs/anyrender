# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.13.0] - 2026-08-16

### Fixed

- `FilterEffect` size assertion on 32-bit targets in general, not just wasm (#74).

## [0.12.0] - 2026-07-23

### Added

- `CompositeAlphaMode` enum and a unified `RendererConfig` type (with builder-pattern methods) for configuring renderer transparency and background color (#71).

## [0.11.1] - 2026-06-07

### Changed

- Updated the backend list in the crate documentation ([`54fa6b6`](https://github.com/dioxuslabs/anyrender/commit/54fa6b6)).

## [0.11.0] - 2026-06-04

### Added

- Support for filter effects (#64).

## [0.10.0] - 2026-05-15

### Changed

- Upgraded to WGPU v29, Vello 0.9 and Sparse Strips 0.0.8 (#62).

## [0.9.0] - 2026-05-10

### Added

- Render `Context` and resource (e.g. wgpu `Texture`) registration (#58).

### Changed

- Made `WindowRenderer::resume` async (with a callback) for wasm-friendly wgpu initialization (#59).

## [0.8.0] - 2026-03-25

### Added

- `Scene` recording (#2).
- Scene serialization support (#40).
- Default image type for `StrokeCommand` and `FillCommand` ([`1ca48d0`](https://github.com/dioxuslabs/anyrender/commit/1ca48d0)).

## [0.7.0] - 2026-01-15

### Changed

- Upgraded to kurbo 0.13, peniko 0.6, wgpu 27, vello 0.7 and vello_cpu/hybrid 0.0.6 (#37).

## [0.6.2] - 2025-10-30

### Added

- `push_clip_layer` method (#30), with a default implementation (#31).

## [0.6.1] - 2025-10-09

### Added

- `Null` backend ([`9f54cd3`](https://github.com/dioxuslabs/anyrender/commit/9f54cd3)).

### Fixed

- Fixed the `render_to_buffer` function (#15).

## [0.6.0] - 2025-10-06

### Added

- Owned version of `Paint`; the `Scene` API now consistently uses `PaintRef` ([`d7e08b8`](https://github.com/dioxuslabs/anyrender/commit/d7e08b8)).

### Changed

- CPU renderers are now generic (#1).

---

Versions prior to 0.6.0 predate this repository (the crate was previously
developed in the dioxus-native repository) and are not documented here.
