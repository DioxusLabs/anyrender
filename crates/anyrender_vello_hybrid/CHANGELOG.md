# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-07-23

### Added

- Configurable composite alpha mode and base color via `VelloHybridRendererOptions`, with builder-pattern methods (#71).

### Changed

- Marked `VelloHybridRendererOptions` as `#[non_exhaustive]` (#71).

## [0.8.0] - 2026-06-23

### Changed

- Bumped `vello_hybrid` and related backend dependencies ([`e465127`](https://github.com/dioxuslabs/anyrender/commit/e465127)).

## [0.7.0] - 2026-06-04

### Added

- Support for filter effects (#64).

## [0.6.0] - 2026-05-31

### Changed

- Always use a clip path for clips in the WebGL renderer ([`262f36a`](https://github.com/dioxuslabs/anyrender/commit/262f36a)).

## [0.5.2] - 2026-05-25

### Changed

- Use `push_clip_path` for clipping layers ([`324f0b7`](https://github.com/dioxuslabs/anyrender/commit/324f0b7)).

## [0.5.1] - 2026-05-15

### Fixed

- Don't automatically apply a shift to emboldened glyphs ([`0da6d5d`](https://github.com/dioxuslabs/anyrender/commit/0da6d5d)).

## [0.5.0] - 2026-05-15

### Changed

- Upgraded to WGPU v29, Vello 0.9 and Sparse Strips 0.0.8 (#62).

## [0.4.0] - 2026-05-10

### Added

- Render `Context` and resource (e.g. wgpu `Texture`) registration (#58).

### Changed

- Made `WindowRenderer::resume` async (with a callback) for wasm-friendly wgpu initialization (#59).

## [0.3.0] - 2026-03-25

### Added

- WebGL support (#43).
- Exposed the hybrid image manager (#44).

### Changed

- Upgraded to wgpu v28, Vello v0.8 and Sparse Strips v0.0.7 ([`9864394`](https://github.com/dioxuslabs/anyrender/commit/9864394)).

### Fixed

- Handle transient WGPU `SurfaceError`s without panicking (#46).

## [0.2.2] - 2026-02-04

### Fixed

- Use `Rgba8Unorm` rather than `Bgra8Unorm` on Android (via `wgpu_context`) ([`6167f19`](https://github.com/dioxuslabs/anyrender/commit/6167f19)).

## [0.2.1] - 2026-02-02

### Changed

- Set `Scene` dimensions in the `resume` method ([`0142a74`](https://github.com/dioxuslabs/anyrender/commit/0142a74)).
- Use `push_clip_layer` ([`7d82dd2`](https://github.com/dioxuslabs/anyrender/commit/7d82dd2)).

### Fixed

- Fixed a `Timeout` crash when getting the surface texture (via `wgpu_context`) (#38).

## [0.2.0] - 2026-01-15

### Changed

- Upgraded to kurbo 0.13, peniko 0.6, wgpu 27, vello 0.7 and vello_cpu/hybrid 0.0.6 (#37).

## [0.1.1] - 2025-10-30

### Added

- `push_clip_layer` support (#30).

## [0.1.0] - 2025-10-17

### Added

- Initial `vello_hybrid` backend for anyrender (#13).
