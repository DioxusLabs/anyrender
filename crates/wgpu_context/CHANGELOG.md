# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-07-23

### Added

- Alpha blitter for converting between alpha formats when compositing the surface (#71).

### Changed

- The surface renderer now supports configurable composite alpha modes (#71).

## [0.7.0] - 2026-06-23

### Fixed

- Don't reconfigure the surface while a surface output is alive (#66).

## [0.6.0] - 2026-05-15

### Changed

- Upgraded to WGPU v29, Vello 0.9 and Sparse Strips 0.0.8 (#62).

## [0.5.0] - 2026-05-10

### Changed

- Made `WindowRenderer::resume` async (with a callback) for wasm-friendly wgpu initialization (#59).

## [0.4.0] - 2026-03-25

### Fixed

- Handle transient WGPU `SurfaceError`s without panicking (#46).

## [0.3.1] - 2026-02-04

### Changed

- Version bump ([`4f0058f`](https://github.com/dioxuslabs/anyrender/commit/4f0058f)).

## [0.3.0] - 2026-02-02

### Fixed

- Fixed a `Timeout` crash when getting the surface texture (#38).

## [0.2.0] - 2026-01-15

### Changed

- Upgraded to kurbo 0.13, peniko 0.6, wgpu 27, vello 0.7 and vello_cpu/hybrid 0.0.6 (#37).

## [0.1.2] - 2025-11-14

### Changed

- Use `MemoryHints::MemoryUsage` ([`0c59130`](https://github.com/dioxuslabs/anyrender/commit/0c59130)).

## [0.1.1] - 2025-10-17

### Changed

- Version bump ([`dc1b392`](https://github.com/dioxuslabs/anyrender/commit/dc1b392)).

## [0.1.0] - 2025-10-06

### Added

- Initial release: context for managing WGPU surfaces ([`94830de`](https://github.com/dioxuslabs/anyrender/commit/94830de)).
