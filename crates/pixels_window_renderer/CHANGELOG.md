# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-07-23

### Added

- Configurable renderer options (`PixelsRendererOptions`) including base color and composite alpha mode, with builder-pattern methods (#71).

## [0.6.0] - 2026-06-04

### Changed

- Version bump for the AnyRender 0.11 release ([`2c00f1e`](https://github.com/dioxuslabs/anyrender/commit/2c00f1e)).

## [0.5.0] - 2026-05-15

### Changed

- Version bump for the AnyRender 0.10 release ([`21d7c7d`](https://github.com/dioxuslabs/anyrender/commit/21d7c7d)).

## [0.4.0] - 2026-05-10

### Added

- Render `Context` and resource (e.g. wgpu `Texture`) registration (#58).

### Changed

- Made `WindowRenderer::resume` async (with a callback) for wasm-friendly wgpu initialization (#59).

## [0.3.2] - 2026-04-22

### Changed

- Upgraded to `pixels` 0.17 ([`61deb27`](https://github.com/dioxuslabs/anyrender/commit/61deb27)).

## [0.3.1] - 2026-04-21

### Changed

- Upgraded to `pixels` 0.16 ([`c12e3ff`](https://github.com/dioxuslabs/anyrender/commit/c12e3ff)).

## [0.3.0] - 2026-03-25

### Changed

- Version bump for the AnyRender 0.8 release ([`0adf06b`](https://github.com/dioxuslabs/anyrender/commit/0adf06b)).

## [0.2.0] - 2026-01-15

### Changed

- Upgraded to kurbo 0.13, peniko 0.6, wgpu 27, vello 0.7 and vello_cpu/hybrid 0.0.6 (#37).

## [0.1.0] - 2025-10-06

### Added

- Initial release: an AnyRender `WindowRenderer` backed by the `pixels` crate (#1).
