# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-07-23

### Added

- Configurable base color via `SoftbufferRendererOptions`, along with `with_options`/`try_with_options` constructors (#71).

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

## [0.3.0] - 2026-03-25

### Changed

- Version bump for the AnyRender 0.8 release ([`0adf06b`](https://github.com/dioxuslabs/anyrender/commit/0adf06b)).

## [0.2.1] - 2026-02-02

### Changed

- Use a persistent buffer in the renderer ([`e855eee`](https://github.com/dioxuslabs/anyrender/commit/e855eee)).

## [0.2.0] - 2026-01-15

### Changed

- Upgraded to kurbo 0.13, peniko 0.6, wgpu 27, vello 0.7 and vello_cpu/hybrid 0.0.6 (#37).

## [0.1.0] - 2025-10-06

### Added

- Initial release: an AnyRender `WindowRenderer` backed by the `softbuffer` crate (#1).
