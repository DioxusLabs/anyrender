# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.16.0] - 2026-08-16

### Changed

- Upgraded `vello_cpu` to 0.1 (#80).

## [0.15.0] - 2026-07-23

### Added

- Renderer configuration options, including composite alpha mode support (#71).

## [0.14.0] - 2026-06-04

### Added

- Support for filter effects (#64).

## [0.13.0] - 2026-05-31

### Changed

- Version bump ([`efdb47d`](https://github.com/dioxuslabs/anyrender/commit/efdb47d)).

## [0.12.1] - 2026-05-15

### Fixed

- Don't automatically apply a shift to emboldened glyphs ([`0da6d5d`](https://github.com/dioxuslabs/anyrender/commit/0da6d5d)).

## [0.12.0] - 2026-05-15

### Changed

- Upgraded to WGPU v29, Vello 0.9 and Sparse Strips 0.0.8 (#62).

## [0.11.0] - 2026-05-10

### Added

- Render `Context` and resource (e.g. wgpu `Texture`) registration (#58).

## [0.10.0] - 2026-03-25

### Added

- `experimental_image_cache` feature ([`52effbe`](https://github.com/dioxuslabs/anyrender/commit/52effbe)).

## [0.9.0] - 2026-01-15

### Changed

- Upgraded to kurbo 0.13, peniko 0.6, wgpu 27, vello 0.7 and vello_cpu/hybrid 0.0.6 (#37).

### Fixed

- Buffer size fix ([`d747934`](https://github.com/dioxuslabs/anyrender/commit/d747934)).

## [0.8.1] - 2025-10-30

### Added

- `push_clip_layer` support (#30).

## [0.8.0] - 2025-10-17

### Changed

- Version bump ([`53910a5`](https://github.com/dioxuslabs/anyrender/commit/53910a5)).

## [0.7.0] - 2025-10-06

### Added

- Owned version of `Paint`; the `Scene` API now consistently uses `PaintRef` ([`d7e08b8`](https://github.com/dioxuslabs/anyrender/commit/d7e08b8)).
- Debug timings for the `ImageRenderer` ([`97949e9`](https://github.com/dioxuslabs/anyrender/commit/97949e9)).

### Changed

- CPU renderers are now generic (#1).

---

Versions prior to 0.7.0 predate this repository (the crate was previously
developed in the dioxus-native repository) and are not documented here.
