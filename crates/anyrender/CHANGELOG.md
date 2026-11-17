# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.0] - 2026-11-16

### Added

- `Paint::Pattern`, a paint that repeats a recorded `Scene` as a tile, with a stable id for caching the rasterized tile (breaking for code matching on `Paint`).
- `Scene::replay`, which draws a recorded scene into any `PaintScene`.
- `ColorMatrix::SOURCE_ALPHA` and `ColorMatrix::LUMINANCE_TO_ALPHA` constants.
- `PaintScene::push_mask_layer` with `MaskMode::Alpha` and `MaskMode::Luminance`, and the recorded `MaskLayerCommand` (breaking for `PaintScene` implementors).
- `Filter::region`, an optional filter region that clips the source graphic and every primitive result, and is the default subregion of primitives that fill an area. `None` (the default) is unbounded.
- `NonUniformRoundedRect` for CSS-compatible box shadows ([#117](https://github.com/DioxusLabs/anyrender/pull/117)).

### Changed

- The filter types now live in the new `image_filters_types` crate, generic over the content of `feImage` primitives, and `anyrender::filters` re-exports them. `Filter`, `FilterGraphNode` and `FilterEffect` are aliases that use `ExternalImageSource` for `feImage`, so existing code keeps compiling.
- `Filter::add` now makes the added primitive the graph's output, so graphs built with `Filter::empty` and `add` output their last node instead of their first.
- Added an `AntiAliasing` parameter to `fill`, `stroke` and `draw_glyphs` (breaking) ([#198](https://github.com/DioxusLabs/anyrender/pull/198)).
- Changed `draw_box_shadow` to follow CSS geometry and spread semantics (breaking) ([#117](https://github.com/DioxusLabs/anyrender/pull/117)).
- Filter regions, node regions and primitive subregions are now required; filter constructors require bounds (breaking) ([#188](https://github.com/DioxusLabs/anyrender/pull/188), [#194](https://github.com/DioxusLabs/anyrender/pull/194), [#196](https://github.com/DioxusLabs/anyrender/pull/196)).
- `feImage` content is now a recorded `Scene`, rather than an external image source (breaking) ([#176](https://github.com/DioxusLabs/anyrender/pull/176)).
- Expanded filter primitives with per-primitive colour spaces, separate X/Y parameters, full convolution parameters, lighting colour, stitched turbulence, primitive subregions and tiling (breaking) ([#168](https://github.com/DioxusLabs/anyrender/pull/168), [#169](https://github.com/DioxusLabs/anyrender/pull/169), [#170](https://github.com/DioxusLabs/anyrender/pull/170), [#171](https://github.com/DioxusLabs/anyrender/pull/171), [#172](https://github.com/DioxusLabs/anyrender/pull/172), [#174](https://github.com/DioxusLabs/anyrender/pull/174)).
- Added `CompositeOperator::Lighter`; turbulence seeds are now `f32`, truncated toward zero (breaking) ([#186](https://github.com/DioxusLabs/anyrender/pull/186), [#187](https://github.com/DioxusLabs/anyrender/pull/187)).

### Fixed

- Fixed the `FilterEffect` size assertion on 32-bit x86 ([#120](https://github.com/DioxusLabs/anyrender/pull/120)).

## [0.14.0] - 2026-10-03

### Added

- Explicit clip fill rules on `PaintScene::push_layer` and `push_clip_layer`, and `fill` fields on recorded `LayerCommand` and `ClipCommand` (breaking). Older serialized scenes default to the nonzero fill rule (#105).
- `ColorMatrix` constructors for CSS brightness, contrast, invert and opacity filters (#104).

### Changed

- CSS brightness, contrast, invert and opacity filters now use `ColorMatrix` instead of `ComponentTransfer`, enabling them in the Vello CPU/GPU backends (#104).
- Raised the workspace MSRV to Rust 1.90 (#94).

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
