# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-16

### Added

- Initial release of generic SVG/CSS filter graph types, extracted from AnyRender and generic over `feImage` content ([#181](https://github.com/DioxusLabs/anyrender/pull/181)).
- Required filter/node regions and primitive subregions, region-aware graph construction and tiling ([#174](https://github.com/DioxusLabs/anyrender/pull/174), [#188](https://github.com/DioxusLabs/anyrender/pull/188), [#194](https://github.com/DioxusLabs/anyrender/pull/194), [#196](https://github.com/DioxusLabs/anyrender/pull/196)).
- Per-primitive colour spaces, independent X/Y parameters, full convolution parameters, lighting colour and stitched turbulence ([#168](https://github.com/DioxusLabs/anyrender/pull/168), [#169](https://github.com/DioxusLabs/anyrender/pull/169), [#170](https://github.com/DioxusLabs/anyrender/pull/170), [#171](https://github.com/DioxusLabs/anyrender/pull/171), [#172](https://github.com/DioxusLabs/anyrender/pull/172)).
- `CompositeOperator::Lighter` and floating-point turbulence seeds truncated toward zero ([#186](https://github.com/DioxusLabs/anyrender/pull/186), [#187](https://github.com/DioxusLabs/anyrender/pull/187)).
