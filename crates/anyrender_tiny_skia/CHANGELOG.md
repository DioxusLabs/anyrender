# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-16

### Added

- Initial release of the tiny-skia backend, with image rendering and optional pixels/softbuffer window renderers ([#179](https://github.com/DioxusLabs/anyrender/pull/179)).
- Pattern paints, masks and SVG filter effects with required regions and recorded-scene image content ([#176](https://github.com/DioxusLabs/anyrender/pull/176), [#188](https://github.com/DioxusLabs/anyrender/pull/188), [#191](https://github.com/DioxusLabs/anyrender/pull/191), [#194](https://github.com/DioxusLabs/anyrender/pull/194), [#196](https://github.com/DioxusLabs/anyrender/pull/196)).
- Explicit antialiasing control on fills, strokes and glyphs ([#198](https://github.com/DioxusLabs/anyrender/pull/198)).

### Fixed

- Clip filter regions precisely, run convolution on the filter region, and allocate filter layers using their region rather than canvas bounds ([#193](https://github.com/DioxusLabs/anyrender/pull/193), [#195](https://github.com/DioxusLabs/anyrender/pull/195), [#197](https://github.com/DioxusLabs/anyrender/pull/197)).
- Run filters in scale-only space under rotation/skew and retain a 2px margin on masked clip layers ([#199](https://github.com/DioxusLabs/anyrender/pull/199), [#201](https://github.com/DioxusLabs/anyrender/pull/201), [#205](https://github.com/DioxusLabs/anyrender/pull/205)).
