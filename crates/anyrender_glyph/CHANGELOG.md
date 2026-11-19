# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial release: a fork of `glifo` 0.5 without its glyph atlas and without a dependency on `vello_common`. It prepares outline, COLR and bitmap glyphs and draws them through the `GlyphRenderer` trait, which also lets a renderer supply its own raster cache. `GlyphRasterCache` is a per-glyph raster cache with a byte budget for renderers that keep one image per glyph.
