# Changelog

All notable changes to this crate are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-07-23

### Changed

- Version bump for the AnyRender 0.12 release (#71).

## [0.5.0] - 2026-06-04

### Added

- Support for filter effects (#64).

## [0.4.0] - 2026-05-24

### Changed

- Bumped fontations dependencies ([`2dc40e1`](https://github.com/dioxuslabs/anyrender/commit/2dc40e1)).

## [0.3.0] - 2026-05-15

### Changed

- Upgraded to WGPU v29, Vello 0.9 and Sparse Strips 0.0.8 (#62).

## [0.2.0] - 2026-05-10

### Added

- Render `Context` and resource (e.g. wgpu `Texture`) registration (#58).

## [0.1.1] - 2026-04-21

### Changed

- Improved font serialization (#41).
- Use runtime config for serialization (#42).
- Use `skera` from crates.io ([`e78e467`](https://github.com/dioxuslabs/anyrender/commit/e78e467)).

### Fixed

- Pinned `skera` to avoid build breakage caused by a fontations update ([`2f12eb0`](https://github.com/dioxuslabs/anyrender/commit/2f12eb0)).

## [0.1.0] - 2026-02-07

### Added

- Initial release: serialization of recorded scenes to a portable zip format (#40).
