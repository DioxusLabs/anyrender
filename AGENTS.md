# AnyRender

## Crate dependencies

- Backend crates (`anyrender_vello`, `anyrender_vello_cpu`, `anyrender_vello_gpu`, `anyrender_skia`, `anyrender_tiny_skia`) must not depend on one another. This includes `[dev-dependencies]`: cargo keeps versioned dev-dependencies in the published manifest, so they constrain publish order and couple one backend's tests to another backend's behaviour.
- Backends may depend on the core and shared crates (`anyrender`, `image_filters_types`, `wgpu_context`, `pixels_window_renderer`, `softbuffer_window_renderer`).
- Test a backend against itself, for example by comparing against the same scene drawn a different way through the same backend.
