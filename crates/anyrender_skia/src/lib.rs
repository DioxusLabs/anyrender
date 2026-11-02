mod image_renderer;
mod scene;
mod window_renderer;

// Backends
mod cache;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod metal;
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
mod opengl;
#[cfg(feature = "vulkan")]
mod vulkan;

pub use image_renderer::SkiaImageRenderer;
pub use scene::{SkiaSceneCache, SkiaScenePainter};
pub use window_renderer::*;

/// Whether Skia rasterizes glyphs with FreeType here (Linux, Android, BSDs).
const FREETYPE_TEXT: bool = !cfg!(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "windows"
));

/// Text contrast and gamma that Chromium builds Skia with on each platform (`SK_GAMMA_CONTRAST`
/// and `SK_GAMMA_EXPONENT` in Chromium's `skia/BUILD.gn`). A gamma of 0 means sRGB.
///
/// Skia's own defaults (contrast 0.5, sRGB gamma) render text noticeably lighter and blurrier
/// than Chrome and Firefox.
const TEXT_CONTRAST_AND_GAMMA: Option<(f32, f32)> = if cfg!(target_os = "ios") {
    None
} else if cfg!(target_os = "macos") {
    Some((0.0, 0.0))
} else if cfg!(target_os = "windows") {
    Some((1.0, 0.0))
} else if cfg!(target_os = "android") {
    Some((0.0, 1.4))
} else {
    Some((0.2, 1.2))
};

/// Surface properties for every surface we draw into.
pub(crate) fn surface_props() -> skia_safe::SurfaceProps {
    let props = skia_safe::SurfaceProps::default();
    let Some((contrast, gamma)) = TEXT_CONTRAST_AND_GAMMA else {
        return props;
    };
    skia_safe::SurfaceProps::new_with_text_properties(
        props.flags(),
        props.pixel_geometry(),
        contrast,
        gamma,
    )
}
