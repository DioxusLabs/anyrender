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

/// Whether to override Skia's default text contrast and gamma (everywhere except Apple, where
/// CoreText font smoothing is used instead).
const CHROME_TEXT_GAMMA: bool = !cfg!(any(target_os = "macos", target_os = "ios"));

/// Surface properties for every surface we draw into.
///
/// With FreeType and DirectWrite, Skia's default text contrast (0.5) and sRGB gamma render text
/// noticeably lighter and blurrier than Chrome and Firefox. Use the contrast and gamma that Chrome
/// builds Skia with (`SK_GAMMA_CONTRAST=0.2`, `SK_GAMMA_EXPONENT=1.2`).
pub(crate) fn surface_props() -> skia_safe::SurfaceProps {
    let props = skia_safe::SurfaceProps::default();
    if !CHROME_TEXT_GAMMA {
        return props;
    }
    skia_safe::SurfaceProps::new_with_text_properties(
        props.flags(),
        props.pixel_geometry(),
        0.2,
        1.2,
    )
}
