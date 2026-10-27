use anyrender::{RenderContext, WindowRenderer};
use debug_timer::debug_timer;
use skia_safe::{Color, Surface, graphics};
use std::any::Any;
use std::sync::Arc;

use crate::{BackendError, SkiaScenePainter, scene::SkiaSceneCache};

/// GPU backend to use for a window. Auto uses D3D12 on Windows, Metal on Apple
/// platforms, and OpenGL elsewhere. Windows Auto and D3D try hardware D3D12,
/// then OpenGL, then software D3D12 (WARP or Microsoft Basic Render Driver).
/// Transparent Windows surfaces always use OpenGL.
///
/// `ANYRENDER_SKIA_BACKEND` overrides this preference at resume time. Values are
/// case-insensitive: `gl`/`opengl`, `d3d`/`d3d12`, or `metal`. Unknown or unavailable backends warn
/// and use Auto instead.
/// `ANYRENDER_SKIA_D3D_ALLOW_WARP=1` moves software D3D12 ahead of OpenGL.
/// Explicit OpenGL and transparent Windows surfaces never fall back to D3D12.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SkiaBackendPreference {
    #[default]
    Auto,
    #[cfg(windows)]
    D3D,
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    OpenGL,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Metal,
}

impl SkiaBackendPreference {
    fn from_env_value(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            #[cfg(windows)]
            "d3d" | "d3d12" => Some(Self::D3D),
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            "gl" | "opengl" => Some(Self::OpenGL),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }

    fn with_env_override(self) -> Self {
        match std::env::var("ANYRENDER_SKIA_BACKEND") {
            Ok(value) => Self::from_env_value(&value).unwrap_or_else(|| {
                tracing::warn!(
                    "Unknown or unavailable ANYRENDER_SKIA_BACKEND value {value:?}; using Auto"
                );
                Self::Auto
            }),
            Err(std::env::VarError::NotPresent) => self,
            Err(std::env::VarError::NotUnicode(_)) => {
                tracing::warn!("Invalid ANYRENDER_SKIA_BACKEND value; using Auto");
                Self::Auto
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SkiaBackendPreference, SkiaRendererOptions};

    #[test]
    fn backend_options() {
        assert_eq!(
            SkiaRendererOptions::new().backend,
            SkiaBackendPreference::Auto
        );
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let preference = SkiaBackendPreference::OpenGL;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let preference = SkiaBackendPreference::Metal;
        assert_eq!(
            SkiaRendererOptions::new().backend(preference).backend,
            preference
        );
    }

    #[test]
    fn backend_env_values() {
        for value in ["", "unknown"] {
            assert_eq!(SkiaBackendPreference::from_env_value(value), None);
        }
        #[cfg(windows)]
        for value in ["d3d", "D3D12"] {
            assert_eq!(
                SkiaBackendPreference::from_env_value(value),
                Some(SkiaBackendPreference::D3D)
            );
        }
        #[cfg(not(windows))]
        assert_eq!(SkiaBackendPreference::from_env_value("d3d"), None);
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            for value in ["gl", "GL", "OpenGL"] {
                assert_eq!(
                    SkiaBackendPreference::from_env_value(value),
                    Some(SkiaBackendPreference::OpenGL)
                );
            }
            assert_eq!(SkiaBackendPreference::from_env_value("metal"), None);
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            assert_eq!(
                SkiaBackendPreference::from_env_value("MeTaL"),
                Some(SkiaBackendPreference::Metal)
            );
            assert_eq!(SkiaBackendPreference::from_env_value("gl"), None);
            assert_eq!(SkiaBackendPreference::from_env_value("opengl"), None);
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_backend_fallback_order() {
        use super::{WindowsBackendAttempt::*, try_windows_backends};
        for (early_software, expected) in [
            (false, [HardwareD3D, OpenGL, SoftwareD3D]),
            (true, [HardwareD3D, SoftwareD3D, OpenGL]),
        ] {
            for success_index in 0..expected.len() {
                let mut attempted = Vec::new();
                let selected = try_windows_backends(early_software, |attempt| {
                    attempted.push(attempt);
                    if attempted.len() == success_index + 1 {
                        Ok(attempt)
                    } else {
                        Err(crate::BackendError::new("unavailable"))
                    }
                })
                .unwrap();
                assert_eq!(attempted, expected[..=success_index]);
                assert_eq!(selected, expected[success_index]);
            }
            let mut attempted = Vec::new();
            let error = try_windows_backends::<()>(early_software, |attempt| {
                attempted.push(attempt);
                Err(crate::BackendError::new("unavailable"))
            })
            .unwrap_err();
            assert_eq!(attempted, expected);
            assert_eq!(error.to_string().matches("unavailable").count(), 3);
        }
    }
}

#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowsBackendAttempt {
    HardwareD3D,
    SoftwareD3D,
    OpenGL,
}

#[cfg(windows)]
fn try_windows_backends<T>(
    software_before_gl: bool,
    mut create: impl FnMut(WindowsBackendAttempt) -> Result<T, BackendError>,
) -> Result<T, BackendError> {
    use WindowsBackendAttempt::*;
    let attempts = if software_before_gl {
        [HardwareD3D, SoftwareD3D, OpenGL]
    } else {
        [HardwareD3D, OpenGL, SoftwareD3D]
    };
    let mut errors = Vec::new();
    for attempt in attempts {
        let name = match attempt {
            HardwareD3D => "D3D12 hardware",
            SoftwareD3D => "D3D12 software (WARP/Basic Render Driver)",
            OpenGL => "OpenGL",
        };
        match create(attempt) {
            Ok(backend) => {
                log::info!("Selected Skia {name} backend");
                return Ok(backend);
            }
            Err(error) => {
                log::warn!("Skia {name} initialization failed: {error}");
                errors.push(format!("{name}: {error}"));
            }
        }
    }
    Err(BackendError::new(errors.join("; ")))
}

fn create_backend(
    preference: SkiaBackendPreference,
    window: Arc<dyn anyrender::WindowHandle>,
    width: u32,
    height: u32,
    composite_alpha_mode: anyrender::CompositeAlphaMode,
) -> Result<Box<dyn SkiaBackend>, BackendError> {
    #[cfg(windows)]
    let preference = match preference {
        SkiaBackendPreference::Auto | SkiaBackendPreference::D3D => {
            if matches!(
                composite_alpha_mode,
                anyrender::CompositeAlphaMode::Opaque | anyrender::CompositeAlphaMode::Auto
            ) {
                return try_windows_backends(
                    std::env::var("ANYRENDER_SKIA_D3D_ALLOW_WARP").as_deref() == Ok("1"),
                    |attempt| -> Result<Box<dyn SkiaBackend>, BackendError> {
                        match attempt {
                            WindowsBackendAttempt::HardwareD3D
                            | WindowsBackendAttempt::SoftwareD3D => {
                                Ok(Box::new(crate::d3d::D3DBackend::new(
                                    window.clone(),
                                    width,
                                    height,
                                    attempt == WindowsBackendAttempt::SoftwareD3D,
                                )?))
                            }
                            WindowsBackendAttempt::OpenGL => {
                                Ok(Box::new(crate::opengl::OpenGLBackend::new(
                                    window.clone(),
                                    width,
                                    height,
                                    composite_alpha_mode,
                                )?))
                            }
                        }
                    },
                );
            } else {
                log::info!("Skia D3D12 does not support per-pixel HWND alpha; using OpenGL");
            }
            SkiaBackendPreference::OpenGL
        }
        preference => preference,
    };
    match preference {
        #[cfg(windows)]
        SkiaBackendPreference::D3D => unreachable!(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        SkiaBackendPreference::Auto | SkiaBackendPreference::Metal => {
            let backend =
                crate::metal::MetalBackend::new(window, width, height, composite_alpha_mode)?;
            tracing::info!("Selected Skia Metal backend");
            Ok(Box::new(backend))
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        SkiaBackendPreference::Auto | SkiaBackendPreference::OpenGL => {
            let backend =
                crate::opengl::OpenGLBackend::new(window, width, height, composite_alpha_mode)
                    .inspect_err(|error| {
                        log::warn!("Skia OpenGL initialization failed: {error}")
                    })?;
            tracing::info!("Selected Skia OpenGL backend");
            Ok(Box::new(backend))
        }
    }
}

pub(crate) trait SkiaBackend: Any {
    fn is_lost(&self) -> bool {
        false
    }
    fn set_size(&mut self, width: u32, height: u32);

    fn prepare(&mut self) -> Option<Surface>;

    fn flush(&mut self, surface: Surface);
}

enum RenderState {
    Active(Box<ActiveRenderState>),
    Suspended,
}

struct ActiveRenderState {
    backend: Option<Box<dyn SkiaBackend>>,
    window: Arc<dyn anyrender::WindowHandle>,
    size: (u32, u32),
    scene_cache: SkiaSceneCache,
}

/// Options for configuring the Skia renderer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SkiaRendererOptions {
    /// Preferred GPU backend, overridden by `ANYRENDER_SKIA_BACKEND` when set.
    pub backend: SkiaBackendPreference,
    /// Background color used to clear the canvas.
    pub base_color: Color,
    /// Alpha mode used when compositing the window surface.
    pub composite_alpha_mode: anyrender::CompositeAlphaMode,
}

impl Default for SkiaRendererOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SkiaRendererOptions {
    pub const fn new() -> Self {
        Self {
            backend: SkiaBackendPreference::Auto,
            base_color: Color::WHITE,
            composite_alpha_mode: anyrender::CompositeAlphaMode::Auto,
        }
    }

    pub const fn base_color(self, base_color: Color) -> Self {
        Self { base_color, ..self }
    }

    pub const fn backend(self, backend: SkiaBackendPreference) -> Self {
        Self { backend, ..self }
    }

    pub const fn composite_alpha_mode(
        self,
        composite_alpha_mode: anyrender::CompositeAlphaMode,
    ) -> Self {
        Self {
            composite_alpha_mode,
            ..self
        }
    }
}

impl From<anyrender::RendererConfig> for SkiaRendererOptions {
    fn from(config: anyrender::RendererConfig) -> Self {
        let mut options = Self::default();
        if let Some(color) = config.base_color {
            let rgba8 = color.to_rgba8();
            options.base_color = skia_safe::Color::from_argb(rgba8.a, rgba8.r, rgba8.g, rgba8.b);
        }
        if let Some(mode) = config.composite_alpha_mode {
            options.composite_alpha_mode = mode;
        }
        options
    }
}

pub struct SkiaWindowRenderer {
    render_state: RenderState,
    options: SkiaRendererOptions,
}

impl Default for SkiaWindowRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl SkiaWindowRenderer {
    pub fn new() -> Self {
        Self {
            render_state: RenderState::Suspended,
            options: SkiaRendererOptions::default(),
        }
    }
    pub fn with_options(options: impl Into<SkiaRendererOptions>) -> Self {
        Self {
            render_state: RenderState::Suspended,
            options: options.into(),
        }
    }
}

impl SkiaWindowRenderer {}

impl RenderContext for SkiaWindowRenderer {}
impl WindowRenderer for SkiaWindowRenderer {
    type ScenePainter<'a>
        = SkiaScenePainter<'a>
    where
        Self: 'a;

    fn resume<F: FnOnce() + 'static>(
        &mut self,
        window: Arc<dyn anyrender::WindowHandle>,
        width: u32,
        height: u32,
        on_ready: F,
    ) {
        graphics::set_font_cache_count_limit(100);
        graphics::set_typeface_cache_count_limit(100);
        graphics::set_resource_cache_total_bytes_limit(10485760);

        let backend = create_backend(
            self.options.backend.with_env_override(),
            window.clone(),
            width,
            height,
            self.options.composite_alpha_mode,
        )
        .unwrap_or_else(|error| panic!("Failed to initialize Skia backend: {error}"));

        self.render_state = RenderState::Active(Box::new(ActiveRenderState {
            backend: Some(backend),
            window,
            size: (width, height),
            scene_cache: SkiaSceneCache::default(),
        }));
        on_ready();
    }

    fn complete_resume(&mut self) -> bool {
        true
    }

    fn suspend(&mut self) {
        self.render_state = RenderState::Suspended;
    }

    fn is_active(&self) -> bool {
        matches!(self.render_state, RenderState::Active(..))
    }

    fn set_size(&mut self, width: u32, height: u32) {
        if let RenderState::Active(state) = &mut self.render_state {
            state.size = (width, height);
            if let Some(backend) = &mut state.backend {
                backend.set_size(width, height);
            }
        }
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        if state.size.0 == 0 || state.size.1 == 0 {
            return;
        }
        if state
            .backend
            .as_ref()
            .is_some_and(|backend| backend.is_lost())
        {
            state.backend = None;
        }
        if state.backend.is_none() {
            state.backend = Some(
                create_backend(
                    self.options.backend.with_env_override(),
                    state.window.clone(),
                    state.size.0,
                    state.size.1,
                    self.options.composite_alpha_mode,
                )
                .unwrap_or_else(|error| panic!("Failed to reinitialize Skia backend: {error}")),
            );
        }
        let backend = state.backend.as_mut().unwrap();

        debug_timer!(timer, feature = "log_frame_times");

        let mut surface = match backend.prepare() {
            Some(it) => it,
            None => return,
        };

        surface.canvas().restore_to_count(1);
        surface.canvas().clear(self.options.base_color);

        draw_fn(&mut SkiaScenePainter {
            inner: surface.canvas(),
            cache: &mut state.scene_cache,
        });
        timer.record_time("cmd");

        backend.flush(surface);
        timer.record_time("render");

        state.scene_cache.next_gen();
        timer.record_time("cache next gen");

        timer.print_times("skia: ");
    }
}

#[cfg(any(
    feature = "pixels_window_renderer",
    feature = "softbuffer_window_renderer"
))]
pub mod raster {
    #[cfg(feature = "pixels_window_renderer")]
    pub use pixels_window_renderer::PixelsRendererOptions;
    #[cfg(feature = "pixels_window_renderer")]
    pub use pixels_window_renderer::PixelsWindowRenderer;

    #[cfg(feature = "softbuffer_window_renderer")]
    pub use softbuffer_window_renderer::{SoftbufferRendererOptions, SoftbufferWindowRenderer};

    #[cfg(feature = "pixels_window_renderer")]
    pub type SkiaRasterWindowRenderer =
        PixelsWindowRenderer<crate::image_renderer::SkiaImageRenderer>;
    #[cfg(feature = "pixels_window_renderer")]
    pub type SkiaRasterRendererOptions = PixelsRendererOptions;

    #[cfg(all(
        feature = "softbuffer_window_renderer",
        not(feature = "pixels_window_renderer")
    ))]
    pub type SkiaRasterWindowRenderer =
        SoftbufferWindowRenderer<crate::image_renderer::SkiaImageRenderer>;
    #[cfg(all(
        feature = "softbuffer_window_renderer",
        not(feature = "pixels_window_renderer")
    ))]
    pub type SkiaRasterRendererOptions = SoftbufferRendererOptions;
}
