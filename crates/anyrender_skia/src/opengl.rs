use std::{ffi::CString, num::NonZeroU32, sync::Arc};

use glutin::display::DisplayApiPreference;
use glutin::{
    config::{ConfigTemplateBuilder, GetGlConfig, GlConfig},
    context::{ContextAttributesBuilder, PossiblyCurrentContext},
    display::{Display, GetGlDisplay},
    prelude::{GlDisplay, NotCurrentGlContext, PossiblyCurrentGlContext},
    surface::{GlSurface, SurfaceAttributesBuilder, WindowSurface},
};
use skia_safe::{
    Surface,
    gpu::{
        DirectContext, direct_contexts,
        gl::{FramebufferInfo, Interface},
    },
};

use crate::{BackendError, window_renderer::SkiaBackend};

pub(crate) struct OpenGLBackend {
    surface: Option<Surface>,
    gr_context: DirectContext,
    gl_surface: glutin::surface::Surface<WindowSurface>,
    gl_context: PossiblyCurrentContext,
    fb_info: FramebufferInfo,
}

impl OpenGLBackend {
    pub(crate) fn new(
        window: Arc<dyn anyrender::WindowHandle>,
        width: u32,
        height: u32,
        composite_alpha_mode: anyrender::CompositeAlphaMode,
    ) -> Result<Self, BackendError> {
        let raw_display_handle = window.display_handle().map_err(BackendError::new)?.as_raw();
        let raw_window_handle = window.window_handle().map_err(BackendError::new)?.as_raw();

        let gl_display = unsafe {
            Display::new(
                raw_display_handle,
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                DisplayApiPreference::Cgl,
                #[cfg(target_os = "windows")]
                DisplayApiPreference::Wgl(Some(raw_window_handle)),
                #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "ios")))]
                DisplayApiPreference::Egl,
            )
            .map_err(BackendError::new)?
        };

        let transparency = !matches!(
            composite_alpha_mode,
            anyrender::CompositeAlphaMode::Opaque | anyrender::CompositeAlphaMode::Auto
        );
        let mut template = ConfigTemplateBuilder::new().with_transparency(transparency);
        if transparency {
            template = template.with_alpha_size(8);
        } else {
            template = template.with_alpha_size(0);
        }
        let gl_config_template = template.build();

        let gl_config = unsafe {
            gl_display
                .find_configs(gl_config_template)
                .map_err(BackendError::new)?
                .reduce(|accum, config| {
                    let config_transparency = config.supports_transparency().unwrap_or(false);
                    let accum_transparency = accum.supports_transparency().unwrap_or(false);

                    if config_transparency == transparency && accum_transparency != transparency {
                        config
                    } else if accum_transparency == transparency
                        && config_transparency != transparency
                    {
                        accum
                    } else {
                        let config_alpha = config.alpha_size();
                        let accum_alpha = accum.alpha_size();
                        if config_alpha != accum_alpha {
                            let prefer_config = if transparency {
                                config_alpha > accum_alpha
                            } else {
                                config_alpha < accum_alpha
                            };
                            if prefer_config { config } else { accum }
                        } else if config.num_samples() < accum.num_samples() {
                            config
                        } else {
                            accum
                        }
                    }
                })
                .ok_or_else(|| BackendError::new("No suitable OpenGL configuration"))?
        };

        let gl_context_attrs = ContextAttributesBuilder::new().build(Some(raw_window_handle));
        let gl_surface_attrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            raw_window_handle,
            NonZeroU32::new(width).ok_or_else(|| BackendError::new("Width must be positive"))?,
            NonZeroU32::new(height).ok_or_else(|| BackendError::new("Height must be positive"))?,
        );

        let gl_not_current_context = unsafe {
            gl_display
                .create_context(&gl_config, &gl_context_attrs)
                .map_err(BackendError::new)?
        };

        let gl_surface = unsafe {
            gl_config
                .display()
                .create_window_surface(&gl_config, &gl_surface_attrs)
                .map_err(BackendError::new)?
        };

        let gl_context = gl_not_current_context
            .make_current(&gl_surface)
            .map_err(BackendError::new)?;

        gl::load_with(|s| {
            CString::new(s).map_or(std::ptr::null(), |s| {
                gl_config.display().get_proc_address(&s)
            })
        });

        let interface = Interface::new_load_with(|name| {
            if name == "eglGetCurrentDisplay" {
                return std::ptr::null();
            }
            CString::new(name).map_or(std::ptr::null(), |name| {
                gl_config.display().get_proc_address(&name)
            })
        })
        .ok_or_else(|| BackendError::new("Failed to load OpenGL interface"))?;

        let mut gr_context = direct_contexts::make_gl(interface, None)
            .ok_or_else(|| BackendError::new("Failed to create Skia OpenGL context"))?;

        let mut fb_info = {
            let mut fboid: gl::types::GLint = 0;
            unsafe {
                gl::GetIntegerv(gl::FRAMEBUFFER_BINDING, &mut fboid);
            }

            FramebufferInfo {
                fboid: fboid.try_into().map_err(BackendError::new)?,
                format: skia_safe::gpu::gl::Format::RGBA8.into(),
                ..Default::default()
            }
        };

        Ok(Self {
            surface: Some(Self::create_surface(
                width,
                height,
                &mut gr_context,
                &gl_surface,
                &gl_context,
                &mut fb_info,
            )?),
            gr_context,
            gl_surface,
            gl_context,
            fb_info,
        })
    }

    fn create_surface(
        width: u32,
        height: u32,
        gr_context: &mut DirectContext,
        gl_surface: &glutin::surface::Surface<WindowSurface>,
        gl_context: &PossiblyCurrentContext,
        fb_info: &mut FramebufferInfo,
    ) -> Result<Surface, BackendError> {
        gl_surface.resize(
            gl_context,
            NonZeroU32::new(width).ok_or_else(|| BackendError::new("Width must be positive"))?,
            NonZeroU32::new(height).ok_or_else(|| BackendError::new("Height must be positive"))?,
        );

        let backend_render_target = skia_safe::gpu::backend_render_targets::make_gl(
            (width as i32, height as i32),
            gl_context.config().num_samples() as usize,
            gl_context.config().stencil_size() as usize,
            *fb_info,
        );

        skia_safe::gpu::surfaces::wrap_backend_render_target(
            gr_context,
            &backend_render_target,
            skia_safe::gpu::SurfaceOrigin::BottomLeft,
            skia_safe::ColorType::RGBA8888,
            None,
            None,
        )
        .ok_or_else(|| BackendError::new("Failed to wrap OpenGL render target"))
    }
}

impl SkiaBackend for OpenGLBackend {
    fn set_size(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface = Some(
            Self::create_surface(
                width,
                height,
                &mut self.gr_context,
                &self.gl_surface,
                &self.gl_context,
                &mut self.fb_info,
            )
            .expect("Failed to resize OpenGL surface"),
        );
    }

    fn prepare(&mut self) -> Option<Surface> {
        self.gl_context.make_current(&self.gl_surface).unwrap();
        self.surface.take()
    }

    fn flush(&mut self, mut surface: Surface) {
        self.gr_context
            .flush_and_submit()
            .expect("Skia GPU flush failed");
        self.gl_surface.swap_buffers(&self.gl_context).unwrap();
        surface.canvas().discard();

        self.surface = Some(surface);
    }
}
