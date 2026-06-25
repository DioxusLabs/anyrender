use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::{ffi::CString, num::NonZeroU32, sync::Arc};

use glutin::config::Config;
use glutin::display::DisplayApiPreference;
use glutin::{
    config::{ConfigTemplateBuilder, GetGlConfig, GlConfig},
    context::{ContextAttributesBuilder, PossiblyCurrentContext},
    display::{Display, GetGlDisplay},
    prelude::{GlDisplay, NotCurrentGlContext, PossiblyCurrentGlContext},
    surface::{GlSurface, SurfaceAttributesBuilder, WindowSurface},
};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use skia_safe::{
    Surface,
    gpu::{
        DirectContext, direct_contexts,
        gl::{FramebufferInfo, Interface},
    },
};

use crate::window_renderer::SkiaBackend;

fn build_gl_display(
    raw_display_handle: RawDisplayHandle,
    _raw_window: Option<RawWindowHandle>,
) -> Display {
    unsafe {
        Display::new(
            raw_display_handle,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            DisplayApiPreference::Cgl,
            #[cfg(target_os = "windows")]
            DisplayApiPreference::Wgl(_raw_window),
            #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "ios")))]
            DisplayApiPreference::Egl,
        )
        .unwrap()
    }
}

fn pick_gl_config(display: &Display, raw_window: Option<RawWindowHandle>) -> Option<Config> {
    let mut template = ConfigTemplateBuilder::new().with_transparency(true);
    if let Some(rwh) = raw_window {
        template = template.compatible_with_native_window(rwh);
    }
    let template = template.build();
    unsafe {
        display
            .find_configs(template)
            .ok()?
            .reduce(|accum, config| {
                let transparency_check = config.supports_transparency().unwrap_or(false)
                    & !accum.supports_transparency().unwrap_or(false);

                if transparency_check || config.num_samples() < accum.num_samples() {
                    config
                } else {
                    accum
                }
            })
    }
}

/// Cache of glutin `Display`s keyed by the raw X display/connection pointer, to avoid
/// double-initializing EGL for the same X server connection (once in `pick_x11_gl_visual`
/// and again in `OpenGLBackend::new`).
#[cfg(all(
    unix,
    not(any(target_os = "macos", target_os = "ios", target_os = "android"))
))]
fn display_cache() -> &'static Mutex<HashMap<usize, Display>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Display>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(all(
    unix,
    not(any(target_os = "macos", target_os = "ios", target_os = "android"))
))]
fn display_cache_key(raw_display_handle: RawDisplayHandle) -> Option<usize> {
    match raw_display_handle {
        RawDisplayHandle::Xlib(h) => h.display.map(|p| p.as_ptr() as usize),
        RawDisplayHandle::Xcb(h) => h.connection.map(|p| p.as_ptr() as usize),
        _ => None,
    }
}

#[cfg(all(
    unix,
    not(any(target_os = "macos", target_os = "ios", target_os = "android"))
))]
fn get_or_build_gl_display(
    raw_display_handle: RawDisplayHandle,
    raw_window: Option<RawWindowHandle>,
) -> Display {
    if let Some(key) = display_cache_key(raw_display_handle) {
        let mut cache = display_cache().lock().unwrap();
        if let Some(d) = cache.get(&key) {
            return d.clone();
        }
        let display = build_gl_display(raw_display_handle, raw_window);
        cache.insert(key, display.clone());
        return display;
    }
    build_gl_display(raw_display_handle, raw_window)
}

/// Pick an X11 visual ID compatible with the OpenGL backend used by [`SkiaWindowRenderer`].
///
/// Apply the returned visual to the winit window with [`WindowAttributesExtX11::with_x11_visual`]
/// before creating the window. This avoids `EGL_BAD_MATCH` at surface creation, which happens
/// when winit's default visual doesn't match any EGL config.
///
/// Returns `None` on non-Xlib/Xcb display handles, if no compatible EGL config can be
/// found, or if the platform's EGL implementation doesn't expose an X11 visual ID for the
/// picked config.
///
/// [`SkiaWindowRenderer`]: crate::SkiaWindowRenderer
/// [`WindowAttributesExtX11::with_x11_visual`]: https://docs.rs/winit/latest/winit/platform/x11/trait.WindowAttributesExtX11.html#tymethod.with_x11_visual
#[cfg(all(
    unix,
    not(any(target_os = "macos", target_os = "ios", target_os = "android"))
))]
pub fn pick_x11_gl_visual(raw_display_handle: RawDisplayHandle) -> Option<u32> {
    if !matches!(
        raw_display_handle,
        RawDisplayHandle::Xlib(_) | RawDisplayHandle::Xcb(_)
    ) {
        return None;
    }
    let display = get_or_build_gl_display(raw_display_handle, None);
    let config = pick_gl_config(&display, None)?;
    use glutin::platform::x11::X11GlConfigExt;
    config.x11_visual().map(|v| v.visual_id() as u32)
}

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
    ) -> OpenGLBackend {
        let raw_display_handle = window.display_handle().unwrap().as_raw();
        let raw_window_handle = window.window_handle().unwrap().as_raw();

        #[cfg(all(
            unix,
            not(any(target_os = "macos", target_os = "ios", target_os = "android"))
        ))]
        let gl_display = get_or_build_gl_display(raw_display_handle, Some(raw_window_handle));
        #[cfg(not(all(
            unix,
            not(any(target_os = "macos", target_os = "ios", target_os = "android"))
        )))]
        let gl_display = build_gl_display(raw_display_handle, Some(raw_window_handle));

        let gl_config = pick_gl_config(&gl_display, Some(raw_window_handle))
            .expect("no GL config found for display");

        let gl_context_attrs = ContextAttributesBuilder::new().build(Some(raw_window_handle));
        let gl_surface_attrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(
            raw_window_handle,
            NonZeroU32::new(width).expect("width should be a positive value"),
            NonZeroU32::new(height).expect("height should be a positive value"),
        );

        let gl_not_current_context = unsafe {
            gl_display
                .create_context(&gl_config, &gl_context_attrs)
                .unwrap()
        };

        let gl_surface = unsafe {
            gl_config
                .display()
                .create_window_surface(&gl_config, &gl_surface_attrs)
                .unwrap()
        };

        let gl_context = gl_not_current_context.make_current(&gl_surface).unwrap();

        gl::load_with(|s| {
            gl_config
                .display()
                .get_proc_address(CString::new(s).unwrap().as_c_str())
        });

        let interface = Interface::new_load_with(|name| {
            if name == "eglGetCurrentDisplay" {
                return std::ptr::null();
            }
            gl_config
                .display()
                .get_proc_address(CString::new(name).unwrap().as_c_str())
        })
        .unwrap();

        let mut gr_context = direct_contexts::make_gl(interface, None).unwrap();

        let mut fb_info = {
            let mut fboid: gl::types::GLint = 0;
            unsafe {
                gl::GetIntegerv(gl::FRAMEBUFFER_BINDING, &mut fboid);
            }

            FramebufferInfo {
                fboid: fboid.try_into().unwrap(),
                format: skia_safe::gpu::gl::Format::RGBA8.into(),
                ..Default::default()
            }
        };

        OpenGLBackend {
            surface: Some(Self::create_surface(
                width,
                height,
                &mut gr_context,
                &gl_surface,
                &gl_context,
                &mut fb_info,
            )),
            gr_context,
            gl_surface,
            gl_context,
            fb_info,
        }
    }

    fn create_surface(
        width: u32,
        height: u32,
        gr_context: &mut DirectContext,
        gl_surface: &glutin::surface::Surface<WindowSurface>,
        gl_context: &PossiblyCurrentContext,
        fb_info: &mut FramebufferInfo,
    ) -> Surface {
        gl_surface.resize(
            gl_context,
            NonZeroU32::new(width).unwrap(),
            NonZeroU32::new(height).unwrap(),
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
        .unwrap()
    }
}

impl SkiaBackend for OpenGLBackend {
    fn set_size(&mut self, width: u32, height: u32) {
        self.surface = Some(Self::create_surface(
            width,
            height,
            &mut self.gr_context,
            &self.gl_surface,
            &self.gl_context,
            &mut self.fb_info,
        ));
    }

    fn prepare(&mut self) -> Option<Surface> {
        self.gl_context.make_current(&self.gl_surface).unwrap();
        self.surface.take()
    }

    fn flush(&mut self, mut surface: Surface) {
        self.gr_context.flush_and_submit();
        self.gl_surface.swap_buffers(&self.gl_context).unwrap();
        surface.canvas().discard();

        self.surface = Some(surface);
    }
}
