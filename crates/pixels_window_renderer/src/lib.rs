//! An AnyRender WindowRenderer for rendering pixel buffers using the pixels crate

#![cfg_attr(docsrs, feature(doc_cfg))]

use anyrender::{
    ImageRenderer, RenderContext, WindowHandle, WindowRenderer, WindowRendererContext,
};
use debug_timer::debug_timer;
use pixels::{
    Pixels, PixelsBuilder, SurfaceTexture,
    wgpu::{Color, CompositeAlphaMode},
};
use std::marker::PhantomData;
use std::sync::Arc;

// Simple struct to hold the state of the renderer
pub struct ActiveRenderState {
    // surface: SurfaceTexture<Arc<dyn WindowHandle>>,
    pixels: Pixels<'static>,
}

#[allow(clippy::large_enum_variant)]
pub enum RenderState {
    Active(ActiveRenderState),
    Suspended,
}

/// Configuration options for the Pixels renderer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PixelsRendererOptions {
    /// Background color used to clear the frame.
    pub base_color: Color,
    /// Alpha mode used when compositing the window surface.
    pub composite_alpha_mode: anyrender::CompositeAlphaMode,
}

impl Default for PixelsRendererOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl PixelsRendererOptions {
    pub const fn new() -> Self {
        Self {
            base_color: Color::WHITE,
            composite_alpha_mode: anyrender::CompositeAlphaMode::Auto,
        }
    }

    pub const fn base_color(self, base_color: Color) -> Self {
        Self { base_color, ..self }
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

impl From<anyrender::RendererConfig> for PixelsRendererOptions {
    fn from(config: anyrender::RendererConfig) -> Self {
        let mut options = Self::default();
        if let Some(color) = config.base_color {
            let rgba8 = color.to_rgba8();
            options.base_color = pixels::wgpu::Color {
                r: rgba8.r as f64 / 255.0,
                g: rgba8.g as f64 / 255.0,
                b: rgba8.b as f64 / 255.0,
                a: rgba8.a as f64 / 255.0,
            };
        }
        options.composite_alpha_mode(config.composite_alpha_mode.unwrap_or_default())
    }
}

/// Creates [`PixelsWindowRenderer`]s that all use the same options. This backend has no state to share
/// between windows.
pub struct PixelsWindowRendererContext<Renderer: ImageRenderer> {
    options: PixelsRendererOptions,
    renderer: PhantomData<fn() -> Renderer>,
}

impl<Renderer: ImageRenderer> PixelsWindowRendererContext<Renderer> {
    pub fn new(options: impl Into<PixelsRendererOptions>) -> Self {
        Self {
            options: options.into(),
            renderer: PhantomData,
        }
    }
}

impl<Renderer: ImageRenderer> Default for PixelsWindowRendererContext<Renderer> {
    fn default() -> Self {
        Self::new(PixelsRendererOptions::default())
    }
}

impl<Renderer: ImageRenderer> Clone for PixelsWindowRendererContext<Renderer> {
    fn clone(&self) -> Self {
        Self::new(self.options.clone())
    }
}

impl<Renderer: ImageRenderer> WindowRendererContext for PixelsWindowRendererContext<Renderer> {
    type Renderer = PixelsWindowRenderer<Renderer>;

    fn create_renderer(&self) -> PixelsWindowRenderer<Renderer> {
        PixelsWindowRenderer::with_options(self.options.clone())
    }
}

pub struct PixelsWindowRenderer<Renderer: ImageRenderer> {
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,
    renderer: Renderer,
    config: PixelsRendererOptions,
}

impl<Renderer: ImageRenderer> PixelsWindowRenderer<Renderer> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_renderer(Renderer::new(0, 0))
    }

    pub fn with_renderer<R: ImageRenderer>(renderer: R) -> PixelsWindowRenderer<R> {
        PixelsWindowRenderer {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer,
            config: PixelsRendererOptions::default(),
        }
    }

    pub fn with_options(config: impl Into<PixelsRendererOptions>) -> Self {
        Self {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer: Renderer::new(0, 0),
            config: config.into(),
        }
    }

    pub fn with_options_and_renderer<R: ImageRenderer>(
        renderer: R,
        config: impl TryInto<PixelsRendererOptions, Error = impl std::error::Error>,
    ) -> PixelsWindowRenderer<R> {
        PixelsWindowRenderer {
            render_state: RenderState::Suspended,
            window_handle: None,
            renderer,
            config: config
                .try_into()
                .expect("Invalid Pixels renderer configuration"),
        }
    }
}

impl<Renderer: ImageRenderer> RenderContext for PixelsWindowRenderer<Renderer> {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn std::any::Any>,
    ) -> Result<anyrender::ResourceId, anyrender::RegisterResourceError> {
        self.renderer.try_register_custom_resource(resource)
    }

    fn unregister_resource(&mut self, resource_id: anyrender::ResourceId) {
        self.renderer.unregister_resource(resource_id);
    }
}
impl<Renderer: ImageRenderer> WindowRenderer for PixelsWindowRenderer<Renderer> {
    type ScenePainter<'a>
        = <Renderer as ImageRenderer>::ScenePainter<'a>
    where
        Renderer: 'a;

    fn is_active(&self) -> bool {
        matches!(self.render_state, RenderState::Active(_))
    }

    fn resume<F: FnOnce() + 'static>(
        &mut self,
        window_handle: Arc<dyn WindowHandle>,
        width: u32,
        height: u32,
        on_ready: F,
    ) {
        let composite_alpha_mode = match self.config.composite_alpha_mode {
            anyrender::CompositeAlphaMode::Auto => CompositeAlphaMode::Auto,
            anyrender::CompositeAlphaMode::Opaque => CompositeAlphaMode::Opaque,
            anyrender::CompositeAlphaMode::Transparent => {
                #[cfg(target_vendor = "apple")]
                {
                    // wgpu is lying in apple's case it uses PreMultiplied in reality
                    // (do not modify shaders for PostMultiplied)
                    CompositeAlphaMode::PostMultiplied
                }
                #[cfg(not(target_vendor = "apple"))]
                {
                    CompositeAlphaMode::PreMultiplied
                }
            }
        };
        let surface = SurfaceTexture::new(width, height, window_handle.clone());
        let pixels = PixelsBuilder::new(width, height, surface)
            .enable_vsync(true)
            .alpha_mode(composite_alpha_mode)
            .clear_color(self.config.base_color)
            .build()
            .unwrap();
        #[cfg(target_os = "macos")]
        tag_metal_layers_as_srgb(&*window_handle);
        self.render_state = RenderState::Active(ActiveRenderState { pixels });
        self.window_handle = Some(window_handle);

        self.set_size(width, height);
        on_ready();
    }

    fn complete_resume(&mut self) -> bool {
        true
    }

    fn suspend(&mut self) {
        self.render_state = RenderState::Suspended;
    }

    fn set_size(&mut self, physical_width: u32, physical_height: u32) {
        if let RenderState::Active(state) = &mut self.render_state {
            state
                .pixels
                .resize_buffer(physical_width, physical_height)
                .unwrap();
            state
                .pixels
                .resize_surface(physical_width, physical_height)
                .unwrap();
            self.renderer.resize(physical_width, physical_height);
        };
    }

    fn render<F: FnOnce(&mut Renderer::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        debug_timer!(timer, feature = "log_frame_times");

        // Paint
        self.renderer.render(draw_fn, state.pixels.frame_mut());
        timer.record_time("render");
        state.pixels.render().unwrap();
        timer.record_time("present");
        timer.print_times("pixels: ");

        // Reset the renderer ready for the next render
        self.renderer.reset();
    }
}

/// wgpu leaves the colorspace of the `CAMetalLayer` it adds to the window's view unset,
/// so macOS shows the sRGB values we render unconverted in the display's native space
/// (Display P3 on most Macs). `pixels` keeps its wgpu surface private, so find the layer
/// through the view.
#[cfg(target_os = "macos")]
fn tag_metal_layers_as_srgb(window: &dyn WindowHandle) {
    use objc2_app_kit::NSView;
    use objc2_core_graphics::{CGColorSpace, kCGColorSpaceSRGB};
    use objc2_quartz_core::CAMetalLayer;
    use raw_window_handle::RawWindowHandle;

    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
        return;
    };
    // SAFETY: the AppKit handle points to a live `NSView`.
    let view = unsafe { appkit.ns_view.cast::<NSView>().as_ref() };
    let Some(root) = view.layer() else {
        return;
    };
    let colorspace = CGColorSpace::with_name(Some(unsafe { kCGColorSpaceSRGB }));
    let mut layers = vec![root];
    while let Some(layer) = layers.pop() {
        if let Some(metal_layer) = layer.downcast_ref::<CAMetalLayer>() {
            metal_layer.setColorspace(colorspace.as_deref());
        }
        // SAFETY: the layer tree is only read, on the main thread that owns the view.
        if let Some(sublayers) = unsafe { layer.sublayers() } {
            layers.extend(sublayers.iter());
        }
    }
}
