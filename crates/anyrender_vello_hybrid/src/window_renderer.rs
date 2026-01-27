use anyrender::{WindowHandle, WindowRenderer};
use debug_timer::debug_timer;
use rustc_hash::FxHashMap;
use std::sync::{
    Arc,
    // atomic::{AtomicU64},
};
use vello_common::paint::ImageId;
use vello_hybrid::{
    RenderSettings, RenderSize, RenderTargetConfig, Renderer as VelloHybridRenderer,
    Scene as VelloHybridScene,
};
use wgpu::{
    CommandEncoderDescriptor, Features, Limits, PresentMode, TextureFormat, TextureViewDescriptor,
};
use wgpu_context::{DeviceHandle, SurfaceRenderer, SurfaceRendererConfiguration, WGPUContext};

use crate::{VelloHybridScenePainter, scene::ImageManager};
// use crate::CustomPaintSource;

// static PAINT_SOURCE_ID: AtomicU64 = AtomicU64::new(0);

// Simple struct to hold the state of the renderer
struct ActiveRenderState {
    renderer: VelloHybridRenderer,
    render_surface: SurfaceRenderer<'static>,
}

#[allow(clippy::large_enum_variant)]
enum RenderState {
    Active(ActiveRenderState),
    Suspended,
}

impl RenderState {
    fn current_device_handle(&self) -> Option<&DeviceHandle> {
        let RenderState::Active(state) = self else {
            return None;
        };
        Some(&state.render_surface.device_handle)
    }
}

#[derive(Clone, Default)]
pub struct VelloHybridRendererOptions {
    pub features: Option<Features>,
    pub limits: Option<Limits>,
    pub render_settings: RenderSettings,
}

pub struct VelloHybridWindowRenderer {
    // The fields MUST be in this order, so that the surface is dropped before the window
    // Window is cached even when suspended so that it can be reused when the app is resumed after being suspended
    render_state: RenderState,
    window_handle: Option<Arc<dyn WindowHandle>>,

    // Vello
    wgpu_context: WGPUContext,
    scene: VelloHybridScene,
    config: VelloHybridRendererOptions,
    // custom_paint_sources: FxHashMap<u64, Box<dyn CustomPaintSource>>,
    cached_images: FxHashMap<u64, ImageId>,
}
impl VelloHybridWindowRenderer {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self::with_options(VelloHybridRendererOptions::default())
    }

    pub fn with_options(config: VelloHybridRendererOptions) -> Self {
        let features = config.features.unwrap_or_default()
            | Features::CLEAR_TEXTURE
            | Features::PIPELINE_CACHE;
        let render_settings = config.render_settings;
        Self {
            wgpu_context: WGPUContext::with_features_and_limits(
                Some(features),
                config.limits.clone(),
            ),
            config,
            render_state: RenderState::Suspended,
            window_handle: None,
            scene: VelloHybridScene::new_with(0, 0, render_settings),
            // custom_paint_sources: FxHashMap::default(),
            cached_images: FxHashMap::default(),
        }
    }

    pub fn current_device_handle(&self) -> Option<&DeviceHandle> {
        self.render_state.current_device_handle()
    }

    // pub fn register_custom_paint_source(&mut self, mut source: Box<dyn CustomPaintSource>) -> u64 {
    //     if let Some(device_handle) = self.render_state.current_device_handle() {
    //         source.resume(device_handle);
    //     }
    //     let id = PAINT_SOURCE_ID.fetch_add(1, atomic::Ordering::SeqCst);
    //     self.custom_paint_sources.insert(id, source);

    //     id
    // }

    // pub fn unregister_custom_paint_source(&mut self, id: u64) {
    //     if let Some(mut source) = self.custom_paint_sources.remove(&id) {
    //         source.suspend();
    //         drop(source);
    //     }
    // }
}

impl WindowRenderer for VelloHybridWindowRenderer {
    type ScenePainter<'a>
        = VelloHybridScenePainter<'a>
    where
        Self: 'a;

    fn is_active(&self) -> bool {
        matches!(self.render_state, RenderState::Active(_))
    }

    fn resume(&mut self, window_handle: Arc<dyn WindowHandle>, width: u32, height: u32) {
        // Create wgpu_context::SurfaceRenderer
        let render_surface = pollster::block_on(self.wgpu_context.create_surface(
            window_handle.clone(),
            SurfaceRendererConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                formats: vec![
                    /*TextureFormat::Rgba8Unorm, */ TextureFormat::Bgra8Unorm,
                ],
                width,
                height,
                present_mode: PresentMode::AutoVsync,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
            },
            None,
            // Some(TextureConfiguration {
            //     usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            // }),
        ))
        .expect("Error creating surface");

        // Create vello::Renderer
        let renderer = VelloHybridRenderer::new(
            render_surface.device(),
            &RenderTargetConfig {
                // TODO: Make configurable?
                format: TextureFormat::Bgra8Unorm,
                width,
                height,
            },
        );

        // Resume custom paint sources
        // let device_handle = &render_surface.device_handle;
        // for source in self.custom_paint_sources.values_mut() {
        //     source.resume(device_handle)
        // }

        // Set state to Active
        self.window_handle = Some(window_handle);
        self.render_state = RenderState::Active(ActiveRenderState {
            renderer,
            render_surface,
        });
    }

    fn suspend(&mut self) {
        // Suspend custom paint sources
        // for source in self.custom_paint_sources.values_mut() {
        //     source.suspend()
        // }

        // Set state to Suspended
        self.render_state = RenderState::Suspended;
    }

    fn set_size(&mut self, width: u32, height: u32) {
        if width as u16 != self.scene.width() || height as u16 != self.scene.height() {
            self.scene = VelloHybridScene::new_with(
                width as u16,
                height as u16,
                self.config.render_settings,
            );
            if let RenderState::Active(state) = &mut self.render_state {
                state.render_surface.resize(width, height);
            };
        }
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F) {
        let RenderState::Active(state) = &mut self.render_state else {
            return;
        };

        let render_surface = &state.render_surface;

        debug_timer!(timer, feature = "log_frame_times");

        let mut encoder =
            render_surface
                .device()
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Render scene"),
                });

        let image_manager = ImageManager {
            renderer: &mut state.renderer,
            device: render_surface.device(),
            queue: render_surface.queue(),
            encoder: &mut encoder,
            cache: &mut self.cached_images,
        };

        // Regenerate the vello scene
        draw_fn(&mut VelloHybridScenePainter {
            scene: &mut self.scene,
            image_manager: Some(image_manager),
        });
        timer.record_time("cmd");

        let surface_texture = render_surface.current_surface_texture();
        let texture_view = surface_texture
            .texture
            .create_view(&TextureViewDescriptor::default());

        state
            .renderer
            .render(
                &self.scene,
                render_surface.device(),
                render_surface.queue(),
                &mut encoder,
                &RenderSize {
                    width: render_surface.config.width,
                    height: render_surface.config.height,
                },
                &texture_view,
            )
            .expect("failed to render to texture");
        render_surface.queue().submit([encoder.finish()]);
        timer.record_time("render");

        surface_texture.present();
        timer.record_time("present");

        render_surface
            .device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        timer.record_time("wait");
        timer.print_times("vello_hybrid: ");

        // static COUNTER: AtomicU64 = AtomicU64::new(0);
        // println!("FRAME {}", COUNTER.fetch_add(1, atomic::Ordering::Relaxed));

        // Empty the Vello scene (memory optimisation)
        self.scene.reset();
    }
}
