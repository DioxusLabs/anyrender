use anyrender::{Backdrop, ImageRenderer, RenderContext, ResourceId};
use peniko::ImageData;
use rustc_hash::FxHashMap;
use vello::{Renderer as VelloRenderer, RendererOptions, Scene as VelloScene};
use wgpu::TextureUsages;
use wgpu_context::{BufferRenderer, BufferRendererConfig, WGPUContext};

use crate::{DEFAULT_THREADS, VelloScenePainter};

pub struct VelloImageRenderer {
    buffer_renderer: BufferRenderer,
    vello_renderer: VelloRenderer,
    scene: VelloScene,
    texture_handles: FxHashMap<ResourceId, ImageData>,
}

impl VelloImageRenderer {
    /// Like [`ImageRenderer::new`], but reusing compiled pipelines from a cache
    /// wgpu built earlier. Creating the cache, persisting its data and deciding
    /// when it is stale are all the caller's responsibility.
    pub fn with_pipeline_cache(
        width: u32,
        height: u32,
        pipeline_cache: Option<wgpu::PipelineCache>,
    ) -> Self {
        // Create WGPUContext
        let mut context = WGPUContext::new();

        // Create wgpu_context::BufferRenderer
        let buffer_renderer =
            pollster::block_on(context.create_buffer_renderer(BufferRendererConfig {
                width,
                height,
                usage: TextureUsages::STORAGE_BINDING,
            }))
            .expect("No compatible device found");

        // Create vello::Renderer
        let vello_renderer = VelloRenderer::new(
            buffer_renderer.device(),
            RendererOptions {
                use_cpu: false,
                num_init_threads: DEFAULT_THREADS,
                antialiasing_support: vello::AaSupport::area_only(),
                pipeline_cache,
            },
        )
        .expect("Got non-Send/Sync error from creating renderer");

        Self {
            buffer_renderer,
            vello_renderer,
            scene: VelloScene::new(),
            texture_handles: FxHashMap::default(),
        }
    }
}

impl RenderContext for VelloImageRenderer {}
impl ImageRenderer for VelloImageRenderer {
    type ScenePainter<'a>
        = VelloScenePainter<'a, 'a>
    where
        Self: 'a;

    fn new(width: u32, height: u32) -> Self {
        Self::with_pipeline_cache(width, height, None)
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.buffer_renderer.resize(width, height);
    }

    fn reset(&mut self) {
        self.scene.reset();
    }

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        backdrop: Backdrop,
        draw_fn: F,
        cpu_buffer: &mut Vec<u8>,
    ) {
        let size = self.buffer_renderer.size();
        cpu_buffer.resize((size.width * size.height * 4) as usize, 0);
        self.render(backdrop, draw_fn, cpu_buffer);
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        backdrop: Backdrop,
        draw_fn: F,
        cpu_buffer: &mut [u8],
    ) {
        draw_fn(&mut VelloScenePainter {
            inner: &mut self.scene,
            renderer: Some(&mut self.vello_renderer),
            device_handle: Some(&self.buffer_renderer.device_handle),
            texture_handles: Some(&mut self.texture_handles),
        });
        let texture_view = self.buffer_renderer.target_texture_view();
        let backdrop_image = if matches!(backdrop, Backdrop::Preserve) {
            let mut scene = VelloScene::new();
            let image_data = self
                .vello_renderer
                .register_texture(texture_view.texture().clone());
            let image = peniko::ImageBrush::new(image_data.clone());
            scene.draw_image(&image, kurbo::Affine::IDENTITY);
            std::mem::swap(&mut self.scene, &mut scene);
            self.scene.append(&scene, None);
            Some(image_data)
        } else {
            None
        };

        for handle in self.texture_handles.values() {
            self.vello_renderer.mark_override_image_dirty(handle);
        }

        let size = self.buffer_renderer.size();
        self.vello_renderer
            .render_to_texture(
                self.buffer_renderer.device(),
                self.buffer_renderer.queue(),
                &self.scene,
                &texture_view,
                &vello::RenderParams {
                    base_color: match backdrop {
                        Backdrop::Preserve => vello::peniko::Color::TRANSPARENT,
                        Backdrop::Clear(color) => color,
                    },
                    width: size.width,
                    height: size.height,
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .expect("Got non-Send/Sync error from rendering");

        if let Some(image) = backdrop_image {
            self.vello_renderer.unregister_texture(image);
        }

        self.buffer_renderer.copy_texture_to_buffer(cpu_buffer);

        // Empty the Vello scene (memory optimisation)
        self.scene.reset();
    }
}
