use anyrender::{ImageRenderer, ImageResource, RenderContext, ResourceId};
use peniko::ImageData;
use rustc_hash::FxHashMap;
use vello::{Renderer as VelloRenderer, RendererOptions, Scene as VelloScene};
use wgpu::TextureUsages;
use wgpu_context::{BufferRenderer, BufferRendererConfig, WGPUContext};

use crate::{DEFAULT_THREADS, VelloRenderContext, VelloScenePainter};

pub struct VelloImageRenderer {
    ctx: VelloRenderContext,
    buffer_renderer: BufferRenderer,
    vello_renderer: VelloRenderer,
    scene: VelloScene,
}

impl RenderContext for VelloImageRenderer {
    fn register_image(&mut self, image: ImageData) -> ImageResource {
        self.ctx.register_image(image)
    }

    fn unregister_resource(&mut self, id: ResourceId) {
        self.ctx.unregister_resource(id)
    }
}

impl ImageRenderer for VelloImageRenderer {
    type ScenePainter<'a>
        = VelloScenePainter<'a, 'a>
    where
        Self: 'a;

    fn new(width: u32, height: u32) -> Self {
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
                pipeline_cache: None,
            },
        )
        .expect("Got non-Send/Sync error from creating renderer");

        Self {
            ctx: VelloRenderContext::new(),
            buffer_renderer,
            vello_renderer,
            scene: VelloScene::new(),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.buffer_renderer.resize(width, height);
    }

    fn reset(&mut self) {
        self.scene.reset();
    }

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        cpu_buffer: &mut Vec<u8>,
    ) {
        let size = self.buffer_renderer.size();
        cpu_buffer.resize((size.width * size.height * 4) as usize, 0);
        self.render(draw_fn, cpu_buffer);
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        cpu_buffer: &mut [u8],
    ) {
        draw_fn(&mut VelloScenePainter {
            ctx: &self.ctx,
            inner: &mut self.scene,
            renderer: Some(&mut self.vello_renderer),
            custom_paint_sources: Some(&mut FxHashMap::default()),
        });

        let size = self.buffer_renderer.size();
        self.vello_renderer
            .render_to_texture(
                self.buffer_renderer.device(),
                self.buffer_renderer.queue(),
                &self.scene,
                &self.buffer_renderer.target_texture_view(),
                &vello::RenderParams {
                    base_color: vello::peniko::Color::TRANSPARENT,
                    width: size.width,
                    height: size.height,
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .expect("Got non-Send/Sync error from rendering");

        self.buffer_renderer.copy_texture_to_buffer(cpu_buffer);

        // Empty the Vello scene (memory optimisation)
        self.scene.reset();
    }
}
