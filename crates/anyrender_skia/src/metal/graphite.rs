use std::ffi::c_void;
use std::sync::Arc;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_core_foundation::CGSize;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLDrawable,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use skia_safe::{
    ColorType, Surface,
    gpu::graphite::{self, InsertRecordingInfo, mtl},
};

use crate::window_renderer::SkiaBackend;

pub struct MetalGraphiteBackend {
    metal_layer: Retained<CAMetalLayer>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    // Field order matters: the recorder must be dropped before the context.
    recorder: graphite::Recorder,
    context: graphite::Context,
    _backend_context: mtl::BackendContext,
    prepared_drawable: Option<Retained<ProtocolObject<dyn MTLDrawable>>>,
}

impl MetalGraphiteBackend {
    pub fn new(
        window: Arc<dyn anyrender::WindowHandle>,
        width: u32,
        height: u32,
        composite_alpha_mode: anyrender::CompositeAlphaMode,
    ) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("no device found");

        let metal_layer =
            super::create_metal_layer(&device, window, width, height, composite_alpha_mode);

        let command_queue = device
            .newCommandQueue()
            .expect("unable to get command queue");

        let backend_context = unsafe {
            mtl::BackendContext::new(
                Retained::as_ptr(&device) as mtl::Handle,
                Retained::as_ptr(&command_queue) as mtl::Handle,
            )
        };

        let mut context = mtl::context_factory::make_metal(&backend_context, None)
            .expect("unable to create Graphite context");
        let recorder = context
            .make_recorder(None)
            .expect("unable to create Graphite recorder");

        Self {
            metal_layer,
            command_queue,
            recorder,
            context,
            _backend_context: backend_context,
            prepared_drawable: None,
        }
    }
}

impl SkiaBackend for MetalGraphiteBackend {
    fn set_size(&mut self, width: u32, height: u32) {
        self.metal_layer
            .setDrawableSize(CGSize::new(width as f64, height as f64));
    }

    fn prepare(&mut self) -> Option<Surface> {
        let drawable = self.metal_layer.nextDrawable()?;

        let size = self.metal_layer.drawableSize();
        let texture = drawable.texture();
        // SAFETY: the texture is owned by `drawable`, which is kept alive in
        // `prepared_drawable` until the frame has been submitted in `flush`.
        let backend_texture = unsafe {
            mtl::backend_textures::make_metal(
                (size.width as i32, size.height as i32),
                Retained::as_ptr(&texture) as *mut c_void,
            )
        };

        let surface = graphite::surfaces::wrap_backend_texture(
            &mut self.recorder,
            &backend_texture,
            ColorType::BGRA8888,
            None,
            None,
        )?;

        self.prepared_drawable = Some((&drawable).into());

        Some(surface)
    }

    fn flush(&mut self, surface: Surface) {
        let recording = self.recorder.snap();
        drop(surface);

        if let Some(mut recording) = recording {
            self.context
                .insert_recording(&InsertRecordingInfo::new(&mut recording));
        }
        self.context.submit(None);

        let drawable = self.prepared_drawable.take().unwrap();
        let command_buffer = self
            .command_queue
            .commandBuffer()
            .expect("unable to get command buffer");
        command_buffer.presentDrawable(&drawable);
        command_buffer.commit();
    }

    fn recorder(&mut self) -> Option<&mut graphite::Recorder> {
        Some(&mut self.recorder)
    }
}
