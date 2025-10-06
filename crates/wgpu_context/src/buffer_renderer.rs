use crate::{DeviceHandle, block_on_wgpu, util::create_texture};
use wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, Extent3d, Queue,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TextureFormat, TextureUsages, TextureView,
};

#[derive(Clone, Debug)]
pub struct BufferRendererConfig {
    pub width: u32,
    pub height: u32,
    pub usage: TextureUsages,
}

/// Utility struct for rendering to `Vec<u8>`
pub struct BufferRenderer {
    // The device and queue for rendering to the surface
    pub dev_id: usize,
    pub device_handle: DeviceHandle,

    config: BufferRendererConfig,
    texture_view: wgpu::TextureView,
    gpu_buffer: wgpu::Buffer,
}

impl std::fmt::Debug for BufferRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceRenderer")
            .field("dev_id", &self.dev_id)
            .field("config", &self.config)
            .finish()
    }
}

impl BufferRenderer {
    /// Creates a new render surface for the specified window and dimensions.
    pub fn new(config: BufferRendererConfig, device_handle: DeviceHandle, dev_id: usize) -> Self {
        let texture_view = create_texture(
            config.width,
            config.height,
            TextureFormat::Rgba8Unorm,
            config.usage | TextureUsages::COPY_SRC,
            &device_handle.device,
        );

        let padded_byte_width = (config.width * 4).next_multiple_of(256);
        let buffer_size = padded_byte_width as u64 * config.height as u64;
        let gpu_buffer = device_handle.device.create_buffer(&BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            dev_id,
            device_handle,
            config,
            texture_view,
            gpu_buffer,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device_handle.device
    }

    pub fn queue(&self) -> &Queue {
        &self.device_handle.queue
    }

    pub fn size(&self) -> Extent3d {
        Extent3d {
            width: self.config.width,
            height: self.config.height,
            depth_or_array_layers: 1,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.texture_view = create_texture(
            self.config.width,
            self.config.height,
            TextureFormat::Rgba8Unorm,
            self.config.usage | TextureUsages::COPY_SRC,
            &self.device_handle.device,
        );

        let padded_byte_width = (width * 4).next_multiple_of(256);
        let buffer_size = padded_byte_width as u64 * height as u64;
        self.gpu_buffer = self.device_handle.device.create_buffer(&BufferDescriptor {
            label: None,
            size: buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }

    // /// Resizes the surface to the new dimensions.
    // pub fn resize(&mut self, width: u32, height: u32) {
    //     // TODO: Use clever resize semantics to avoid thrashing the memory allocator during a resize
    //     // especially important on metal.
    //     if let Some(intermediate_texture_stuff) = &mut self.intermediate_texture {
    //         intermediate_texture_stuff.texture_view = create_intermediate_texture(
    //             width,
    //             height,
    //             intermediate_texture_stuff.config.usage,
    //             &self.device_handle.device,
    //         );
    //     }
    //     self.config.width = width;
    //     self.config.height = height;
    //     self.configure();
    // }

    pub fn target_texture_view(&self) -> TextureView {
        self.texture_view.clone()
    }

    pub fn copy_texture_to_vec(&self, cpu_buffer: &mut Vec<u8>) {
        cpu_buffer.clear();
        cpu_buffer.reserve((self.config.width * self.config.height * 4) as usize);
        self.copy_texture_to_buffer(&mut *cpu_buffer);
    }

    pub fn copy_texture_to_buffer(&self, cpu_buffer: &mut [u8]) {
        let mut encoder = self
            .device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Copy out buffer"),
            });
        let row_byte_width = self.config.width as usize * 4;
        let padded_row_byte_width = row_byte_width.next_multiple_of(256);

        let texture = self.texture_view.texture();
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &self.gpu_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_byte_width as u32),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );

        self.queue().submit([encoder.finish()]);
        let buf_slice = self.gpu_buffer.slice(..);

        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        if let Ok(recv_result) =
            block_on_wgpu(self.device(), receiver.receive()).inspect_err(|err| {
                panic!("channel inaccessible: {:#}", err);
            })
        {
            let _ = recv_result.unwrap();
        }

        let data = buf_slice.get_mapped_range();

        // Pad result
        for row in 0..(self.config.height as usize) {
            let src_start = row * padded_row_byte_width;
            let src = &data[src_start..(src_start + row_byte_width)];

            let dest_start = row * row_byte_width;
            cpu_buffer[dest_start..(dest_start + row_byte_width)].clone_from_slice(src);
        }

        // Unmap buffer
        drop(data);
        self.gpu_buffer.unmap();
    }
}
