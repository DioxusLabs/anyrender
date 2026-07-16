//! A blitter that converts between premultiplied and straight
//! (non-premultiplied) alpha while copying one texture into another.
//!
//! This is needed when the alpha representation of a renderer's output does not
//! match what the window surface expects:
//!
//! - A [`wgpu::CompositeAlphaMode::PostMultiplied`] surface expects *straight*
//!   alpha, because the OS compositor multiplies the colour by alpha itself.
//! - A [`wgpu::CompositeAlphaMode::PreMultiplied`] surface expects colour that
//!   has *already* been multiplied by alpha.
//!
//! Renderers differ: Vello Hybrid emits premultiplied alpha, whereas Vello
//! Classic emits straight alpha. Feeding the wrong representation to a surface
//! either darkens (premultiplied into PostMultiplied) or brightens (straight
//! into PreMultiplied) partially-transparent pixels. This blitter applies the
//! matching correction (`rgb / a` or `rgb * a`) while copying.

use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, ColorTargetState, ColorWrites,
    CommandEncoder, Device, FragmentState, LoadOp, MultisampleState, Operations,
    PipelineLayoutDescriptor, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, ShaderStages, StoreOp, TextureFormat,
    TextureSampleType, TextureView, TextureViewDimension, VertexState,
};

/// The direction of alpha conversion to apply while blitting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AlphaConversion {
    /// Convert straight alpha to premultiplied (`rgb * a`). Use when the
    /// renderer emits straight alpha and the surface expects premultiplied
    /// (i.e. [`wgpu::CompositeAlphaMode::PreMultiplied`]).
    Premultiply,
    /// Convert premultiplied alpha to straight (`rgb / a`). Use when the
    /// renderer emits premultiplied alpha and the surface expects straight
    /// (i.e. [`wgpu::CompositeAlphaMode::PostMultiplied`]).
    Unpremultiply,
}

const SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

// Fullscreen triangle: 3 vertices covering the whole clip space.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var src: texture_2d<f32>;

// Premultiplied alpha -> straight alpha.
@fragment
fn fs_unpremultiply(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(in.position.xy), 0);
    let a = c.a;
    var rgb = vec3<f32>(0.0, 0.0, 0.0);
    if (a > 0.0) {
        rgb = c.rgb / a;
    }
    return vec4<f32>(rgb, a);
}

// Straight alpha -> premultiplied alpha.
@fragment
fn fs_premultiply(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = textureLoad(src, vec2<i32>(in.position.xy), 0);
    return vec4<f32>(c.rgb * c.a, c.a);
}
"#;

/// Blits a texture into another texture, converting between premultiplied and
/// straight alpha in the process.
///
/// Mirrors the API of [`wgpu::util::TextureBlitter`] so it can be used as a
/// drop-in replacement.
pub struct AlphaConvertBlitter {
    pipeline: RenderPipeline,
    bind_group_layout: BindGroupLayout,
}

impl AlphaConvertBlitter {
    /// Create a blitter that applies `conversion` and writes to a target
    /// texture of the given `format`.
    pub fn new(device: &Device, format: TextureFormat, conversion: AlphaConversion) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("alpha convert blit shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let entry_point = match conversion {
            AlphaConversion::Premultiply => "fs_premultiply",
            AlphaConversion::Unpremultiply => "fs_unpremultiply",
        };

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("alpha convert blit bind group layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("alpha convert blit pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("alpha convert blit pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Copy `source` into `target`, applying the alpha conversion.
    ///
    /// `source` and `target` must have the same dimensions.
    pub fn copy(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        source: &TextureView,
        target: &TextureView,
    ) {
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("alpha convert blit bind group"),
            layout: &self.bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(source),
            }],
        });

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("alpha convert blit pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
