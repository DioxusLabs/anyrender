// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simple helpers for managing wgpu state and surfaces.

use wgpu::{
    Adapter, Device, Features, Instance, Limits, MemoryHints, Queue, Surface, SurfaceTarget,
};

mod buffer_renderer;
mod error;
mod surface_renderer;
mod util;

pub use buffer_renderer::{BufferRenderer, BufferRendererConfig};
pub use error::WgpuContextError;
pub use surface_renderer::{SurfaceRenderer, SurfaceRendererConfiguration, TextureConfiguration};
pub use util::block_on_wgpu;

/// A wgpu `Device`, it's associated `Queue`, and the `Adapter` and `Instance` used to create them
#[derive(Clone, Debug)]
pub struct DeviceHandle {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl DeviceHandle {
    /// Creates a `DeviceHandle` with `Device` that's compatible with the specified `Surface`
    pub async fn new_from_compatible_surface(
        instance: Instance,
        compatible_surface: Option<&Surface<'_>>,
        extra_features: Option<Features>,
        override_limits: Option<Limits>,
    ) -> Result<Self, WgpuContextError> {
        let adapter =
            wgpu::util::initialize_adapter_from_env_or_default(&instance, compatible_surface)
                .await?;

        // Determine features to request
        // The user may request additional features
        let requested_features = extra_features.unwrap_or(Features::empty());
        let available_features = adapter.features();
        let required_features = requested_features & available_features;

        // Determine limits to request
        // The user may override the limits
        let required_limits = override_limits.clone().unwrap_or_else(|| Limits {
            // Fix iOS simulator
            max_inter_stage_shader_variables: 15,
            ..Limits::default()
        });

        // Create the device and the queue
        let descripter = wgpu::DeviceDescriptor {
            label: None,
            required_features,
            required_limits,
            memory_hints: MemoryHints::MemoryUsage,
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::default(),
        };
        let (device, queue) = adapter.request_device(&descripter).await?;

        // Create the device handle and store in the pool
        Ok(DeviceHandle {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Creates a new surface for the specified window and dimensions.
    pub async fn create_surface<'w>(
        &mut self,
        window: impl Into<SurfaceTarget<'w>>,
        surface_config: SurfaceRendererConfiguration,
        intermediate_texture_config: Option<TextureConfiguration>,
    ) -> Result<SurfaceRenderer<'w>, WgpuContextError> {
        // Create a surface from the window handle
        let surface = self.instance.create_surface(window.into())?;
        SurfaceRenderer::new(
            surface,
            surface_config,
            intermediate_texture_config,
            self.clone(),
        )
    }
}

/// Simple render context that maintains wgpu state for rendering the pipeline.
pub struct WGPUContext {
    /// A WGPU `Instance`. This only needs to be created once per application.
    pub instance: Instance,
    /// A pool of already-created devices so that we can avoid recreating devices
    /// when we already have a suitable one available
    pub device_pool: Vec<DeviceHandle>,

    // Config
    extra_features: Option<Features>,
    override_limits: Option<Limits>,
}

impl Default for WGPUContext {
    fn default() -> Self {
        Self::new()
    }
}

impl WGPUContext {
    pub fn new() -> Self {
        Self::with_features_and_limits(None, None)
    }

    pub fn with_features_and_limits(
        extra_features: Option<Features>,
        override_limits: Option<Limits>,
    ) -> Self {
        Self {
            instance: Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::from_env().unwrap_or_default(),
                flags: wgpu::InstanceFlags::from_build_config().with_env(),
                backend_options: wgpu::BackendOptions::from_env_or_default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),

                // TODO: support passing display handle
                // Needed for opengl/webgl
                display: None,
            }),
            device_pool: Vec::new(),
            extra_features,
            override_limits,
        }
    }

    pub fn extra_features(&self) -> Option<Features> {
        self.extra_features
    }

    pub fn override_limits(&self) -> Option<Limits> {
        self.override_limits.clone()
    }

    /// Creates a new surface for the specified window and dimensions.
    pub fn create_surface<'w>(
        &self,
        window: impl Into<SurfaceTarget<'w>>,
    ) -> Result<Surface<'w>, WgpuContextError> {
        Ok(self.instance.create_surface(window.into())?)
    }

    /// Creates a new surface for the specified window and dimensions.
    pub async fn create_surface_renderer<'w>(
        &mut self,
        window: impl Into<SurfaceTarget<'w>>,
        surface_config: SurfaceRendererConfiguration,
        intermediate_texture_config: Option<TextureConfiguration>,
    ) -> Result<SurfaceRenderer<'w>, WgpuContextError> {
        // Create a surface from the window handle
        let surface = self.create_surface(window.into())?;

        // Find or create a suitable device for rendering to the surface
        let dev_id = self
            .find_or_create_device(Some(&surface))
            .await
            .or(Err(WgpuContextError::NoCompatibleDevice))?;
        let device_handle = self.device_pool[dev_id].clone();

        SurfaceRenderer::new(
            surface,
            surface_config,
            intermediate_texture_config,
            device_handle,
        )
    }

    /// Creates a new `BufferRenderer` for the specified dimensions.
    pub async fn create_buffer_renderer(
        &mut self,
        config: BufferRendererConfig,
    ) -> Result<BufferRenderer, WgpuContextError> {
        // Find or create a suitable device for rendering to the surface
        let dev_id = self
            .find_or_create_device(None)
            .await
            .or(Err(WgpuContextError::NoCompatibleDevice))?;
        let device_handle = self.device_pool[dev_id].clone();

        Ok(BufferRenderer::new(config, device_handle, dev_id))
    }

    /// Finds or creates a compatible device handle id.
    pub async fn find_or_create_device(
        &mut self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Result<usize, WgpuContextError> {
        match self.find_existing_device_idx(compatible_surface) {
            Some(device_id) => Ok(device_id),
            None => self.create_device(compatible_surface).await,
        }
    }

    /// Finds or creates a compatible device handle id.
    pub fn find_compatible_device_handle(
        &mut self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Option<DeviceHandle> {
        self.find_existing_device_idx(compatible_surface)
            .map(|idx| self.device_pool[idx].clone())
    }

    /// Finds  a compatible device handle id.
    fn find_existing_device_idx(&self, compatible_surface: Option<&Surface<'_>>) -> Option<usize> {
        match compatible_surface {
            Some(s) => self
                .device_pool
                .iter()
                .enumerate()
                .find(|(_, d)| d.adapter.is_surface_supported(s))
                .map(|(i, _)| i),
            None => (!self.device_pool.is_empty()).then_some(0),
        }
    }

    pub fn create_device_handle(
        &self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> impl Future<Output = Result<DeviceHandle, WgpuContextError>> {
        DeviceHandle::new_from_compatible_surface(
            self.instance.clone(),
            compatible_surface,
            self.extra_features,
            self.override_limits.clone(),
        )
    }

    /// Creates a compatible device handle id.
    async fn create_device(
        &mut self,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Result<usize, WgpuContextError> {
        let device_handle = self.create_device_handle(compatible_surface).await?;
        self.device_pool.push(device_handle);
        Ok(self.device_pool.len() - 1)
    }
}
