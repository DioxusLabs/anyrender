use std::sync::Arc;

use skia_safe::{
    ColorType, Surface,
    gpu::{self, BackendRenderTarget, DirectContext, Protected, SurfaceOrigin, d3d},
};
use windows::{
    Win32::{
        Foundation::{HANDLE, HWND},
        Graphics::{
            Direct3D::D3D_FEATURE_LEVEL_11_0,
            Direct3D12::*,
            Dxgi::{Common::*, *},
        },
    },
    core::Interface,
};

use crate::{BackendError, window_renderer::SkiaBackend};

pub(crate) struct D3DBackend {
    surfaces: [Option<(Surface, BackendRenderTarget)>; 2],
    context: Option<DirectContext>,
    swapchain: Option<IDXGISwapChain3>,
    backend: d3d::BackendContext,
    fence: ID3D12Fence,
    fence_value: u64,
    size: (u32, u32),
    lost: bool,
}

fn create_device(adapter: &IDXGIAdapter1) -> Result<ID3D12Device, BackendError> {
    let mut device = None;
    unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, &mut device) }
        .map_err(BackendError::new)?;
    device.ok_or_else(|| BackendError::new("D3D12CreateDevice returned no device"))
}

fn is_software_adapter(desc: &DXGI_ADAPTER_DESC1) -> bool {
    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
        || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c)
}

fn choose_adapter(
    factory: &IDXGIFactory4,
    software: bool,
) -> Result<(IDXGIAdapter1, ID3D12Device), BackendError> {
    for index in 0.. {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => adapter,
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(error) => return Err(BackendError::new(error)),
        };
        let desc = unsafe { adapter.GetDesc1() }.map_err(BackendError::new)?;
        if is_software_adapter(&desc) != software {
            continue;
        }
        if let Ok(device) = create_device(&adapter) {
            return Ok((adapter, device));
        }
    }
    if software {
        let adapter =
            unsafe { factory.EnumWarpAdapter::<IDXGIAdapter1>() }.map_err(BackendError::new)?;
        let device = create_device(&adapter)?;
        return Ok((adapter, device));
    }
    Err(BackendError::new("No hardware D3D12 adapter available"))
}

impl D3DBackend {
    pub(crate) fn new(
        window: Arc<dyn anyrender::WindowHandle>,
        width: u32,
        height: u32,
        software: bool,
    ) -> Result<Self, BackendError> {
        if width == 0 || height == 0 {
            return Err(BackendError::new("Surface dimensions must be positive"));
        }
        let raw_window = window.window_handle().map_err(BackendError::new)?.as_raw();
        let raw_window_handle::RawWindowHandle::Win32(handle) = raw_window else {
            return Err(BackendError::new("D3D requires a Win32 window handle"));
        };
        let hwnd = HWND(handle.hwnd.get() as *mut _);
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory1() }.map_err(BackendError::new)?;
        let (adapter, device) = choose_adapter(&factory, software)?;
        let desc = unsafe { adapter.GetDesc1() }.map_err(BackendError::new)?;
        let end = desc
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(desc.Description.len());
        tracing::info!(
            "Skia D3D adapter: {}",
            String::from_utf16_lossy(&desc.Description[..end])
        );
        let queue = unsafe {
            device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
        }
        .map_err(BackendError::new)?;
        let fence =
            unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.map_err(BackendError::new)?;
        let backend = d3d::BackendContext {
            adapter,
            device,
            queue,
            memory_allocator: None,
            protected_context: Protected::No,
        };
        let context = unsafe { gpu::direct_contexts::make_d3d(&backend, None) }
            .ok_or_else(|| BackendError::new("Failed to create Skia D3D context"))?;
        let swapchain: IDXGISwapChain3 = unsafe {
            factory.CreateSwapChainForHwnd(
                &backend.queue,
                hwnd,
                &DXGI_SWAP_CHAIN_DESC1 {
                    Width: width,
                    Height: height,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                    BufferCount: 2,
                    SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                },
                None,
                None,
            )
        }
        .map_err(BackendError::new)?
        .cast()
        .map_err(BackendError::new)?;
        unsafe { factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) }
            .map_err(BackendError::new)?;
        Ok(Self {
            surfaces: [None, None],
            context: Some(context),
            swapchain: Some(swapchain),
            backend,
            fence,
            fence_value: 0,
            size: (width, height),
            lost: false,
        })
    }

    fn wait_for_gpu(&mut self) -> Result<(), BackendError> {
        self.fence_value += 1;
        unsafe {
            self.backend
                .queue
                .Signal(&self.fence, self.fence_value)
                .map_err(BackendError::new)?;
            // A null event makes SetEventOnCompletion wait synchronously.
            self.fence
                .SetEventOnCompletion(self.fence_value, HANDLE::default())
                .map_err(BackendError::new)?;
        }
        Ok(())
    }

    fn surface(&mut self) -> Result<Surface, BackendError> {
        let swapchain = self.swapchain.as_ref().unwrap();
        let index = unsafe { swapchain.GetCurrentBackBufferIndex() } as usize;
        if self.surfaces[index].is_none() {
            let resource =
                unsafe { swapchain.GetBuffer(index as u32) }.map_err(BackendError::new)?;
            let target = gpu::backend_render_targets::make_d3d(
                (self.size.0 as i32, self.size.1 as i32),
                &d3d::TextureResourceInfo {
                    resource,
                    alloc: None,
                    resource_state: D3D12_RESOURCE_STATE_PRESENT,
                    format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    sample_count: 1,
                    level_count: 1,
                    sample_quality_pattern: DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN,
                    protected: Protected::No,
                },
            );
            let surface = gpu::surfaces::wrap_backend_render_target(
                self.context.as_mut().unwrap(),
                &target,
                SurfaceOrigin::TopLeft,
                ColorType::RGBA8888,
                None,
                None,
            )
            .ok_or_else(|| BackendError::new("Failed to wrap D3D render target"))?;
            self.surfaces[index] = Some((surface, target));
        }
        Ok(self.surfaces[index].as_ref().unwrap().0.clone())
    }

    fn mark_lost(&mut self, error: impl std::fmt::Display) {
        tracing::warn!("Skia D3D backend lost: {error}");
        self.lost = true;
    }
}

impl SkiaBackend for D3DBackend {
    fn set_size(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || self.lost || self.size == (width, height) {
            return;
        }
        self.surfaces = [None, None];
        if let Err(error) = self.context.as_mut().unwrap().flush_submit_and_sync_cpu() {
            self.mark_lost(error);
            return;
        }
        if let Err(error) = self.wait_for_gpu() {
            self.mark_lost(error);
            return;
        }
        let result = unsafe {
            self.swapchain.as_ref().unwrap().ResizeBuffers(
                0,
                width,
                height,
                DXGI_FORMAT_UNKNOWN,
                DXGI_SWAP_CHAIN_FLAG(0),
            )
        };
        match result {
            Ok(()) => self.size = (width, height),
            Err(error) => self.mark_lost(error),
        }
    }

    fn prepare(&mut self) -> Option<Surface> {
        if self.lost {
            return None;
        }
        match self.surface() {
            Ok(surface) => Some(surface),
            Err(error) => {
                self.mark_lost(error);
                None
            }
        }
    }

    fn flush(&mut self, mut surface: Surface) {
        if let Err(error) = self
            .context
            .as_mut()
            .unwrap()
            .flush_and_submit_surface(&mut surface, None)
        {
            self.mark_lost(error);
            return;
        }
        let result = unsafe { self.swapchain.as_ref().unwrap().Present(1, DXGI_PRESENT(0)) };
        if let Err(error) = result.ok() {
            self.mark_lost(error);
            return;
        }
        if let Err(error) = self.wait_for_gpu() {
            self.mark_lost(error);
        }
    }

    fn is_lost(&self) -> bool {
        self.lost
    }
}

impl Drop for D3DBackend {
    fn drop(&mut self) {
        if !self.lost
            && let Err(error) = self.wait_for_gpu()
        {
            self.mark_lost(error);
        }
        self.surfaces = [None, None];
        if let Some(mut context) = self.context.take() {
            if self.lost {
                context.abandon();
            }
            drop(context);
        }
        self.swapchain.take();
    }
}

#[cfg(test)]
mod tests {
    use super::{D3DBackend, is_software_adapter};
    use raw_window_handle::{
        DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
    };
    use std::sync::Arc;
    use windows::Win32::Graphics::Dxgi::{DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG_SOFTWARE};

    struct UnavailableWindow;

    impl HasWindowHandle for UnavailableWindow {
        fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
            Err(HandleError::Unavailable)
        }
    }

    impl HasDisplayHandle for UnavailableWindow {
        fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
            Err(HandleError::Unavailable)
        }
    }

    #[test]
    fn invalid_dimensions_are_errors() {
        for (width, height) in [(0, 0), (0, 100), (100, 0)] {
            let error = D3DBackend::new(Arc::new(UnavailableWindow), width, height, false)
                .err()
                .unwrap();
            assert_eq!(error.to_string(), "Surface dimensions must be positive");
        }
    }

    #[test]
    fn unavailable_window_is_an_error() {
        assert!(D3DBackend::new(Arc::new(UnavailableWindow), 100, 100, false).is_err());
    }

    #[test]
    fn software_adapter_detection() {
        assert!(!is_software_adapter(&DXGI_ADAPTER_DESC1::default()));
        assert!(is_software_adapter(&DXGI_ADAPTER_DESC1 {
            Flags: DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32,
            ..Default::default()
        }));
        assert!(is_software_adapter(&DXGI_ADAPTER_DESC1 {
            VendorId: 0x1414,
            DeviceId: 0x8c,
            ..Default::default()
        }));
        assert!(!is_software_adapter(&DXGI_ADAPTER_DESC1 {
            VendorId: 0x1414,
            DeviceId: 0x8d,
            ..Default::default()
        }));
    }
}
