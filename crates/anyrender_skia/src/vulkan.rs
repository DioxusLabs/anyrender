use std::sync::Arc;

use skia_safe::{
    ColorType, Surface,
    gpu::{
        DirectContext, SurfaceOrigin, direct_contexts,
        ganesh::vk::backend_render_targets,
        surfaces,
        vk::{self},
    },
};
use vulkano::{
    Handle, Validated, VulkanError, VulkanObject,
    device::{DeviceExtensions, Queue, QueueCreateInfo, physical::PhysicalDeviceType},
    image::{Image, view::ImageView},
    instance::{InstanceCreateFlags, InstanceCreateInfo},
    swapchain::{
        self, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo,
        acquire_next_image,
    },
    sync::GpuFuture,
};

use crate::window_renderer::SkiaBackend;

pub(crate) struct VulkanBackend {
    gr_context: DirectContext,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_image_views: Vec<Arc<ImageView>>,
    next_frame: Option<(u32, SwapchainAcquireFuture)>,
    last_render: Option<Box<dyn GpuFuture>>,
    swapchain_is_valid: bool,
    window_size: [u32; 2],
}

impl VulkanBackend {
    pub(crate) fn new(
        window: Arc<dyn anyrender::WindowHandle>,
        width: u32,
        height: u32,
    ) -> VulkanBackend {
        let library = vulkano::VulkanLibrary::new().expect("vulkan library is available on system");

        let required_extensions = swapchain::Surface::required_extensions(&window).unwrap();

        let instance = vulkano::instance::Instance::new(
            library.clone(),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("instance supporting required extensions available");

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let surface =
            unsafe { swapchain::Surface::from_window_ref(instance.clone(), &window) }.unwrap();

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags
                            .intersects(vulkano::device::QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("suitable physical device available");

        let (device, mut queues) = vulkano::device::Device::new(
            physical_device,
            vulkano::device::DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("device initializes");

        let queue = queues.next().unwrap();

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            let (image_format, _) = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()
                .into_iter()
                .find(|(format, _)| *format == vulkano::format::Format::B8G8R8A8_UNORM)
                .unwrap();

            swapchain::Swapchain::new(
                device.clone(),
                surface,
                swapchain::SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_extent: [width, height],
                    image_usage: vulkano::image::ImageUsage::INPUT_ATTACHMENT,
                    image_format,
                    present_mode: swapchain::PresentMode::Mailbox,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
        }
        .unwrap();

        let last_render = Some(vulkano::sync::now(device.clone()).boxed());

        let gr_context = unsafe {
            let get_proc = |gpo| {
                let get_device_proc_addr = instance.fns().v1_0.get_device_proc_addr;

                match gpo {
                    vk::GetProcOf::Instance(instance, name) => {
                        let vk_instance = ash::vk::Instance::from_raw(instance as _);
                        library.get_instance_proc_addr(vk_instance, name)
                    }
                    vk::GetProcOf::Device(device, name) => {
                        let vk_device = ash::vk::Device::from_raw(device as _);
                        get_device_proc_addr(vk_device, name)
                    }
                }
                .map(|f| f as _)
                .unwrap()
            };

            direct_contexts::make_vulkan(
                &vk::BackendContext::new(
                    instance.handle().as_raw() as _,
                    device.physical_device().handle().as_raw() as _,
                    device.handle().as_raw() as _,
                    (queue.handle().as_raw() as _, queue.queue_index() as usize),
                    &get_proc,
                ),
                None,
            )
        }
        .unwrap();

        let mut image_views: Vec<Arc<ImageView>> = Vec::with_capacity(images.len());
        for image in &images {
            image_views.push(ImageView::new_default(image.clone()).unwrap());
        }

        VulkanBackend {
            gr_context,
            queue,
            swapchain,
            swapchain_images: images,
            swapchain_image_views: image_views,
            last_render,
            swapchain_is_valid: true,
            next_frame: None,
            window_size: [width, height],
        }
    }

    fn prepare_swapchain(&mut self) {
        if self.swapchain_is_valid {
            return;
        }

        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: self.window_size.clone(),

                ..self.swapchain.create_info()
            })
            .unwrap();

        let mut new_image_views: Vec<Arc<ImageView>> = Vec::with_capacity(new_images.len());
        for image in &new_images {
            new_image_views.push(ImageView::new_default(image.clone()).unwrap());
        }

        self.swapchain = new_swapchain;
        self.swapchain_images = new_images;
        self.swapchain_image_views = new_image_views;
        self.swapchain_is_valid = true;
    }

    fn next_frame(&mut self) -> Option<(u32, SwapchainAcquireFuture)> {
        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(it) => it,
                Err(VulkanError::OutOfDate) => {
                    self.swapchain_is_valid = false;
                    return None;
                }
                Err(e) => panic!("failed to acquire next image: {e:?}"),
            };

        if suboptimal {
            self.swapchain_is_valid = false;
        }

        Some((image_index, acquire_future))
    }

    fn create_surface_from_image_view(&mut self, image_view: Arc<ImageView>) -> Surface {
        let image = image_view.image();
        let [width, height, _] = image.extent();

        let alloc = vk::Alloc::default();
        let image_info = &unsafe {
            vk::ImageInfo::new(
                image.handle().as_raw() as _,
                alloc,
                vk::ImageTiling::OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::Format::B8G8R8A8_UNORM,
                1,
                None,
                None,
                None,
                None,
            )
        };

        let render_target =
            backend_render_targets::make_vk((width as i32, height as i32), image_info);

        surfaces::wrap_backend_render_target(
            &mut self.gr_context,
            &render_target,
            SurfaceOrigin::TopLeft,
            ColorType::BGRA8888,
            None,
            None,
        )
        .unwrap()
    }
}

impl SkiaBackend for VulkanBackend {
    fn set_size(&mut self, width: u32, height: u32) {
        self.window_size = [width, height];
        self.swapchain_is_valid = false;
    }

    fn prepare(&mut self) -> Option<Surface> {
        if let Some(last_render) = self.last_render.as_mut() {
            last_render.cleanup_finished();
        }

        self.prepare_swapchain();

        if let Some((image_index, acquire_future)) = self.next_frame() {
            self.next_frame = Some((image_index, acquire_future));

            return Some(self.create_surface_from_image_view(
                self.swapchain_image_views[image_index as usize].clone(),
            ));
        } else {
            None
        }
    }

    // In vulkan implementation we do not reuse surface so we just drop it straight away
    fn flush(&mut self, _: Surface) {
        self.gr_context.flush_and_submit();

        let (image_index, acquire_future) = self.next_frame.take().unwrap();

        let future = self
            .last_render
            .take()
            .unwrap()
            .join(acquire_future)
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.last_render = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.swapchain_is_valid = false;
                self.last_render = Some(vulkano::sync::now(self.queue.device().clone()).boxed());
            }
            Err(e) => {
                self.last_render = Some(vulkano::sync::now(self.queue.device().clone()).boxed());
                println!("skia vlk: failed to flush future: {e:?}")
            }
        };
    }
}
