use anyrender::{RenderContext, ResourceId};
use peniko::ImageData;
use rustc_hash::FxHashMap;
use vello::Renderer as VelloRenderer;
use wgpu::Texture;
pub use wgpu_context::DeviceHandle;

pub trait CustomPaintSource: 'static {
    fn resume(&mut self, device_handle: &DeviceHandle);
    fn suspend(&mut self);
    fn render(
        &mut self,
        ctx: CustomPaintCtx<'_>,
        width: u32,
        height: u32,
        scale: f64,
    ) -> Option<TextureHandle>;
}

pub struct CustomPaintCtx<'r> {
    pub(crate) renderer: &'r mut VelloRenderer,
    pub(crate) texture_handles: &'r mut FxHashMap<ResourceId, ImageData>,
}

pub type TextureHandle = ResourceId;

impl CustomPaintCtx<'_> {
    pub(crate) fn new<'a>(
        renderer: &'a mut VelloRenderer,
        texture_handles: &'a mut FxHashMap<ResourceId, ImageData>,
    ) -> CustomPaintCtx<'a> {
        CustomPaintCtx {
            renderer,
            texture_handles,
        }
    }

    pub fn register_texture(&mut self, texture: Texture) -> TextureHandle {
        let id = ResourceId::new();
        self.texture_handles
            .insert(id, self.renderer.register_texture(texture));
        id
    }

    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        if let Some(handle) = self.texture_handles.remove(&handle) {
            self.renderer.unregister_texture(handle);
        }
    }
}

impl RenderContext for CustomPaintCtx<'_> {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn std::any::Any>,
    ) -> Result<ResourceId, anyrender::RegisterResourceError> {
        if let Ok(texture) = resource.downcast::<Texture>() {
            Ok(self.register_texture(*texture))
        } else {
            Err(anyrender::RegisterResourceErrorKind::UnsupportedResourceKind.into())
        }
    }

    fn unregister_resource(&mut self, resource_id: ResourceId) {
        self.unregister_texture(resource_id);
    }
}
