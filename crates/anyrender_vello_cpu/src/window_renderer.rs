#[cfg(feature = "softbuffer_window_renderer")]
pub use softbuffer_window_renderer::{SoftbufferRendererOptions, SoftbufferWindowRenderer};

#[cfg(feature = "pixels_window_renderer")]
pub use pixels_window_renderer::PixelsRendererOptions;
#[cfg(feature = "pixels_window_renderer")]
pub use pixels_window_renderer::PixelsWindowRenderer;

#[cfg(feature = "pixels_window_renderer")]
pub type VelloCpuWindowRenderer = PixelsWindowRenderer<crate::VelloCpuImageRenderer>;
#[cfg(feature = "pixels_window_renderer")]
pub type VelloCpuRendererOptions = PixelsRendererOptions;

#[cfg(all(
    feature = "softbuffer_window_renderer",
    not(feature = "pixels_window_renderer")
))]
pub type VelloCpuWindowRenderer = SoftbufferWindowRenderer<crate::VelloCpuImageRenderer>;
#[cfg(all(
    feature = "softbuffer_window_renderer",
    not(feature = "pixels_window_renderer")
))]
pub type VelloCpuRendererOptions = SoftbufferRendererOptions;
