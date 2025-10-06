#[cfg(feature = "softbuffer_window_renderer")]
#[cfg_attr(docsrs, doc(cfg(feature = "softbuffer_window_renderer")))]
mod softbuffer_window_renderer;

#[cfg(feature = "pixels_window_renderer")]
#[cfg_attr(docsrs, doc(cfg(feature = "pixels_window_renderer")))]
mod pixels_window_renderer;

#[cfg(feature = "softbuffer_window_renderer")]
pub use softbuffer_window_renderer::SoftbufferWindowRenderer;

#[cfg(feature = "pixels_window_renderer")]
pub use pixels_window_renderer::PixelsWindowRenderer;

#[cfg(feature = "pixels_window_renderer")]
pub type VelloCpuWindowRenderer = PixelsWindowRenderer<crate::VelloCpuImageRenderer>;
#[cfg(all(
    feature = "softbuffer_window_renderer",
    not(feature = "pixels_window_renderer")
))]
pub type VelloCpuWindowRenderer = SoftbufferWindowRenderer<crate::VelloCpuImageRenderer>;
