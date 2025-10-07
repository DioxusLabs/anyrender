#[cfg(feature = "softbuffer_window_renderer")]
pub use softbuffer_window_renderer::SoftbufferWindowRenderer;

#[cfg(feature = "pixels_window_renderer")]
pub use pixels_window_renderer::PixelsWindowRenderer;

#[cfg(feature = "pixels_window_renderer")]
pub type TinySkiaWindowRenderer = PixelsWindowRenderer<crate::TinySkiaImageRenderer>;
#[cfg(all(
    feature = "softbuffer_window_renderer",
    not(feature = "pixels_window_renderer")
))]
pub type VelloCpuWindowRenderer = SoftbufferWindowRenderer<crate::TinySkiaImageRenderer>;
