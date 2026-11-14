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
pub type TinySkiaWindowRenderer = SoftbufferWindowRenderer<crate::TinySkiaImageRenderer>;

/// Creates [`TinySkiaWindowRenderer`]s.
#[cfg(feature = "pixels_window_renderer")]
pub type TinySkiaWindowRendererContext =
    pixels_window_renderer::PixelsWindowRendererContext<crate::TinySkiaImageRenderer>;
/// Creates [`TinySkiaWindowRenderer`]s.
#[cfg(all(
    feature = "softbuffer_window_renderer",
    not(feature = "pixels_window_renderer")
))]
pub type TinySkiaWindowRendererContext =
    softbuffer_window_renderer::SoftbufferWindowRendererContext<crate::TinySkiaImageRenderer>;
