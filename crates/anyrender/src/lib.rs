//! 2D drawing abstraction that allows applications/frameworks to support many rendering backends through
//! a unified API.
//!
//! ### Painting a scene
//!
//! The core abstraction in AnyRender is the [`PaintScene`] trait.
//!
//! [`PaintScene`] is a "sink" which accepts drawing commands:
//!
//!   - Applications and libraries draw by pushing commands into a [`PaintScene`]
//!   - Backends execute those commands to produce an output
//!
//! ### Rendering to surface or buffer
//!
//! In addition to PaintScene, there is:
//!
//!   - The [`ImageRenderer`] trait which provides an abstraction for rendering to a `Vec<u8>` RGBA8 buffer.
//!   - The [`WindowRenderer`] trait which provides an abstraction for rendering to a surface/window
//!
//! ### SVG
//!
//! The [anyrender_svg](https://docs.rs/anyrender_svg) crate allows SVGs to be rendered using AnyRender
//!
//! ### WASM support
//!
//! Wgpu adapter/device/surface initialization is fundamentally async on the web. To avoid
//! deadlocking the JS event loop, [`WindowRenderer::resume`] takes an `on_ready` callback:
//! GPU backends spawn the init on `wasm_bindgen_futures::spawn_local` and invoke the callback
//! once the surface is live. The embedder then calls [`WindowRenderer::complete_resume`] to
//! transition the renderer to the active state. On native targets the same code path runs
//! inline (`pollster::block_on` on the GPU backends), so callers see no behavioural difference.
//!
//! ### Backends
//!
//! Currently existing backends are:
//!  - [anyrender_vello](https://docs.rs/anyrender_vello)
//!  - [anyrender_vello_cpu](https://docs.rs/anyrender_vello_cpu)

#![allow(clippy::collapsible_if)]

use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrushRef, StyleRef};
use recording::RenderCommand;
use std::{any::Any, sync::Arc};

pub mod wasm_send_sync;
pub use wasm_send_sync::*;
pub mod types;
pub use types::*;
mod null_backend;
pub use null_backend::*;
pub mod recording;
pub use recording::Scene;

mod resource_id;
pub use resource_id::ResourceId;

#[cfg(feature = "serde")]
mod svg_path_parser;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RegisterResourceErrorKind {
    /// The `RenderContext` you tried to register the resource with does not support the kind of resource
    UnsupportedResourceKind,
    /// Some other kind of error occured
    Other,
    /// This backend has not implemented resource registration
    Unimplemented,
    /// The `RenderContext` you tried to register the resource is not currently active
    NotActive,
}

#[derive(Debug, Clone)]
pub struct RegisterResourceError {
    /// The kind of error that occurred when registering the resource
    pub kind: RegisterResourceErrorKind,
    /// An optional detailed error message
    pub message: Option<String>,
}

impl From<RegisterResourceErrorKind> for RegisterResourceError {
    fn from(kind: RegisterResourceErrorKind) -> Self {
        Self {
            kind,
            message: None,
        }
    }
}

pub trait RenderContext {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn Any>,
    ) -> Result<ResourceId, RegisterResourceError> {
        let _ = resource;
        Err(RegisterResourceErrorKind::Unimplemented.into())
    }
    fn unregister_resource(&mut self, resource_id: ResourceId) {
        let _ = resource_id;
    }

    /// Return a type-erased context type that is passed to custom widgets
    /// in order to enable them to render renderer-specific content
    fn renderer_specific_context(&self) -> Option<Box<dyn Any>> {
        None
    }
}

/// Abstraction for rendering a scene to a window
pub trait WindowRenderer: RenderContext {
    type ScenePainter<'a>: PaintScene
    where
        Self: 'a;

    /// Begin resuming the renderer. `on_ready` fires when initialization completes —
    /// synchronously inside `resume` on native, asynchronously (via
    /// `wasm_bindgen_futures::spawn_local`) on `wasm32-unknown-unknown`. After it
    /// fires, the embedder must call [`complete_resume`](Self::complete_resume) to
    /// transition the renderer to the active state.
    fn resume<F: FnOnce() + 'static>(
        &mut self,
        window: Arc<dyn WindowHandle>,
        width: u32,
        height: u32,
        on_ready: F,
    );

    /// Finalize a previously-initiated resume. Returns `true` once the renderer is
    /// active and ready to render. Idempotent on already-active renderers; returns
    /// `false` if a pending init has not yet produced a result.
    ///
    /// Backends whose `resume` finishes synchronously inline should return `true`
    /// directly. There is intentionally no default: forgetting to override this on
    /// an async-init backend would silently no-op rendering.
    fn complete_resume(&mut self) -> bool;

    fn suspend(&mut self);
    fn is_active(&self) -> bool;

    /// Returns `true` while an asynchronous resume is in flight (after `resume`
    /// but before `complete_resume` has succeeded). Defaults to `false` for
    /// backends with synchronous initialization.
    fn is_pending(&self) -> bool {
        false
    }

    fn set_size(&mut self, width: u32, height: u32);
    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F);
}

/// Abstraction for rendering a scene to an image buffer
pub trait ImageRenderer: RenderContext {
    type ScenePainter<'a>: PaintScene
    where
        Self: 'a;
    fn new(width: u32, height: u32) -> Self;
    fn resize(&mut self, width: u32, height: u32);
    fn reset(&mut self);
    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        vec: &mut Vec<u8>,
    );
    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(&mut self, draw_fn: F, buffer: &mut [u8]);
}

/// Draw a scene to a buffer using an `ImageRenderer`
pub fn render_to_buffer<R: ImageRenderer, F: FnOnce(&mut R::ScenePainter<'_>)>(
    draw_fn: F,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut buf = Vec::with_capacity((width * height * 4) as usize);
    let mut renderer = R::new(width, height);
    renderer.render_to_vec(draw_fn, &mut buf);

    buf
}

/// Abstraction for drawing a 2D scene
pub trait PaintScene: RenderContext {
    /// Removes all content from the scene
    fn reset(&mut self);

    /// Pushes a new layer clipped by the specified shape and composed with previous layers using the specified blend mode.
    /// Every drawing command after this call will be clipped by the shape until the layer is popped.
    /// However, the transforms are not saved or modified by the layer stack.
    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
    );

    /// Pushes a new clip layer clipped by the specified shape.
    /// Every drawing command after this call will be clipped by the shape until the layer is popped.
    /// However, the transforms are not saved or modified by the layer stack.
    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape);

    /// Pops the current layer.
    fn pop_layer(&mut self);

    /// Strokes a shape using the specified style and brush.
    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    );

    /// Fills a shape using the specified style and brush.
    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        brush: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    );

    /// Draws a run of glyphs
    #[allow(clippy::too_many_arguments)]
    fn draw_glyphs<'a, 's: 'a>(
        &'s mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'a>>,
        brush: impl Into<PaintRef<'a>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
    );

    /// Draw a rounded rectangle blurred with a gaussian filter.
    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        brush: Color,
        radius: f64,
        std_dev: f64,
    );

    // --- Provided methods

    /// Append a recorded Scene Fragment to the current scene
    fn append_scene(&mut self, scene: Scene, scene_transform: Affine) {
        for cmd in scene.commands {
            match cmd {
                RenderCommand::PushLayer(cmd) => self.push_layer(
                    cmd.blend,
                    cmd.alpha,
                    scene_transform * cmd.transform,
                    &cmd.clip,
                ),
                RenderCommand::PushClipLayer(cmd) => {
                    self.push_clip_layer(scene_transform * cmd.transform, &cmd.clip)
                }
                RenderCommand::PopLayer => self.pop_layer(),
                RenderCommand::Stroke(cmd) => self.stroke(
                    &cmd.style,
                    scene_transform * cmd.transform,
                    match cmd.brush {
                        Paint::Solid(alpha_color) => Paint::Solid(alpha_color),
                        Paint::Gradient(ref gradient) => Paint::Gradient(gradient),
                        Paint::Image(ref image) => Paint::Image(image.as_ref()),
                        Paint::Resource(id) => Paint::Resource(id),
                        Paint::Custom(ref custom) => Paint::Custom(custom.as_ref()),
                    },
                    cmd.brush_transform,
                    &cmd.shape,
                ),
                RenderCommand::Fill(cmd) => self.fill(
                    cmd.fill,
                    scene_transform * cmd.transform,
                    match cmd.brush {
                        Paint::Solid(alpha_color) => Paint::Solid(alpha_color),
                        Paint::Gradient(ref gradient) => Paint::Gradient(gradient),
                        Paint::Image(ref image) => Paint::Image(image.as_ref()),
                        Paint::Resource(id) => Paint::Resource(id),
                        Paint::Custom(ref custom) => Paint::Custom(custom.as_ref()),
                    },
                    cmd.brush_transform,
                    &cmd.shape,
                ),
                RenderCommand::GlyphRun(cmd) => self.draw_glyphs(
                    &cmd.font_data,
                    cmd.font_size,
                    cmd.hint,
                    &cmd.normalized_coords,
                    cmd.embolden,
                    &cmd.style,
                    match cmd.brush {
                        Paint::Solid(alpha_color) => Paint::Solid(alpha_color),
                        Paint::Gradient(ref gradient) => Paint::Gradient(gradient),
                        Paint::Image(ref image) => Paint::Image(image.as_ref()),
                        Paint::Resource(id) => Paint::Resource(id),
                        Paint::Custom(ref custom) => Paint::Custom(custom.as_ref()),
                    },
                    cmd.brush_alpha,
                    scene_transform * cmd.transform,
                    cmd.glyph_transform,
                    cmd.glyphs.into_iter(),
                ),
                RenderCommand::BoxShadow(cmd) => self.draw_box_shadow(
                    scene_transform * cmd.transform,
                    cmd.rect,
                    cmd.brush,
                    cmd.radius,
                    cmd.std_dev,
                ),
            }
        }
    }

    /// Utility method to draw an image at it's natural size. For more advanced image drawing use the `fill` method
    fn draw_image(&mut self, image: ImageBrushRef, transform: Affine) {
        self.fill(
            Fill::NonZero,
            transform,
            image,
            None,
            &Rect::new(
                0.0,
                0.0,
                image.image.width as f64,
                image.image.height as f64,
            ),
        );
    }
}
