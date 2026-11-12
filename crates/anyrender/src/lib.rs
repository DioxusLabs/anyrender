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
//!
//! - [anyrender_vello_hybrid](https://docs.rs/anyrender_vello_hybrid) which draws using [vello_hybrid](https://docs.rs/vello_hybrid)
//! - [anyrender_vello_cpu](https://docs.rs/anyrender_vello_cpu) which draws using [vello_cpu](https://docs.rs/vello_cpu)
//! - [anyrender_vello](https://docs.rs/anyrender_vello) which draws using [vello](https://docs.rs/vello)
//! - [anyrender_skia](https://crates.io/crates/anyrender_skia) which draws using Skia (via the [skia-safe](https://github.com/rust-skia/rust-skia) crate)

#![allow(clippy::collapsible_if)]

use kurbo::{Affine, Rect, Stroke, Vec2};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrushRef, StyleRef};
use recording::RenderCommand;
use std::{any::Any, sync::Arc};

pub mod filters;
pub use filters::Filter;
pub mod wasm_send_sync;
pub use wasm_send_sync::*;
pub mod types;
pub use types::*;
mod shapes;
pub use shapes::*;
mod box_shadow;
pub use box_shadow::*;
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
    /// transition the renderer to the active state. Backends that are polled
    /// instead don't call `on_ready`; see [`is_pending`](Self::is_pending).
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

    /// Returns `true` while a resume is in flight (after `resume` but before
    /// `complete_resume` has succeeded). Defaults to `false` for backends with
    /// synchronous initialization.
    ///
    /// Some backends (e.g. WebGL, whose shaders compile in the background) do
    /// their remaining setup inside `complete_resume` and don't call `on_ready`.
    /// While this returns `true`, the embedder should keep calling
    /// `complete_resume` (e.g. once per redraw) until it returns `true`.
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

    /// Backend implementation of [`push_layer`](Self::push_layer).
    #[allow(clippy::too_many_arguments)]
    fn push_layer_shape(
        &mut self,
        fill: Fill,
        blend: BlendMode,
        alpha: f32,
        transform: Affine,
        clip: PaintShape<'_>,
        filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    );

    /// Backend implementation of [`push_clip_layer`](Self::push_clip_layer).
    fn push_clip_layer_shape(&mut self, fill: Fill, transform: Affine, clip: PaintShape<'_>);

    /// Backend implementation of [`push_mask_layer`](Self::push_mask_layer).
    fn push_mask_layer_shape(
        &mut self,
        fill: Fill,
        mode: MaskMode,
        transform: Affine,
        clip: PaintShape<'_>,
    );

    /// Pops the current layer.
    fn pop_layer(&mut self);

    /// Backend implementation of [`stroke`](Self::stroke).
    fn stroke_shape(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: PaintRef<'_>,
        brush_transform: Option<Affine>,
        shape: PaintShape<'_>,
    );

    /// Backend implementation of [`fill`](Self::fill).
    fn fill_shape(
        &mut self,
        style: Fill,
        transform: Affine,
        brush: PaintRef<'_>,
        brush_transform: Option<Affine>,
        shape: PaintShape<'_>,
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

    /// Draw a box shadow cast by `box_shape`, as specified by CSS `box-shadow`.
    ///
    /// The shape of the shadow is `box_shape` grown by `spread` (or for an inset shadow, shrunk by
    /// `spread`), with its corner radii adjusted as specified by CSS (see
    /// [`NonUniformRoundedRect::spread`]), and translated by `offset`. It is then blurred by a
    /// gaussian filter with a standard deviation of `std_dev`. A CSS blur radius `b` corresponds
    /// to a `std_dev` of `b / 2`.
    ///
    /// `kind` determines where the shadow is painted: see [`BoxShadowKind`]. Everything is
    /// transformed by `transform`.
    #[allow(clippy::too_many_arguments)]
    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        box_shape: &NonUniformRoundedRect,
        offset: Vec2,
        spread: f64,
        std_dev: f64,
        color: Color,
        kind: BoxShadowKind,
    );

    // --- Provided methods

    /// Pushes a new layer clipped by the specified shape and composed with previous layers using the specified blend mode.
    /// The fill rule determines which parts of the clip shape are inside the clipping region.
    /// Every drawing command after this call will be clipped by the shape until the layer is popped.
    /// However, the transforms are not saved or modified by the layer stack.
    #[allow(clippy::too_many_arguments)]
    fn push_layer<'s>(
        &mut self,
        fill: Fill,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: impl Into<PaintShape<'s>>,
        filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    ) where
        Self: Sized,
    {
        self.push_layer_shape(
            fill,
            blend.into(),
            alpha,
            transform,
            clip.into(),
            filter,
            backdrop_filter,
        );
    }

    /// Pushes a new clip layer clipped by the specified shape.
    /// The fill rule determines which parts of the clip shape are inside the clipping region.
    /// Every drawing command after this call will be clipped by the shape until the layer is popped.
    /// However, the transforms are not saved or modified by the layer stack.
    fn push_clip_layer<'s>(
        &mut self,
        fill: Fill,
        transform: Affine,
        clip: impl Into<PaintShape<'s>>,
    ) where
        Self: Sized,
    {
        self.push_clip_layer_shape(fill, transform, clip.into());
    }

    /// Pushes a new mask layer clipped by the specified shape.
    /// Content drawn until the layer is popped is used as a mask for the content drawn before it
    /// in the enclosing layer, converted to a mask according to `mode`. Everything outside of the
    /// clip shape counts as fully masked, so content of the enclosing layer outside of it is removed.
    /// However, the transforms are not saved or modified by the layer stack.
    fn push_mask_layer<'s>(
        &mut self,
        fill: Fill,
        mode: MaskMode,
        transform: Affine,
        clip: impl Into<PaintShape<'s>>,
    ) where
        Self: Sized,
    {
        self.push_mask_layer_shape(fill, mode, transform, clip.into());
    }

    /// Strokes a shape using the specified style and brush.
    fn stroke<'a, 's>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        brush: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: impl Into<PaintShape<'s>>,
    ) where
        Self: Sized,
    {
        self.stroke_shape(
            style,
            transform,
            brush.into(),
            brush_transform,
            shape.into(),
        );
    }

    /// Fills a shape using the specified style and brush.
    fn fill<'a, 's>(
        &mut self,
        style: Fill,
        transform: Affine,
        brush: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: impl Into<PaintShape<'s>>,
    ) where
        Self: Sized,
    {
        self.fill_shape(
            style,
            transform,
            brush.into(),
            brush_transform,
            shape.into(),
        );
    }

    /// Append a recorded Scene Fragment to the current scene
    fn append_scene(&mut self, scene: Scene, scene_transform: Affine) {
        for cmd in scene.commands {
            match cmd {
                RenderCommand::PushLayer(cmd) => self.push_layer_shape(
                    cmd.fill,
                    cmd.blend,
                    cmd.alpha,
                    scene_transform * cmd.transform,
                    (&cmd.clip).into(),
                    cmd.filter,
                    cmd.backdrop_filter,
                ),
                RenderCommand::PushClipLayer(cmd) => self.push_clip_layer_shape(
                    cmd.fill,
                    scene_transform * cmd.transform,
                    (&cmd.clip).into(),
                ),
                RenderCommand::PushMaskLayer(cmd) => self.push_mask_layer_shape(
                    cmd.fill,
                    cmd.mode,
                    scene_transform * cmd.transform,
                    (&cmd.clip).into(),
                ),
                RenderCommand::PopLayer => self.pop_layer(),
                RenderCommand::Stroke(cmd) => self.stroke_shape(
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
                    (&cmd.shape).into(),
                ),
                RenderCommand::Fill(cmd) => self.fill_shape(
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
                    (&cmd.shape).into(),
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
                    &cmd.box_shape,
                    cmd.offset,
                    cmd.spread,
                    cmd.std_dev,
                    cmd.color,
                    cmd.kind,
                ),
            }
        }
    }

    /// Utility method to draw an image at it's natural size. For more advanced image drawing use the `fill` method
    fn draw_image(&mut self, image: ImageBrushRef, transform: Affine) {
        let rect = Rect::new(
            0.0,
            0.0,
            image.image.width as f64,
            image.image.height as f64,
        );
        self.fill_shape(
            Fill::NonZero,
            transform,
            image.into(),
            None,
            PaintShape::Rect(&rect),
        );
    }
}
