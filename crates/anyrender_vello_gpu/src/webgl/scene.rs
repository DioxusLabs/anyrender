//! WebGL-compatible [`PaintScene`] implementation for [`vello_gpu::Scene`].

use anyrender::{
    AntiAliasing, Filter, Glyph, MaskMode, NormalizedCoord, Paint, PaintRef, PaintScene,
    RenderContext,
};
use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Compose, Fill, FontData, Mix, StyleRef};
use vello_common::paint::PaintType;

use crate::common::{DEFAULT_TOLERANCE, ImageCache, LayerKind};
use anyrender::ResourceId;
use peniko::ImageBrush;
use vello_common::TextureId;
use vello_common::paint::{ImageSource, TextureRegion};
use vello_gpu::{
    WebGlDataTransferOperation, WebGlError, WebGlOperation, WebGlResourceKind, WebGlTextureBindings,
};
use web_sys::WebGl2RenderingContext;

use std::sync::Arc;

/// The glyph caches of a WebGL renderer, with rasterized glyphs packed into one texture.
///
/// Call `clear` when the WebGL context it was used with is lost or replaced.
pub type WebGlGlyphAtlas = crate::common::GlyphAtlas<web_sys::WebGlTexture>;

pub struct WebGlImageManager<'a> {
    pub(crate) renderer: &'a mut vello_gpu::WebGlRenderer,
    pub(crate) cache: &'a mut ImageCache,
    pub(crate) texture_bindings: &'a mut WebGlTextureBindings,
    pub(crate) glyphs: &'a mut WebGlGlyphAtlas,
}

impl<'a> WebGlImageManager<'a> {
    pub fn new(
        renderer: &'a mut vello_gpu::WebGlRenderer,
        cache: &'a mut ImageCache,
        texture_bindings: &'a mut WebGlTextureBindings,
        glyphs: &'a mut WebGlGlyphAtlas,
    ) -> Self {
        Self {
            renderer,
            cache,
            texture_bindings,
            glyphs,
        }
    }

    /// Write the glyphs rasterized since the last call to the glyph atlas texture, creating
    /// and binding the texture first if there is none.
    fn upload_glyphs(&mut self) -> Result<(), WebGlError> {
        if !self.glyphs.has_uploads() {
            return Ok(());
        }
        let upload_error = |e: wasm_bindgen::JsValue| WebGlError::OperationFailed {
            operation: WebGlOperation::DataTransfer(WebGlDataTransferOperation::TextureUpload),
            message: e.as_string(),
        };
        let gl = self.renderer.gl_context();
        gl.active_texture(WebGl2RenderingContext::TEXTURE0);
        // The glyph pixels are already premultiplied.
        gl.pixel_storei(WebGl2RenderingContext::UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0);
        gl.pixel_storei(WebGl2RenderingContext::UNPACK_FLIP_Y_WEBGL, 0);
        if self.glyphs.texture().is_none() {
            let (width, height) = self.glyphs.texture_size();
            let texture = gl.create_texture().ok_or(WebGlError::OperationFailed {
                operation: WebGlOperation::ResourceCreation(WebGlResourceKind::Texture),
                message: None,
            })?;
            gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
            gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                WebGl2RenderingContext::RGBA8 as i32,
                i32::from(width),
                i32::from(height),
                0,
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                None,
            )
            .map_err(upload_error)?;
            for (parameter, value) in [
                (
                    WebGl2RenderingContext::TEXTURE_MIN_FILTER,
                    WebGl2RenderingContext::LINEAR,
                ),
                (
                    WebGl2RenderingContext::TEXTURE_MAG_FILTER,
                    WebGl2RenderingContext::LINEAR,
                ),
                (
                    WebGl2RenderingContext::TEXTURE_WRAP_S,
                    WebGl2RenderingContext::CLAMP_TO_EDGE,
                ),
                (
                    WebGl2RenderingContext::TEXTURE_WRAP_T,
                    WebGl2RenderingContext::CLAMP_TO_EDGE,
                ),
            ] {
                gl.tex_parameteri(WebGl2RenderingContext::TEXTURE_2D, parameter, value as i32);
            }
            self.texture_bindings
                .insert(self.glyphs.texture_id(), texture.clone(), width, height);
            self.glyphs.set_texture(texture);
        }
        gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, self.glyphs.texture());
        for upload in self.glyphs.take_uploads() {
            gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_u8_array(
                WebGl2RenderingContext::TEXTURE_2D,
                0,
                i32::from(upload.rect.x),
                i32::from(upload.rect.y),
                i32::from(upload.rect.width),
                i32::from(upload.rect.height),
                WebGl2RenderingContext::RGBA,
                WebGl2RenderingContext::UNSIGNED_BYTE,
                Some(&upload.data),
            )
            .map_err(upload_error)?;
        }
        Ok(())
    }

    /// Upload `image` to its own texture and bind it for rendering, returning
    /// its [`TextureId`] and transparency hint.
    pub(crate) fn upload_image(
        &mut self,
        image: &peniko::ImageData,
    ) -> Result<Option<(TextureId, bool)>, WebGlError> {
        upload_image(self.renderer, self.cache, self.texture_bindings, image)
    }

    /// Evict unused images after rendering a frame, including skipped frames.
    pub fn finish_frame(&mut self) -> Result<(), WebGlError> {
        for id in self.cache.finish_frame() {
            if let Some(texture) = self.texture_bindings.remove(id) {
                self.renderer.gl_context().delete_texture(Some(&texture));
            }
        }
        self.glyphs.finish_frame();
        Ok(())
    }
}

/// Upload `image` to its own texture and bind it for rendering, returning
/// its [`TextureId`] and transparency hint.
fn upload_image(
    renderer: &mut vello_gpu::WebGlRenderer,
    cache: &mut ImageCache,
    texture_bindings: &mut WebGlTextureBindings,
    image: &peniko::ImageData,
) -> Result<Option<(TextureId, bool)>, WebGlError> {
    if !cache.fits(image.width, image.height) {
        return Ok(None);
    }
    let peniko_id = image.data.id();

    if let Some(cached) = cache.get(peniko_id) {
        return Ok(Some(cached));
    }

    let ImageSource::Pixmap(pixmap) = ImageSource::from_peniko_image_data(image) else {
        unreachable!();
    };
    let may_have_transparency = pixmap.may_have_transparency();

    let gl = renderer.gl_context();
    let texture = gl.create_texture().ok_or(WebGlError::OperationFailed {
        operation: WebGlOperation::ResourceCreation(WebGlResourceKind::Texture),
        message: None,
    })?;
    gl.active_texture(WebGl2RenderingContext::TEXTURE0);
    gl.bind_texture(WebGl2RenderingContext::TEXTURE_2D, Some(&texture));
    // The pixmap data is already premultiplied.
    gl.pixel_storei(WebGl2RenderingContext::UNPACK_PREMULTIPLY_ALPHA_WEBGL, 0);
    gl.pixel_storei(WebGl2RenderingContext::UNPACK_FLIP_Y_WEBGL, 0);
    gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_u8_array(
        WebGl2RenderingContext::TEXTURE_2D,
        0,
        WebGl2RenderingContext::RGBA8 as i32,
        i32::from(pixmap.width()),
        i32::from(pixmap.height()),
        0,
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(pixmap.data_as_u8_slice()),
    )
    .map_err(|e| WebGlError::OperationFailed {
        operation: WebGlOperation::DataTransfer(WebGlDataTransferOperation::TextureUpload),
        message: e.as_string(),
    })?;
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_MIN_FILTER,
        WebGl2RenderingContext::LINEAR as i32,
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_MAG_FILTER,
        WebGl2RenderingContext::LINEAR as i32,
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_WRAP_S,
        WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
    );
    gl.tex_parameteri(
        WebGl2RenderingContext::TEXTURE_2D,
        WebGl2RenderingContext::TEXTURE_WRAP_T,
        WebGl2RenderingContext::CLAMP_TO_EDGE as i32,
    );

    let texture_id = TextureId(ResourceId::new().into_ffi());
    texture_bindings.insert(texture_id, texture, pixmap.width(), pixmap.height());
    cache.insert(
        &image.data,
        texture_id,
        may_have_transparency,
        image.width,
        image.height,
    );
    Ok(Some((texture_id, may_have_transparency)))
}

pub struct WebGlScenePainter<'s> {
    scene: &'s mut vello_gpu::Scene,
    layer_stack: Vec<LayerKind>,
    image_manager: WebGlImageManager<'s>,
    glyph_caching: bool,
}

impl<'s> WebGlScenePainter<'s> {
    fn cull_rect(&self) -> Option<Rect> {
        Some(Rect::new(
            0.0,
            0.0,
            f64::from(self.scene.width()),
            f64::from(self.scene.height()),
        ))
    }

    pub fn new(scene: &'s mut vello_gpu::Scene, image_manager: WebGlImageManager<'s>) -> Self {
        Self {
            scene,
            layer_stack: Vec::with_capacity(16),
            image_manager,
            glyph_caching: true,
        }
    }

    /// Enable or disable caching of rasterized glyphs in the glyph atlas.
    ///
    /// Defaults to `true`. Turning it off is meant for debugging and tests: every outline and
    /// COLR glyph is then drawn as paths, and a bitmap glyph from a texture of its own.
    pub fn with_glyph_caching(mut self, enabled: bool) -> Self {
        self.glyph_caching = enabled;
        self
    }
}

impl WebGlScenePainter<'_> {
    fn convert_paint(&mut self, paint: PaintRef<'_>) -> PaintType {
        match paint {
            Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
            Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
            Paint::Image(image_brush) => self.convert_image_paint(image_brush),

            // TODO: custom paint
            Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
            Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
            // Patterns are drawn with `fill_pattern_tiles`
            Paint::Pattern(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        }
    }

    fn convert_image_paint(&mut self, image_brush: peniko::ImageBrushRef<'_>) -> PaintType {
        let Ok(Some((texture_id, may_have_transparency))) =
            self.image_manager.upload_image(image_brush.image)
        else {
            return PaintType::Solid(Color::TRANSPARENT);
        };
        PaintType::Image(ImageBrush {
            image: ImageSource::external_texture(
                texture_id,
                TextureRegion::Full,
                may_have_transparency,
            ),
            sampler: image_brush.sampler,
        })
    }
}

impl RenderContext for WebGlScenePainter<'_> {}
impl PaintScene for WebGlScenePainter<'_> {
    fn reset(&mut self) {
        self.scene.reset();
    }

    fn push_layer(
        &mut self,
        fill: Fill,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        filter: Option<Arc<Filter>>,
        _backdrop_filter: Option<Arc<Filter>>,
    ) {
        let (filter, region) = match filter.as_deref().and_then(crate::common::convert_filter) {
            Some(converted) => (
                Some(converted),
                filter.and_then(|f| Some(f.nodes().first()?.region)),
            ),
            None => (None, None),
        };
        self.scene.set_transform(transform);
        if let Some(region) = &region {
            self.scene.push_clip_rect(region);
        }
        self.scene.set_fill_rule(fill);
        if let Some(rect) = clip.as_rect() {
            self.scene.push_clip_rect(&rect);
        } else {
            self.scene
                .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
        }
        self.scene
            .push_layer(None, Some(blend.into()), Some(alpha), None, filter);
        match &region {
            Some(region) => {
                self.scene.push_clip_rect(region);
                self.layer_stack.push(LayerKind::FilterRegionLayer);
            }
            None => self.layer_stack.push(LayerKind::Layer),
        }
    }

    fn push_mask_layer(
        &mut self,
        fill: Fill,
        _mode: MaskMode,
        transform: Affine,
        clip: &impl Shape,
    ) {
        let blend = BlendMode::new(Mix::Normal, Compose::DestIn);

        // Everything outside of the clip is masked out
        let everything = Rect::new(-1e6, -1e6, 1e6, 1e6);
        self.push_layer(
            Fill::NonZero,
            blend,
            1.0,
            Affine::IDENTITY,
            &everything,
            None,
            None,
        );
        self.fill(
            fill,
            transform,
            Color::BLACK,
            None,
            clip,
            AntiAliasing::Enabled,
        );
        self.pop_layer();

        // Luminance masks aren't supported yet, so they are treated as alpha masks
        self.push_layer(fill, blend, 1.0, transform, clip, None, None);
    }

    fn push_clip_layer(&mut self, fill: Fill, transform: Affine, clip: &impl Shape) {
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(fill);
        self.layer_stack.push(LayerKind::Clip);
        if let Some(rect) = clip.as_rect() {
            self.scene.push_clip_rect(&rect);
        } else {
            self.scene
                .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
        }
        // Pattern antialiasing applies to this clip, not the tile content.
        self.scene.set_aliasing_threshold(None);
    }

    fn pop_layer(&mut self) {
        if let Some(kind) = self.layer_stack.pop() {
            match kind {
                LayerKind::Layer => {
                    self.scene.pop_layer();
                    self.scene.pop_clip();
                }
                LayerKind::Clip => self.scene.pop_clip(),
                LayerKind::FilterRegionLayer => {
                    self.scene.pop_clip();
                    self.scene.pop_layer();
                    self.scene.pop_clip();
                    self.scene.pop_clip();
                }
            }
        }
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
        anti_aliasing: AntiAliasing,
    ) {
        let paint: anyrender::PaintRef<'a> = paint.into();
        if let anyrender::Paint::Pattern(pattern) = paint {
            let cull_rect = self.cull_rect();
            self.scene
                .set_aliasing_threshold((!anti_aliasing.is_enabled()).then_some(127));
            anyrender::stroke_pattern_tiles(
                self,
                pattern,
                style,
                transform,
                brush_transform,
                shape,
                cull_rect,
            );
            self.scene.set_aliasing_threshold(None);
            return;
        }
        self.scene.set_transform(transform);
        self.scene.set_stroke(style.clone());
        self.scene
            .set_aliasing_threshold((!anti_aliasing.is_enabled()).then_some(127));
        let paint = self.convert_paint(paint);
        self.scene.set_paint(paint);
        self.scene
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.scene.stroke_path(&shape.into_path(DEFAULT_TOLERANCE));
        self.scene.set_aliasing_threshold(None);
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
        anti_aliasing: AntiAliasing,
    ) {
        let paint: anyrender::PaintRef<'a> = paint.into();
        if let anyrender::Paint::Pattern(pattern) = paint {
            let cull_rect = self.cull_rect();
            self.scene
                .set_aliasing_threshold((!anti_aliasing.is_enabled()).then_some(127));
            anyrender::fill_pattern_tiles(
                self,
                pattern,
                style,
                transform,
                brush_transform,
                shape,
                cull_rect,
            );
            self.scene.set_aliasing_threshold(None);
            return;
        }
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(style);
        self.scene
            .set_aliasing_threshold((!anti_aliasing.is_enabled()).then_some(127));
        let paint = self.convert_paint(paint);
        self.scene.set_paint(paint);
        self.scene
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        if let Some(rect) = shape.as_rect() {
            self.scene.fill_rect(&rect);
        } else {
            self.scene.fill_path(&shape.into_path(DEFAULT_TOLERANCE));
        }
        self.scene.set_aliasing_threshold(None);
    }

    fn draw_glyphs<'a, 's2: 'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
        anti_aliasing: AntiAliasing,
    ) {
        let paint: PaintRef<'a> = paint.into();
        if let anyrender::Paint::Pattern(pattern) = paint {
            let cull_rect = self.cull_rect();
            anyrender::draw_glyphs_pattern_tiles(
                self,
                pattern,
                font,
                font_size,
                hint,
                normalized_coords,
                embolden,
                style,
                brush_alpha,
                transform,
                glyph_transform,
                glyphs,
                cull_rect,
            );
            return;
        }
        let paint = self.convert_paint(paint);
        self.scene.set_paint(paint);
        let WebGlImageManager {
            renderer,
            cache,
            texture_bindings,
            glyphs: glyph_atlas,
        } = &mut self.image_manager;
        glyph_atlas.draw_glyphs(
            self.scene,
            &mut |image| {
                upload_image(renderer, cache, texture_bindings, image)
                    .ok()
                    .flatten()
            },
            self.glyph_caching,
            font,
            font_size,
            hint,
            normalized_coords,
            embolden,
            style.into(),
            transform,
            glyph_transform,
            glyphs.map(|g| anyrender_glyph::Glyph {
                id: g.id,
                x: g.x,
                y: g.y,
            }),
            anti_aliasing,
        );
        if let Err(error) = self.image_manager.upload_glyphs() {
            // The glyphs' atlas regions stay transparent.
            web_sys::console::warn_1(&format!("failed to upload glyphs: {error:?}").into());
        }
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        box_shape: &anyrender::NonUniformRoundedRect,
        offset: kurbo::Vec2,
        spread: f64,
        std_dev: f64,
        color: Color,
        kind: anyrender::BoxShadowKind,
    ) {
        let geometry = anyrender::BoxShadowGeometry::new(box_shape, offset, spread, std_dev, kind);
        self.scene.set_transform(transform);
        self.scene.set_paint(PaintType::Solid(color));
        self.scene.reset_paint_transform();
        self.scene.set_fill_rule(Fill::NonZero);
        if geometry.std_dev == 0.0 {
            // A single fill, so a non-isolated clip anti-aliases the clip edge only once.
            let clip = geometry.needs_clip();
            if clip {
                self.scene.push_clip_path(&geometry.area);
            }
            self.scene.fill_path(&geometry.unblurred_path());
            if clip {
                self.scene.pop_clip();
            }
            return;
        }
        // TODO: draw shadows with matching individual radii instead of averaging them
        self.scene.fill_blurred_rounded_rect_in(
            &geometry.area,
            &geometry.shadow.rect,
            geometry.shadow.average_radius() as f32,
            geometry.std_dev as f32,
            geometry.is_inset(),
        );
    }
}
