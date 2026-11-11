//! WebGL-compatible [`PaintScene`] implementation for [`vello_gpu::Scene`].

use anyrender::{Filter, Glyph, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, StyleRef};
use vello_common::paint::PaintType;

use crate::common::{DEFAULT_TOLERANCE, ImageCache, LayerKind, supports_glyph_caching};
use anyrender::ResourceId;
use peniko::ImageBrush;
use vello_common::TextureId;
use vello_common::paint::{ImageSource, TextureRegion};
use vello_gpu::{
    WebGlDataTransferOperation, WebGlError, WebGlOperation, WebGlResourceKind, WebGlTextureBindings,
};
use web_sys::WebGl2RenderingContext;

use std::sync::Arc;

pub struct WebGlImageManager<'a> {
    pub(crate) renderer: &'a mut vello_gpu::WebGlRenderer,
    pub(crate) resources: &'a mut vello_gpu::Resources,
    pub(crate) cache: &'a mut ImageCache,
    pub(crate) texture_bindings: &'a mut WebGlTextureBindings,
}

impl<'a> WebGlImageManager<'a> {
    pub fn new(
        renderer: &'a mut vello_gpu::WebGlRenderer,
        resources: &'a mut vello_gpu::Resources,
        cache: &'a mut ImageCache,
        texture_bindings: &'a mut WebGlTextureBindings,
    ) -> Self {
        Self {
            renderer,
            resources,
            cache,
            texture_bindings,
        }
    }

    /// Upload `image` to its own texture and bind it for rendering, returning
    /// its [`TextureId`] and transparency hint.
    pub(crate) fn upload_image(
        &mut self,
        image: &peniko::ImageData,
    ) -> Result<Option<(TextureId, bool)>, WebGlError> {
        if !self.cache.fits(image.width, image.height) {
            return Ok(None);
        }
        let peniko_id = image.data.id();

        if let Some(cached) = self.cache.get(peniko_id) {
            return Ok(Some(cached));
        }

        let ImageSource::Pixmap(pixmap) = ImageSource::from_peniko_image_data(image) else {
            unreachable!();
        };
        let may_have_transparency = pixmap.may_have_transparency();

        let gl = self.renderer.gl_context();
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
        self.texture_bindings
            .insert(texture_id, texture, pixmap.width(), pixmap.height());
        self.cache.insert(
            &image.data,
            texture_id,
            may_have_transparency,
            image.width,
            image.height,
        );
        Ok(Some((texture_id, may_have_transparency)))
    }

    /// Evict unused images after rendering a frame, including skipped frames.
    pub fn finish_frame(&mut self) -> Result<(), WebGlError> {
        for id in self.cache.finish_frame() {
            if let Some(texture) = self.texture_bindings.remove(id) {
                self.renderer.gl_context().delete_texture(Some(&texture));
            }
        }
        Ok(())
    }
}

pub struct WebGlScenePainter<'s> {
    scene: &'s mut vello_gpu::Scene,
    layer_stack: Vec<LayerKind>,
    image_manager: WebGlImageManager<'s>,
    glyph_caching: bool,
}

impl<'s> WebGlScenePainter<'s> {
    pub fn new(scene: &'s mut vello_gpu::Scene, image_manager: WebGlImageManager<'s>) -> Self {
        Self {
            scene,
            layer_stack: Vec::with_capacity(16),
            image_manager,
            glyph_caching: false,
        }
    }

    /// Enable or disable caching of rasterized glyphs in Vello's glyph atlas.
    ///
    /// Defaults to `false`.
    ///
    /// Note: Vello considers atlas-backed glyph caching experimental.
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
        let filter = filter.and_then(crate::common::convert_filter);
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(fill);
        self.layer_stack.push(LayerKind::Layer);
        if let Some(rect) = clip.as_rect() {
            self.scene.push_clip_rect(&rect);
        } else {
            self.scene
                .push_clip_path(&clip.into_path(DEFAULT_TOLERANCE));
        }
        self.scene
            .push_layer(None, Some(blend.into()), Some(alpha), None, filter);
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
    }

    fn pop_layer(&mut self) {
        if let Some(kind) = self.layer_stack.pop() {
            match kind {
                LayerKind::Layer => {
                    self.scene.pop_layer();
                    self.scene.pop_clip();
                }
                LayerKind::Clip => self.scene.pop_clip(),
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
    ) {
        self.scene.set_transform(transform);
        self.scene.set_stroke(style.clone());
        let paint = self.convert_paint(paint.into());
        self.scene.set_paint(paint);
        self.scene
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.scene.stroke_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.scene.set_transform(transform);
        self.scene.set_fill_rule(style);
        let paint = self.convert_paint(paint.into());
        self.scene.set_paint(paint);
        self.scene
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        if let Some(rect) = shape.as_rect() {
            self.scene.fill_rect(&rect);
        } else {
            self.scene.fill_path(&shape.into_path(DEFAULT_TOLERANCE));
        }
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
        _brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
    ) {
        let paint = self.convert_paint(paint.into());
        self.scene.set_paint(paint);
        self.scene.set_transform(transform);

        let glyph_caching =
            self.glyph_caching && supports_glyph_caching(transform, glyph_transform);
        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.scene.set_fill_rule(fill);
                let _ = self
                    .scene
                    .glyph_run(self.image_manager.resources, font)
                    .atlas_cache(glyph_caching)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)))
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(|g| glifo::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
            StyleRef::Stroke(stroke) => {
                self.scene.set_stroke(stroke.clone());
                let _ = self
                    .scene
                    .glyph_run(self.image_manager.resources, font)
                    .atlas_cache(glyph_caching)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(|g| glifo::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
        }
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        color: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.scene.set_transform(transform);
        self.scene.set_paint(PaintType::Solid(color));
        self.scene.reset_paint_transform();
        self.scene
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32);
    }
}
