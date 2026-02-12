//! Serialization of recorded scenes to a portable zip archive format.
//!
//! # Archive Format
//!
//! The serialized scene is a zip archive containing:
//!
//! - `resources.json` - Metadata mapping resource files to IDs
//! - `draw_commands.json` - Serialized draw commands referencing resources by ID
//! - `images/<sha256_hash>.png` - Image files (PNG format)
//! - `fonts/<sha256_hash>.{woff2,ttf}` - Font data files (optionally WOFF2-compressed and subsetted)

use std::io::{Read, Seek, Write};

use image::{ImageBuffer, ImageEncoder, RgbaImage};
use peniko::{Blob, Brush, FontData, ImageAlphaType, ImageBrush, ImageData, ImageFormat};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;
use zip::{ZipArchive, ZipWriter};

use anyrender::recording::{FillCommand, GlyphRunCommand, RenderCommand, Scene, StrokeCommand};
use anyrender::{ImageResource, RecordingRenderContext, RenderContext, ResourceId};

mod font_writer;
mod json_formatter;

use font_writer::FontWriter;

/// A render command with resources replaced by IDs.
pub type SerializableRenderCommand = RenderCommand<SerializedFontResourceId, SerializedResourceId>;

/// A brush with images replaced by IDs.
pub type SerializableBrush = Brush<ImageBrush<SerializedResourceId>>;

/// A unique identifier for a serialized resource.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SerializedResourceId(pub usize);

/// A reference to a font in a serialized scene.
///
/// Pairs a [`ResourceId`] (which identifies the font file) with a collection index
/// (which identifies a specific face).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializedFontResourceId {
    pub resource_id: SerializedResourceId,
    pub index: u32,
}

/// A scene archive that can be serialized to/from a zip file.
#[derive(Clone)]
pub struct SceneArchive {
    pub manifest: ResourceManifest,
    pub commands: Vec<SerializableRenderCommand>,
    /// Font data (one per font resource, optionally WOFF2-compressed and/or subsetted).
    pub fonts: Vec<Blob<u8>>,
    pub images: Vec<ImageData>,
}

/// The resources manifest stored in the archive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceManifest {
    /// Version of the archive format
    pub version: u32,
    /// Scene tolerance (used for path flattening)
    pub tolerance: f64,
    pub images: Vec<ImageMetadata>,
    pub fonts: Vec<FontMetadata>,
}

impl ResourceManifest {
    /// Current archive format version. Bump this when the format changes.
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(tolerance: f64) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            tolerance,
            images: Vec::new(),
            fonts: Vec::new(),
        }
    }
}

/// Metadata for an image resource.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageMetadata {
    #[serde(flatten)]
    pub entry: ResourceEntry,
    /// The original image format. Images are stored as RGBA8 in the archive
    /// and converted back to the original format on reconstruction.
    pub format: ImageFormat,
    pub alpha_type: ImageAlphaType,
    pub width: u32,
    pub height: u32,
}

/// Metadata for a font resource.
///
/// When WOFF2 is enabled via [`SerializeConfig`], fonts are WOFF2-compressed.
/// When subsetting is enabled, TTC fonts are extracted to standalone fonts and
/// subsetted to only the glyphs used.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FontMetadata {
    #[serde(flatten)]
    pub entry: ResourceEntry,
}

/// Metadata for a resource in the archive.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResourceEntry {
    pub id: SerializedResourceId,
    pub kind: ResourceKind,
    /// The size of the raw decompressed resource data in bytes.
    pub size: usize,
    /// SHA-256 hash of the resource's raw content
    pub sha256_hash: String,
    /// Path to the resource file within the archive
    pub path: String,
}

/// The type of resource stored in the archive.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    Image,
    Font,
}

/// Collects and deduplicates resources from a scene.
struct ResourceCollector {
    fonts: FontWriter,
    /// Maps ResourceId to SerializedResourceId for images
    image_id_map: FxHashMap<ResourceId, SerializedResourceId>,
    /// Collected images
    images: Vec<ImageData>,
}

impl ResourceCollector {
    fn new(config: SerializeConfig) -> Self {
        Self {
            fonts: FontWriter::new(config),
            image_id_map: FxHashMap::default(),
            images: Vec::new(),
        }
    }

    /// Register an image and return its [`SerializedResourceId`].
    fn register_image(
        &mut self,
        ctx: &RecordingRenderContext,
        resource: &ImageResource,
    ) -> SerializedResourceId {
        if let Some(&id) = self.image_id_map.get(&resource.id) {
            return id;
        }

        let image_data = ctx.image_data().get(&resource.id).unwrap();
        let id = SerializedResourceId(self.images.len());
        self.image_id_map.insert(resource.id, id);
        self.images.push(image_data.clone());
        id
    }

    /// Convert a [`Brush<ImageBrush<ImageResource>>`] to a [`SerializableBrush`] by registering images.
    fn convert_brush(
        &mut self,
        ctx: &RecordingRenderContext,
        brush: &Brush<ImageBrush<ImageResource>>,
    ) -> SerializableBrush {
        match brush {
            Brush::Solid(color) => Brush::Solid(*color),
            Brush::Gradient(gradient) => Brush::Gradient(gradient.clone()),
            Brush::Image(image_brush) => {
                let id = self.register_image(ctx, &image_brush.image);
                Brush::Image(ImageBrush {
                    image: id,
                    sampler: image_brush.sampler,
                })
            }
        }
    }

    /// Convert a [`RenderCommand`] to a [`SerializableRenderCommand`].
    fn convert_command(
        &mut self,
        ctx: &RecordingRenderContext,
        cmd: &RenderCommand,
    ) -> SerializableRenderCommand {
        match cmd {
            RenderCommand::PushLayer(layer) => SerializableRenderCommand::PushLayer(layer.clone()),
            RenderCommand::PushClipLayer(clip) => {
                SerializableRenderCommand::PushClipLayer(clip.clone())
            }
            RenderCommand::PopLayer => SerializableRenderCommand::PopLayer,
            RenderCommand::Stroke(stroke) => SerializableRenderCommand::Stroke(StrokeCommand {
                style: stroke.style.clone(),
                transform: stroke.transform,
                brush: self.convert_brush(ctx, &stroke.brush),
                brush_transform: stroke.brush_transform,
                shape: stroke.shape.clone(),
            }),
            RenderCommand::Fill(fill) => SerializableRenderCommand::Fill(FillCommand {
                fill: fill.fill,
                transform: fill.transform,
                brush: self.convert_brush(ctx, &fill.brush),
                brush_transform: fill.brush_transform,
                shape: fill.shape.clone(),
            }),
            RenderCommand::GlyphRun(glyph_run) => {
                let resource_id = self.fonts.register(&glyph_run.font_data);
                self.fonts.record_glyphs(resource_id, &glyph_run.glyphs);
                let brush = self.convert_brush(ctx, &glyph_run.brush);
                SerializableRenderCommand::GlyphRun(GlyphRunCommand {
                    font_data: SerializedFontResourceId {
                        resource_id,
                        index: self.fonts.face_index(&glyph_run.font_data),
                    },
                    font_size: glyph_run.font_size,
                    hint: glyph_run.hint,
                    normalized_coords: glyph_run.normalized_coords.clone(),
                    style: glyph_run.style.clone(),
                    brush,
                    brush_alpha: glyph_run.brush_alpha,
                    transform: glyph_run.transform,
                    glyph_transform: glyph_run.glyph_transform,
                    glyphs: glyph_run.glyphs.clone(),
                })
            }
            RenderCommand::BoxShadow(shadow) => {
                SerializableRenderCommand::BoxShadow(shadow.clone())
            }
        }
    }
}

/// Reconstructs resources from deserialized data.
struct ResourceReconstructor {
    fonts: Vec<FontData>,
    image_resources: Vec<ImageResource>,
}

impl ResourceReconstructor {
    fn new(ctx: &mut impl RenderContext, fonts: Vec<FontData>, images: Vec<ImageData>) -> Self {
        let image_resources: Vec<ImageResource> = images
            .into_iter()
            .map(|image| ctx.register_image(image))
            .collect();

        Self {
            fonts,
            image_resources,
        }
    }

    fn get_font(&self, id: SerializedResourceId) -> Result<&FontData, ArchiveError> {
        self.fonts
            .get(id.0)
            .ok_or(ArchiveError::ResourceNotFound(id))
    }

    fn get_image_resource(&self, id: SerializedResourceId) -> Result<ImageResource, ArchiveError> {
        self.image_resources
            .get(id.0)
            .copied()
            .ok_or(ArchiveError::ResourceNotFound(id))
    }

    /// Convert a [`SerializableBrush`] back to a [`Brush<ImageBrush<ImageResource>>`].
    fn convert_brush(
        &self,
        brush: &SerializableBrush,
    ) -> Result<Brush<ImageBrush<ImageResource>>, ArchiveError> {
        Ok(match brush {
            Brush::Solid(color) => Brush::Solid(*color),
            Brush::Gradient(gradient) => Brush::Gradient(gradient.clone()),
            Brush::Image(image_brush) => {
                let resource = self.get_image_resource(image_brush.image)?;
                Brush::Image(ImageBrush {
                    image: resource,
                    sampler: image_brush.sampler,
                })
            }
        })
    }

    /// Convert a [`SerializableRenderCommand`] back to a [`RenderCommand`].
    fn convert_command(
        &self,
        cmd: &SerializableRenderCommand,
    ) -> Result<RenderCommand, ArchiveError> {
        Ok(match cmd {
            SerializableRenderCommand::PushLayer(layer) => RenderCommand::PushLayer(layer.clone()),
            SerializableRenderCommand::PushClipLayer(clip) => {
                RenderCommand::PushClipLayer(clip.clone())
            }
            SerializableRenderCommand::PopLayer => RenderCommand::PopLayer,
            SerializableRenderCommand::Stroke(stroke) => RenderCommand::Stroke(StrokeCommand {
                style: stroke.style.clone(),
                transform: stroke.transform,
                brush: self.convert_brush(&stroke.brush)?,
                brush_transform: stroke.brush_transform,
                shape: stroke.shape.clone(),
            }),
            SerializableRenderCommand::Fill(fill) => RenderCommand::Fill(FillCommand {
                fill: fill.fill,
                transform: fill.transform,
                brush: self.convert_brush(&fill.brush)?,
                brush_transform: fill.brush_transform,
                shape: fill.shape.clone(),
            }),
            SerializableRenderCommand::GlyphRun(glyph_run) => {
                let font = self.get_font(glyph_run.font_data.resource_id)?;
                let font_data = FontData::new(font.data.clone(), glyph_run.font_data.index);
                let brush = self.convert_brush(&glyph_run.brush)?;
                RenderCommand::GlyphRun(GlyphRunCommand {
                    font_data,
                    font_size: glyph_run.font_size,
                    hint: glyph_run.hint,
                    normalized_coords: glyph_run.normalized_coords.clone(),
                    style: glyph_run.style.clone(),
                    brush,
                    brush_alpha: glyph_run.brush_alpha,
                    transform: glyph_run.transform,
                    glyph_transform: glyph_run.glyph_transform,
                    glyphs: glyph_run.glyphs.clone(),
                })
            }
            SerializableRenderCommand::BoxShadow(shadow) => {
                RenderCommand::BoxShadow(shadow.clone())
            }
        })
    }
}

pub(crate) fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex_encode(&result)
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut hex = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        hex.push(HEX_CHARS[(byte >> 4) as usize] as char);
        hex.push(HEX_CHARS[(byte & 0xf) as usize] as char);
    }
    hex
}

fn encode_rgba_to_png(rgba_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, ArchiveError> {
    let img: RgbaImage = ImageBuffer::from_raw(width, height, rgba_data.to_vec())
        .ok_or_else(|| ArchiveError::InvalidFormat("Failed to create image buffer".to_string()))?;

    let mut png_data = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut png_data);
    encoder.write_image(&img, width, height, image::ExtendedColorType::Rgba8)?;

    Ok(png_data)
}

fn decode_png_to_rgba(png_data: &[u8]) -> Result<Vec<u8>, ArchiveError> {
    let img = image::load_from_memory_with_format(png_data, image::ImageFormat::Png)?;
    Ok(img.into_rgba8().into_raw())
}

/// Convert RGBA8 data to the target [`ImageFormat`].
fn convert_from_rgba(rgba_blob: &Blob<u8>, target: ImageFormat) -> Result<Blob<u8>, ArchiveError> {
    match target {
        ImageFormat::Rgba8 => Ok(rgba_blob.clone()),
        ImageFormat::Bgra8 => {
            // Swap R and B channels
            let mut bgra = rgba_blob.data().to_vec();
            for chunk in bgra.chunks_exact_mut(4) {
                chunk.swap(0, 2);
            }
            Ok(Blob::from(bgra))
        }
        other => Err(ArchiveError::InvalidFormat(format!(
            "Unsupported image format: {other:?}"
        ))),
    }
}

/// Convert [`ImageData`] to RGBA8 format.
fn convert_to_rgba(image: &ImageData) -> Result<Blob<u8>, ArchiveError> {
    match image.format {
        ImageFormat::Rgba8 => Ok(image.data.clone()),
        ImageFormat::Bgra8 => {
            // Swap B and R channels
            let mut rgba = image.data.data().to_vec();
            for chunk in rgba.chunks_exact_mut(4) {
                chunk.swap(0, 2);
            }
            Ok(Blob::from(rgba))
        }
        // ImageFormat is non_exhaustive, so simply error out if we encounter an unknown format
        other => Err(ArchiveError::InvalidFormat(format!(
            "Unsupported image format: {other:?}"
        ))),
    }
}

impl SceneArchive {
    /// Create a new SceneArchive from a recorded Scene.
    pub fn from_scene(
        ctx: &RecordingRenderContext,
        scene: &Scene,
        config: &SerializeConfig,
    ) -> Result<Self, ArchiveError> {
        let mut manifest = ResourceManifest::new(scene.tolerance);
        let mut collector = ResourceCollector::new(config.clone());

        let commands: Vec<_> = scene
            .commands
            .iter()
            .map(|cmd| collector.convert_command(ctx, cmd))
            .collect();

        // Normalize all images to RGBA8
        let images: Vec<ImageData> = collector
            .images
            .iter()
            .map(|image| {
                let data = convert_to_rgba(image)?;
                Ok(ImageData {
                    data,
                    format: ImageFormat::Rgba8,
                    alpha_type: image.alpha_type,
                    width: image.width,
                    height: image.height,
                })
            })
            .collect::<Result<Vec<_>, ArchiveError>>()?;

        // Add image metadata
        for (idx, (original, normalized)) in collector.images.iter().zip(images.iter()).enumerate()
        {
            let data = normalized.data.data();
            let hash = sha256_hex(data);
            let path = format!("images/{}.png", hash);

            manifest.images.push(ImageMetadata {
                entry: ResourceEntry {
                    id: SerializedResourceId(idx),
                    kind: ResourceKind::Image,
                    size: data.len(),
                    sha256_hash: hash,
                    path,
                },
                format: original.format,
                alpha_type: original.alpha_type,
                width: original.width,
                height: original.height,
            });
        }

        // Add font metadata.
        let mut fonts = Vec::new();
        for (idx, result) in collector.fonts.into_processed().enumerate() {
            let font = result?;
            manifest.fonts.push(FontMetadata {
                entry: ResourceEntry {
                    id: SerializedResourceId(idx),
                    kind: ResourceKind::Font,
                    size: font.raw_size,
                    sha256_hash: font.hash,
                    path: font.path,
                },
            });
            fonts.push(Blob::from(font.stored_data));
        }

        Ok(Self {
            manifest,
            commands,
            fonts,
            images,
        })
    }

    /// Convert this archive back to a Scene.
    pub fn to_scene(&self, ctx: &mut impl RenderContext) -> Result<Scene, ArchiveError> {
        // Convert images back to their original format
        let images: Vec<ImageData> = self
            .images
            .iter()
            .zip(self.manifest.images.iter())
            .map(|(image, meta)| {
                let data = convert_from_rgba(&image.data, meta.format)?;
                Ok(ImageData {
                    data,
                    format: meta.format,
                    alpha_type: image.alpha_type,
                    width: image.width,
                    height: image.height,
                })
            })
            .collect::<Result<Vec<_>, ArchiveError>>()?;

        // Decode fonts.
        let fonts_ttf: Vec<FontData> = self
            .fonts
            .iter()
            .map(|font_blob| {
                let data = font_blob.data();
                let ttf_data = if data.starts_with(b"wOF2") {
                    wuff::decompress_woff2(data).map_err(|e| {
                        ArchiveError::FontProcessing(format!("WOFF2 decoding failed: {e}"))
                    })?
                } else {
                    data.to_vec()
                };
                Ok(FontData::new(Blob::from(ttf_data), 0))
            })
            .collect::<Result<Vec<_>, ArchiveError>>()?;

        let reconstructor = ResourceReconstructor::new(ctx, fonts_ttf, images);

        let commands: Result<Vec<_>, _> = self
            .commands
            .iter()
            .map(|cmd| reconstructor.convert_command(cmd))
            .collect();

        Ok(Scene {
            tolerance: self.manifest.tolerance,
            commands: commands?,
        })
    }

    /// Serialize the archive to a zip file.
    pub fn serialize<W: Write + Seek>(&self, writer: W) -> Result<(), ArchiveError> {
        let mut zip = ZipWriter::new(writer);
        let options = SimpleFileOptions::default();

        // Write resources.json
        {
            zip.start_file("resources.json", options)?;
            let manifest_json = serde_json::to_string_pretty(&self.manifest)?;
            zip.write_all(manifest_json.as_bytes())?;
        }

        // Write draw_commands.json
        {
            zip.start_file("draw_commands.json", options)?;
            let commands_json = json_formatter::to_json_depth_limited(&self.commands, 3)?;
            zip.write_all(commands_json.as_bytes())?;
        }

        // Write image files as PNG
        for (idx, image) in self.images.iter().enumerate() {
            let path = &self.manifest.images[idx].entry.path;
            let png_data = encode_rgba_to_png(image.data.data(), image.width, image.height)?;
            zip.start_file(path, options)?;
            zip.write_all(&png_data)?;
        }

        // Write font files
        for (idx, font_data) in self.fonts.iter().enumerate() {
            let path = &self.manifest.fonts[idx].entry.path;
            zip.start_file(path, options)?;
            zip.write_all(font_data.data())?;
        }

        zip.finish()?;
        Ok(())
    }

    /// Deserialize an archive from a zip file.
    pub fn deserialize<R: Read + Seek>(reader: R) -> Result<Self, ArchiveError> {
        let mut zip = ZipArchive::new(reader)?;

        // Read resources.json
        let manifest: ResourceManifest = {
            let mut file = zip.by_name("resources.json")?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            serde_json::from_str(&contents)?
        };

        // Check version
        if manifest.version != ResourceManifest::CURRENT_VERSION {
            return Err(ArchiveError::UnsupportedVersion(manifest.version));
        }

        // Read draw_commands.json
        let commands: Vec<SerializableRenderCommand> = {
            let mut file = zip.by_name("draw_commands.json")?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            serde_json::from_str(&contents)?
        };

        // Read images
        let mut images = Vec::with_capacity(manifest.images.len());
        for meta in &manifest.images {
            let mut file = zip.by_name(&meta.entry.path)?;
            let mut png_data = Vec::new();
            file.read_to_end(&mut png_data)?;
            let rgba_data = decode_png_to_rgba(&png_data)?;

            // Verify hash
            let hash = sha256_hex(&rgba_data);
            if hash != meta.entry.sha256_hash {
                return Err(ArchiveError::InvalidFormat(format!(
                    "Hash mismatch for {}: expected {}, got {}",
                    meta.entry.path, meta.entry.sha256_hash, hash
                )));
            }

            images.push(ImageData {
                data: Blob::from(rgba_data),
                format: ImageFormat::Rgba8,
                alpha_type: meta.alpha_type,
                width: meta.width,
                height: meta.height,
            });
        }

        // Read fonts (may be WOFF2-compressed or raw TTF/OTF)
        let mut fonts: Vec<Blob<u8>> = Vec::with_capacity(manifest.fonts.len());
        for meta in &manifest.fonts {
            let mut file = zip.by_name(&meta.entry.path)?;
            let mut raw_data = Vec::new();
            file.read_to_end(&mut raw_data)?;

            // Verify hash
            let hash = sha256_hex(&raw_data);
            if hash != meta.entry.sha256_hash {
                return Err(ArchiveError::InvalidFormat(format!(
                    "Hash mismatch for {}: expected {}, got {}",
                    meta.entry.path, meta.entry.sha256_hash, hash
                )));
            }
            fonts.push(Blob::from(raw_data));
        }

        Ok(Self {
            manifest,
            commands,
            fonts,
            images,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct SerializeConfig {
    subset_fonts: bool,
    woff2_fonts: bool,
}

impl SerializeConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Subset fonts to only include glyphs used in the scene.
    pub fn with_subset_fonts(mut self, subset_fonts: bool) -> Self {
        self.subset_fonts = subset_fonts;
        self
    }

    /// WOFF2-compress font data.
    pub fn with_woff2_fonts(mut self, woff2_fonts: bool) -> Self {
        self.woff2_fonts = woff2_fonts;
        self
    }
}

#[derive(Debug)]
pub enum ArchiveError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Zip(zip::result::ZipError),
    Image(image::ImageError),
    FontProcessing(String),
    InvalidFormat(String),
    ResourceNotFound(SerializedResourceId),
    UnsupportedVersion(u32),
}

impl std::fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchiveError::Io(e) => write!(f, "IO error: {}", e),
            ArchiveError::Json(e) => write!(f, "JSON error: {}", e),
            ArchiveError::Zip(e) => write!(f, "Zip error: {}", e),
            ArchiveError::Image(e) => write!(f, "Image error: {}", e),
            ArchiveError::FontProcessing(msg) => write!(f, "Font processing error: {}", msg),
            ArchiveError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ArchiveError::ResourceNotFound(id) => write!(f, "Resource not found: {:?}", id),
            ArchiveError::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
        }
    }
}

impl std::error::Error for ArchiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ArchiveError::Io(e) => Some(e),
            ArchiveError::Json(e) => Some(e),
            ArchiveError::Zip(e) => Some(e),
            ArchiveError::Image(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ArchiveError {
    fn from(e: std::io::Error) -> Self {
        ArchiveError::Io(e)
    }
}

impl From<serde_json::Error> for ArchiveError {
    fn from(e: serde_json::Error) -> Self {
        ArchiveError::Json(e)
    }
}

impl From<zip::result::ZipError> for ArchiveError {
    fn from(e: zip::result::ZipError) -> Self {
        ArchiveError::Zip(e)
    }
}

impl From<image::ImageError> for ArchiveError {
    fn from(e: image::ImageError) -> Self {
        ArchiveError::Image(e)
    }
}
