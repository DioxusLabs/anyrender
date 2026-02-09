//! Write-side font processing: collection, deduplication, subsetting, and encoding.

use std::collections::{HashMap, HashSet};

use klippa::{Plan, SubsetFlags};
use peniko::FontData;
use read_fonts::FontRef;
use read_fonts::collections::int_set::IntSet;
use read_fonts::types::GlyphId;

use crate::{ArchiveError, ResourceId, SerializeConfig, sha256_hex};

/// A font that has been processed (optionally subsetted and/or WOFF2-encoded) and is
/// ready to be written into the archive.
pub(crate) struct ProcessedFont {
    /// Size of the raw (uncompressed) font data in bytes.
    pub raw_size: usize,
    /// The stored font data (WOFF2-compressed or raw TTF/OTF depending on config).
    pub stored_data: Vec<u8>,
    /// SHA-256 hex hash of `stored_data`.
    pub hash: String,
    /// Archive-relative path (e.g. `fonts/<hash>.woff2` or `fonts/<hash>.ttf`).
    pub path: String,
}

/// Collects, deduplicates, and processes fonts for writing into a scene archive.
///
/// When subsetting is enabled, each `(blob, face index)` pair is treated as a distinct
/// resource because subsetting extracts each face into a standalone font.
///
/// When disabled, fonts are deduplicated by blob alone. Multiple faces sharing the same TTC
/// are stored together.
pub(crate) struct FontWriter {
    config: SerializeConfig,
    /// Map `(Blob ID, face index)` to [`ResourceId`].
    /// When subsetting is disabled, the face in the `(Blob ID, face index)` tuple is always 0.
    /// This is because multiple faces sharing the same TTC should be keyed together.
    id_map: HashMap<(u64, u32), ResourceId>,
    fonts: Vec<FontData>,
    glyph_ids: Vec<HashSet<u32>>,
}

impl FontWriter {
    pub fn new(config: SerializeConfig) -> Self {
        Self {
            config,
            id_map: HashMap::new(),
            fonts: Vec::new(),
            glyph_ids: Vec::new(),
        }
    }

    /// Register a font and return its [`ResourceId`].
    pub fn register(&mut self, font: &FontData) -> ResourceId {
        let key = if self.config.subset_fonts {
            (font.data.id(), font.index)
        } else {
            // When subsetting is disabled, the face index is always 0 so that
            // multiple faces sharing the same TTC are keyed together.
            (font.data.id(), 0)
        };

        if let Some(&id) = self.id_map.get(&key) {
            return id;
        }

        let id = ResourceId(self.fonts.len());
        self.id_map.insert(key, id);
        self.fonts.push(font.clone());
        self.glyph_ids.push(HashSet::new());
        id
    }

    /// Record glyph IDs used for a font resource (used for subsetting).
    pub fn record_glyphs(&mut self, id: ResourceId, glyphs: &[anyrender::Glyph]) {
        if self.config.subset_fonts {
            let glyph_set = &mut self.glyph_ids[id.0];
            for glyph in glyphs {
                glyph_set.insert(glyph.id);
            }
        }
    }

    /// The face index to store in [`crate::FontResourceId`].
    ///
    /// When subsetting is enabled, faces are extracted into standalone fonts so the index
    /// is always 0. Otherwise the original face index is preserved.
    pub fn face_index(&self, font: &FontData) -> u32 {
        if self.config.subset_fonts {
            0
        } else {
            font.index
        }
    }

    /// Consume the writer, returning an iterator of processed fonts ready for the archive.
    pub fn into_processed(self) -> impl Iterator<Item = Result<ProcessedFont, ArchiveError>> {
        self.fonts.into_iter().enumerate().map(move |(idx, font)| {
            let raw_data = if self.config.subset_fonts {
                let glyph_ids = &self.glyph_ids[idx];

                let font_ref = FontRef::from_index(font.data.data(), font.index).map_err(|e| {
                    ArchiveError::FontProcessing(format!("Failed to parse font: {e}"))
                })?;

                let mut input_gids: IntSet<GlyphId> = IntSet::empty();
                for &gid in glyph_ids {
                    input_gids.insert(GlyphId::new(gid));
                }

                let plan = Plan::new(
                    &input_gids,
                    &IntSet::empty(),
                    &font_ref,
                    // Keep original glyph IDs so we don't need to remap them in draw commands.
                    SubsetFlags::SUBSET_FLAGS_RETAIN_GIDS,
                    &IntSet::empty(),
                    &IntSet::empty(),
                    &IntSet::empty(),
                    &IntSet::empty(),
                    &IntSet::empty(),
                );

                klippa::subset_font(&font_ref, &plan).map_err(|e| {
                    ArchiveError::FontProcessing(format!("Font subsetting failed: {e}"))
                })?
            } else {
                font.data.data().to_vec()
            };

            let raw_size = raw_data.len();

            // Conditionally WOFF2 compress.
            let stored_data = if self.config.woff2_fonts {
                ttf2woff2::encode_no_transform(&raw_data, ttf2woff2::BrotliQuality::default())
                    .map_err(|e| {
                        ArchiveError::FontProcessing(format!("WOFF2 encoding failed: {e}"))
                    })?
            } else {
                raw_data
            };

            let hash = sha256_hex(&stored_data);
            let extension = if self.config.woff2_fonts {
                "woff2"
            } else {
                "ttf"
            };
            let path = format!("fonts/{}.{}", hash, extension);

            Ok(ProcessedFont {
                raw_size,
                stored_data,
                hash,
                path,
            })
        })
    }
}
