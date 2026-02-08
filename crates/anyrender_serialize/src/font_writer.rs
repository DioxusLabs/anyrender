//! Write-side font processing: collection, deduplication, subsetting, and encoding.

use std::collections::HashMap;
#[cfg(feature = "subsetting")]
use std::collections::HashSet;

#[cfg(feature = "subsetting")]
use klippa::{Plan, SubsetFlags};
use peniko::FontData;
#[cfg(feature = "subsetting")]
use read_fonts::FontRef;
#[cfg(feature = "subsetting")]
use read_fonts::collections::int_set::IntSet;
#[cfg(feature = "subsetting")]
use read_fonts::types::GlyphId;
use sha2::{Digest, Sha256};

use crate::{ArchiveError, ResourceId};

/// A font that has been processed (optionally subsetted and/or WOFF2-encoded) and is
/// ready to be written into the archive.
pub(crate) struct ProcessedFont {
    /// Size of the raw (uncompressed) font data in bytes.
    pub raw_size: usize,
    /// The stored font data (WOFF2-compressed or raw TTF/OTF depending on features).
    pub stored_data: Vec<u8>,
    /// SHA-256 hex hash of `stored_data`.
    pub hash: String,
    /// Archive-relative path (e.g. `fonts/<hash>.woff2` or `fonts/<hash>.ttf`).
    pub path: String,
}

/// Collects, deduplicates, and processes fonts for writing into a scene archive.
///
/// When the `subsetting` feature is enabled, each `(blob, face index)` pair is treated
/// as a distinct resource because subsetting extracts each face into a standalone font.
///
/// When disabled, fonts are deduplicated by blob alone. Multiple faces sharing the same TTC
/// are stored together.
pub(crate) struct FontWriter {
    /// Map `(Blob ID, face index)` to [`ResourceId`].
    #[cfg(feature = "subsetting")]
    id_map: HashMap<(u64, u32), ResourceId>,
    /// Map `Blob ID` to [`ResourceId`].
    #[cfg(not(feature = "subsetting"))]
    id_map: HashMap<u64, ResourceId>,

    fonts: Vec<FontData>,

    #[cfg(feature = "subsetting")]
    glyph_ids: Vec<HashSet<u32>>,
}

impl FontWriter {
    pub fn new() -> Self {
        Self {
            id_map: HashMap::new(),
            fonts: Vec::new(),
            #[cfg(feature = "subsetting")]
            glyph_ids: Vec::new(),
        }
    }

    /// Register a font and return its [`ResourceId`].
    pub fn register(&mut self, font: &FontData) -> ResourceId {
        #[cfg(feature = "subsetting")]
        let key = (font.data.id(), font.index);
        #[cfg(not(feature = "subsetting"))]
        let key = font.data.id();

        if let Some(&id) = self.id_map.get(&key) {
            return id;
        }

        let id = ResourceId(self.fonts.len());
        self.id_map.insert(key, id);
        self.fonts.push(font.clone());
        #[cfg(feature = "subsetting")]
        self.glyph_ids.push(HashSet::new());
        id
    }

    /// Record glyph IDs used for a font resource (used for subsetting).
    pub fn record_glyphs(&mut self, id: ResourceId, glyphs: &[anyrender::Glyph]) {
        #[cfg(feature = "subsetting")]
        {
            let glyph_set = &mut self.glyph_ids[id.0];
            for glyph in glyphs {
                glyph_set.insert(glyph.id);
            }
        }
        #[cfg(not(feature = "subsetting"))]
        {
            let _ = (id, glyphs);
        }
    }

    /// The face index to store in [`crate::FontResourceId`].
    ///
    /// When subsetting is enabled, faces are extracted into standalone fonts so the index
    /// is always 0. Otherwise the original face index is preserved.
    pub fn face_index(&self, font: &FontData) -> u32 {
        #[cfg(feature = "subsetting")]
        {
            let _ = font;
            0
        }
        #[cfg(not(feature = "subsetting"))]
        {
            font.index
        }
    }

    /// Consume the writer, returning an iterator of processed fonts ready for the archive.
    pub fn into_processed(self) -> impl Iterator<Item = Result<ProcessedFont, ArchiveError>> {
        #[cfg(feature = "subsetting")]
        let glyph_ids = self.glyph_ids;

        self.fonts.into_iter().enumerate().map(move |(_idx, font)| {
            // Conditionally subset.
            #[cfg(feature = "subsetting")]
            let raw_data = {
                let font_glyph_ids = &glyph_ids[_idx];

                let font_ref = FontRef::from_index(font.data.data(), font.index).map_err(|e| {
                    ArchiveError::FontProcessing(format!("Failed to parse font: {e}"))
                })?;

                let mut input_gids: IntSet<GlyphId> = IntSet::empty();
                for &gid in font_glyph_ids {
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
            };
            #[cfg(not(feature = "subsetting"))]
            let raw_data = font.data.data().to_vec();

            let raw_size = raw_data.len();

            // Conditionally WOFF2 compress.
            #[cfg(feature = "woff2")]
            let stored_data =
                ttf2woff2::encode_no_transform(&raw_data, ttf2woff2::BrotliQuality::default())
                    .map_err(|e| {
                        ArchiveError::FontProcessing(format!("WOFF2 encoding failed: {e}"))
                    })?;
            #[cfg(not(feature = "woff2"))]
            let stored_data = raw_data;

            let hash = sha256_hex(&stored_data);
            let extension = if cfg!(feature = "woff2") {
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

fn sha256_hex(data: &[u8]) -> String {
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
