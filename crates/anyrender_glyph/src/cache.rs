// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A raster cache that stores one image per glyph.

use crate::key::GlyphCacheKey;
use crate::raster::RasterMetrics;
use alloc::vec::Vec;
use hashbrown::HashMap;

/// Counters describing how a [`GlyphRasterCache`] has been used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlyphRasterCacheStats {
    /// Lookups that found an image.
    pub hits: u64,
    /// Lookups that found nothing.
    pub misses: u64,
    /// Images stored.
    pub insertions: u64,
    /// Images dropped to make room.
    pub evictions: u64,
    /// Images that were not stored because no room could be made.
    pub failed_insertions: u64,
}

#[derive(Debug)]
struct Entry<I> {
    image: I,
    metrics: RasterMetrics,
    bytes: usize,
    last_used: u64,
}

/// A raster cache that stores one image of type `I` per glyph, within a byte budget.
///
/// This is the storage for renderers that draw a cached glyph from its own image, such as CPU
/// renderers. (Renderers that pack glyphs into a shared texture need a texture atlas instead.) The
/// cache knows nothing about `I`: the renderer rasterizes the glyph, says how many bytes the
/// image occupies, and gets the image back on later lookups.
///
/// When an insertion would exceed the budget, the least recently used images that were not
/// used in the current frame are dropped. Call [`begin_frame`](Self::begin_frame) once per
/// frame for that to work.
#[derive(Debug)]
pub struct GlyphRasterCache<I> {
    entries: HashMap<GlyphCacheKey, Entry<I>>,
    max_bytes: usize,
    bytes: usize,
    frame: u64,
    stats: GlyphRasterCacheStats,
    /// Scratch space for choosing eviction candidates.
    ages: Vec<u64>,
}

impl<I> Default for GlyphRasterCache<I> {
    /// A cache with a budget of 16 MiB.
    fn default() -> Self {
        Self::new(16 * 1024 * 1024)
    }
}

impl<I> GlyphRasterCache<I> {
    /// Creates an empty cache that holds at most `max_bytes` of images.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_bytes,
            bytes: 0,
            frame: 0,
            stats: GlyphRasterCacheStats::default(),
            ages: Vec::new(),
        }
    }

    /// The byte budget.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// The number of bytes of images currently stored, as reported on insertion.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The number of images stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no images.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The usage counters.
    pub fn stats(&self) -> GlyphRasterCacheStats {
        self.stats
    }

    /// Starts a new frame. Images used before this call become eligible for eviction.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Looks up a glyph, marking it as used in the current frame.
    pub fn get(&mut self, key: &GlyphCacheKey) -> Option<(&I, RasterMetrics)> {
        match self.entries.get_mut(key) {
            Some(entry) => {
                entry.last_used = self.frame;
                self.stats.hits += 1;
                Some((&entry.image, entry.metrics))
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Stores an image that occupies `bytes` bytes, replacing any image under the same key.
    ///
    /// Returns the image back if it does not fit in the budget even after evicting every image
    /// that was not used in the current frame.
    pub fn insert(
        &mut self,
        key: GlyphCacheKey,
        image: I,
        metrics: RasterMetrics,
        bytes: usize,
    ) -> Result<(), I> {
        self.remove(&key);
        while self.bytes + bytes > self.max_bytes {
            if bytes > self.max_bytes || !self.evict_batch() {
                self.stats.failed_insertions += 1;
                return Err(image);
            }
        }
        self.bytes += bytes;
        self.stats.insertions += 1;
        self.entries.insert(
            key,
            Entry {
                image,
                metrics,
                bytes,
                last_used: self.frame,
            },
        );
        Ok(())
    }

    /// Removes a glyph, returning its image.
    pub fn remove(&mut self, key: &GlyphCacheKey) -> Option<I> {
        let entry = self.entries.remove(key)?;
        self.bytes -= entry.bytes;
        Some(entry.image)
    }

    /// Removes every image.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    /// Drops every image that has not been used in the last `max_idle_frames` frames.
    pub fn evict_idle(&mut self, max_idle_frames: u64) {
        let cutoff = self.frame.saturating_sub(max_idle_frames);
        self.evict_older_than(cutoff);
    }

    /// Evicts the least recently used quarter of the images that were not used in the current
    /// frame (at least one). Returns `false` if there are none.
    fn evict_batch(&mut self) -> bool {
        self.ages.clear();
        self.ages.extend(
            self.entries
                .values()
                .map(|entry| entry.last_used)
                .filter(|&last_used| last_used < self.frame),
        );
        if self.ages.is_empty() {
            return false;
        }
        let batch_len = (self.ages.len() / 4).max(1);
        let (_, &mut cutoff, _) = self.ages.select_nth_unstable(batch_len - 1);
        self.evict_older_than((cutoff + 1).min(self.frame));
        true
    }

    /// Drops every image last used before frame `cutoff`.
    fn evict_older_than(&mut self, cutoff: u64) {
        let before = self.entries.len();
        let mut freed = 0;
        self.entries.retain(|_, entry| {
            let keep = entry.last_used >= cutoff;
            if !keep {
                freed += entry.bytes;
            }
            keep
        });
        self.bytes -= freed;
        self.stats.evictions += (before - self.entries.len()) as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glyph::FontEmbolden;
    use crate::peniko::color::palette::css::BLACK;

    const METRICS: RasterMetrics = RasterMetrics {
        width: 4,
        height: 4,
        bearing_x: 0,
        bearing_y: 0,
    };

    fn key(glyph_id: u32) -> GlyphCacheKey {
        GlyphCacheKey::new(
            1,
            0,
            glyph_id,
            16.0,
            false,
            0.0,
            BLACK,
            0,
            FontEmbolden::default(),
            &[],
        )
    }

    #[test]
    fn insert_then_get() {
        let mut cache = GlyphRasterCache::new(100);
        assert!(cache.get(&key(1)).is_none());
        cache.insert(key(1), "a", METRICS, 64).unwrap();
        assert_eq!(cache.get(&key(1)), Some((&"a", METRICS)));
        assert_eq!((cache.len(), cache.bytes()), (1, 64));
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.insertions), (1, 1, 1));
    }

    #[test]
    fn evicts_least_recently_used_first() {
        let mut cache = GlyphRasterCache::new(4 * 64);
        for id in 0..4 {
            cache.insert(key(id), id, METRICS, 64).unwrap();
            cache.begin_frame();
        }
        // Glyph 0 is the oldest, but it is used again.
        cache.get(&key(0));
        cache.begin_frame();
        cache.insert(key(4), 4, METRICS, 64).unwrap();
        assert!(cache.get(&key(1)).is_none());
        for id in [0, 2, 3, 4] {
            assert!(cache.get(&key(id)).is_some(), "glyph {id} should be kept");
        }
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn never_evicts_images_used_this_frame() {
        let mut cache = GlyphRasterCache::new(2 * 64);
        cache.insert(key(0), 0, METRICS, 64).unwrap();
        cache.insert(key(1), 1, METRICS, 64).unwrap();
        assert_eq!(cache.insert(key(2), 2, METRICS, 64), Err(2));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().failed_insertions, 1);
    }

    #[test]
    fn rejects_images_larger_than_the_budget() {
        let mut cache = GlyphRasterCache::new(64);
        cache.insert(key(0), 0, METRICS, 64).unwrap();
        cache.begin_frame();
        assert_eq!(cache.insert(key(1), 1, METRICS, 65), Err(1));
        assert_eq!(
            cache.len(),
            1,
            "nothing is evicted for an image that can never fit"
        );
    }

    #[test]
    fn replacing_a_key_frees_the_old_image() {
        let mut cache = GlyphRasterCache::new(64);
        cache.insert(key(0), 0, METRICS, 64).unwrap();
        cache.insert(key(0), 1, METRICS, 32).unwrap();
        assert_eq!(cache.bytes(), 32);
        assert_eq!(cache.get(&key(0)), Some((&1, METRICS)));
    }

    #[test]
    fn evict_idle_and_clear() {
        let mut cache = GlyphRasterCache::new(1024);
        cache.insert(key(0), 0, METRICS, 64).unwrap();
        cache.begin_frame();
        cache.insert(key(1), 1, METRICS, 64).unwrap();
        cache.begin_frame();
        cache.evict_idle(1);
        assert!(cache.get(&key(0)).is_none());
        assert!(cache.get(&key(1)).is_some());
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn keys_differ_by_variation_coordinates() {
        let mut a = key(0);
        let mut b = key(0);
        a.var_coords
            .push(skrifa::instance::NormalizedCoord::from_bits(1));
        b.var_coords
            .push(skrifa::instance::NormalizedCoord::from_bits(2));
        let mut cache = GlyphRasterCache::new(1024);
        cache.insert(a.clone(), "a", METRICS, 64).unwrap();
        assert!(cache.get(&b).is_none());
        assert!(cache.get(&a).is_some());
    }
}
