//! A texture atlas: many small images packed into one large one.
//!
//! [`Atlas`] decides *where* images go and which ones to drop when space runs out. It does not
//! hold any pixels itself: they are written through the [`AtlasStorage`] trait, which a renderer
//! implements for its own texture type (a GPU texture, a CPU pixmap, ...).
//!
//! - Packing uses [`etagere`]'s shelf allocator.
//! - Every entry is surrounded by a transparent gutter ([`AtlasConfig::padding`]), so that
//!   filtered sampling at an entry's edge never picks up a neighbour's pixels.
//! - The atlas has a fixed size. When it is full, the least recently used entries that were not
//!   used in the current frame are evicted. If there is still no room, [`Atlas::insert`] returns
//!   `None` and the caller is expected to draw without the atlas.
//!
//! The atlas is generic over its key and over a per-entry payload, so it knows nothing about
//! what the images are.
//!
//! ```
//! use anyrender_atlas::{Atlas, AtlasConfig, AtlasStorage, PixelRect};
//! use pixmap::Pixmap;
//!
//! struct Texture(Pixmap);
//! impl AtlasStorage for Texture {
//!     fn write_pixels(&mut self, rect: PixelRect, data: &[u8]) {
//!         // Copy `data` into `self.0` at `rect`, or upload it to a GPU texture.
//! #       let _ = (rect, data);
//!     }
//! }
//!
//! let config = AtlasConfig::default();
//! let mut texture = Texture(Pixmap::new(config.width, config.height));
//! let mut atlas: Atlas<&str> = Atlas::new(config);
//!
//! // Once per frame.
//! atlas.begin_frame();
//!
//! let image = Pixmap::new(8, 8);
//! let rect = match atlas.get(&"key") {
//!     Some(entry) => entry.rect,
//!     None => atlas.insert(&mut texture, "key", &image, ()).expect("atlas is full"),
//! };
//! assert_eq!((rect.width, rect.height), (8, 8));
//! ```

use std::hash::Hash;

// hashbrown's default hasher is much cheaper than std's SipHash, and lookups happen once or
// more per drawn item.
use hashbrown::HashMap;

use etagere::{AllocId, AtlasAllocator, size2};
use pixmap::{BYTES_PER_PIXEL, Pixmap};

/// The size and layout of an [`Atlas`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AtlasConfig {
    /// The width of the atlas in pixels.
    pub width: u16,
    /// The height of the atlas in pixels.
    pub height: u16,
    /// The width in pixels of the transparent gutter on each side of every entry.
    pub padding: u16,
}

impl Default for AtlasConfig {
    /// A 2048x2048 atlas with a one pixel gutter.
    fn default() -> Self {
        Self {
            width: 2048,
            height: 2048,
            padding: 1,
        }
    }
}

/// A rectangle of pixels in an atlas page.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PixelRect {
    /// The page the rectangle is in. Always 0 while atlases have a single page.
    pub page: u16,
    /// The left edge in pixels.
    pub x: u16,
    /// The top edge in pixels.
    pub y: u16,
    /// The width in pixels.
    pub width: u16,
    /// The height in pixels.
    pub height: u16,
}

/// Where an atlas's pixels live.
pub trait AtlasStorage {
    /// Replaces the pixels in `rect` with `data`.
    ///
    /// `data` is tightly packed rows of premultiplied RGBA8 pixels, exactly as large as `rect`.
    fn write_pixels(&mut self, rect: PixelRect, data: &[u8]);
}

/// An entry found by [`Atlas::get`].
#[derive(Copy, Clone, Debug)]
pub struct AtlasEntry<'a, T> {
    /// Where the image is in the atlas, excluding its gutter.
    pub rect: PixelRect,
    /// The payload given to [`Atlas::insert`].
    pub data: &'a T,
}

/// Counters describing how an [`Atlas`] is being used.
///
/// The counters accumulate until [`Atlas::reset_stats`] is called.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AtlasStats {
    /// [`Atlas::get`] calls that found an entry.
    pub hits: u64,
    /// [`Atlas::get`] calls that found nothing.
    pub misses: u64,
    /// Images inserted.
    pub insertions: u64,
    /// Entries evicted to make room.
    pub evictions: u64,
    /// [`Atlas::insert`] calls that failed because there was no room.
    pub failed_insertions: u64,
}

struct Slot<T> {
    alloc: AllocId,
    rect: PixelRect,
    last_used: u64,
    data: T,
}

/// A fixed-size texture atlas keyed by `K`, with a payload of type `T` per entry.
///
/// See the [crate documentation](crate) for an overview.
pub struct Atlas<K, T = ()> {
    config: AtlasConfig,
    allocator: AtlasAllocator,
    slots: HashMap<K, Slot<T>>,
    frame: u64,
    stats: AtlasStats,
    /// Reused buffer for an image plus its gutter.
    scratch: Vec<u8>,
    /// Reused buffer for the `last_used` frames of eviction candidates.
    ages: Vec<u64>,
}

impl<K: Hash + Eq, T> Atlas<K, T> {
    /// Creates an empty atlas.
    pub fn new(config: AtlasConfig) -> Self {
        Self {
            config,
            allocator: AtlasAllocator::new(size2(config.width as i32, config.height as i32)),
            slots: HashMap::new(),
            frame: 0,
            stats: AtlasStats::default(),
            scratch: Vec::new(),
            ages: Vec::new(),
        }
    }

    /// The configuration the atlas was created with.
    pub fn config(&self) -> AtlasConfig {
        self.config
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the atlas has no entries.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The number of pixels allocated to entries, including their gutters.
    pub fn allocated_pixels(&self) -> u64 {
        self.allocator.allocated_space() as u64
    }

    /// Usage counters since the atlas was created or [`reset_stats`](Self::reset_stats) was
    /// called.
    pub fn stats(&self) -> AtlasStats {
        self.stats
    }

    /// Resets the usage counters to zero.
    pub fn reset_stats(&mut self) {
        self.stats = AtlasStats::default();
    }

    /// Starts a new frame.
    ///
    /// Entries looked up or inserted since the last call are never evicted, so everything a
    /// frame draws stays valid until the next frame begins.
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Looks up an entry and marks it as used in the current frame.
    pub fn get(&mut self, key: &K) -> Option<AtlasEntry<'_, T>> {
        match self.slots.get_mut(key) {
            Some(slot) => {
                self.stats.hits += 1;
                slot.last_used = self.frame;
                Some(AtlasEntry {
                    rect: slot.rect,
                    data: &slot.data,
                })
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Adds an image to the atlas and writes its pixels and gutter to `storage`.
    ///
    /// Returns where the image was placed, excluding its gutter, or `None` if no room could be
    /// made for it. An existing entry with the same key is replaced.
    pub fn insert(
        &mut self,
        storage: &mut impl AtlasStorage,
        key: K,
        image: &Pixmap,
        data: T,
    ) -> Option<PixelRect> {
        self.remove(&key);

        let padding = self.config.padding as u32;
        let padded_width = image.width() as u32 + 2 * padding;
        let padded_height = image.height() as u32 + 2 * padding;
        let fits = image.width() > 0
            && image.height() > 0
            && padded_width <= self.config.width as u32
            && padded_height <= self.config.height as u32;
        let Some(allocation) = fits
            .then(|| self.allocate(size2(padded_width as i32, padded_height as i32)))
            .flatten()
        else {
            self.stats.failed_insertions += 1;
            return None;
        };

        // etagere may round the allocation up; only the requested size is used.
        let padded = PixelRect {
            page: 0,
            x: allocation.rectangle.min.x as u16,
            y: allocation.rectangle.min.y as u16,
            width: padded_width as u16,
            height: padded_height as u16,
        };
        let rect = PixelRect {
            page: 0,
            x: padded.x + self.config.padding,
            y: padded.y + self.config.padding,
            width: image.width(),
            height: image.height(),
        };

        if padding == 0 {
            storage.write_pixels(rect, image.data());
        } else {
            self.write_padded(storage, padded, image);
        }

        self.slots.insert(
            key,
            Slot {
                alloc: allocation.id,
                rect,
                last_used: self.frame,
                data,
            },
        );
        self.stats.insertions += 1;
        Some(rect)
    }

    /// Removes an entry, returning its payload.
    ///
    /// The entry's pixels are left in place; they are overwritten when the space is reused.
    pub fn remove(&mut self, key: &K) -> Option<T> {
        let slot = self.slots.remove(key)?;
        self.allocator.deallocate(slot.alloc);
        Some(slot.data)
    }

    /// Removes every entry, for example after the storage's contents were lost.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.allocator.clear();
    }

    /// Removes entries that have not been used for more than `max_idle_frames` frames.
    pub fn evict_idle(&mut self, max_idle_frames: u64) {
        let Some(cutoff) = self.frame.checked_sub(max_idle_frames) else {
            return;
        };
        self.evict_older_than(cutoff);
    }

    /// Allocates space, evicting least recently used entries as needed.
    fn allocate(&mut self, size: etagere::Size) -> Option<etagere::Allocation> {
        loop {
            if let Some(allocation) = self.allocator.allocate(size) {
                return Some(allocation);
            }

            // Evict the oldest quarter of the entries not used this frame, then retry. Evicting
            // in batches keeps a full atlas from evicting on every insertion.
            self.ages.clear();
            self.ages.extend(
                self.slots
                    .values()
                    .map(|slot| slot.last_used)
                    .filter(|&last_used| last_used < self.frame),
            );
            if self.ages.is_empty() {
                return None;
            }
            let batch_len = (self.ages.len() / 4).max(1);
            let (_, &mut cutoff, _) = self.ages.select_nth_unstable(batch_len - 1);
            self.evict_older_than(cutoff + 1);
        }
    }

    /// Removes entries last used before frame `cutoff`.
    fn evict_older_than(&mut self, cutoff: u64) {
        let allocator = &mut self.allocator;
        let stats = &mut self.stats;
        self.slots.retain(|_, slot| {
            let keep = slot.last_used >= cutoff;
            if !keep {
                allocator.deallocate(slot.alloc);
                stats.evictions += 1;
            }
            keep
        });
    }

    /// Writes `image` surrounded by a transparent gutter into `padded`.
    fn write_padded(&mut self, storage: &mut impl AtlasStorage, padded: PixelRect, image: &Pixmap) {
        let bpp = BYTES_PER_PIXEL;
        let padding = self.config.padding as usize;
        let src_stride = image.width() as usize * bpp;
        let dst_stride = padded.width as usize * bpp;

        self.scratch.clear();
        self.scratch.resize(dst_stride * padded.height as usize, 0);
        for (row, src) in image.data().chunks_exact(src_stride).enumerate() {
            let start = (row + padding) * dst_stride + padding * bpp;
            self.scratch[start..start + src_stride].copy_from_slice(src);
        }
        storage.write_pixels(padded, &self.scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Storage backed by a pixmap, recording nothing but the pixels.
    struct PixmapStorage(Pixmap);

    impl AtlasStorage for PixmapStorage {
        fn write_pixels(&mut self, rect: PixelRect, data: &[u8]) {
            let bpp = BYTES_PER_PIXEL;
            let stride = self.0.width() as usize * bpp;
            let row_len = rect.width as usize * bpp;
            assert_eq!(data.len(), row_len * rect.height as usize);
            for (row, src) in data.chunks_exact(row_len).enumerate() {
                let start = (rect.y as usize + row) * stride + rect.x as usize * bpp;
                self.0.data_mut()[start..start + row_len].copy_from_slice(src);
            }
        }
    }

    fn setup(size: u16, padding: u16) -> (Atlas<u32, u8>, PixmapStorage) {
        let config = AtlasConfig {
            width: size,
            height: size,
            padding,
        };
        (Atlas::new(config), PixmapStorage(Pixmap::new(size, size)))
    }

    fn solid(width: u16, height: u16, value: u8) -> Pixmap {
        let len = width as usize * height as usize * BYTES_PER_PIXEL;
        Pixmap::from_parts(vec![value; len], width, height)
    }

    fn pixel(storage: &PixmapStorage, x: u16, y: u16) -> u8 {
        storage.0.data()[(y as usize * storage.0.width() as usize + x as usize) * 4]
    }

    fn overlaps(a: PixelRect, b: PixelRect) -> bool {
        a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
    }

    #[test]
    fn insert_then_get() {
        let (mut atlas, mut storage) = setup(64, 1);
        atlas.begin_frame();
        assert!(atlas.get(&1).is_none());
        let rect = atlas
            .insert(&mut storage, 1, &solid(8, 6, 255), 42)
            .unwrap();
        assert_eq!((rect.width, rect.height), (8, 6));

        let entry = atlas.get(&1).unwrap();
        assert_eq!(entry.rect, rect);
        assert_eq!(*entry.data, 42);
        assert_eq!(
            atlas.stats(),
            AtlasStats {
                hits: 1,
                misses: 1,
                insertions: 1,
                ..Default::default()
            }
        );
    }

    #[test]
    fn entries_do_not_overlap_and_keep_a_gutter() {
        let (mut atlas, mut storage) = setup(64, 1);
        atlas.begin_frame();
        let rects: Vec<_> = (0..12)
            .map(|key| {
                atlas
                    .insert(&mut storage, key, &solid(9, 7, 255), 0)
                    .unwrap()
            })
            .collect();

        for (i, &a) in rects.iter().enumerate() {
            let grown = PixelRect {
                x: a.x - 1,
                y: a.y - 1,
                width: a.width + 2,
                height: a.height + 2,
                ..a
            };
            for &b in &rects[i + 1..] {
                assert!(!overlaps(grown, b), "{a:?} is too close to {b:?}");
            }
            // Content is written, and the gutter around it is transparent.
            assert_eq!(pixel(&storage, a.x, a.y), 255);
            assert_eq!(pixel(&storage, a.x + a.width - 1, a.y + a.height - 1), 255);
            assert_eq!(pixel(&storage, a.x - 1, a.y), 0);
            assert_eq!(pixel(&storage, a.x, a.y - 1), 0);
            assert_eq!(pixel(&storage, a.x + a.width, a.y), 0);
            assert_eq!(pixel(&storage, a.x, a.y + a.height), 0);
        }
    }

    #[test]
    fn reused_space_gets_a_clean_gutter() {
        let (mut atlas, mut storage) = setup(16, 1);
        atlas.begin_frame();
        atlas
            .insert(&mut storage, 1, &solid(14, 14, 255), 0)
            .unwrap();
        atlas.begin_frame();
        let rect = atlas.insert(&mut storage, 2, &solid(4, 4, 200), 0).unwrap();

        assert!(atlas.get(&1).is_none());
        assert_eq!(pixel(&storage, rect.x, rect.y), 200);
        assert_eq!(pixel(&storage, rect.x - 1, rect.y), 0);
        assert_eq!(pixel(&storage, rect.x + rect.width, rect.y), 0);
        assert_eq!(pixel(&storage, rect.x, rect.y + rect.height), 0);
    }

    #[test]
    fn evicts_least_recently_used_first() {
        let (mut atlas, mut storage) = setup(32, 0);
        // Four 16x16 images fill the atlas.
        for key in 0..4 {
            atlas.begin_frame();
            atlas
                .insert(&mut storage, key, &solid(16, 16, 1), 0)
                .unwrap();
        }
        // Touch the oldest entry so that key 1 becomes the least recently used.
        atlas.begin_frame();
        assert!(atlas.get(&0).is_some());

        atlas.begin_frame();
        atlas.insert(&mut storage, 4, &solid(16, 16, 1), 0).unwrap();
        assert_eq!(atlas.stats().evictions, 1);
        assert!(atlas.get(&1).is_none());
        for key in [0, 2, 3, 4] {
            assert!(atlas.get(&key).is_some(), "key {key} was evicted");
        }
    }

    #[test]
    fn never_evicts_entries_used_this_frame() {
        let (mut atlas, mut storage) = setup(32, 0);
        atlas.begin_frame();
        for key in 0..4 {
            atlas
                .insert(&mut storage, key, &solid(16, 16, 1), 0)
                .unwrap();
        }
        assert!(
            atlas
                .insert(&mut storage, 4, &solid(16, 16, 1), 0)
                .is_none()
        );
        assert_eq!(atlas.len(), 4);
        assert_eq!(atlas.stats().failed_insertions, 1);
        assert_eq!(atlas.stats().evictions, 0);
    }

    #[test]
    fn rejects_images_that_cannot_fit() {
        let (mut atlas, mut storage) = setup(32, 1);
        atlas.begin_frame();
        assert!(atlas.insert(&mut storage, 1, &solid(31, 4, 1), 0).is_none());
        assert!(atlas.insert(&mut storage, 2, &solid(0, 4, 1), 0).is_none());
        assert!(
            atlas
                .insert(&mut storage, 3, &solid(30, 30, 1), 0)
                .is_some()
        );
    }

    #[test]
    fn replacing_a_key_frees_the_old_space() {
        let (mut atlas, mut storage) = setup(32, 0);
        atlas.begin_frame();
        atlas.insert(&mut storage, 1, &solid(32, 32, 1), 1).unwrap();
        atlas.insert(&mut storage, 1, &solid(32, 32, 2), 2).unwrap();
        assert_eq!(atlas.len(), 1);
        assert_eq!(*atlas.get(&1).unwrap().data, 2);
    }

    #[test]
    fn evict_idle_and_clear() {
        let (mut atlas, mut storage) = setup(64, 1);
        atlas.begin_frame();
        atlas.insert(&mut storage, 1, &solid(8, 8, 1), 0).unwrap();
        for _ in 0..3 {
            atlas.begin_frame();
        }
        atlas.insert(&mut storage, 2, &solid(8, 8, 1), 0).unwrap();

        atlas.evict_idle(2);
        assert!(atlas.get(&1).is_none());
        assert!(atlas.get(&2).is_some());

        atlas.clear();
        assert!(atlas.is_empty());
        assert_eq!(atlas.allocated_pixels(), 0);
    }
}
