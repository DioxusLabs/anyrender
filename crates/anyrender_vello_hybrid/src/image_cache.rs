use peniko::{Blob, WeakBlob};
use rustc_hash::FxHashMap;
use vello_common::paint::ImageId;

const UNUSED_IMAGE_BUDGET: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
struct CachedImage {
    blob: WeakBlob<u8>,
    atlas_id: ImageId,
    bytes: u64,
    last_used: u64,
}

/// Persistent image residency shared by successive scene painters.
#[derive(Debug, Default)]
pub struct ImageCache {
    entries: FxHashMap<u64, CachedImage>,
    frame: u64,
}

impl ImageCache {
    /// Discard identities when their renderer resources are replaced.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.frame = 0;
    }

    pub(crate) fn get(&mut self, id: u64) -> Option<ImageId> {
        let entry = self.entries.get_mut(&id)?;
        entry.last_used = self.frame;
        Some(entry.atlas_id)
    }

    pub(crate) fn insert(&mut self, blob: &Blob<u8>, atlas_id: ImageId, width: u32, height: u32) {
        self.entries.insert(
            blob.id(),
            CachedImage {
                blob: blob.downgrade(),
                atlas_id,
                bytes: u64::from(width) * u64::from(height) * 4,
                last_used: self.frame,
            },
        );
    }

    pub(crate) fn finish_frame(&mut self) -> Vec<ImageId> {
        self.evict(UNUSED_IMAGE_BUDGET)
    }

    fn evict(&mut self, budget: u64) -> Vec<ImageId> {
        let mut evicted = Vec::new();
        self.entries.retain(|_, entry| {
            if entry.blob.upgrade().is_none() {
                evicted.push(entry.atlas_id);
                false
            } else {
                true
            }
        });
        let mut unused: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.last_used != self.frame)
            .map(|(&id, entry)| (id, entry.last_used))
            .collect();
        unused.sort_unstable_by_key(|&(id, last_used)| (std::cmp::Reverse(last_used), id));
        let mut remaining = budget;
        for (id, _) in unused {
            let entry = &self.entries[&id];
            if entry.bytes <= remaining {
                remaining -= entry.bytes;
            } else {
                evicted.push(self.entries.remove(&id).unwrap().atlas_id);
            }
        }
        if self.frame == u64::MAX {
            for entry in self.entries.values_mut() {
                entry.last_used = 0;
            }
            self.frame = 1;
        } else {
            self.frame += 1;
        }
        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_frames() {
        let mut cache = ImageCache::default();
        assert!(cache.evict(0).is_empty());
        assert_eq!(cache.frame, 1);
    }

    #[test]
    fn current_images_are_kept_even_over_budget() {
        let mut cache = ImageCache::default();
        let blob = Blob::from(vec![0]);
        cache.insert(&blob, ImageId::new(1), 100, 100);
        assert!(cache.evict(0).is_empty());
        assert_eq!(cache.get(blob.id()), Some(ImageId::new(1)));
        assert!(cache.evict(0).is_empty());
    }

    #[test]
    fn unused_images_are_retained_in_lru_order() {
        let mut cache = ImageCache::default();
        let first = Blob::from(vec![0]);
        let second = Blob::from(vec![0]);
        cache.insert(&first, ImageId::new(1), 2, 2);
        cache.insert(&second, ImageId::new(2), 2, 2);
        cache.evict(32);
        cache.get(second.id());
        cache.evict(32);
        assert_eq!(cache.evict(16), vec![ImageId::new(1)]);
        assert_eq!(cache.get(first.id()), None);
        assert_eq!(cache.get(second.id()), Some(ImageId::new(2)));
    }

    #[test]
    fn clear_invalidates_ids_and_resets_frame() {
        let mut cache = ImageCache::default();
        let blob = Blob::from(vec![0]);
        cache.insert(&blob, ImageId::new(1), 1, 1);
        cache.evict(4);
        cache.clear();
        assert_eq!(cache.get(blob.id()), None);
        assert_eq!(cache.frame, 0);
    }

    #[test]
    fn oversized_unused_image_does_not_displace_smaller_images() {
        let mut cache = ImageCache::default();
        let first = Blob::from(vec![0]);
        let second = Blob::from(vec![0]);
        cache.insert(&first, ImageId::new(1), 10, 10);
        cache.insert(&second, ImageId::new(2), 1, 1);
        cache.evict(4);
        assert_eq!(cache.evict(4), vec![ImageId::new(1)]);
        assert_eq!(cache.get(second.id()), Some(ImageId::new(2)));
    }

    #[test]
    fn dropped_blob_is_evicted_next_frame_within_budget() {
        let mut cache = ImageCache::default();
        let blob = Blob::from(vec![0]);
        let id = blob.id();
        let weak = blob.downgrade();
        cache.insert(&blob, ImageId::new(1), 1, 1);
        assert_eq!(blob.strong_count(), 1);
        assert!(cache.finish_frame().is_empty());
        drop(blob);
        assert!(weak.upgrade().is_none());
        assert_eq!(cache.finish_frame(), vec![ImageId::new(1)]);
        assert_eq!(cache.get(id), None);
        assert!(cache.finish_frame().is_empty());
    }

    #[test]
    fn dropped_current_blob_is_evicted_after_rendering() {
        let mut cache = ImageCache::default();
        let blob = Blob::from(vec![0]);
        cache.insert(&blob, ImageId::new(1), 1, 1);
        drop(blob);
        assert_eq!(cache.finish_frame(), vec![ImageId::new(1)]);
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn surviving_clone_keeps_unused_image_resident() {
        let mut cache = ImageCache::default();
        let blob = Blob::from(vec![0]);
        let clone = blob.clone();
        cache.insert(&blob, ImageId::new(1), 1, 1);
        cache.finish_frame();
        drop(blob);
        assert!(cache.finish_frame().is_empty());
        assert_eq!(cache.get(clone.id()), Some(ImageId::new(1)));
        drop(clone);
        assert_eq!(cache.finish_frame(), vec![ImageId::new(1)]);
    }
}
