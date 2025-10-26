use std::{fmt::Debug, hash::Hash};

use rustc_hash::FxHashMap;

struct CacheEntry<T> {
    resource: T,
    generation: usize,
}

pub(crate) struct GenerationalCache<K, T>
where
    K: Eq + Hash + Clone + Debug,
{
    resources: FxHashMap<K, CacheEntry<T>>,
    current_generation: usize,
    max_age: usize,
}

impl<K, T> Default for GenerationalCache<K, T>
where
    K: Eq + Hash + Clone + Debug,
{
    fn default() -> Self {
        Self {
            resources: Default::default(),
            current_generation: 0,
            max_age: 2,
        }
    }
}

impl<K, T> GenerationalCache<K, T>
where
    K: Eq + Hash + Clone + Debug,
{
    pub(crate) fn new(max_age: usize) -> Self {
        GenerationalCache {
            resources: FxHashMap::default(),
            current_generation: 0,
            max_age,
        }
    }

    pub(crate) fn next_gen(&mut self) -> usize {
        let extracted_count = self
            .resources
            .extract_if(|_, entry| {
                self.current_generation.wrapping_sub(entry.generation) >= self.max_age
            })
            .count();

        self.current_generation = self.current_generation.wrapping_add(1);

        return extracted_count;
    }

    pub(crate) fn contains_key(&self, key: &K) -> bool {
        self.resources.contains_key(key)
    }

    pub(crate) fn hit(&mut self, key: &K) -> Option<&T> {
        if let Some(entry) = self.resources.get_mut(key) {
            entry.generation = self.current_generation;
            Some(&entry.resource)
        } else {
            None
        }
    }

    pub(crate) fn insert(&mut self, key: K, resource: T) {
        let entry = CacheEntry {
            resource,
            generation: self.current_generation,
        };
        self.resources.insert(key, entry);
    }
}
