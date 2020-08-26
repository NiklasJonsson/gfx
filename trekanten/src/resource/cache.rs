use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use super::storage::Handle;

/// Cache for descriptor to Handle<T>
#[derive(Debug)]
pub struct Cache<D: Hash + Eq, T> {
    ty: PhantomData<T>,
    cache: HashMap<D, Handle<T>>,
}

impl<D: Hash + Eq, T> Cache<D, T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get(&self, desc: &D) -> Option<Handle<T>> {
        self.cache.get(desc).cloned()
    }

    pub fn add(&mut self, desc: D, h: Handle<T>) {
        self.cache.insert(desc, h);
    }
}

impl<D: Hash + Eq, T> std::default::Default for Cache<D, T> {
    fn default() -> Self {
        Cache {
            ty: PhantomData {},
            cache: HashMap::new(),
        }
    }
}
