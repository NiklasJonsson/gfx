use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use super::Handle;

/// Cache for descriptor to Handle\<T\>
#[derive(Debug)]
pub struct Cache<D: Hash + Eq, T> {
    ty: PhantomData<T>,
    cache: HashMap<D, Handle<T>>,
}

impl<D: Hash + Eq, T> Cache<D, T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get(&self, desc: &D) -> Option<&Handle<T>> {
        self.cache.get(desc)
    }

    pub fn insert(&mut self, desc: D, h: Handle<T>) {
        self.cache.insert(desc, h);
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn iter(&mut self) -> impl Iterator<Item = (&D, &Handle<T>)> {
        self.cache.iter()
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
