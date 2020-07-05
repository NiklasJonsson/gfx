use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;

use super::storage::Handle;

/// Cache for descriptor to Handle<Asset>
#[derive(Debug)]
pub struct Cache<D: Hash + Eq, A> {
    ty: PhantomData<A>,
    cache: HashMap<D, Handle<A>>,
}

impl<D: Hash + Eq, A> Cache<D, A> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get(&self, desc: &D) -> Option<Handle<A>> {
        self.cache.get(desc).copied()
    }

    pub fn add(&mut self, desc: D, h: Handle<A>) {
        self.cache.insert(desc, h);
    }
}

impl<D: Hash + Eq, A> Default for Cache<D, A> {
    fn default() -> Self {
        Cache::<D, A> {
            ty: PhantomData {},
            cache: HashMap::new(),
        }
    }
}
