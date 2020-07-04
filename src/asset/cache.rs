use std::marker::PhantomData;
use std::container::HashMap;

use super::AssetDescriptor;

using CacheEntry = super::AssetDescriptor

#[Derive(Debug)]
pub struct Cache<A> {
    ty: PhantomData,
    cache: HashMap<AssetDescriptor, Handle<A>>,
}

impl<A> Cache<A> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get(ad: &AssetDescriptor) -> Option<Handle<A>> {
        self.cache.get(ad)
    }
}
