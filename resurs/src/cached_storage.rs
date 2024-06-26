use std::hash::Hash;

use super::cache::*;
use super::storage::*;
use super::Handle;

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub hits: usize,
    pub misses: usize,
}

pub struct CachedStorage<ResourceDescriptor, Resource>
where
    ResourceDescriptor: Hash + Eq,
{
    cache: Cache<ResourceDescriptor, Resource>,
    storage: Storage<Resource>,
    stats: Stats,
}

impl<ResourceDescriptor, Resource> CachedStorage<ResourceDescriptor, Resource>
where
    ResourceDescriptor: Hash + Eq,
{
    pub fn new() -> Self {
        Default::default()
    }

    pub fn cached(&self, descriptor: &ResourceDescriptor) -> Option<Handle<Resource>> {
        self.cache.get(descriptor).cloned()
    }

    pub fn get_or_add<Create, Error>(
        &mut self,
        descriptor: ResourceDescriptor,
        create: Create,
    ) -> Result<Handle<Resource>, Error>
    where
        Create: FnOnce(&ResourceDescriptor) -> Result<Resource, Error>,
    {
        let h = match self.cache.get(&descriptor) {
            Some(h) => {
                self.stats.hits += 1;
                *h
            }
            None => {
                self.stats.misses += 1;
                let resource = create(&descriptor)?;
                let h = self.storage.add(resource);
                self.cache.add(descriptor, h);
                h
            }
        };

        Ok(h)
    }

    pub fn get(&self, h: &Handle<Resource>) -> Option<&Resource> {
        self.storage.get(h)
    }

    pub fn add(&mut self, descriptor: ResourceDescriptor, r: Resource) -> Handle<Resource> {
        let h = self.storage.add(r);
        self.cache.add(descriptor, h);

        h
    }

    pub fn has(&self, h: &Handle<Resource>) -> bool {
        self.get(h).is_some()
    }

    pub fn get_mut(&mut self, h: &Handle<Resource>) -> Option<&mut Resource> {
        self.storage.get_mut(h)
    }

    /// Iterate over the storage contents mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&ResourceDescriptor, &mut Resource)> {
        assert_eq!(self.cache.len(), self.storage.len());
        self.cache
            .iter()
            .zip(self.storage.iter_mut())
            .map(|((desc, _handle), resource)| (desc, resource))
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn flush_stats(&mut self) -> Stats {
        let stats = self.stats;
        self.stats = Stats::default();
        stats
    }
}

impl<ResourceDescriptor, Resource> std::default::Default
    for CachedStorage<ResourceDescriptor, Resource>
where
    ResourceDescriptor: Hash + Eq,
{
    fn default() -> Self {
        Self {
            cache: Default::default(),
            storage: Default::default(),
            stats: Default::default(),
        }
    }
}
