pub mod async_storage;
pub mod buffered_storage;
pub mod cache;
pub mod cached_storage;
pub mod storage;

pub use async_storage::Async;
pub use buffered_storage::BufferedStorage;
pub use cache::Cache;
pub use cached_storage::CachedStorage;
pub use storage::Storage;

use std::marker::PhantomData;

#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct ID {
    index: usize,
}

impl std::fmt::Display for ID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.index)
    }
}

// Can't derive things on Handle because of PhantomData + generic
// https://github.com/rust-lang/rust/issues/26925
// so implement all of them manually :(
pub struct Handle<T> {
    id: ID,
    ty: PhantomData<T>,
}

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Handle<{}>", std::any::type_name::<T>())?;
        f.debug_struct("").field("id", &self.id).finish()
    }
}

impl<T> Default for Handle<T> {
    fn default() -> Self {
        Self {
            id: ID::default(),
            ty: PhantomData {},
        }
    }
}
impl<T> std::cmp::PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> std::cmp::Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> Handle<T> {
    fn new(id: ID) -> Self {
        Handle::<T> {
            id,
            ty: PhantomData {},
        }
    }

    fn index(&self) -> usize {
        self.id.index
    }

    pub fn id(&self) -> ID {
        self.id
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle::<T>::new(self.id)
    }
}
impl<T> Copy for Handle<T> {}
