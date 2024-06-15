pub mod async_storage;
pub mod buffered_storage;
pub mod cache;
pub mod cached_storage;
pub mod storage;

pub use async_storage::Async;
pub use buffered_storage::BufferedStorage;
pub use cache::Cache;
pub use cached_storage::CachedStorage;
pub use storage::ID;
pub use storage::{DrainFilter, Storage};

use std::marker::PhantomData;

// Can't derive things on Handle because of PhantomData + generic
// https://github.com/rust-lang/rust/issues/26925
// so implement all of them manually :(
pub struct Handle<T> {
    id: ID,
    ty: PhantomData<T>,
}

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = format!("Handle<{}>", std::any::type_name::<T>());
        f.debug_struct(&name).field("id", &self.id).finish()
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

impl<T> std::cmp::Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl<T> std::cmp::PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Handle<T> {
    fn new(id: ID) -> Self {
        Handle::<T> {
            id,
            ty: PhantomData {},
        }
    }

    pub fn id(&self) -> ID {
        self.id
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Handle<T> {}
