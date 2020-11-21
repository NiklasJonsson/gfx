pub mod async_storage;
pub mod buffered_storage;
pub mod cache;
pub mod cached_storage;
pub mod storage;

pub use async_storage::AsyncStorage;
pub use buffered_storage::BufferedStorage;
pub use cache::Cache;
pub use cached_storage::CachedStorage;
pub use storage::Handle;
pub use storage::Storage;

pub trait ResourceManager<Descriptor, Resource, Handle> {
    type Error;
    fn get_resource(&self, handle: &Handle) -> Option<&Resource>;
    fn create_resource(&mut self, descriptor: Descriptor) -> Result<Handle, Self::Error>;
}

pub trait MutResourceManager<Descriptor, Resource, Handle> {
    type Error;
    fn get_resource_mut(&mut self, handle: &Handle) -> Option<&mut Resource>;
    fn recreate_resource(
        &mut self,
        handle: Handle,
        descriptor: Descriptor,
    ) -> Result<Handle, Self::Error>;
}
