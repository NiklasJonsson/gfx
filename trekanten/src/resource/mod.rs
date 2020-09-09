pub mod buffered_storage;
pub mod cache;
pub mod cached_storage;
pub mod storage;

pub use buffered_storage::BufferedStorage;
pub use cached_storage::CachedStorage;
pub use storage::Handle;
pub use storage::Storage;

pub trait ResourceManager<Descriptor, Resource, Error> {
    type Handle;
    fn get_resource(&self, handle: &Self::Handle) -> Option<&Resource>;
    fn create_resource(&mut self, descriptor: Descriptor) -> Result<Self::Handle, Error>;
}
