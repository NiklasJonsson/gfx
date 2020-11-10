use crate::mem::BufferHandle;
use crate::mesh::{
    IndexBuffer, OwningIndexBufferDescriptor, OwningVertexBufferDescriptor, VertexBuffer,
};
use crate::resource::{AsyncResources, Handle, ResourceCommand};
use crate::texture::{Texture, TextureDescriptor};
use crate::uniform::{OwningUniformBufferDescriptor, UniformBuffer};

use std::sync::{mpsc::Receiver, mpsc::Sender, Arc, Mutex};

pub type ResourceCommandSender = Sender<ResourceCommand>;
pub type ResourceCommandReceiver = Receiver<ResourceCommand>;

#[derive(Clone)]
pub struct Loader {
    send_channel: Arc<Mutex<ResourceCommandSender>>,
    resources: Arc<AsyncResources>,
}

impl Loader {
    pub fn new(send_channel: ResourceCommandSender, resources: Arc<AsyncResources>) -> Self {
        Self {
            send_channel: Arc::new(Mutex::new(send_channel)),
            resources,
        }
    }
}

pub trait ResourceLoader<D, H> {
    fn load(&self, descriptor: D) -> H;
    fn is_done(&self, h: &H) -> Option<bool>;
}

macro_rules! impl_loader {
    ($desc:ty, $handle:ty, $storage:ident, $cmd_enum:ident) => {
        impl ResourceLoader<$desc, $handle> for Loader {
            fn load(&self, descriptor: $desc) -> $handle {
                if let Some(handle) = self.resources.$storage.cache(&descriptor) {
                    return handle;
                }

                let handle = self.resources.$storage.allocate(&descriptor);
                let cmd = ResourceCommand::$cmd_enum { descriptor, handle };
                self.send_channel
                    .lock()
                    .expect("Failed to lock loader send channel")
                    .send(cmd)
                    .expect("loader send fail on channel");

                handle
            }

            fn is_done(&self, h: &$handle) -> Option<bool> {
                self.resources.$storage.is_done(h)
            }
        }
    };
}

impl_loader!(
    OwningIndexBufferDescriptor,
    BufferHandle<IndexBuffer>,
    index_buffers,
    CreateIndexBuffer
);
impl_loader!(
    OwningVertexBufferDescriptor,
    BufferHandle<VertexBuffer>,
    vertex_buffers,
    CreateVertexBuffer
);
impl_loader!(
    OwningUniformBufferDescriptor,
    BufferHandle<UniformBuffer>,
    uniform_buffers,
    CreateUniformBuffer
);
impl_loader!(TextureDescriptor, Handle<Texture>, textures, CreateTexture);
