use crate::device::Device;
use crate::mem::BufferHandle;
use crate::mem::{
    BufferDescriptor, DeviceBuffer, DrainIterator, IndexBuffer, OwningIndexBufferDescriptor,
    OwningUniformBufferDescriptor, OwningVertexBufferDescriptor, UniformBuffer, VertexBuffer,
};
use crate::resource::{Async, AsyncResources, Handle, Resources};
use crate::texture::{Texture, TextureDescriptor};
use crate::Renderer;
use crate::{
    backend::{
        AllocatorHandle, CommandBuffer, CommandPool, Fence, HasVkDevice, Queue, VkDeviceHandle,
    },
    BufferMutability,
};

// TODO: Don't use vk directly here
use ash::vk;

use std::sync::{Mutex, MutexGuard};

use thiserror::Error;

use crate::backend::{command, queue, sync};
#[derive(Debug, Error)]
pub enum LoaderError {
    Command(#[from] command::CommandError),
    Queue(#[from] queue::QueueError),
    Sync(#[from] sync::SyncError),
    Mutex,
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub enum AsyncResourceCommand {
    CreateVertexBuffer {
        descriptor: OwningVertexBufferDescriptor,
        handle: BufferHandle<Async<VertexBuffer>>,
    },
    CreateIndexBuffer {
        descriptor: OwningIndexBufferDescriptor,
        handle: BufferHandle<Async<IndexBuffer>>,
    },
    CreateUniformBuffer {
        descriptor: OwningUniformBufferDescriptor,
        handle: BufferHandle<Async<UniformBuffer>>,
    },
    CreateTexture {
        descriptor: TextureDescriptor,
        handle: Handle<Async<Texture>>,
    },
}

enum PendingResourceCommand {
    CreateVertexBuffer {
        descriptor: OwningVertexBufferDescriptor,
        handle: BufferHandle<Async<VertexBuffer>>,
        buffer0: VertexBuffer,
        buffer1: Option<VertexBuffer>, // For double buffering
        transients: [Option<DeviceBuffer>; 2],
    },
    CreateIndexBuffer {
        descriptor: OwningIndexBufferDescriptor,
        handle: BufferHandle<Async<IndexBuffer>>,
        buffer0: IndexBuffer,
        buffer1: Option<IndexBuffer>, // For double buffering
        transients: [Option<DeviceBuffer>; 2],
    },
    CreateUniformBuffer {
        descriptor: OwningUniformBufferDescriptor,
        handle: BufferHandle<Async<UniformBuffer>>,
        buffer0: UniformBuffer,
        buffer1: Option<UniformBuffer>, // For double buffering
        transients: [Option<DeviceBuffer>; 2],
    },
    CreateTexture {
        descriptor: TextureDescriptor,
        handle: Handle<Async<Texture>>,
        image: Texture,
        transients: DeviceBuffer,
    },
}

pub enum HandleMapping {
    UniformBuffer {
        old: BufferHandle<Async<UniformBuffer>>,
        new: BufferHandle<UniformBuffer>,
    },
    VertexBuffer {
        old: BufferHandle<Async<VertexBuffer>>,
        new: BufferHandle<VertexBuffer>,
    },
    IndexBuffer {
        old: BufferHandle<Async<IndexBuffer>>,
        new: BufferHandle<IndexBuffer>,
    },
    Texture {
        old: Handle<Async<Texture>>,
        new: Handle<Texture>,
    },
}

struct NonSync {
    queue: Queue,
    command_pool: CommandPool,
    pending_resource_jobs: Vec<PendingResourceJob>,
    resources: AsyncResources,
}

pub struct Loader {
    allocator: AllocatorHandle,
    vk_device: VkDeviceHandle,
    locked: Mutex<NonSync>,
}

struct PendingResourceJob {
    commands: Vec<PendingResourceCommand>,
    done: Fence,
}

macro_rules! process_buffer_creation {
    ($cmd:ident, $desc:ident, $self:ident, $cmd_buffer:ident, $handle:ident) => {{
        let (buf0, buf1) = $desc
            .enqueue(
                &$self.allocator,
                $cmd_buffer.expect("This needs a command buffer"),
            )
            .expect("Fail");

        let (buffer1, transient1) = if let Some(buf1) = buf1 {
            (Some(buf1.buffer), buf1.transient)
        } else {
            (None, None)
        };

        let buffer0 = buf0.buffer;
        let transients = [buf0.transient, transient1];
        Some(PendingResourceCommand::$cmd {
            descriptor: $desc,
            handle: $handle,
            buffer0,
            buffer1,
            transients,
        })
    }};
}

impl Loader {
    fn process_command(
        &self,
        command: AsyncResourceCommand,
        cmd_buffer: Option<&mut CommandBuffer>,
    ) -> Option<PendingResourceCommand> {
        match command {
            AsyncResourceCommand::CreateVertexBuffer { handle, descriptor } => {
                process_buffer_creation!(CreateVertexBuffer, descriptor, self, cmd_buffer, handle)
            }
            AsyncResourceCommand::CreateIndexBuffer { handle, descriptor } => {
                process_buffer_creation!(CreateIndexBuffer, descriptor, self, cmd_buffer, handle)
            }
            AsyncResourceCommand::CreateUniformBuffer { handle, descriptor } => {
                process_buffer_creation!(CreateUniformBuffer, descriptor, self, cmd_buffer, handle)
            }
            AsyncResourceCommand::CreateTexture { handle, descriptor } => {
                let (image, transients) = descriptor
                    .enqueue(
                        &self.allocator,
                        &self.vk_device,
                        cmd_buffer.expect("texture creation needs command buffer"),
                    )
                    .expect("Fail");
                Some(PendingResourceCommand::CreateTexture {
                    descriptor,
                    handle,
                    image,
                    transients,
                })
            }
        }
    }
}

// Good reads for mutex + iterating over contents
// https://users.rust-lang.org/t/creating-an-iterator-over-mutex-contents-cannot-infer-an-appropriate-lifetime/24458/7
// https://www.reddit.com/r/rust/comments/7l97u0/iterator_struct_of_an_iterable_inside_a_lock_from/

macro_rules! map_buffer {
    ($resources:ident, $buf:ident, $mapping_enum:ident, $storage:ident) => {{
        let (h, b0, b1) = $buf;
        let b0 = b0.expect("should be avail");
        let n_elems = b0.n_elems();
        let mutability = if b1.is_some() {
            BufferMutability::Mutable
        } else {
            BufferMutability::Immutable
        };
        let new = $resources
            .$storage
            .add(b0, b1.map(|b| b.expect("should be avail")));
        let (old, new) = unsafe {
            (
                BufferHandle::from_buffer(h, 0, n_elems, mutability),
                BufferHandle::from_buffer(new, 0, n_elems, mutability),
            )
        };
        HandleMapping::$mapping_enum { old, new }
    }};
}

impl AsyncResources {
    fn drain_available<'i, 's: 'i, 'r: 'i>(
        &'s mut self,
        resources: &'r mut Resources,
    ) -> impl Iterator<Item = HandleMapping> + 'i {
        let vbufs = self
            .vertex_buffers
            .drain_available()
            .map(|x| IntermediateIteratorItem::Vertex(x));

        let ubufs = self
            .uniform_buffers
            .drain_available()
            .map(|x| IntermediateIteratorItem::Uniform(x));

        let ibufs = self
            .index_buffers
            .drain_available()
            .map(IntermediateIteratorItem::Index);

        vbufs.chain(ubufs).chain(ibufs).map(move |item| match item {
            IntermediateIteratorItem::Vertex(buf) => {
                map_buffer!(resources, buf, VertexBuffer, vertex_buffers)
            }
            IntermediateIteratorItem::Index(buf) => {
                map_buffer!(resources, buf, IndexBuffer, index_buffers)
            }
            IntermediateIteratorItem::Uniform(buf) => {
                map_buffer!(resources, buf, UniformBuffer, uniform_buffers)
            }
        })
    }
}

pub struct TransferGuard<'mutex, 'renderer> {
    guard: MutexGuard<'mutex, NonSync>,
    resources: &'renderer mut Resources,
}

enum IntermediateIteratorItem {
    Vertex(<DrainIterator<'static, VertexBuffer> as Iterator>::Item),
    Index(<DrainIterator<'static, IndexBuffer> as Iterator>::Item),
    Uniform(<DrainIterator<'static, UniformBuffer> as Iterator>::Item),
}

impl<'m, 'r> TransferGuard<'m, 'r> {
    pub fn iter(&mut self) -> impl Iterator<Item = HandleMapping> + '_ {
        self.guard.resources.drain_available(self.resources)
    }
}

impl Loader {
    pub fn new(device: &mut Device) -> Self {
        let queue = device
            .take_transfer_queue()
            .expect("No transfer queue for the loader");
        let command_pool =
            CommandPool::new(device, queue.family().clone()).expect("TODO: Return error");
        let allocator = device.allocator();
        let vk_device = device.vk_device();
        let locked = Mutex::new(NonSync {
            queue,
            command_pool,
            pending_resource_jobs: Vec::with_capacity(16),
            resources: AsyncResources::default(),
        });
        Self {
            vk_device,
            allocator,
            locked,
        }
    }

    pub fn poll(&mut self) {
        // Query finished
        // TODO: Use drain_filter here when not nightly
        let mut guard = self.locked.lock().expect("Failed to unlock");
        let mut i = 0;
        while i < guard.pending_resource_jobs.len() {
            if guard.pending_resource_jobs[i]
                .done
                .is_signaled()
                .expect("Failed to check fence")
            {
                let PendingResourceJob {
                    commands,
                    done: _done,
                } = guard.pending_resource_jobs.remove(i);

                for pending in commands.into_iter() {
                    match pending {
                        PendingResourceCommand::CreateVertexBuffer {
                            handle,
                            buffer0,
                            buffer1,
                            transients: _transients,
                            descriptor: _descriptor,
                        } => guard
                            .resources
                            .vertex_buffers
                            .insert(&handle, buffer0, buffer1),
                        PendingResourceCommand::CreateIndexBuffer {
                            handle,
                            buffer0,
                            buffer1,
                            transients: _transients,
                            descriptor: _descriptor,
                        } => guard
                            .resources
                            .index_buffers
                            .insert(&handle, buffer0, buffer1),
                        PendingResourceCommand::CreateUniformBuffer {
                            handle,
                            buffer0,
                            buffer1,
                            transients: _transients,
                            descriptor: _descriptor,
                        } => guard
                            .resources
                            .uniform_buffers
                            .insert(&handle, buffer0, buffer1),
                        PendingResourceCommand::CreateTexture {
                            handle,
                            image,
                            transients: _transients,
                            descriptor: _descriptor,
                        } => {
                            let loc = guard
                                .resources
                                .textures
                                .get_mut(&handle)
                                .expect("This should exist");
                            *loc = Async::Available(image);
                        }
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    pub fn transfer<'mutex, 'loader: 'mutex, 'renderer>(
        &'loader mut self,
        renderer: &'renderer mut Renderer,
    ) -> TransferGuard<'mutex, 'renderer> {
        self.poll();
        let guard = self.locked.lock().expect("Failed to lock mutex");
        let resources = renderer.resources_mut();
        TransferGuard { guard, resources }
    }
}

pub trait ResourceLoader<D, H> {
    fn load(&self, descriptor: D) -> Result<H, LoaderError>;
}

macro_rules! impl_loader {
    ($desc:ty, $handle:ty, $storage:ident, $cmd_enum:ident) => {
        impl ResourceLoader<$desc, $handle> for Loader {
            fn load(&self, descriptor: $desc) -> Result<$handle, LoaderError> {
                let mut guard = self.locked.lock().map_err(|_| LoaderError::Mutex)?;
                if let Some(handle) = guard.resources.$storage.cached(&descriptor) {
                    return Ok(handle);
                }

                let handle = guard.resources.$storage.allocate(&descriptor);
                let cmd = AsyncResourceCommand::$cmd_enum { descriptor, handle };

                let mut cmd_buffer = guard.command_pool.begin_single_submit()?;

                // TODO: Allocation. Switch to small vec
                let mut commands = Vec::new();
                if let Some(cmd) = self.process_command(cmd, Some(&mut cmd_buffer)) {
                    commands.push(cmd);
                }

                cmd_buffer.end()?;
                let done = Fence::unsignaled(&self.vk_device)?;
                let buffers = [*cmd_buffer.vk_command_buffer()];
                let info = vk::SubmitInfo::builder().command_buffers(&buffers);
                let job = PendingResourceJob { commands, done };

                guard.queue.submit(&info, &job.done)?;
                guard.pending_resource_jobs.push(job);

                Ok(handle)
            }
        }
    };
}

impl_loader!(
    OwningIndexBufferDescriptor,
    BufferHandle<Async<IndexBuffer>>,
    index_buffers,
    CreateIndexBuffer
);
impl_loader!(
    OwningVertexBufferDescriptor,
    BufferHandle<Async<VertexBuffer>>,
    vertex_buffers,
    CreateVertexBuffer
);
impl_loader!(
    OwningUniformBufferDescriptor,
    BufferHandle<Async<UniformBuffer>>,
    uniform_buffers,
    CreateUniformBuffer
);
impl_loader!(
    TextureDescriptor,
    Handle<Async<Texture>>,
    textures,
    CreateTexture
);
