use crate::backend;

use crate::buffer::{
    BufferHandle, DeviceIndexBuffer, DeviceUniformBuffer, DeviceVertexBuffer,
    DrainIterator as BufferDrainIterator, IndexBufferDescriptor, UniformBufferDescriptor,
    VertexBufferDescriptor,
};
use crate::resource::{Async, AsyncResources, Handle, Resources};
use crate::texture::{DrainIterator as TextureDrainIterator, Texture, TextureDescriptor};
use crate::Renderer;
use crate::{
    backend::{
        AllocatorHandle, CommandBuffer, CommandPool, Fence, HasVkDevice, Queue, VkDeviceHandle,
    },
    BufferMutability,
};
use backend::buffer::Buffer;
use backend::device::Device;

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
    InvalidDescriptor(String),
    Mutex,
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub enum AsyncResourceCommand {
    CreateVertexBuffer {
        descriptor: VertexBufferDescriptor<'static>,
        handle: BufferHandle<Async<DeviceVertexBuffer>>,
    },
    CreateIndexBuffer {
        descriptor: IndexBufferDescriptor<'static>,
        handle: BufferHandle<Async<DeviceIndexBuffer>>,
    },
    CreateUniformBuffer {
        descriptor: UniformBufferDescriptor<'static>,
        handle: BufferHandle<Async<DeviceUniformBuffer>>,
    },
}

#[allow(clippy::enum_variant_names)]
enum PendingResourceCommand {
    CreateVertexBuffer {
        descriptor: VertexBufferDescriptor<'static>,
        handle: BufferHandle<Async<DeviceVertexBuffer>>,
        buffer0: DeviceVertexBuffer,
        buffer1: Option<DeviceVertexBuffer>, // For double buffering
        transients: [Option<Buffer>; 2],
    },
    CreateIndexBuffer {
        descriptor: IndexBufferDescriptor<'static>,
        handle: BufferHandle<Async<DeviceIndexBuffer>>,
        buffer0: DeviceIndexBuffer,
        buffer1: Option<DeviceIndexBuffer>, // For double buffering
        transients: [Option<Buffer>; 2],
    },
    CreateUniformBuffer {
        descriptor: UniformBufferDescriptor<'static>,
        handle: BufferHandle<Async<DeviceUniformBuffer>>,
        buffer0: DeviceUniformBuffer,
        buffer1: Option<DeviceUniformBuffer>, // For double buffering
        transients: [Option<Buffer>; 2],
    },
    CreateTexture {
        handle: Handle<Async<Texture>>,
        texture: Texture,
        _transients: Buffer,
    },
}

pub enum HandleMapping {
    UniformBuffer {
        old: BufferHandle<Async<DeviceUniformBuffer>>,
        new: BufferHandle<DeviceUniformBuffer>,
    },
    VertexBuffer {
        old: BufferHandle<Async<DeviceVertexBuffer>>,
        new: BufferHandle<DeviceVertexBuffer>,
    },
    IndexBuffer {
        old: BufferHandle<Async<DeviceIndexBuffer>>,
        new: BufferHandle<DeviceIndexBuffer>,
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
    command: PendingResourceCommand,
    done: Fence,
}

macro_rules! process_buffer_creation {
    ($cmd:ident, $desc:ident, $self:ident, $cmd_buffer:ident, $handle:ident) => {{
        let (buf0, buf1) = $desc.enqueue(&$self.allocator, $cmd_buffer).expect("Fail");

        let (buffer1, transient1) = if let Some(buf1) = buf1 {
            (Some(buf1.buffer), buf1.transient)
        } else {
            (None, None)
        };

        let buffer0 = buf0.buffer;
        let transients = [buf0.transient, transient1];
        PendingResourceCommand::$cmd {
            descriptor: $desc,
            handle: $handle,
            buffer0,
            buffer1,
            transients,
        }
    }};
}

impl Loader {
    fn process_command(
        &self,
        command: AsyncResourceCommand,
        cmd_buffer: &mut CommandBuffer,
    ) -> PendingResourceCommand {
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
            .map(IntermediateIteratorItem::Vertex);

        let ubufs = self
            .uniform_buffers
            .drain_available()
            .map(IntermediateIteratorItem::Uniform);

        let ibufs = self
            .index_buffers
            .drain_available()
            .map(IntermediateIteratorItem::Index);

        let textures = self
            .textures
            .drain_available()
            .map(IntermediateIteratorItem::Texture);

        vbufs
            .chain(ubufs)
            .chain(ibufs)
            .chain(textures)
            .map(move |item| match item {
                IntermediateIteratorItem::Vertex(buf) => {
                    map_buffer!(resources, buf, VertexBuffer, vertex_buffers)
                }
                IntermediateIteratorItem::Index(buf) => {
                    map_buffer!(resources, buf, IndexBuffer, index_buffers)
                }
                IntermediateIteratorItem::Uniform(buf) => {
                    map_buffer!(resources, buf, UniformBuffer, uniform_buffers)
                }
                IntermediateIteratorItem::Texture((handle, tex)) => {
                    let new_handle = resources.textures.add(tex.expect("Should be available"));
                    HandleMapping::Texture {
                        old: handle,
                        new: new_handle,
                    }
                }
            })
    }
}

pub struct TransferGuard<'mutex, 'renderer> {
    guard: MutexGuard<'mutex, NonSync>,
    resources: &'renderer mut Resources,
}

enum IntermediateIteratorItem {
    Vertex(<BufferDrainIterator<'static, DeviceVertexBuffer> as Iterator>::Item),
    Index(<BufferDrainIterator<'static, DeviceIndexBuffer> as Iterator>::Item),
    Uniform(<BufferDrainIterator<'static, DeviceUniformBuffer> as Iterator>::Item),
    Texture(<TextureDrainIterator<'static> as Iterator>::Item),
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

    pub fn poll(&self) {
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
                    command,
                    done: _done,
                } = guard.pending_resource_jobs.remove(i);

                match command {
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
                        handle, texture, ..
                    } => {
                        let loc = guard
                            .resources
                            .textures
                            .get_mut(&handle)
                            .expect("This should exist");
                        *loc = Async::Available(texture);
                    }
                }
            } else {
                i += 1;
            }
        }
    }

    pub fn transfer<'mutex, 'loader: 'mutex, 'renderer>(
        &'loader self,
        renderer: &'renderer mut Renderer,
    ) -> TransferGuard<'mutex, 'renderer> {
        self.poll();
        let guard = self.locked.lock().expect("Failed to lock mutex");
        let resources = renderer.resources_mut();
        TransferGuard { guard, resources }
    }
}

impl Loader {
    fn submit_commands<F, R>(
        vk_device: &VkDeviceHandle,
        guard: &MutexGuard<'_, NonSync>,
        f: F,
    ) -> Result<(R, Fence), LoaderError>
    where
        F: FnOnce(&mut command::CommandBuffer) -> R,
    {
        let mut command_buffer = guard.command_pool.begin_single_submit()?;
        let r = f(&mut command_buffer);
        command_buffer.end()?;
        let done = Fence::unsignaled(vk_device)?;
        let buffers = [*command_buffer.vk_command_buffer()];
        let info = vk::SubmitInfo::builder().command_buffers(&buffers);
        guard.queue.submit(&info, &done)?;
        Ok((r, done))
    }
}

pub trait ResourceLoader<D, H> {
    fn load(&self, descriptor: D) -> Result<H, LoaderError>;
}

fn validate_texture_descriptor(d: &TextureDescriptor) -> Result<(), LoaderError> {
    if d.mipmaps() == crate::texture::MipMaps::Generate {
        return Err(LoaderError::InvalidDescriptor(String::from(
            "Can't generate mipmaps on loader queue",
        )));
    }

    Ok(())
}

macro_rules! impl_buffer_loader {
    ($desc:ty, $handle:ty, $storage:ident, $cmd_enum:ident) => {
        impl ResourceLoader<$desc, $handle> for Loader {
            fn load(&self, descriptor: $desc) -> Result<$handle, LoaderError> {
                let mut guard = self.locked.lock().map_err(|_| LoaderError::Mutex)?;

                let handle = guard.resources.$storage.allocate(&descriptor);
                let (command, done) =
                    Loader::submit_commands(&self.vk_device, &guard, |command_buffer| {
                        let cmd = AsyncResourceCommand::$cmd_enum { descriptor, handle };
                        self.process_command(cmd, command_buffer)
                    })
                    .expect("Failed to submit command");
                let job = PendingResourceJob { command, done };
                guard.pending_resource_jobs.push(job);

                Ok(handle)
            }
        }
    };
}

impl_buffer_loader!(
    IndexBufferDescriptor<'static>,
    BufferHandle<Async<DeviceIndexBuffer>>,
    index_buffers,
    CreateIndexBuffer
);
impl_buffer_loader!(
    VertexBufferDescriptor<'static>,
    BufferHandle<Async<DeviceVertexBuffer>>,
    vertex_buffers,
    CreateVertexBuffer
);
impl_buffer_loader!(
    UniformBufferDescriptor<'static>,
    BufferHandle<Async<DeviceUniformBuffer>>,
    uniform_buffers,
    CreateUniformBuffer
);

impl Loader {
    pub fn load_texture(
        &self,
        descriptor: TextureDescriptor<'static>,
    ) -> Result<Handle<Async<Texture>>, LoaderError> {
        validate_texture_descriptor(&descriptor)?;

        let (desc, mipmaps, data) = descriptor
            .split_desc_data()
            .expect("Failed to load descriptor data");
        if let Some(data) = data {
            let mut guard = self.locked.lock().map_err(|_| LoaderError::Mutex)?;
            let handle = guard.resources.textures.allocate();
            let (result, done) =
                Loader::submit_commands(&self.vk_device, &guard, |command_buffer| {
                    crate::texture::load_texture_from_data(
                        &self.vk_device,
                        &self.allocator,
                        command_buffer,
                        desc,
                        data.data(),
                        mipmaps,
                    )
                })
                .expect("Failed to submit command");
            let (tex, buf) = result.expect("Failed to load texture from data");
            let job = PendingResourceJob {
                command: PendingResourceCommand::CreateTexture {
                    handle,
                    texture: tex,
                    _transients: buf,
                },
                done,
            };

            guard.pending_resource_jobs.push(job);
            Ok(handle)
        } else {
            Err(LoaderError::InvalidDescriptor(String::from(
                "Can't load empty textures in the loader (for now)",
            )))
        }
    }
}
