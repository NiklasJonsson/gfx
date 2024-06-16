use crate::backend;

use crate::buffer::{
    AsyncBufferHandle, BufferDescriptor, BufferHandle, DeviceBuffer,
    DrainIterator as BufferDrainIterator,
};
use crate::resource::{Async, Handle, Resources};
use crate::texture::{
    AsyncTextures, DrainIterator as TextureDrainIterator, Texture, TextureDescriptor,
};
use crate::Renderer;
use crate::{
    backend::{AllocatorHandle, CommandPool, Fence, HasVkDevice, Queue, VkDeviceHandle},
    BufferMutability,
};
use backend::buffer::Buffer;
use backend::device::Device;

// TODO: Don't use vk directly here
use ash::vk;

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use thiserror::Error;

use crate::backend::{command, queue, sync};
#[derive(Debug, Error)]
pub enum LoaderError {
    Command(#[from] command::CommandError),
    Queue(#[from] queue::QueueError),
    Sync(#[from] sync::SyncError),
    Texture(#[from] image::ImageError),
    InvalidDescriptor(String),
    Mutex,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(&'static str);

#[derive(Default)]
struct AsyncResources {
    pub buffers: crate::buffer::AsyncBuffers,
    pub textures: crate::texture::AsyncTextures,
}

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[allow(clippy::enum_variant_names)]
enum PendingResourceCommand {
    CreateBuffer {
        descriptor: BufferDescriptor<'static>,
        handle: AsyncBufferHandle,
        buffer0: DeviceBuffer,
        buffer1: Option<DeviceBuffer>, // For double buffering
        transients: [Option<Buffer>; 2],
    },
    CreateTexture {
        handle: Handle<Async<Texture>>,
        texture: Texture,
        _transients: Buffer,
    },
}

pub enum HandleMapping {
    Buffer {
        old: AsyncBufferHandle,
        new: BufferHandle,
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
    done_resource_jobs: HashMap<RequestId, Vec<HandleMapping>>,
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

// Good reads for mutex + iterating over contents
// https://users.rust-lang.org/t/creating-an-iterator-over-mutex-contents-cannot-infer-an-appropriate-lifetime/24458/7
// https://www.reddit.com/r/rust/comments/7l97u0/iterator_struct_of_an_iterable_inside_a_lock_from/

impl AsyncResources {
    fn drain_available<'i, 's: 'i, 'r: 'i>(
        &'s mut self,
        resources: &'r mut Resources,
    ) -> impl Iterator<Item = HandleMapping> + 'i {
        let bufs = self
            .buffers
            .drain_available()
            .map(IntermediateIteratorItem::Buffer);

        let textures = self
            .textures
            .drain_available()
            .map(IntermediateIteratorItem::Texture);

        bufs.chain(textures).map(move |item| match item {
            IntermediateIteratorItem::Buffer(buf) => {
                let (h, b0, b1) = buf;
                let b0 = b0.expect("should be avail");
                let ty = b0.buffer_type().ty();
                let n_elems = b0.n_elems();
                let mutability = if b1.is_some() {
                    BufferMutability::Mutable
                } else {
                    BufferMutability::Immutable
                };
                let new = resources
                    .buffers
                    .add(b0, b1.map(|b| b.expect("should be avail")));
                let (old, new) = unsafe {
                    (
                        AsyncBufferHandle::from_buffer(h, 0, n_elems, mutability, ty),
                        BufferHandle::from_buffer(new, 0, n_elems, mutability, ty),
                    )
                };
                HandleMapping::Buffer { old, new }
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
    Buffer(<BufferDrainIterator<'static, DeviceBuffer> as Iterator>::Item),
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
            done_resource_jobs: HashMap::with_capacity(16),
            resources: AsyncResources::default(),
        });
        Self {
            vk_device,
            allocator,
            locked,
        }
    }

    fn progress_pending(&self) {
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
                    PendingResourceCommand::CreateBuffer {
                        handle,
                        buffer0,
                        buffer1,
                        transients: _transients,
                        descriptor: _descriptor,
                    } => guard.resources.buffers.insert(handle, buffer0, buffer1),
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
        self.progress_pending();
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

impl Loader {
    pub fn load_buffer_2(
        &self,
        descriptor: BufferDescriptor<'static>,
        request_id: RequestId,
    ) -> Result<AsyncBufferHandle, LoaderError> {
        log::trace!("Loading buffer with descriptor {descriptor:?}");
        assert!(!descriptor.is_empty());

        let mut guard = self.locked.lock().map_err(|_| LoaderError::Mutex)?;
        let handle = guard.resources.buffers.allocate(&descriptor);

        #[derive(Debug, Default)]
        struct Requests {
            buffers: Vec<AsyncBufferHandle>,
            textures: Vec<Handle<Async<Texture>>>,
        }
        let mut request_map: std::collections::HashMap<RequestId, Requests> =
            std::collections::HashMap::new();

        let requests = request_map.entry(request_id).or_default();
        requests.buffers.push(handle);

        todo!()
    }

    pub fn flush(&self, request_id: RequestId) -> Vec<HandleMapping> {
        let mut done_map: std::collections::HashMap<RequestId, Vec<HandleMapping>> =
            std::collections::HashMap::new();

        done_map.remove(&request_id).unwrap_or_default()
    }

    pub fn transfer_2<'mutex, 'loader: 'mutex, 'renderer>(
        &'loader self,
        renderer: &'renderer mut Renderer,
    ) -> TransferGuard<'mutex, 'renderer> {
        self.progress_pending();
        let guard = self.locked.lock().expect("Failed to lock mutex");
        let resources = renderer.resources_mut();
        TransferGuard { guard, resources }
    }
}
impl Loader {
    pub fn load_buffer(
        &self,
        descriptor: BufferDescriptor<'static>,
    ) -> Result<AsyncBufferHandle, LoaderError> {
        log::trace!("Loading buffer with descriptor {descriptor:?}");
        assert!(!descriptor.is_empty());

        let mut guard = self.locked.lock().map_err(|_| LoaderError::Mutex)?;

        let handle = guard.resources.buffers.allocate(&descriptor);
        let (result, done) = Loader::submit_commands(&self.vk_device, &guard, |cmd_buf| {
            descriptor.enqueue(&self.allocator, cmd_buf)
        })
        .expect("Failed to submit command to create buffer");

        let (buf0, buf1) = result.expect("Failed to create buffers");

        let (buffer1, transient1) = if let Some(buf1) = buf1 {
            (Some(buf1.buffer), buf1.transient)
        } else {
            (None, None)
        };

        let buffer0 = buf0.buffer;
        let transients = [buf0.transient, transient1];
        let job = PendingResourceJob {
            command: PendingResourceCommand::CreateBuffer {
                descriptor,
                handle,
                buffer0,
                buffer1,
                transients,
            },
            done,
        };
        guard.pending_resource_jobs.push(job);

        Ok(handle)
    }

    pub fn load_texture(
        &self,
        descriptor: TextureDescriptor<'static>,
    ) -> Result<Handle<Async<Texture>>, LoaderError> {
        log::trace!("Loading texture with descriptor {descriptor:?}");
        if descriptor.mipmaps() == crate::texture::MipMaps::Generate {
            return Err(LoaderError::InvalidDescriptor(String::from(
                "Can't generate mipmaps on loader queue",
            )));
        }

        let (desc, mipmaps, data) = descriptor.split_desc_data()?;

        let Some(data) = data else {
            return Err(LoaderError::InvalidDescriptor(String::from(
                "Can't load empty textures in the loader (for now)",
            )));
        };

        let mut guard = self.locked.lock().map_err(|_| LoaderError::Mutex)?;
        let handle = guard.resources.textures.allocate();
        let (result, done) = Loader::submit_commands(&self.vk_device, &guard, |command_buffer| {
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
    }
}
