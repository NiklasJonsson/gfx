use crate::backend;

use crate::buffer::{BufferDescriptor, BufferHandle, BufferTypeId, DeviceBuffer};
use crate::resource::Handle;
use crate::texture::{Texture, TextureDescriptor};
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
use std::num::Wrapping;
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
pub struct LoadId(pub &'static str);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct PendingResourceHandle(pub(crate) u64);

impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PendingBufferHandle {
    handle: PendingResourceHandle,
    mutability: BufferMutability,
    idx: u32,
    n_elems: u32,
    ty: BufferTypeId,
}

impl PendingBufferHandle {
    pub fn sub_buffer(h: Self, idx: u32, n_elems: u32) -> Self {
        assert!((idx + n_elems) <= (h.idx + h.n_elems));
        Self { idx, n_elems, ..h }
    }

    pub fn is_same_resource(&self, other: &Self) -> bool {
        self.handle == other.handle
    }

    pub fn idx(&self) -> u32 {
        self.idx
    }

    pub fn n_elems(&self) -> u32 {
        self.n_elems
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PendingTextureHandle(PendingResourceHandle);

#[derive(Debug, Default)]
struct HandleGenerator {
    counter: Wrapping<u64>,
}

impl HandleGenerator {
    fn new_handle(&mut self) -> PendingResourceHandle {
        let out = self.counter.0;
        self.counter += 1;
        PendingResourceHandle(out)
    }

    fn new_buffer(&mut self) -> PendingResourceHandle {
        self.new_handle()
    }

    fn new_texture(&mut self) -> PendingResourceHandle {
        self.new_handle()
    }
}

#[allow(clippy::enum_variant_names)]
enum PendingResourceCommand {
    CreateBuffer {
        // TODO: Do we need to keep this around?
        _descriptor: BufferDescriptor<'static>,
        handle: PendingBufferHandle,
        buffer0: DeviceBuffer,
        buffer1: Option<DeviceBuffer>, // For double buffering
        _transients: [Option<Buffer>; 2],
    },
    CreateTexture {
        handle: PendingTextureHandle,
        texture: Texture,
        _transients: Buffer,
    },
}

pub enum HandleMapping {
    Buffer {
        old: PendingBufferHandle,
        new: BufferHandle,
    },
    Texture {
        old: PendingTextureHandle,
        new: Handle<Texture>,
    },
}

struct NonSync {
    queue: Queue,
    command_pool: CommandPool,
    pending_resource_jobs: Vec<PendingResourceJob>,
    done_resource_jobs: HashMap<LoadId, Vec<HandleMapping>>,
    handle_generator: HandleGenerator,
}

/// Asynchronous loading of GPU resources.
///
/// The general loop is:
/// 1. Request to load something with one of the load_* functions. These take a load id that identifies a set of load requests.
/// 2. The main loop of the application should call `Loader::progress` periodically (once per frame makes sense).
/// 3. Call `Loader::flush` with the same load id as was used for the request.
///
pub struct Loader {
    allocator: AllocatorHandle,
    vk_device: VkDeviceHandle,
    locked: Mutex<NonSync>,
}

impl PendingResourceJob {
    fn is_done(&self) -> bool {
        self.done.is_signaled().expect("Failed to query fence")
    }
}

struct PendingResourceJob {
    command: PendingResourceCommand,
    load_id: LoadId,
    done: Fence,
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
            handle_generator: HandleGenerator::default(),
        });
        Self {
            vk_device,
            allocator,
            locked,
        }
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
    pub fn load_buffer(
        &self,
        descriptor: BufferDescriptor<'static>,
        load_id: LoadId,
    ) -> Result<PendingBufferHandle, LoaderError> {
        log::trace!("Loading buffer with descriptor {descriptor:?}");
        assert!(!descriptor.is_empty());

        let mut guard = self.locked.lock().map_err(|_| LoaderError::Mutex)?;

        let raw_handle = guard.handle_generator.new_buffer();
        let handle = PendingBufferHandle {
            handle: raw_handle,
            idx: 0,
            n_elems: descriptor.n_elems(),
            mutability: descriptor.mutability(),
            ty: descriptor.buffer_type().ty(),
        };

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
                _descriptor: descriptor,
                handle,
                buffer0,
                buffer1,
                _transients: transients,
            },
            load_id,
            done,
        };
        guard.pending_resource_jobs.push(job);

        Ok(handle)
    }

    pub fn load_texture(
        &self,
        descriptor: TextureDescriptor<'static>,
        load_id: LoadId,
    ) -> Result<PendingTextureHandle, LoaderError> {
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
        let handle = PendingTextureHandle(guard.handle_generator.new_texture());
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
            load_id,
        };

        guard.pending_resource_jobs.push(job);
        Ok(handle)
    }

    pub fn flush(&self, load_id: LoadId) -> Vec<HandleMapping> {
        let mut guard = self.locked.lock().expect("Mutex poison");
        guard
            .done_resource_jobs
            .remove(&load_id)
            .unwrap_or_default()
    }

    /// Progresses the loader.
    /// This checks all pending jobs that have been queued to the loader and if they are done,
    /// resources are inserted into the renderer.
    pub fn progress(&self, renderer: &mut Renderer) {
        let mut guard = self.locked.lock().expect("Failed to unlock");
        let mut i = 0;
        while i < guard.pending_resource_jobs.len() {
            if guard.pending_resource_jobs[i].is_done() {
                let PendingResourceJob {
                    command,
                    load_id: request_id,
                    ..
                } = guard.pending_resource_jobs.remove(i);

                let mapping: HandleMapping = match command {
                    PendingResourceCommand::CreateBuffer {
                        handle,
                        buffer0,
                        buffer1,
                        ..
                    } => {
                        let mutability = if buffer1.is_some() {
                            BufferMutability::Mutable
                        } else {
                            BufferMutability::Immutable
                        };
                        let ty = buffer0.buffer_type().ty();
                        let n_elems = buffer0.n_elems();

                        let raw_handle: Handle<DeviceBuffer> =
                            renderer.resources.buffers.add(buffer0, buffer1);
                        let new_handle = unsafe {
                            BufferHandle::from_buffer(raw_handle, 0, n_elems, mutability, ty)
                        };

                        HandleMapping::Buffer {
                            old: handle,
                            new: new_handle,
                        }
                    }
                    PendingResourceCommand::CreateTexture {
                        handle, texture, ..
                    } => {
                        let new_handle = renderer.resources.textures.add(texture);
                        HandleMapping::Texture {
                            old: handle,
                            new: new_handle,
                        }
                    }
                };

                guard
                    .done_resource_jobs
                    .entry(request_id)
                    .or_default()
                    .push(mapping);
            } else {
                i += 1;
            }
        }
    }
}
