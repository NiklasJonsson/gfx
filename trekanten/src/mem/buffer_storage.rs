use crate::command::CommandBuffer;
use crate::device::Device;
use crate::resource::{BufferedStorage, Storage};

use super::MemoryError;
use super::{BufferDescriptor, BufferHandle, BufferMutability};

pub struct DeviceBufferStorage<T> {
    buffered: BufferedStorage<T>,
    unbuffered: Storage<T>,
}

impl<T> Default for DeviceBufferStorage<T> {
    fn default() -> Self {
        Self {
            buffered: BufferedStorage::<T>::default(),
            unbuffered: Storage::<T>::default(),
        }
    }
}

impl<T> DeviceBufferStorage<T> {
    pub fn get_all(&self, h: &BufferHandle<T>) -> Option<(&T, Option<&T>)> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get(h.handle()).map(|x| (x, None)),
            BufferMutability::Mutable => {
                self.buffered.get_all(h.handle()).map(|[x, y]| (x, Some(y)))
            }
        }
    }

    pub fn get_buffered(&self, h: &BufferHandle<T>, idx: usize) -> Option<&T> {
        assert_eq!(h.mutability(), BufferMutability::Mutable);
        self.buffered.get(h.handle(), idx)
    }

    pub fn get_buffered_mut(&mut self, h: &BufferHandle<T>, idx: usize) -> Option<&mut T> {
        assert_eq!(h.mutability(), BufferMutability::Mutable);
        self.buffered.get_mut(h.handle(), idx)
    }

    pub fn get_unbuffered(&self, h: &BufferHandle<T>) -> Option<&T> {
        assert_eq!(h.mutability(), BufferMutability::Immutable);
        self.unbuffered.get(h.handle())
    }

    pub fn get_either(&self, h: &BufferHandle<T>, idx: usize) -> Option<&T> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get(h.handle()),
            BufferMutability::Mutable => self.buffered.get(h.handle(), idx),
        }
    }

    pub fn create<BD>(
        &mut self,
        device: &Device,
        command_buffer: &mut CommandBuffer,
        desc: &BD,
    ) -> Result<(BufferHandle<BD::Buffer>, Vec<Option<super::DeviceBuffer>>), MemoryError>
    where
        BD: BufferDescriptor<Buffer = T>,
    {
        let mut staging_buffers = Vec::new();
        let stride = desc.stride(device);
        let h = match desc.mutability() {
            BufferMutability::Immutable => {
                log::trace!("Creating immutable buffer");
                let (buf, staging) = desc.create(device, command_buffer)?;
                staging_buffers.push(staging);
                self.unbuffered.add(buf)
            }
            BufferMutability::Mutable => {
                log::trace!("Creating buffered buffer");
                let (buf0, staging0) = desc.create(device, command_buffer)?;
                let (buf1, staging1) = desc.create(device, command_buffer)?;
                let bufs = [buf0, buf1];
                staging_buffers.push(staging0);
                staging_buffers.push(staging1);
                self.buffered.add(bufs)
            }
        };

        Ok((
            unsafe {
                BufferHandle::from_buffer(
                    h,
                    0,
                    desc.n_elems(),
                    desc.elem_size(),
                    stride,
                    desc.mutability(),
                )
            },
            staging_buffers,
        ))
    }

    pub fn recreate<BD>(
        &mut self,
        device: &Device,
        command_buffer: &mut CommandBuffer,
        handle: BufferHandle<BD::Buffer>,
        idx: usize,
        desc: &BD,
    ) -> Result<BufferHandle<BD::Buffer>, MemoryError>
    where
        BD: BufferDescriptor<Buffer = T>,
    {
        assert_eq!(desc.mutability(), handle.mutability());
        if let BufferMutability::Mutable = desc.mutability() {
            log::trace!("Recreating buffered buffer at {}", idx);
            let (new, staging) = desc.create(device, command_buffer)?;
            assert!(staging.is_none(), "Can't recreate with staging");
            *self
                .buffered
                .get_mut(handle.handle(), idx)
                .expect("Bad handle") = new;
            Ok(handle)
        } else {
            unreachable!("Can't recreate immutable buffer");
        }
    }
}
