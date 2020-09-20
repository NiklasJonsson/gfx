use crate::command::CommandPool;
use crate::device::Device;
use crate::queue::Queue;
use crate::resource::{BufferedStorage, Storage};

use super::MemoryError;
use super::{Buffer, BufferDescriptor, BufferHandle, BufferMutability};

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

impl<T> DeviceBufferStorage<T>
where
    T: Buffer,
{
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
        queue: &Queue,
        command_pool: &CommandPool,
        desc: &BD,
    ) -> Result<BufferHandle<BD::Buffer>, MemoryError>
    where
        BD: BufferDescriptor<Buffer = T>,
    {
        let (h, stride) = match desc.mutability() {
            BufferMutability::Immutable => {
                log::trace!("Creating immutable buffer");
                let buf = desc.create(device, queue, command_pool)?;
                let stride = buf.stride();
                (self.unbuffered.add(buf), stride)
            }
            BufferMutability::Mutable => {
                log::trace!("Creating buffered buffer");
                let bufs = [
                    desc.create(device, queue, command_pool)?,
                    desc.create(device, queue, command_pool)?,
                ];
                let stride = bufs[0].stride();

                (self.buffered.add(bufs), stride)
            }
        };

        Ok(unsafe {
            BufferHandle::from_buffer(
                h,
                0,
                desc.n_elems(),
                desc.elem_size(),
                stride,
                desc.mutability(),
            )
        })
    }

    pub fn recreate<BD>(
        &mut self,
        device: &Device,
        queue: &Queue,
        command_pool: &CommandPool,
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
            let new = desc.create(device, queue, command_pool)?;
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
