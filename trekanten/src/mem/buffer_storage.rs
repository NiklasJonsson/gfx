use crate::resource::{BufferedStorage, Handle, Storage};

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

    pub fn get_all_mut(&mut self, h: &BufferHandle<T>) -> Option<(&mut T, Option<&mut T>)> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get_mut(h.handle()).map(|x| (x, None)),
            BufferMutability::Mutable => self
                .buffered
                .get_all_mut(h.handle())
                .map(|[x, y]| (x, Some(y))),
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

    // TODO: BufferHandle here. Needs buffer type which exposes sizes etc.
    pub fn add(&mut self, data0: T, data1: Option<T>) -> resurs::Handle<T> {
        match data1 {
            Some(data1) => self.buffered.add([data0, data1]),
            None => self.unbuffered.add(data0),
        }
    }

    pub fn get(&self, h: &BufferHandle<T>, idx: usize) -> Option<&T> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get(h.handle()),
            BufferMutability::Mutable => self.buffered.get(h.handle(), idx),
        }
    }

    pub fn get_mut(&mut self, h: &BufferHandle<T>, idx: usize) -> Option<&mut T> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get_mut(h.handle()),
            BufferMutability::Mutable => self.buffered.get_mut(h.handle(), idx),
        }
    }

    pub fn has(&self, h: &BufferHandle<T>) -> bool {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.has(h.handle()),
            BufferMutability::Mutable => self.buffered.has(h.handle()),
        }
    }

    pub fn drain_filter<F1, F2>(&mut self, f1: F1, f2: F2) -> DrainFilter<'_, F1, F2, T>
    where
        F1: FnMut(&mut T) -> bool,
        F2: FnMut(&mut [T; 2]) -> bool,
    {
        DrainFilter {
            unbuffered_iter: self.unbuffered.drain_filter(f1),
            buffered_iter: self.buffered.drain_filter(f2),
        }
    }
}

pub struct DrainFilter<'a, F1, F2, T>
where
    F1: FnMut(&mut T) -> bool,
    F2: FnMut(&mut [T; 2]) -> bool,
{
    unbuffered_iter: resurs::storage::DrainFilter<'a, F1, T>,
    buffered_iter: resurs::storage::DrainFilter<'a, F2, [T; 2]>,
}

impl<'a, F1, F2, T> Iterator for DrainFilter<'a, F1, F2, T>
where
    F1: FnMut(&mut T) -> bool,
    F2: FnMut(&mut [T; 2]) -> bool,
{
    type Item = (Handle<T>, T, Option<T>);
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if let Some((handle, item)) = self.unbuffered_iter.next() {
            return Some((handle, item, None));
        }

        if let Some((handle, [item0, item1])) = self.buffered_iter.next() {
            return Some((handle.as_unbuffered(), item0, Some(item1)));
        }

        return None;
    }
}

use crate::resource::Async;
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

pub type BufferStorageReadGuard<'a, T> = RwLockReadGuard<'a, DeviceBufferStorage<T>>;
pub type BufferStorageWriteGuard<'a, T> = RwLockWriteGuard<'a, DeviceBufferStorage<T>>;

pub struct AsyncDeviceBufferStorage<T> {
    inner: DeviceBufferStorage<Async<T>>,
}

impl<T> Default for AsyncDeviceBufferStorage<T> {
    fn default() -> Self {
        Self {
            inner: DeviceBufferStorage::default(),
        }
    }
}

impl<T> AsyncDeviceBufferStorage<T> {
    pub fn get_buffered(&self, h: &BufferHandle<Async<T>>, idx: usize) -> Option<&Async<T>> {
        self.inner.get_buffered(h, idx)
    }

    pub fn get_buffered_mut(
        &mut self,
        h: &BufferHandle<Async<T>>,
        idx: usize,
    ) -> Option<&mut Async<T>> {
        self.inner.get_buffered_mut(h, idx)
    }

    pub fn allocate<BD>(&mut self, desc: &BD) -> BufferHandle<Async<T>>
    where
        BD: BufferDescriptor<Buffer = T>,
    {
        let buffer1 = if let BufferMutability::Immutable = desc.mutability() {
            None
        } else {
            Some(Async::<T>::Pending)
        };
        let inner_handle = self.inner.add(Async::<T>::Pending, buffer1);
        unsafe { BufferHandle::from_buffer(inner_handle, 0, desc.n_elems(), desc.mutability()) }
    }

    pub fn cached<BD>(&self, _desc: &BD) -> Option<BufferHandle<Async<T>>>
    where
        BD: BufferDescriptor<Buffer = T>,
    {
        None
    }

    pub fn get(&self, h: &BufferHandle<Async<T>>, idx: usize) -> Option<&Async<T>> {
        self.inner.get(h, idx)
    }

    pub fn is_done(&self, h: &BufferHandle<Async<T>>) -> Option<bool> {
        self.inner
            .get_all(h)
            .map(|(buf0, buf1)| !buf0.is_pending() && buf1.map(|x| !x.is_pending()).unwrap_or(true))
    }

    pub fn insert(&mut self, h: &BufferHandle<Async<T>>, buf0: T, buf1: Option<T>) {
        if let Some((slot0, slot1)) = self.inner.get_all_mut(&h) {
            *slot0 = Async::Available(buf0);
            match (buf1, slot1) {
                (Some(buf1), Some(slot1)) => *slot1 = Async::Available(buf1),
                (None, None) => (),
                _ => unreachable!(
                    "mutability mismatch, expected buffers & pre-allocted slots to match"
                ),
            }
        }
    }

    pub fn drain_available(&mut self) -> DrainIterator<'_, T> {
        let f1 = |x: &mut Async<T>| std::matches!(x, Async::Available(_));
        let f2 = |x: &mut [Async<T>; 2]| std::matches!(x[0], Async::Available(_));
        self.inner.drain_filter(f1, f2)
    }
}

pub type DrainIterator<'a, T> =
    DrainFilter<'a, fn(&mut Async<T>) -> bool, fn(&mut [Async<T>; 2]) -> bool, Async<T>>;

use super::buffer::*;
pub type UniformBuffers = DeviceBufferStorage<UniformBuffer>;
pub type AsyncUniformBuffers = AsyncDeviceBufferStorage<UniformBuffer>;
pub type VertexBuffers = DeviceBufferStorage<VertexBuffer>;
pub type AsyncVertexBuffers = AsyncDeviceBufferStorage<VertexBuffer>;
pub type IndexBuffers = DeviceBufferStorage<IndexBuffer>;
pub type AsyncIndexBuffers = AsyncDeviceBufferStorage<IndexBuffer>;
