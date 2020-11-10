use crate::resource::{BufferedStorage, Storage};

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
}

use crate::resource::Async;
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, RwLock, RwLockReadGuard, RwLockWriteGuard,
};
use std::sync::Arc;

pub type BufferStorageReadGuard<'a, T> = RwLockReadGuard<'a, DeviceBufferStorage<T>>;

pub struct AsyncDeviceBufferStorage<T> {
    inner: Arc<RwLock<DeviceBufferStorage<Async<T>>>>,
}

impl<T> Default for AsyncDeviceBufferStorage<T> {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(DeviceBufferStorage::default())),
        }
    }
}

impl<T> AsyncDeviceBufferStorage<T> {
    pub fn get_buffered(
        &self,
        h: &BufferHandle<T>,
        idx: usize,
    ) -> Option<MappedRwLockReadGuard<'_, Async<T>>> {
        let g = self.inner.read();
        let h = h.wrap_async();
        if !g.has(&h) {
            return None;
        }

        Some(RwLockReadGuard::map(g, |x| {
            x.get_buffered(&h, idx).unwrap()
        }))
    }

    pub fn allocate<BD>(&self, desc: &BD) -> BufferHandle<T>
    where
        BD: BufferDescriptor<Buffer = T>,
    {
        let buffer1 = if let BufferMutability::Immutable = desc.mutability() {
            None
        } else {
            Some(Async::<T>::Pending)
        };
        let inner_handle = self
            .inner
            .write()
            .add(Async::<T>::Pending, buffer1)
            .unwrap_async();
        unsafe { BufferHandle::from_buffer(inner_handle, 0, desc.n_elems(), desc.mutability()) }
    }

    pub fn read(&self) -> BufferStorageReadGuard<'_, Async<T>> {
        self.inner.read()
    }

    pub fn cache<BD>(&self, _desc: &BD) -> Option<BufferHandle<T>>
    where
        BD: BufferDescriptor<Buffer = T>,
    {
        None
    }

    pub fn get_buffered_mut(
        &self,
        h: &BufferHandle<T>,
        idx: usize,
    ) -> Option<MappedRwLockWriteGuard<'_, Async<T>>> {
        let g = self.inner.write();
        let h = h.wrap_async();
        if !g.has(&h) {
            return None;
        }

        Some(RwLockWriteGuard::map(g, |inner| {
            inner.get_buffered_mut(&h, idx).unwrap()
        }))
    }

    pub fn get(
        &self,
        h: &BufferHandle<T>,
        idx: usize,
    ) -> Option<MappedRwLockReadGuard<'_, Async<T>>> {
        let g = self.inner.read();
        let h = h.wrap_async();
        if !g.has(&h) {
            return None;
        }

        Some(RwLockReadGuard::map(g, |inner| inner.get(&h, idx).unwrap()))
    }

    pub fn is_done(&self, h: &BufferHandle<T>) -> Option<bool> {
        self.inner
            .read()
            .get_all(&h.wrap_async())
            .map(|(buf0, buf1)| !buf0.is_pending() && buf1.map(|x| !x.is_pending()).unwrap_or(true))
    }

    pub fn insert(&self, h: &BufferHandle<T>, buf0: T, buf1: Option<T>) {
        if let Some((slot0, slot1)) = self.inner.write().get_all_mut(&h.wrap_async()) {
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
}
