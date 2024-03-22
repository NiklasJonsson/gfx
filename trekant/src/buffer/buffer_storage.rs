use crate::resource::{BufferedStorage, Handle, Storage};

use super::{AsyncBufferHandle, BufferDescriptor, BufferHandle, BufferMutability, DeviceBuffer};

pub struct DeviceBufferStorage {
    buffered: BufferedStorage<DeviceBuffer>,
    unbuffered: Storage<DeviceBuffer>,
}

impl Default for DeviceBufferStorage {
    fn default() -> Self {
        Self {
            buffered: BufferedStorage::<DeviceBuffer>::default(),
            unbuffered: Storage::<DeviceBuffer>::default(),
        }
    }
}

impl DeviceBufferStorage {
    pub fn get_all(&self, h: BufferHandle) -> Option<(&DeviceBuffer, Option<&DeviceBuffer>)> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get(h.handle()).map(|x| (x, None)),
            BufferMutability::Mutable => {
                self.buffered.get_all(h.handle()).map(|[x, y]| (x, Some(y)))
            }
        }
    }

    pub fn get_all_mut(
        &mut self,
        h: BufferHandle,
    ) -> Option<(&mut DeviceBuffer, Option<&mut DeviceBuffer>)> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get_mut(h.handle()).map(|x| (x, None)),
            BufferMutability::Mutable => self
                .buffered
                .get_all_mut(h.handle())
                .map(|[x, y]| (x, Some(y))),
        }
    }

    pub fn get_buffered(&self, h: BufferHandle, idx: usize) -> Option<&DeviceBuffer> {
        assert_eq!(h.mutability(), BufferMutability::Mutable);
        self.buffered.get(h.handle(), idx)
    }

    pub fn get_buffered_mut(&mut self, h: BufferHandle, idx: usize) -> Option<&mut DeviceBuffer> {
        assert_eq!(h.mutability(), BufferMutability::Mutable);
        self.buffered.get_mut(h.handle(), idx)
    }

    pub fn get_unbuffered(&self, h: BufferHandle) -> Option<&DeviceBuffer> {
        assert_eq!(h.mutability(), BufferMutability::Immutable);
        self.unbuffered.get(h.handle())
    }

    // TODO: BufferHandle here. Needs buffer type which exposes sizes etc.
    pub fn add(
        &mut self,
        data0: DeviceBuffer,
        data1: Option<DeviceBuffer>,
    ) -> resurs::Handle<DeviceBuffer> {
        match data1 {
            Some(data1) => self.buffered.add([data0, data1]),
            None => self.unbuffered.add(data0),
        }
    }

    pub fn get(&self, h: BufferHandle, idx: usize) -> Option<&DeviceBuffer> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get(h.handle()),
            BufferMutability::Mutable => self.buffered.get(h.handle(), idx),
        }
    }

    pub fn get_mut(&mut self, h: BufferHandle, idx: usize) -> Option<&mut DeviceBuffer> {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.get_mut(h.handle()),
            BufferMutability::Mutable => self.buffered.get_mut(h.handle(), idx),
        }
    }

    pub fn has(&self, h: BufferHandle) -> bool {
        match h.mutability() {
            BufferMutability::Immutable => self.unbuffered.has(h.handle()),
            BufferMutability::Mutable => self.buffered.has(h.handle()),
        }
    }

    pub fn drain_filter<F1, F2>(&mut self, f1: F1, f2: F2) -> DrainFilter<'_, F1, F2, DeviceBuffer>
    where
        F1: FnMut(&mut DeviceBuffer) -> bool,
        F2: FnMut(&mut [DeviceBuffer; 2]) -> bool,
    {
        DrainFilter {
            unbuffered_iter: self.unbuffered.drain_filter(f1),
            buffered_iter: self.buffered.drain_filter(f2),
        }
    }
}

pub struct DrainFilter<'a, F1, F2, DeviceBuffer>
where
    F1: FnMut(&mut DeviceBuffer) -> bool,
    F2: FnMut(&mut [DeviceBuffer; 2]) -> bool,
{
    unbuffered_iter: resurs::storage::DrainFilter<'a, F1, DeviceBuffer>,
    buffered_iter: resurs::storage::DrainFilter<'a, F2, [DeviceBuffer; 2]>,
}

impl<'a, F1, F2, DeviceBuffer> Iterator for DrainFilter<'a, F1, F2, DeviceBuffer>
where
    F1: FnMut(&mut DeviceBuffer) -> bool,
    F2: FnMut(&mut [DeviceBuffer; 2]) -> bool,
{
    type Item = (Handle<DeviceBuffer>, DeviceBuffer, Option<DeviceBuffer>);
    fn next(&mut self) -> Option<<Self as Iterator>::Item> {
        if let Some((handle, item)) = self.unbuffered_iter.next() {
            return Some((handle, item, None));
        }

        if let Some((handle, [item0, item1])) = self.buffered_iter.next() {
            return Some((handle.as_unbuffered(), item0, Some(item1)));
        }

        None
    }
}

use crate::resource::Async;

pub struct AsyncDeviceBufferStorage {
    buffered: BufferedStorage<Async<DeviceBuffer>>,
    unbuffered: Storage<Async<DeviceBuffer>>,
}

impl Default for AsyncDeviceBufferStorage {
    fn default() -> Self {
        Self {
            buffered: Default::default(),
            unbuffered: Default::default(),
        }
    }
}

impl AsyncDeviceBufferStorage {
    pub fn get_buffered(&self, h: AsyncBufferHandle, idx: usize) -> Option<&Async<DeviceBuffer>> {
        self.buffered.get(&h.h, idx)
    }

    pub fn get_buffered_mut(
        &mut self,
        h: AsyncBufferHandle,
        idx: usize,
    ) -> Option<&mut Async<DeviceBuffer>> {
        self.buffered.get_mut(&h.h, idx)
    }

    pub fn allocate<'a>(&mut self, desc: &BufferDescriptor<'a>) -> AsyncBufferHandle {
        let raw_handle = match desc.mutability() {
            BufferMutability::Immutable => self.unbuffered.add(Async::<DeviceBuffer>::Pending),
            BufferMutability::Mutable => self.buffered.add([
                Async::<DeviceBuffer>::Pending,
                Async::<DeviceBuffer>::Pending,
            ]),
        };
        unsafe {
            AsyncBufferHandle::from_buffer(
                raw_handle,
                desc.mutability(),
                desc.n_elems(),
                desc.buffer_type().ty(),
            )
        }
    }

    pub fn insert(&mut self, h: AsyncBufferHandle, buf0: DeviceBuffer, buf1: Option<DeviceBuffer>) {
        todo!()
        /*
               if let Some((slot0, slot1)) = self.inner.get_all_mut(h) {
                   *slot0 = Async::Available(buf0);
                   match (buf1, slot1) {
                       (Some(buf1), Some(slot1)) => *slot1 = Async::Available(buf1),
                       (None, None) => (),
                       _ => unreachable!(
                           "mutability mismatch, expected buffers & pre-allocted slots to match"
                       ),
                   }
               }
        */
    }

    pub fn drain_available(&mut self) -> DrainIterator<'_, DeviceBuffer> {
        /*
        let f1 = |x: &mut Async<T>| std::matches!(x, Async::Available(_));
        let f2 = |x: &mut [Async<T>; 2]| std::matches!(x[0], Async::Available(_));
        self.inner.drain_filter(f1, f2)
        */
        todo!()
    }
}

pub type DrainIterator<'a, T> =
    DrainFilter<'a, fn(&mut Async<T>) -> bool, fn(&mut [Async<T>; 2]) -> bool, Async<T>>;
