pub mod extent;
pub mod ffi;
pub mod format;
pub mod lifetime;
pub mod offset;
pub mod rect;
pub mod viewport;
pub mod vk_debug;

pub use extent::*;
pub use format::*;
pub use offset::*;
pub use rect::*;
pub use viewport::*;

pub fn clamp<T: Ord>(v: T, min: T, max: T) -> T {
    std::cmp::max(min, std::cmp::min(v, max))
}

pub fn as_byte_slice<T: Copy>(slice: &[T]) -> &[u8] {
    let ptr = slice.as_ptr() as *const u8;
    let size = std::mem::size_of::<T>() * slice.len();
    unsafe { std::slice::from_raw_parts(ptr, size) }
}

pub fn as_bytes<T: Copy>(v: &T) -> &[u8] {
    let ptr = (v as *const T) as *const u8;
    let size = std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, size) }
}

/// Only deallocated memory
pub struct ByteBuffer {
    data: Vec<u8>,
    layout: std::alloc::Layout,
}

impl std::ops::Deref for ByteBuffer {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::fmt::Debug for ByteBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.data.len();
        let ptr = self.data.as_ptr();
        write!(f, "({:?}, {:?})[{}]", ptr, ptr.wrapping_add(len), len)
    }
}

impl ByteBuffer {
    // Requiring copy enforces that there is no custom drop that is needed
    // Very unsafe. Does not call drop() for elements
    pub unsafe fn from_vec<T: Copy>(mut v: Vec<T>) -> Self {
        if v.is_empty() {
            return Self::empty();
        }
        // TODO: Static assertion
        assert!(std::mem::size_of::<T>() > 0, "ZST are not supported");
        let ptr = v.as_mut_ptr();
        let orig_len = v.len();
        let orig_cap = v.capacity();
        let len = orig_len * std::mem::size_of::<T>();
        let cap = orig_cap * std::mem::size_of::<T>();
        std::mem::forget(v);

        // TODO: Use RawVec directly?
        // From the implementation of alloc::raw_vec::RawVec which std::alloc::Vec uses
        // internally:

        // We have an allocated chunk of memory, so we can bypass runtime
        // checks to get our current layout.
        let align = std::mem::align_of::<T>();
        let size = std::mem::size_of::<T>() * orig_cap;
        let layout = std::alloc::Layout::from_size_align_unchecked(size, align);

        let data: Vec<u8> = Vec::from_raw_parts(ptr as *mut u8, len, cap);
        Self { data, layout }
    }

    pub fn empty() -> Self {
        let layout = unsafe { std::alloc::Layout::from_size_align_unchecked(0, 0) };
        Self {
            data: Vec::new(),
            layout,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl Drop for ByteBuffer {
    fn drop(&mut self) {
        if self.data.is_empty() {
            return;
        }
        let mut owned = std::mem::take(&mut self.data);
        let ptr = owned.as_mut_ptr();
        std::mem::forget(owned);
        unsafe {
            std::alloc::dealloc(ptr as *mut u8, self.layout);
        }
    }
}
