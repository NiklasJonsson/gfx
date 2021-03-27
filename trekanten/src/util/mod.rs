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

fn drop_as_vec<T>(ptr: *const u8, len: usize, cap: usize) {
    assert!(len % std::mem::size_of::<T>() == 0);
    assert!(cap % std::mem::size_of::<T>() == 0);
    std::mem::drop(unsafe {
        Vec::from_raw_parts(
            ptr as *mut T,
            len / std::mem::size_of::<T>(),
            cap / std::mem::size_of::<T>(),
        )
    })
}

pub struct ByteBuffer {
    ptr: *const u8,
    len: usize,
    cap: usize,
    drop: fn(*const u8, usize, usize),
}

impl std::ops::Deref for ByteBuffer {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }
}

impl std::fmt::Debug for ByteBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = self.len;
        let ptr = self.ptr;
        write!(f, "({:?}, {:?})[{}]", ptr, ptr.wrapping_add(len), len)
    }
}

impl ByteBuffer {
    // Requiring copy enforces that there is no custom drop that is needed
    // Very unsafe
    pub unsafe fn from_vec<T: Copy + 'static>(mut v: Vec<T>) -> Self {
        let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
        let drop = drop_as_vec::<T>;
        std::mem::forget(v);

        Self {
            ptr: ptr as *const u8,
            len: len * std::mem::size_of::<T>(),
            cap: cap * std::mem::size_of::<T>(),
            drop,
        }
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
}

unsafe impl Send for ByteBuffer {}
unsafe impl Sync for ByteBuffer {}

impl Drop for ByteBuffer {
    fn drop(&mut self) {
        (self.drop)(self.ptr, self.len, self.cap);
    }
}
