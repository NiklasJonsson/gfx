pub mod extent;
pub mod ffi;
pub mod format;
pub mod lifetime;
pub mod offset;
pub mod rect;
pub mod viewport;

pub use extent::*;
pub use format::*;
pub use offset::*;
pub use rect::*;
pub use viewport::*;

pub fn clamp<T: Ord>(v: T, min: T, max: T) -> T {
    std::cmp::max(min, std::cmp::min(v, max))
}

pub const fn max(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}

pub const fn round_to_multiple(n: usize, multiple: usize) -> usize {
    if n == 0 {
        multiple
    } else if n % multiple == 0 {
        n
    } else {
        n + (multiple - (n % multiple))
    }
}

/// # Safety
/// # Only call this is the type can be represented with all possible byte values
pub unsafe fn as_byte_slice<T: Copy>(slice: &[T]) -> &[u8] {
    let ptr = slice.as_ptr() as *const u8;
    let size = std::mem::size_of::<T>() * slice.len();
    std::slice::from_raw_parts(ptr, size)
}

/// # Safety
/// # Only call this is the type can be represented with all possible byte values
pub unsafe fn as_bytes<T: Copy>(v: &T) -> &[u8] {
    let ptr = (v as *const T) as *const u8;
    let size = std::mem::size_of::<T>();
    std::slice::from_raw_parts(ptr, size)
}

/// # Safety
///  Only call this if A & B are transparent (repr(transparent))
pub unsafe fn cast_transparent_slice<A, B>(a: &[A]) -> &[B] {
    let ptr = a.as_ptr() as *const B;
    let len = a.len();
    std::slice::from_raw_parts(ptr, len)
}

// SAFETY: Only call this on the components that come from a call to std::Vec::into_raw_parts
// Note that len and cap should is expected to be in *bytes*, this function will make the conversion
// to len/cap in number of *T*.
unsafe fn drop_as_vec<T>(ptr: *const u8, len: usize, cap: usize) {
    assert!(len % std::mem::size_of::<T>() == 0);
    assert!(cap % std::mem::size_of::<T>() == 0);
    std::mem::drop(Vec::from_raw_parts(
        ptr as *mut T,
        len / std::mem::size_of::<T>(),
        cap / std::mem::size_of::<T>(),
    ));
}

/// A buffer containing bytes. Essentially a type-erased Box<[T]>.
pub struct ByteBuffer {
    ptr: *const u8,
    len: usize,
    cap: usize,
    drop: unsafe fn(*const u8, usize, usize),
}

impl std::ops::Deref for ByteBuffer {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
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
    /// # Safety
    /// Very unsafe
    /// The vector should really only contain pods or the like, require Copy to kind of enforce this.
    /// Note: after this, the contents of the vector might be passed accross ffi boundaries, e.g. to the gpu.
    pub unsafe fn from_vec<T: Copy + 'static>(mut v: Vec<T>) -> Self {
        // TODO: use the into_raw_parts here when it is not nightly
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

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

unsafe impl Send for ByteBuffer {}
unsafe impl Sync for ByteBuffer {}

impl Drop for ByteBuffer {
    fn drop(&mut self) {
        unsafe {
            (self.drop)(self.ptr, self.len, self.cap);
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_round() {
        use super::round_to_multiple;

        let check = |x, m, e| {
            assert_eq!(round_to_multiple(x, m), e);
        };

        check(0, 100, 100);
        check(15, 100, 100);
        check(100, 100, 100);
        check(101, 100, 200);
        check(4, 7, 7);
        check(8, 7, 14);
        check(12, 16, 16);
        check(12, 4, 12);
    }
}
