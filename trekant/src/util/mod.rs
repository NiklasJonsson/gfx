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
pub fn as_byte_slice<T: bytemuck::Pod>(slice: &[T]) -> &[u8] {
    bytemuck::cast_slice(slice)
}

/// # Safety
/// # Only call this is the type can be represented with all possible byte values
pub fn as_bytes<T: bytemuck::Pod>(v: &T) -> &[u8] {
    bytemuck::cast_slice(std::slice::from_ref(v))
}

/// # Safety
///  Only call this if A & B are transparent (repr(transparent))
pub unsafe fn cast_transparent_slice<A, B>(a: &[A]) -> &[B] {
    let ptr = a.as_ptr() as *const B;
    let len = a.len();
    std::slice::from_raw_parts(ptr, len)
}

/// Compute the stride requirement for elements with size 'elem_size' and alignment 'elem_align'.
///
/// Stride is used when laying out elements in a sequence in memory.
/// Assume a struct has size 9, and its member with the highest alignment requirement needs to be aligned to 8 bytes.
/// Then, to make sure we respect the alignment requirement, the stride needs to be 16.
///
/// In other words, the stride is the closest multiple of the alignment that is larger than the size.
///
pub fn compute_stride(elem_size: u16, elem_align: u16) -> u16 {
    assert!(elem_size != 0 && elem_align != 0);

    let padding = if elem_size % elem_align == 0 { 0 } else { 1 };
    ((elem_size / elem_align) + padding) * elem_align
}

/// Copy the elements in src to dst, assmuming:
/// * Elements in src are laid out linearly, with no padding.
/// * Elements in dst are aligned to elem_align, with possible padding in-between. This padding is not touched.
/// If the alignment and size of the elements are the same, dst is assumed to have no padding and a regular memcpy
/// is performed.
// TODO:
// * dst should be a slice, then we can panic if the copy is not alright
pub unsafe fn copy_nonoverlapping_aligned(
    src: &[u8],
    dst: *mut u8,
    elem_size: u16,
    elem_align: u16,
) {
    let size = src.len();
    let src: *const u8 = src.as_ptr();
    if elem_size == elem_align {
        unsafe {
            std::ptr::copy_nonoverlapping::<u8>(src, dst, size);
        }
    } else {
        let n_elems = size / elem_size as usize;
        let stride = compute_stride(elem_size, elem_align);
        for i in 0..n_elems {
            unsafe {
                let src = src.add(i * (elem_size as usize));
                let dst = dst.add(i * (stride as usize));
                std::ptr::copy_nonoverlapping::<u8>(src, dst, elem_size as usize);
            }
        }
    }
}

// SAFETY: Only call this on the components that come from a call to std::Vec::into_raw_parts
// Note that len and cap should be in *bytes*, this function will make the conversion
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

// SAFETY: Only call this on the components that come from a call to std::Vec::into_raw_parts
// Note that len and cap should be in *bytes*, this function will make the conversion
// to len/cap in number of *T*.
unsafe fn clone_impl<T: Copy + 'static>(ptr: *const u8, len: usize, cap: usize) -> ByteBuffer {
    // This implementation is maybe not ideal conceptually, we should be able to store the layout
    // and realloc/memcpy manually but this
    assert!(len % std::mem::size_of::<T>() == 0);
    assert!(cap % std::mem::size_of::<T>() == 0);
    let cloned = {
        let v = Vec::from_raw_parts(
            ptr as *mut T,
            len / std::mem::size_of::<T>(),
            cap / std::mem::size_of::<T>(),
        );

        let out = v.clone();
        std::mem::forget(v);
        out
    };

    ByteBuffer::from_vec(cloned)
}

/// A buffer containing bytes. Essentially a type-erased Box<\[T\]>``.
pub struct ByteBuffer {
    ptr: *const u8, // TODO: Non-null?
    len: usize,
    cap: usize,
    drop: unsafe fn(*const u8, usize, usize),
    clone: unsafe fn(*const u8, usize, usize) -> ByteBuffer,
}

impl Clone for ByteBuffer {
    fn clone(&self) -> Self {
        unsafe { (self.clone)(self.ptr, self.len, self.cap) }
    }
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
    // TODO: bytemuck bounds
    pub unsafe fn from_vec<T: Copy + 'static>(mut v: Vec<T>) -> Self {
        // TODO: use the into_raw_parts here when it is not nightly
        let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
        let drop = drop_as_vec::<T>;
        let clone = clone_impl::<T>;
        std::mem::forget(v);

        Self {
            ptr: ptr as *const u8,
            len: len * std::mem::size_of::<T>(),
            cap: cap * std::mem::size_of::<T>(),
            drop,
            clone,
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

    #[test]
    fn test_empty_byte_buffer() {
        use super::ByteBuffer;

        let v: Vec<i32> = vec![];
        let b = unsafe { ByteBuffer::from_vec(v) };
        let slice: &[u8] = &b;
        assert_eq!(slice.len(), 0);
        assert!(slice.is_empty());
    }

    #[test]
    fn test_stride() {
        use super::compute_stride;

        assert_eq!(4, compute_stride(4, 4));
        assert_eq!(16, compute_stride(15, 8));
        assert_eq!(16, compute_stride(9, 8));
        assert_eq!(8, compute_stride(3, 8));
        assert_eq!(32, compute_stride(3, 32));
        assert_eq!(32, compute_stride(32, 8));
        assert_eq!(16, compute_stride(16, 8));
    }
}
