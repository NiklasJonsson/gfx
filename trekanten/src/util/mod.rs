pub mod extent;
pub mod ffi;
pub mod format;
pub mod lifetime;
pub mod vk_debug;

pub use extent::*;
pub use format::*;

pub fn clamp<T: Ord>(v: T, min: T, max: T) -> T {
    std::cmp::max(min, std::cmp::min(v, max))
}

pub fn as_byte_slice<T>(slice: &[T]) -> &[u8] {
    let ptr = slice.as_ptr() as *const u8;
    let size = std::mem::size_of::<T>() * slice.len();
    unsafe { std::slice::from_raw_parts(ptr, size) }
}

pub fn as_bytes<T>(v: &T) -> &[u8] {
    let ptr = (v as *const T) as *const u8;
    let size = std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, size) }
}
