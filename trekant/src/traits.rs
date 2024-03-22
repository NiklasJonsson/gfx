/// # Safety
/// Has to fulfill the requirements of vulkan uniform. I.e. no padding in the struct.
pub unsafe trait Uniform: Copy + BufferContents {
    // The size of a uniform in bytes.
    fn size() -> u16;
}

/// # Safety
/// Has to fulfill the requirements of vulkan push constants. I.e. no padding in the struct.
pub unsafe trait PushConstant: Copy {
    // The size of a pust constant in bytes
    fn size() -> u16;
}

use std::convert::TryInto;

unsafe impl<T> Uniform for T
where
    T: crate::std140::Std140 + Copy,
{
    fn size() -> u16 {
        T::SIZE
            .try_into()
            .expect("Size too big, has to fit in 16 bits")
    }
}

unsafe impl<T> PushConstant for T
where
    T: crate::std140::Std140 + Copy,
{
    fn size() -> u16 {
        T::SIZE
            .try_into()
            .expect("Size too big, has to fit in 16 bits")
    }
}
