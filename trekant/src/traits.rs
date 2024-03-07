/// # Safety
/// Has to fulfill the requirements of vulkan uniform. I.e. no padding in the struct.
pub unsafe trait Uniform: bytemuck::Pod {
    // The size of a uniform in bytes.
    fn size() -> u16;
    fn align() -> u16;
}

/// # Safety
/// Has to fulfill the requirements of vulkan push constants. I.e. no padding in the struct.
pub unsafe trait PushConstant: bytemuck::Pod {
    // The size of a push constant in bytes
    fn size() -> u16;
}

use std::convert::TryInto;

unsafe impl<T> Uniform for T
where
    T: crate::std140::Std140 + bytemuck::Pod,
{
    fn size() -> u16 {
        T::SIZE
            .try_into()
            .expect("Size too big, has to fit in 16 bits")
    }

    fn align() -> u16 {
        T::ALIGNMENT
            .try_into()
            .expect("Alignment too bug, has to fit in 16 bits")
    }
}

unsafe impl<T> PushConstant for T
where
    T: crate::std140::Std140 + bytemuck::Pod,
{
    fn size() -> u16 {
        T::SIZE
            .try_into()
            .expect("Size too big, has to fit in 16 bits")
    }
}
