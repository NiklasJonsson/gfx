use crate::generics::True;

// TODO: Hacky way to get compile time bounds on constant generics
pub struct Gt3<const N: usize> {}
macro_rules! impl_gt3 {
    ($n:expr) => {
        impl True for Gt3<$n> {}
    };
    ($x:expr, $($y:expr),+) => {
        impl_gt3!($x);
        impl_gt3!($($y),+);
    };
}

impl_gt3!(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64);

/// Std140 according to the opengl spec. Use the trekant::Std140 to derive
pub unsafe trait Std140 {
    const SIZE: usize;
    const ALIGNMENT: usize;
}

macro_rules! impl_std140 {
    ($ty:ty, $size:expr, $align:expr) => {
        unsafe impl Std140 for $ty {
            const SIZE: usize = $size;
            const ALIGNMENT: usize = $align;
        }
    };
}

macro_rules! impl_std140_scalar {
    ($ty:ty) => {
        // rule 1
        impl_std140!($ty, 4, 4);
        // rule 2
        impl_std140!([$ty; 2], 8, 8);
        // rule 3
        impl_std140!([$ty; 3], 12, 16);
    };
}

impl_std140_scalar!(f32);
impl_std140_scalar!(u32);

// rule 2, 4, 5, 6
unsafe impl<T: Std140, const N: usize> Std140 for [T; N]
where
    Gt3<N>: True,
{
    const SIZE: usize = crate::util::round_to_multiple(T::SIZE * N, Self::ALIGNMENT);
    const ALIGNMENT: usize = crate::util::round_to_multiple(T::ALIGNMENT, 16);
}

/// # Safety
/// Has to fulfill the requirements of vulkan uniform. I.e. no padding in the struct.
pub unsafe trait Uniform: Copy {
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
    T: Std140 + Copy,
{
    fn size() -> u16 {
        T::SIZE
            .try_into()
            .expect("Size too big, has to fit in 16 bits")
    }
}

unsafe impl<T> PushConstant for T
where
    T: Std140 + Copy,
{
    fn size() -> u16 {
        T::SIZE
            .try_into()
            .expect("Size too big, has to fit in 16 bits")
    }
}
