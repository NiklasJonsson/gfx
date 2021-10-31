/// Std140 according to the opengl spec. Use the trekanten::Std140 to derive
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
        // rule 2
        impl_std140!([$ty; 4], 16, 16);
        // rule 5
        impl_std140!([$ty; 16], 16 * 4, 16);

        // rule 4
        unsafe impl<const N: usize> Std140 for [[$ty; 4]; N] {
            const SIZE: usize =
                crate::util::round_to_multiple(<[$ty; 4] as Std140>::SIZE * N, Self::ALIGNMENT);
            const ALIGNMENT: usize =
                crate::util::round_to_multiple(<[$ty; 4] as Std140>::ALIGNMENT, 16);
        }

        // rule 6
        unsafe impl<const N: usize> Std140 for [[$ty; 16]; N] {
            const SIZE: usize =
                crate::util::round_to_multiple(<[$ty; 16] as Std140>::SIZE * N, Self::ALIGNMENT);
            const ALIGNMENT: usize =
                crate::util::round_to_multiple(<[$ty; 16] as Std140>::ALIGNMENT, 16);
        }
    };
}

impl_std140_scalar!(f32);
impl_std140_scalar!(u32);

/* TODO: Support generic args and generic sizes when we can add bounds on N (>= 3)
// Otherwise we will get conflicting impl for e.g vec2
unsafe impl<T: Std140, const N: usize> Std140 for [T; N] {
    const SIZE: usize = crate::util::round_to_multiple(T::SIZE * N, Self::ALIGNMENT);
    const ALIGNMENT: usize = crate::util::round_to_multiple(T::ALIGNMENT, 16);
}
*/

pub unsafe trait Uniform: Copy {
    // The size of a uniform in bytes. Note that there cannot be padding in a struct
    fn size() -> u16;
}

pub unsafe trait PushConstant: Copy {
    // The size of a pust constant in bytes. Note that there cannot be padding in a struct
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
