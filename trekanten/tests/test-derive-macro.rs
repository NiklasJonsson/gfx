use trekanten::traits::Std140;
use trekanten::Std140Compat;

#[allow(unused)]
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_a() {
        #[derive(Clone, Copy, Std140Compat)]
        struct A {
            x: f32,
            y: [f32; 2],
        }

        assert_eq!(A::offset_of_field_x(), 0);
        assert_eq!(A::offset_of_field_y(), 8);
        assert_eq!(A::ALIGNMENT, 16);
        assert_eq!(A::SIZE, 16);
        assert!(A::SIZE >= std::mem::size_of::<A>())
    }

    #[test]
    fn test_b() {
        #[derive(Clone, Copy, Std140Compat)]
        struct B {
            x: [f32; 3],
            y: f32,
        }

        assert_eq!(B::offset_of_field_x(), 0);
        assert_eq!(B::offset_of_field_y(), 12);
        assert_eq!(B::ALIGNMENT, 16);
        assert_eq!(B::SIZE, 16);
        assert!(B::SIZE >= std::mem::size_of::<B>())
    }

    #[test]
    fn test_c() {
        #[derive(Clone, Copy, Std140Compat)]
        struct C {
            a: [f32; 4],
            b: f32,
            c: f32,
            d: [f32; 4],
            e: f32,
        }

        assert_eq!(C::offset_of_field_a(), 0);
        assert_eq!(C::offset_of_field_b(), 16);
        assert_eq!(C::offset_of_field_c(), 20);
        assert_eq!(C::offset_of_field_d(), 32);
        assert_eq!(C::offset_of_field_e(), 48);
        assert_eq!(C::ALIGNMENT, 16);
        assert_eq!(C::SIZE, 64);
        assert!(C::SIZE >= std::mem::size_of::<C>())
    }

    #[test]
    fn test_d() {
        #[derive(Clone, Copy, Std140Compat)]
        struct D {
            a: f32,
            b: [f32; 4],
            c: f32,
            d: f32,
        }

        assert_eq!(D::offset_of_field_a(), 0);
        assert_eq!(D::offset_of_field_b(), 16);
        assert_eq!(D::offset_of_field_c(), 32);
        assert_eq!(D::offset_of_field_d(), 36);
        assert_eq!(D::ALIGNMENT, 16);
        assert_eq!(D::SIZE, 48);
        assert!(D::SIZE >= std::mem::size_of::<D>());
    }

    #[test]
    fn test_e() {
        #[derive(Clone, Copy, Std140Compat)]
        struct E {
            a: [f32; 16],
            b: [f32; 3],
        }

        assert_eq!(E::offset_of_field_a(), 0);
        assert_eq!(E::offset_of_field_b(), 64);
        assert_eq!(E::ALIGNMENT, 16);
        assert_eq!(E::SIZE, 80);
        assert!(E::SIZE >= std::mem::size_of::<E>());
    }

    #[test]
    fn test_nested() {
        #[derive(Clone, Copy, Std140Compat)]
        struct E {
            a: [f32; 16],
            b: [f32; 3],
        }

        #[derive(Clone, Copy, Std140Compat)]
        struct F {
            a: E,
            b: f32,
        }

        assert_eq!(F::offset_of_field_a(), 0);
        assert_eq!(F::offset_of_field_b(), 80);
        assert_eq!(F::ALIGNMENT, 16);
        assert_eq!(F::SIZE, 96);
        assert!(F::SIZE >= std::mem::size_of::<F>());
    }

    #[test]
    fn test_mat() {
        type Mat = [f32; 16];
        const LEN: usize = 16;
        assert_eq!(<Mat as Std140>::SIZE, 64);
        assert_eq!(<[Mat; LEN] as Std140>::SIZE, 1024);

        #[derive(Clone, Copy, Std140Compat)]
        pub struct Matrices {
            pub matrices: [Mat; LEN],
            pub num_matrices: u32,
        }

        assert_eq!(Matrices::offset_of_field_matrices(), 0);
        assert_eq!(Matrices::offset_of_field_num_matrices(), 1024);
        // It is not 1028 but 1040 because the size is rounded up to a multiple of 16
        assert_eq!(Matrices::SIZE, 1040);
        assert_eq!(Matrices::ALIGNMENT, 16);
        assert!(Matrices::SIZE >= std::mem::size_of::<Matrices>());
    }

    #[test]
    fn test_light() {
        const LEN: usize = 16;

        // TODO: structs in arrays (as members of structs)
        pub struct PackedLight {
            pub pos: [f32; 4],         // position for point/spot light
            pub dir_cutoff: [f32; 4], // direction for spot/directional light. .w is the cos(cutoff_angle) of the spotlight
            pub color_range: [f32; 4], // color for all light types. .w is the range of point/spot lights
            pub shadow_idx: [u32; 4],
        }

        pub struct LightingData {
            pub punctual_lights: [PackedLight; LEN],
            pub ambient: [f32; 4],
            pub num_lights: u32,
        }
    }
}
