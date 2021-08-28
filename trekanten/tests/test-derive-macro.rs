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
        assert_eq!(A::SIZE, 16);
        assert_eq!(A::ALIGNMENT, 16);
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
        assert_eq!(B::SIZE, 16);
        assert_eq!(B::ALIGNMENT, 16);
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
        assert_eq!(C::SIZE, 64);
        assert_eq!(C::ALIGNMENT, 16);
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
        assert_eq!(D::SIZE, 48);
        assert_eq!(D::ALIGNMENT, 16);
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
        assert_eq!(E::SIZE, 80);
        assert_eq!(E::ALIGNMENT, 16);
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
        assert_eq!(F::SIZE, 96);
        assert_eq!(F::ALIGNMENT, 16);
    }
}
