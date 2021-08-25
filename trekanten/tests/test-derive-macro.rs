use trekanten::traits::Std140;
use trekanten::Std140Compat;

#[derive(Clone, Copy, Std140Compat)]
#[allow(unused)]
struct A {
    x: f32,
    y: [f32; 2],
}

#[test]
fn test_a() {
    assert_eq!(A::SIZE, 16);
    assert_eq!(A::alignment_of_field_x(), 0);
    assert_eq!(A::alignment_of_field_y(), 8);
}
