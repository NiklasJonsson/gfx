use super::*;
use nalgebra_glm as glm;
use nalgebra_glm::{Mat4, Vec3};
use specs::Component;
use std::ops::{AddAssign, Mul};

// TODO: Auto derive inner type traits
#[derive(Debug, Component, Clone, Copy)]
#[storage(VecStorage)]
pub struct Position(pub Vec3);

impl Position {
    pub fn x(&self) -> f32 {
        self.0.x
    }
    pub fn y(&self) -> f32 {
        self.0.y
    }
    pub fn z(&self) -> f32 {
        self.0.z
    }
}

impl Position {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(glm::vec3(x, y, z))
    }

    pub fn max() -> Self {
        Self::new(f32::MAX, f32::MAX, f32::MAX)
    }

    pub fn min() -> Self {
        Self::new(f32::MIN, f32::MIN, f32::MIN)
    }
}

impl From<[f32; 3]> for Position {
    fn from(x: [f32; 3]) -> Self {
        Position::new(x[0], x[1], x[2])
    }
}

impl From<Vec3> for Position {
    fn from(src: Vec3) -> Self {
        Position::new(src.x, src.y, src.z)
    }
}

impl AddAssign<&Vec3> for Position {
    fn add_assign(&mut self, other: &Vec3) {
        self.0 += other;
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "({:.2}, {:.2}, {:.2})", self.x(), self.y(), self.z())
    }
}

impl Mul<Position> for ModelMatrix {
    type Output = Position;
    fn mul(self, pos: Position) -> Position {
        let v4 = glm::vec4(pos.x(), pos.y(), pos.z(), 1.0);
        let r = self.0 * v4;
        Position(glm::vec4_to_vec3(&r))
    }
}

#[derive(Debug, Component, Copy, Clone)]
#[storage(DenseVecStorage)]
pub struct Transform(Mat4);

impl From<[[f32; 4]; 4]> for Transform {
    fn from(x: [[f32; 4]; 4]) -> Self {
        Transform(x.into())
    }
}

impl From<Mat4> for Transform {
    fn from(x: Mat4) -> Self {
        Transform(x)
    }
}

impl Into<Mat4> for Transform {
    fn into(self) -> Mat4 {
        self.0
    }
}

impl Transform {
    pub fn identity() -> Transform {
        Self(glm::identity::<f32, glm::U4>())
    }
}

#[derive(Debug, Component, Clone, Copy)]
#[storage(DenseVecStorage)]
#[repr(transparent)]
pub struct ModelMatrix(pub Mat4);

impl From<[[f32; 4]; 4]> for ModelMatrix {
    fn from(x: [[f32; 4]; 4]) -> Self {
        Self(x.into())
    }
}

impl Into<[[f32; 4]; 4]> for ModelMatrix {
    fn into(self) -> [[f32; 4]; 4] {
        self.0.into()
    }
}

impl From<Mat4> for ModelMatrix {
    fn from(x: Mat4) -> Self {
        Self(x)
    }
}

impl Into<Mat4> for ModelMatrix {
    fn into(self) -> Mat4 {
        self.0
    }
}

impl ModelMatrix {
    pub fn identity() -> ModelMatrix {
        Self(glm::identity::<f32, glm::U4>())
    }
}

impl std::fmt::Display for ModelMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/*
pub fn to_vertices_and_indices(&self) -> (VertexBuf, IndexData) {
    assert!(self.min.iter().all(|x| x != std::f32::MAX));
    assert!(self.max.iter().all(|x| x != std::f32::MIN));

    let BoundingBox { min, max } = self;

    let mut vbuf: Vec<VertexPosOnly> = Vec::with_capacity(8);
    // Face 1
    vbuf.push(vertex::pos_only(min.x(), min.y(), min.z()));
    vbuf.push(vertex::pos_only(max.x(), min.y(), min.z()));
    vbuf.push(vertex::pos_only(max.x(), max.y(), min.z()));
    vbuf.push(vertex::pos_only(min.x(), max.y(), min.z()));

    // Face 2
    vbuf.push(vertex::pos_only(min.x(), min.y(), max.z()));
    vbuf.push(vertex::pos_only(max.x(), min.y(), max.z()));
    vbuf.push(vertex::pos_only(max.x(), max.y(), max.z()));
    vbuf.push(vertex::pos_only(min.x(), max.y(), max.z()));

    let vbuf = VertexBuf::PosOnly(vbuf);

    // 12 lines make a box, each has two vertices
    let mut ibuf: Vec<u32> = Vec::with_capacity(24);
    // Face 1
    ibuf.push(0);
    ibuf.push(1);
    ibuf.push(1);
    ibuf.push(2);
    ibuf.push(2);
    ibuf.push(3);
    ibuf.push(3);
    ibuf.push(0);

    // One side
    ibuf.push(0);
    ibuf.push(4);

    // Face 2
    ibuf.push(4);
    ibuf.push(5);
    ibuf.push(5);
    ibuf.push(6);
    ibuf.push(6);
    ibuf.push(7);
    ibuf.push(7);
    ibuf.push(4);

    // Rest of the sides
    ibuf.push(7);
    ibuf.push(3);
    ibuf.push(6);
    ibuf.push(2);
    ibuf.push(5);
    ibuf.push(1);
    assert_eq!(ibuf.len(), 24);

    let ibuf = IndexData(ibuf);

    (vbuf, ibuf)
}
*/
