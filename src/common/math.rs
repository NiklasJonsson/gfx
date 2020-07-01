use super::*;
use nalgebra_glm as glm;
use nalgebra_glm::{Mat4, Vec3};
use specs::Component;
use std::ops::{AddAssign, Mul};

// TODO: Auto derive inner type traits
#[derive(Debug, Component, Clone, Copy)]
#[storage(VecStorage)]
pub struct Position(glm::Vec4);

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

    pub fn xyz(&self) -> Vec3 {
        glm::vec3(self.0.x, self.0.y, self.0.z)
    }

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(glm::vec4(x, y, z, 1.0))
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = f32> + 'a {
        PositionIterator::<'a> { pos: self, idx: 0 }
    }
}

/// Iterates over the three coordinates of a position
struct PositionIterator<'a> {
    pos: &'a Position,
    idx: usize,
}

impl<'a> Iterator for PositionIterator<'a> {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx > 2 {
            return None;
        }

        let r = Some(self.pos.0[self.idx]);
        self.idx += 1;
        r
    }
}

impl From<[f32; 3]> for Position {
    fn from(x: [f32; 3]) -> Self {
        Position::new(x[0], x[1], x[2])
    }
}

impl From<glm::Vec3> for Position {
    fn from(src: glm::Vec3) -> Self {
        Position::new(src.x, src.y, src.z)
    }
}

impl AddAssign<&glm::Vec3> for Position {
    fn add_assign(&mut self, other: &glm::Vec3) {
        self.0 += glm::vec4(other.x, other.y, other.z, 0.0);
        assert!(self.0.w == 1.0f32);
    }
}

impl Mul<Position> for ModelMatrix {
    type Output = Position;
    fn mul(self, pos: Position) -> Position {
        Position(self.0 * pos.0)
    }
}

// TODO: Alias this?
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

#[derive(Debug)]
pub struct BoundingBox {
    pub min: Position,
    pub max: Position,
}

impl Default for BoundingBox {
    fn default() -> Self {
        BoundingBox {
            min: Position::new(std::f32::MAX, std::f32::MAX, std::f32::MAX),
            max: Position::new(std::f32::MIN, std::f32::MIN, std::f32::MIN),
        }
    }
}

impl BoundingBox {
    pub fn combine_with(&mut self, other: &Self) {
        // min
        let x = self.min.x().min(other.min.x());
        let y = self.min.y().min(other.min.y());
        let z = self.min.z().min(other.min.z());
        self.min = Position::new(x, y, z);

        // max
        let x = self.max.x().max(other.max.x());
        let y = self.max.y().max(other.max.y());
        let z = self.max.z().max(other.max.z());
        self.max = Position::new(x, y, z);
    }

    pub fn to_vertices_and_indices(self) -> (VertexBuf, IndexData) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_iterator() {
        let pos = Position::new(0.2, 0.3, 0.4);
        let mut it = pos.iter();

        assert_eq!(Some(pos.x()), it.next());
        assert_eq!(Some(pos.y()), it.next());
        assert_eq!(Some(pos.z()), it.next());
        assert_eq!(None, it.next());
    }
}
