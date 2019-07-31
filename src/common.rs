use specs::prelude::*;
use std::ops::AddAssign;

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexBase {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}
impl_vertex!(VertexBase, position, normal);

impl From<([f32; 3], [f32; 3])> for VertexBase {
    fn from(tpl: ([f32; 3], [f32; 3])) -> Self {
        Self {
            position: tpl.0,
            normal: tpl.1,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexUV {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl_vertex!(VertexUV, position, normal, tex_coords);

impl From<([f32; 3], [f32; 3], [f32; 2])> for VertexUV {
    fn from(tpl: ([f32; 3], [f32; 3], [f32; 2])) -> Self {
        Self {
            position: tpl.0,
            normal: tpl.1,
            tex_coords: tpl.2,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexUVCol {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
    pub color: [f32; 4],
}

impl_vertex!(VertexUVCol, position, normal, tex_coords, color);

impl From<([f32; 3], [f32; 3], [f32; 2], [f32; 4])> for VertexUVCol {
    fn from(tpl: ([f32; 3], [f32; 3], [f32; 2], [f32; 4])) -> Self {
        Self {
            position: tpl.0,
            normal: tpl.1,
            tex_coords: tpl.2,
            color: tpl.3,
        }
    }
}

// TODO: Generate the other vertex defs from this
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: Option<[f32; 2]>,
    pub color: Option<[f32; 4]>,
}

pub enum VertexBuf {
    Base(Vec<VertexBase>),
    UV(Vec<VertexUV>),
    UVCol(Vec<VertexUVCol>),
}

// TODO: Auto derive inner type traits
#[derive(Debug, Component)]
#[storage(VecStorage)]
pub struct Position(glm::Vec3);

impl Position {
    pub fn to_vec3(&self) -> glm::Vec3 {
        self.0
    }
}

impl From<glm::Vec3> for Position {
    fn from(src: glm::Vec3) -> Self {
        Position(src)
    }
}

impl AddAssign<&glm::Vec3> for Position {
    fn add_assign(&mut self, other: &glm::Vec3) {
        self.0 += other;
    }
}
