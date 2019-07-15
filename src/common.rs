use specs::prelude::*;
use std::ops::AddAssign;

trait VertexVariant {
    fn get_defines() -> Vec<(String, String)>;
}

#[derive(Copy, Clone, Debug)]
pub struct VertexBase {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}
impl_vertex!(VertexBase, position, normal);

impl VertexVariant for VertexBase {
    fn get_defines() -> Vec<(String, String)> {
        Vec::new()
    }
}

impl From<([f32; 3], [f32; 3])> for VertexBase {
    fn from(tpl: ([f32; 3], [f32; 3])) -> Self {
        Self {position: tpl.0, normal: tpl.1}
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VertexUV {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],

}

impl_vertex!(VertexUV, position, normal, tex_coords);

impl VertexVariant for VertexUV {
    fn get_defines() -> Vec<(String, String)> {
        vec![("HAS_TEX_COORDS".to_owned(), "1".to_owned())]
    }
}

impl From<([f32; 3], [f32; 3], [f32; 2])> for VertexUV {
    fn from(tpl: ([f32; 3], [f32; 3], [f32; 2])) -> Self {
        Self {position: tpl.0, normal: tpl.1, tex_coords: tpl.2}
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: Option<[f32; 2]>,
}

pub enum VertexBuf {
    Base(Vec<VertexBase>),
    UV(Vec<VertexUV>),
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
