pub mod material;
pub mod math;
pub mod render_graph;
pub mod time;

use specs::prelude::*;
use specs::Component;

use vulkano::impl_vertex;

use crate::render::texture::Texture;
pub use material::*;
pub use math::*;
pub use time::*;

use std::path::PathBuf;

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexPosOnly {
    pub position: [f32; 3],
}

impl_vertex!(VertexPosOnly, position);

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

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexUVTan {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: [f32; 2],
    pub tangent: [f32; 4],
}

impl_vertex!(VertexUVTan, position, normal, tex_coords, tangent);

impl From<([f32; 3], [f32; 3], [f32; 2], [f32; 4])> for VertexUVTan {
    fn from(tpl: ([f32; 3], [f32; 3], [f32; 2], [f32; 4])) -> Self {
        Self {
            position: tpl.0,
            normal: tpl.1,
            tex_coords: tpl.2,
            tangent: tpl.3,
        }
    }
}

#[derive(Debug)]
pub enum VertexBuf {
    PosOnly(Vec<VertexPosOnly>),
    Base(Vec<VertexBase>),
    UV(Vec<VertexUV>),
    UVCol(Vec<VertexUVCol>),
    UVTan(Vec<VertexUVTan>),
}

mod vertex {
    pub fn pos_only(x: f32, y: f32, z: f32) -> super::VertexPosOnly {
        super::VertexPosOnly {
            position: [x, y, z],
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub enum ComponentLayout {
    R8G8B8A8,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    Linear,
    Srgb,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct Format {
    pub component_layout: ComponentLayout,
    pub color_space: ColorSpace,
}

use vulkano::format::Format as VkFormat;
impl Into<VkFormat> for Format {
    fn into(self) -> VkFormat {
        match (self.component_layout, self.color_space) {
            (ComponentLayout::R8G8B8A8, ColorSpace::Srgb) => VkFormat::R8G8B8A8Srgb,
            (ComponentLayout::R8G8B8A8, ColorSpace::Linear) => VkFormat::R8G8B8A8Unorm,
        }
    }
}

#[derive(Debug)]
pub struct IndexData(pub Vec<u32>);
#[derive(Debug)]
pub enum MeshType {
    Triangle {
        triangle_indices: IndexData,
        line_indices: IndexData,
    },
    Line {
        indices: IndexData,
    },
}

// One ore more vertices with associated data
#[derive(Debug, Component)]
#[storage(DenseVecStorage)]
pub struct Mesh {
    pub ty: MeshType,
    pub vertex_data: VertexBuf,
    // TODO: Move this to it's own component?
    pub bounding_box: Option<BoundingBox>,
}
