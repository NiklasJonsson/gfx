use specs::prelude::*;

pub mod math;
pub mod render_graph;

pub use math::*;

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

// TODO: Generate the other vertex defs from this
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coords: Option<[f32; 2]>,
    pub color: Option<[f32; 4]>,
}

#[derive(Debug)]
pub enum VertexBuf {
    PosOnly(Vec<VertexPosOnly>),
    Base(Vec<VertexBase>),
    UV(Vec<VertexUV>),
    UVCol(Vec<VertexUVCol>),
}

mod vertex {
    pub fn pos_only(x: f32, y: f32, z: f32) -> super::VertexPosOnly {
        super::VertexPosOnly {
            position: [x, y, z],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Texture {
    pub image: image::RgbaImage,
    pub coord_set: u32,
}

#[derive(Clone, Debug)]
pub enum Material {
    Color {
        color: [f32; 4],
    },
    ColorTexture(Texture),
    GlTFPBR {
        base_color_factor: [f32; 4],
        metallic_factor: f32,
        roughness_factor: f32,
        base_color_texture: Option<Texture>,
    },
    None,
}

#[derive(Debug)]
pub struct IndexData(pub Vec<u32>);

// REFACTOR:
// - Rename
// - Make a single enum and no struct?
// - Create structs for each enum content
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

// One ore more vertices with associated data on how to render them
#[derive(Debug, Component)]
#[storage(DenseVecStorage)]
pub struct PolygonMesh {
    pub ty: MeshType,
    pub vertex_data: VertexBuf,
    pub material: Material,
    // TODO: Move this to it's own component?
    pub bounding_box: Option<BoundingBox>,
}
