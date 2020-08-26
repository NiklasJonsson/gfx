pub mod material;
pub mod math;
pub mod render_graph;
pub mod time;

use specs::prelude::*;
use specs::Component;

pub use material::*;
pub use math::*;
pub use time::*;

use std::path::PathBuf;

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexPosOnly {
    pub position: [f32; 3],
}

#[derive(Debug)]
pub enum VertexBuf {
    PosOnly(Vec<VertexPosOnly>),
}

mod vertex {
    pub fn pos_only(x: f32, y: f32, z: f32) -> super::VertexPosOnly {
        super::VertexPosOnly {
            position: [x, y, z],
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
