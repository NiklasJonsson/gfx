use trekanten::buffer;
use trekanten::util;
use trekanten::vertex::{VertexDefinition, VertexFormat};

use crate::Chunk;

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct Vertex {
    pos: [f32; 3],
    nor: [f32; 3],
}

impl VertexDefinition for Vertex {
    fn format() -> VertexFormat {
        VertexFormat::builder()
            .add_attribute(util::Format::FLOAT3)
            .add_attribute(util::Format::FLOAT3)
            .build()
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

fn add_cube(mesh: &mut Mesh, point: [f32; 3], size: f32) {
    let [x, y, z] = point;

    let mk_pos = |x_rel: f32, y_rel: f32, z_rel: f32| -> [f32; 3] {
        [(x_rel * size + x), (y_rel * size + y), (z_rel * size + z)]
    };

    let vertices = [
        // Bottom
        Vertex {
            pos: mk_pos(-0.5, -0.5, 0.5),
            nor: [0.0, -1.0, 0.0],
        },
        Vertex {
            pos: mk_pos(-0.5, -0.5, -0.5),
            nor: [0.0, -1.0, 0.0],
        },
        Vertex {
            pos: mk_pos(0.5, -0.5, -0.5),
            nor: [0.0, -1.0, 0.0],
        },
        Vertex {
            pos: mk_pos(0.5, -0.5, 0.5),
            nor: [0.0, -1.0, 0.0],
        },
        // Top
        Vertex {
            pos: mk_pos(-0.5, 0.5, 0.5),
            nor: [0.0, 1.0, 0.0],
        },
        Vertex {
            pos: mk_pos(0.5, 0.5, 0.5),
            nor: [0.0, 1.0, 0.0],
        },
        Vertex {
            pos: mk_pos(0.5, 0.5, -0.5),
            nor: [0.0, 1.0, 0.0],
        },
        Vertex {
            pos: mk_pos(-0.5, 0.5, -0.5),
            nor: [0.0, 1.0, 0.0],
        },
        // Front
        Vertex {
            pos: mk_pos(-0.5, -0.5, 0.5),
            nor: [0.0, 0.0, 1.0],
        },
        Vertex {
            pos: mk_pos(0.5, -0.5, 0.5),
            nor: [0.0, 0.0, 1.0],
        },
        Vertex {
            pos: mk_pos(0.5, 0.5, 0.5),
            nor: [0.0, 0.0, 1.0],
        },
        Vertex {
            pos: mk_pos(-0.5, 0.5, 0.5),
            nor: [0.0, 0.0, 1.0],
        },
        // Back
        Vertex {
            pos: mk_pos(-0.5, -0.5, -0.5),
            nor: [0.0, 0.0, -1.0],
        },
        Vertex {
            pos: mk_pos(-0.5, 0.5, -0.5),
            nor: [0.0, 0.0, -1.0],
        },
        Vertex {
            pos: mk_pos(0.5, 0.5, -0.5),
            nor: [0.0, 0.0, -1.0],
        },
        Vertex {
            pos: mk_pos(0.5, -0.5, -0.5),
            nor: [0.0, 0.0, -1.0],
        },
        // Left
        Vertex {
            pos: mk_pos(-0.5, -0.5, -0.5),
            nor: [-1.0, 0.0, 0.0],
        },
        Vertex {
            pos: mk_pos(-0.5, -0.5, 0.5),
            nor: [-1.0, 0.0, 0.0],
        },
        Vertex {
            pos: mk_pos(-0.5, 0.5, 0.5),
            nor: [-1.0, 0.0, 0.0],
        },
        Vertex {
            pos: mk_pos(-0.5, 0.5, -0.5),
            nor: [-1.0, 0.0, 0.0],
        },
        // Right
        Vertex {
            pos: mk_pos(0.5, -0.5, -0.5),
            nor: [1.0, 0.0, 0.0],
        },
        Vertex {
            pos: mk_pos(0.5, 0.5, -0.5),
            nor: [1.0, 0.0, 0.0],
        },
        Vertex {
            pos: mk_pos(0.5, 0.5, 0.5),
            nor: [1.0, 0.0, 0.0],
        },
        Vertex {
            pos: mk_pos(0.5, -0.5, 0.5),
            nor: [1.0, 0.0, 0.0],
        },
    ];

    let indices = [
        0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, // Bottom/top
        8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, // Front/back
        16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20, // Left/right
    ]
    .map(|i| i + mesh.vertices.len() as u32);

    mesh.vertices.extend_from_slice(&vertices);
    mesh.indices.extend_from_slice(&indices);
}

const VOXEL_CUBE_SIDE: f32 = 1.0;

pub fn mesh(chunk: &crate::voxel::Chunk) -> Mesh {
    let mut m = Mesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };

    for z in 0..Chunk::SIDE {
        for y in 0..Chunk::SIDE {
            for x in 0..Chunk::SIDE {
                let i = chunk.index(x, y, z) as usize;
                let data = chunk.data[i];
                if data != 0 {
                    add_cube(&mut m, [x as f32, y as f32, z as f32], VOXEL_CUBE_SIDE);
                }
            }
        }
    }

    m
}
