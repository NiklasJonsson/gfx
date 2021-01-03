use trekanten::mesh::{
    BufferMutability, OwningIndexBufferDescriptor, OwningVertexBufferDescriptor,
};
use trekanten::util::Format;
use trekanten::vertex::{VertexDefinition, VertexFormat};

#[derive(Copy, Clone)]
struct PosVertex {
    _pos: [f32; 3],
}

impl VertexDefinition for PosVertex {
    fn format() -> VertexFormat {
        VertexFormat::builder()
            .add_attribute(Format::FLOAT3)
            .build()
    }
}

/// origin-centered box. x,y,z are the length of the sides.
/// CCW triangles
pub fn box_mesh(
    x: f32,
    y: f32,
    z: f32,
) -> (OwningVertexBufferDescriptor, OwningIndexBufferDescriptor) {
    let vertices = vec![
        // Bottom
        PosVertex {
            _pos: [-0.5 * x, -0.5 * y, 0.5 * z],
        },
        PosVertex {
            _pos: [-0.5 * x, -0.5 * y, -0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, -0.5 * y, -0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, -0.5 * y, 0.5 * z],
        },
        // Top
        PosVertex {
            _pos: [-0.5 * x, 0.5 * y, 0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, 0.5 * y, 0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, 0.5 * y, -0.5 * z],
        },
        PosVertex {
            _pos: [-0.5 * x, 0.5 * y, -0.5 * z],
        },
        // Front
        PosVertex {
            _pos: [-0.5 * x, -0.5 * y, 0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, -0.5 * y, 0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, 0.5 * y, 0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, 0.5 * y, 0.5 * z],
        },
        // Back
        PosVertex {
            _pos: [-0.5 * x, -0.5 * y, -0.5 * z],
        },
        PosVertex {
            _pos: [-0.5 * x, 0.5 * y, -0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, 0.5 * y, -0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, -0.5 * y, -0.5 * z],
        },
        // Left
        PosVertex {
            _pos: [-0.5 * x, -0.5 * y, -0.5 * z],
        },
        PosVertex {
            _pos: [-0.5 * x, -0.5 * y, 0.5 * z],
        },
        PosVertex {
            _pos: [-0.5 * x, 0.5 * y, 0.5 * z],
        },
        PosVertex {
            _pos: [-0.5 * x, 0.5 * y, -0.5 * z],
        },
        // Right
        PosVertex {
            _pos: [0.5 * x, -0.5 * y, -0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, 0.5 * y, -0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, 0.5 * y, 0.5 * z],
        },
        PosVertex {
            _pos: [0.5 * x, -0.5 * y, 0.5 * z],
        },
    ];

    let indices = vec![
        // Top/bottom
        0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, // Front/back
        8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, // Left/right
        16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20,
    ];

    let vertices = OwningVertexBufferDescriptor::from_vec(vertices, BufferMutability::Immutable);
    let indices = OwningIndexBufferDescriptor::from_vec(indices, BufferMutability::Immutable);

    (vertices, indices)
}
