use trekanten::mem::{BufferMutability, OwningIndexBufferDescriptor, OwningVertexBufferDescriptor};
use trekanten::util::Format;
use trekanten::vertex::{VertexDefinition, VertexFormat};

#[derive(Copy, Clone)]
struct PosVertex {
    _pos: [f32; 3],
}

fn pos(x: f32, y: f32, z: f32) -> PosVertex {
    PosVertex { _pos: [x, y, z] }
}

impl VertexDefinition for PosVertex {
    fn format() -> VertexFormat {
        VertexFormat::builder()
            .add_attribute(Format::FLOAT3)
            .build()
    }
}

pub type Mesh = (OwningVertexBufferDescriptor, OwningIndexBufferDescriptor);

/// Right-handed
/// origin-centered box. x,y,z are the length of the sides.
/// CCW triangles
pub fn box_mesh(x: f32, y: f32, z: f32) -> Mesh {
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
        0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, // Top/bottom
        8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, // Front/back
        16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20, // Left/right
    ];

    let vertices = OwningVertexBufferDescriptor::from_vec(vertices, BufferMutability::Immutable);
    let indices = OwningIndexBufferDescriptor::from_vec(indices, BufferMutability::Immutable);

    (vertices, indices)
}

/// Right-handed coordinates
pub fn sphere_mesh(radius: f32) -> Mesh {
    // phi rotates around y. Theta from (0, 1, 0) to (0, -1, 0)
    // ISO Spherical coordinates
    // Note that phi is sampled once for the beginning and once for the end, to provide proper
    // texture coordinates.
    let n_phi_samples = 17u32;
    let n_theta_samples = 9u32;

    let mut vertices = Vec::with_capacity((n_phi_samples * n_theta_samples) as usize);
    let mut indices: Vec<u32> = Vec::new();

    for i in 0..n_theta_samples {
        for j in 0..n_phi_samples {
            let theta_ratio = i as f32 / (n_theta_samples - 1) as f32;
            let phi_ratio = j as f32 / (n_phi_samples - 1) as f32;

            let phi = std::f32::consts::PI * 2.0 * phi_ratio;
            let theta = std::f32::consts::PI * theta_ratio;

            let x = radius * theta.sin() * phi.cos();
            let y = radius * theta.cos();
            let z = -radius * theta.sin() * phi.sin();
            vertices.push(PosVertex { _pos: [x, y, z] });

            if i < n_theta_samples - 1 && j < n_phi_samples - 1 {
                indices.push(n_phi_samples * i + j);
                indices.push(n_phi_samples * i + (j + 1));
                indices.push(n_phi_samples * (i + 1) + (j + 1));

                indices.push(n_phi_samples * i + j);
                indices.push(n_phi_samples * (i + 1) + (j + 1));
                indices.push(n_phi_samples * (i + 1) + j);
            }
        }
    }

    let vertices = OwningVertexBufferDescriptor::from_vec(vertices, BufferMutability::Immutable);
    let indices = OwningIndexBufferDescriptor::from_vec(indices, BufferMutability::Immutable);

    (vertices, indices)
}

// Cone with circle at origin in z/x, height in +y
pub fn cone_mesh(radius: f32, height: f32) -> Mesh {
    let n_angle_samples = 17u32;
    let mut vertices = Vec::with_capacity(n_angle_samples as usize + 2);
    vertices.push(pos(0.0, 0.0, 0.0));
    vertices.push(pos(0.0, height, 0.0));
    for i in 0..n_angle_samples {
        let angle = (i as f32 / (n_angle_samples - 1) as f32) as f32 * std::f32::consts::PI * 2.0;
        vertices.push(pos(radius * angle.cos(), 0.0, -radius * angle.sin()));
    }

    let n_triangles = (n_angle_samples as usize - 1) * 2;
    let mut indices = Vec::with_capacity(n_triangles * 3);
    let circle_vertices_offset = 2;
    for i in 0..(n_angle_samples - 1) {
        indices.push(1);
        indices.push(circle_vertices_offset + i);
        indices.push(circle_vertices_offset + i + 1);

        indices.push(0);
        indices.push(circle_vertices_offset + i + 1);
        indices.push(circle_vertices_offset + i);
    }

    let vertices = OwningVertexBufferDescriptor::from_vec(vertices, BufferMutability::Immutable);
    let indices = OwningIndexBufferDescriptor::from_vec(indices, BufferMutability::Immutable);
    (vertices, indices)
}
