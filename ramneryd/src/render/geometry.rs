use crate::ecs::prelude::*;
use crate::math::Vec3;
use trekanten::buffer::{HostIndexBuffer, HostVertexBuffer};
use trekanten::util::Format;
use trekanten::vertex::{VertexDefinition, VertexFormat};

use super::mesh::Mesh;

use ramneryd_derive::Visitable;

#[derive(Copy, Clone)]
struct Vertex {
    _pos: [f32; 3],
    _nor: [f32; 3],
}

impl VertexDefinition for Vertex {
    fn format() -> VertexFormat {
        VertexFormat::from([Format::FLOAT3; 2])
    }
}

/// Right-handed
/// origin-centered box. x,y,z are the length of the sides.
/// CCW triangles
pub fn box_mesh(x: f32, y: f32, z: f32) -> Mesh {
    assert!(x > f32::EPSILON && y > f32::EPSILON && z > f32::EPSILON);
    let vertices = vec![
        // Bottom
        Vertex {
            _pos: [-0.5 * x, -0.5 * y, 0.5 * z],
            _nor: [0.0, -1.0, 0.0],
        },
        Vertex {
            _pos: [-0.5 * x, -0.5 * y, -0.5 * z],
            _nor: [0.0, -1.0, 0.0],
        },
        Vertex {
            _pos: [0.5 * x, -0.5 * y, -0.5 * z],
            _nor: [0.0, -1.0, 0.0],
        },
        Vertex {
            _pos: [0.5 * x, -0.5 * y, 0.5 * z],
            _nor: [0.0, -1.0, 0.0],
        },
        // Top
        Vertex {
            _pos: [-0.5 * x, 0.5 * y, 0.5 * z],
            _nor: [0.0, 1.0, 0.0],
        },
        Vertex {
            _pos: [0.5 * x, 0.5 * y, 0.5 * z],
            _nor: [0.0, 1.0, 0.0],
        },
        Vertex {
            _pos: [0.5 * x, 0.5 * y, -0.5 * z],
            _nor: [0.0, 1.0, 0.0],
        },
        Vertex {
            _pos: [-0.5 * x, 0.5 * y, -0.5 * z],
            _nor: [0.0, 1.0, 0.0],
        },
        // Front
        Vertex {
            _pos: [-0.5 * x, -0.5 * y, 0.5 * z],
            _nor: [0.0, 0.0, 1.0],
        },
        Vertex {
            _pos: [0.5 * x, -0.5 * y, 0.5 * z],
            _nor: [0.0, 0.0, 1.0],
        },
        Vertex {
            _pos: [0.5 * x, 0.5 * y, 0.5 * z],
            _nor: [0.0, 0.0, 1.0],
        },
        Vertex {
            _pos: [-0.5 * x, 0.5 * y, 0.5 * z],
            _nor: [0.0, 0.0, 1.0],
        },
        // Back
        Vertex {
            _pos: [-0.5 * x, -0.5 * y, -0.5 * z],
            _nor: [0.0, 0.0, -1.0],
        },
        Vertex {
            _pos: [-0.5 * x, 0.5 * y, -0.5 * z],
            _nor: [0.0, 0.0, -1.0],
        },
        Vertex {
            _pos: [0.5 * x, 0.5 * y, -0.5 * z],
            _nor: [0.0, 0.0, -1.0],
        },
        Vertex {
            _pos: [0.5 * x, -0.5 * y, -0.5 * z],
            _nor: [0.0, 0.0, -1.0],
        },
        // Left
        Vertex {
            _pos: [-0.5 * x, -0.5 * y, -0.5 * z],
            _nor: [-1.0, 0.0, 0.0],
        },
        Vertex {
            _pos: [-0.5 * x, -0.5 * y, 0.5 * z],
            _nor: [-1.0, 0.0, 0.0],
        },
        Vertex {
            _pos: [-0.5 * x, 0.5 * y, 0.5 * z],
            _nor: [-1.0, 0.0, 0.0],
        },
        Vertex {
            _pos: [-0.5 * x, 0.5 * y, -0.5 * z],
            _nor: [-1.0, 0.0, 0.0],
        },
        // Right
        Vertex {
            _pos: [0.5 * x, -0.5 * y, -0.5 * z],
            _nor: [1.0, 0.0, 0.0],
        },
        Vertex {
            _pos: [0.5 * x, 0.5 * y, -0.5 * z],
            _nor: [1.0, 0.0, 0.0],
        },
        Vertex {
            _pos: [0.5 * x, 0.5 * y, 0.5 * z],
            _nor: [1.0, 0.0, 0.0],
        },
        Vertex {
            _pos: [0.5 * x, -0.5 * y, 0.5 * z],
            _nor: [1.0, 0.0, 0.0],
        },
    ];

    let indices: Vec<u16> = vec![
        0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4, // Bottom/top
        8, 9, 10, 10, 11, 8, 12, 13, 14, 14, 15, 12, // Front/back
        16, 17, 18, 18, 19, 16, 20, 21, 22, 22, 23, 20, // Left/right
    ];

    let vertices = HostVertexBuffer::from_vec(vertices);
    let indices = HostIndexBuffer::from_vec(indices);

    Mesh::new(vertices, indices)
}

/// Right-handed coordinates
pub fn sphere_mesh(radius: f32) -> Mesh {
    assert!(radius > f32::EPSILON);
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
            let normal = Vec3 { x, y, z }.normalized().into_array();
            vertices.push(Vertex {
                _pos: [x, y, z],
                _nor: normal,
            });

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

    let vertices = HostVertexBuffer::from_vec(vertices);
    let indices = HostIndexBuffer::from_vec(indices);

    Mesh::new(vertices, indices)
}

// Cone with circle at origin in z/x, height in +y
pub fn cone_mesh(radius: f32, height: f32) -> Mesh {
    assert!(radius > f32::EPSILON && height > f32::EPSILON);
    // TODO: revisit prealloc
    let n_angle_samples = 17u32;
    let mut vertices = Vec::with_capacity(n_angle_samples as usize + 2);
    let base_vertex = Vertex {
        _pos: [0.0, 0.0, 0.0],
        _nor: [0.0, -1.0, 0.0],
    };

    vertices.push(base_vertex);

    let n_triangles = (n_angle_samples as usize - 1) * 2;
    let mut indices = Vec::with_capacity(n_triangles * 3);

    // No need to worry about oob for i here as we are constructing an angle, which can be > 360
    let angle_at =
        |i: u32| (i as f32 / (n_angle_samples - 1) as f32) as f32 * std::f32::consts::PI * 2.0;
    let pos_at = |i: u32| Vec3 {
        x: radius * angle_at(i).cos(),
        y: 0.0,
        z: -radius * angle_at(i).sin(),
    };

    // base
    for i in 0..n_angle_samples {
        let pos = pos_at(i);
        vertices.push(Vertex {
            _pos: pos.into_array(),
            _nor: [0.0, -1.0, 0.0],
        });

        // Reuse base vertex as normals are all the same
        if i < n_angle_samples - 1 {
            indices.push(0);
            indices.push(i);
            indices.push(i + 1);
        }
    }

    let apex = Vec3 {
        x: 0.0,
        y: height,
        z: 0.0,
    };
    // angled surface
    for i in 0..n_angle_samples {
        let pos = pos_at(i);
        let next_pos = pos_at(i + 1);
        let tangent0 = next_pos - pos;
        let tangent1 = apex - pos;
        let normal = tangent0.cross(tangent1).normalized();

        // Need to create a new apex vertex as it has different normals
        vertices.push(Vertex {
            _pos: apex.into_array(),
            _nor: normal.into_array(),
        });

        vertices.push(Vertex {
            _pos: pos.into_array(),
            _nor: normal.into_array(),
        });

        if i < n_angle_samples - 1 {
            debug_assert!(vertices.len() >= 2);
            // apex
            indices.push(vertices.len() as u32 - 2);
            // current point
            indices.push(vertices.len() as u32 - 1);
            // Skip the next apex and use the next point on the edge
            indices.push(vertices.len() as u32 + 1);
        }
    }

    let vertices = HostVertexBuffer::from_vec(vertices);
    let indices = HostIndexBuffer::from_vec(indices);
    Mesh::new(vertices, indices)
}

fn box_lines_strip(bounds: [Vec3; 3], center: Vec3) -> [Vec3; 16] {
    let [u, v, w] = bounds;
    let c = center;
    // TODO: Can we shrink this?
    [
        c + u - v + w,
        c + u + v + w,
        c - u + v + w,
        c - u - v + w,
        c - u - v - w,
        c - u + v - w,
        c + u + v - w,
        c + u - v - w,
        c + u - v + w,
        c - u - v + w,
        c - u + v + w,
        c - u + v - w,
        c - u - v - w,
        c + u - v - w,
        c + u + v - w,
        c + u + v + w,
    ]
}

pub fn obb_line_strip(obb: crate::math::Obb) -> [Vec3; 16] {
    box_lines_strip(obb.uvw(), obb.center())
}

pub fn aabb_line_strip(aabb: crate::math::Aabb) -> [Vec3; 16] {
    let dims = (aabb.max - aabb.min) / Vec3::from(2.0);
    let bounds = [
        Vec3::unit_x() * dims.x,
        Vec3::unit_y() * dims.y,
        Vec3::unit_z() * dims.z,
    ];
    let c = (aabb.max + aabb.min) / Vec3::from(2.0);
    box_lines_strip(bounds, c)
}

#[derive(Debug, Component, Visitable, Clone, Copy)]
pub enum Shape {
    Box { width: f32, height: f32, depth: f32 },
    Sphere { radius: f32 },
    Cone { radius: f32, height: f32 },
}

pub struct ShapeMeshCreation;

impl ShapeMeshCreation {
    pub const ID: &'static str = "ShapeMeshCreation";
}

impl<'a> System<'a> for ShapeMeshCreation {
    type SystemData = (Entities<'a>, ReadStorage<'a, Shape>, WriteStorage<'a, Mesh>);
    fn run(&mut self, (entities, shapes, mut meshes): Self::SystemData) {
        for (ent, shape) in (&entities, &shapes).join() {
            if meshes.contains(ent) {
                continue;
            }

            let mesh = match *shape {
                Shape::Box {
                    width,
                    height,
                    depth,
                } => box_mesh(width, height, depth),
                Shape::Sphere { radius } => sphere_mesh(radius),
                Shape::Cone { radius, height } => cone_mesh(radius, height),
            };

            meshes.insert(ent, mesh).expect("This was already checked");
        }
    }
}

pub fn register_systems(builder: ExecutorBuilder) -> ExecutorBuilder {
    builder.with(ShapeMeshCreation, ShapeMeshCreation::ID, &[])
}
