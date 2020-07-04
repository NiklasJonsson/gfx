pub mod math;
pub mod render_graph;
pub mod time;

use specs::prelude::*;
use specs::Component;

use vulkano::impl_vertex;

use crate::render::texture::Texture;
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

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum ComponentLayout {
    R8G8B8A8,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Linear,
    Srgb,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
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

#[derive(Clone, Debug)]
pub struct NormalMap {
    pub tex: Texture,
    pub scale: f32,
}

/// Compile time means that the material will be used to lookup what pre-compiled shader to use.
#[derive(Debug, Clone, Eq)]
pub enum CompilationMode {
    CompileTime,
    RunTime { vs_path: PathBuf, fs_path: PathBuf },
}

impl PartialEq for CompilationMode {
    fn eq(&self, other: &Self) -> bool {
        use CompilationMode::*;
        match (self, other) {
            (CompileTime, CompileTime) => true,
            (RunTime { .. }, RunTime { .. }) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub enum MaterialData {
    Color {
        color: [f32; 4],
    },
    ColorTexture(Texture),
    GlTFPBR {
        base_color_factor: [f32; 4],
        metallic_factor: f32,
        roughness_factor: f32,
        normal_map: Option<NormalMap>,
        base_color_texture: Option<Texture>,
        metallic_roughness_texture: Option<Texture>,
    },
    None,
}

#[derive(Clone, Debug, Component)]
#[storage(DenseVecStorage)]
pub struct Material {
    pub data: MaterialData,
    pub compilation_mode: CompilationMode,
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

pub fn runtime_shaders_for_material(
    world: &World,
    root: Entity,
    vs_path: impl Into<PathBuf>,
    fs_path: impl Into<PathBuf>,
    match_material: impl Fn(&Material) -> bool,
) {
    let mut materials = world.write_storage::<Material>();
    let vs_path = vs_path.into();
    let fs_path = fs_path.into();
    let change_to_runtime = |ent| {
        if let Some(mat) = materials.get_mut(ent) {
            if match_material(&mat) {
                let vs_path = vs_path.clone();
                let fs_path = fs_path.clone();
                (*mat).compilation_mode = CompilationMode::RunTime { vs_path, fs_path };
            }
        }
    };

    render_graph::map(world, root, change_to_runtime);
}
