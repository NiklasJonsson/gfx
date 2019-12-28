use specs::prelude::*;

pub mod math;
pub mod render_graph;

pub use math::*;

use std::fmt;
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

#[derive(Clone)]
pub struct Texture {
    pub image: image::RgbaImage,
    pub coord_set: u32,
    pub format: Format,
}

impl fmt::Debug for Texture {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Texture")
            .field(
                "image",
                &format_args!(
                    "RgbaImage{{ w: {}, h: {}}}",
                    self.image.width(),
                    self.image.height()
                ),
            )
            .field("coord_set", &self.coord_set)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct NormalMap {
    pub tex: Texture,
    pub scale: f32,
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
        normal_map: Option<NormalMap>,
        base_color_texture: Option<Texture>,
        metallic_roughness_texture: Option<Texture>,
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

// One ore more vertices with associated data on how to render them
#[derive(Debug, Component)]
#[storage(DenseVecStorage)]
pub struct PolygonMesh {
    pub ty: MeshType,
    pub vertex_data: VertexBuf,
    pub material: Material,
    pub compilation_mode: CompilationMode,
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
    let mut meshes = world.write_storage::<PolygonMesh>();
    let vs_path = vs_path.into();
    let fs_path = fs_path.into();
    let change_to_runtime = |ent| {
        if let Some(mesh) = meshes.get_mut(ent) {
            if match_material(&mesh.material) {
                let vs_path = vs_path.clone();
                let fs_path = fs_path.clone();
                (*mesh).compilation_mode = CompilationMode::RunTime { vs_path, fs_path };
            }
        }
    };

    render_graph::map(world, root, change_to_runtime);
}
