use specs::prelude::*;
use std::ops::AddAssign;
use specs_hierarchy::{Parent as HParent};
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use nalgebra_glm::{Mat4, U4};

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
    Base(Vec<VertexBase>),
    UV(Vec<VertexUV>),
    UVCol(Vec<VertexUVCol>),
}

// TODO: Auto derive inner type traits
#[derive(Debug, Component)]
#[storage(VecStorage)]
pub struct Position(glm::Vec3);

impl Position {
    pub fn to_vec3(&self) -> glm::Vec3 {
        self.0
    }
    pub fn x(&self) -> f32 {
        self.0.x
    }

    pub fn y(&self) -> f32 {
        self.0.y
    }

    pub fn z(&self) -> f32 {
        self.0.z
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

#[derive(Debug, Component, Copy, Clone)]
#[storage(DenseVecStorage)]
pub struct Transform(Mat4);

impl From<[[f32; 4]; 4]> for Transform {
    fn from(x: [[f32; 4]; 4]) -> Self {
        Transform(x.into())
    }
}

impl From<Mat4> for Transform {
    fn from(x: Mat4) -> Self {
        Transform(x)
    }
}

impl Into<Mat4> for Transform {
    fn into(self) -> Mat4 {
        self.0
    }
}

#[derive(Debug, Component, Clone, Copy)]
#[storage(DenseVecStorage)]
pub struct ModelMatrix(Mat4);

impl From<[[f32; 4]; 4]> for ModelMatrix {
    fn from(x: [[f32; 4]; 4]) -> Self {
        Self(x.into())
    }
}

impl Into<[[f32; 4]; 4]> for ModelMatrix {
    fn into(self) -> [[f32; 4]; 4] {
        self.0.into()
    }
}

impl From<Mat4> for ModelMatrix {
    fn from(x: Mat4) -> Self {
        Self(x)
    }
}

impl Into<Mat4> for ModelMatrix {
    fn into(self) -> Mat4 {
        self.0
    }
}

impl ModelMatrix {
    pub fn identity() -> ModelMatrix {
        Self(glm::identity::<f32, glm::U4>())
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
    GlTFPBRMaterial {
        base_color_factor: [f32; 4],
        metallic_factor: f32,
        roughness_factor: f32,
        base_color_texture: Option<Texture>,
    },
    None,
}

#[derive(Debug)]
pub struct IndexData {
    pub rendering_mode: PrimitiveTopology,
    pub data: Vec<u32>,
}

impl IndexData {
    pub fn line_list(data: Vec<u32>) -> Self {
        IndexData {
            rendering_mode: PrimitiveTopology::LineList,
            data,
        }
    }

    pub fn triangle_list(data: Vec<u32>) -> Self {
        IndexData {
            rendering_mode: PrimitiveTopology::TriangleList,
            data,
        }
    }
}

// One ore more vertices with associated data on how to render them
#[derive(Debug, Component)]
#[storage(DenseVecStorage)]
pub struct GraphicsPrimitive {
    pub triangle_indices: IndexData,
    pub line_indices: IndexData,
    pub vertex_data: VertexBuf,
    pub material: Material,
}

/// Component for defining a node in a render graph
/// Useful when representing 3d models comprised of
/// several GraphicsPrimitive where we might have
/// transforms on several levels that should be concatenated
#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct RenderGraphNode {
    pub parent: Entity,
    pub children: Vec<Entity>,
}

impl RenderGraphNode {
    pub fn leaf(parent: Entity) -> Self {
        RenderGraphNode {parent, children: Vec::new()}
    }
}

impl Component for RenderGraphNode {
    type Storage = FlaggedStorage<Self, DenseVecStorage<Self>>;
}

impl HParent for RenderGraphNode {
    fn parent_entity(&self) -> Entity {
        self.parent
    }
}

/// Component for defining TODO
#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd, Component)]
#[storage(DenseVecStorage)]
pub struct RenderGraphRoot{
    pub children: Vec<Entity>,
}

pub struct TransformPropagation;

impl TransformPropagation {
    fn propagate_transforms_rec<'a>(stack: &mut Vec<Mat4>, ent: Entity,
                                    rgnodes: &ReadStorage<'a, RenderGraphNode>,
                                    transforms: &ReadStorage<'a, Transform>,
                                    model_matrices: &mut WriteStorage<'a, ModelMatrix>) {

        let transform: Mat4 = transforms
            .get(ent)
            .map(|x| *x)
            .unwrap_or(glm::identity::<f32, U4>().into())
            .into();
        let transform = stack.last().unwrap() * transform;

        stack.push(transform);
        model_matrices.insert(ent, ModelMatrix(transform));

        // We got here because this entity is a child of another node
        // This means this have to a RenderGraphComponent
        let children = rgnodes
            .get(ent)
            .expect("Broken graph, child is not node!")
            .children
            .iter();
        for child in children {
            TransformPropagation::propagate_transforms_rec(stack, *child, rgnodes,
                                                           transforms, model_matrices);
        }
        stack.pop();
    }
}

// TODO: Can we use a flagged storage for this?
// E.g. only run propagation for any of the roots
// for which it or children has been changed?
impl<'a> System<'a> for TransformPropagation {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, RenderGraphRoot>,
        ReadStorage<'a, RenderGraphNode>,
        ReadStorage<'a, Transform>,
        WriteStorage<'a, ModelMatrix>);

    fn run(&mut self, (entities, roots, rgnodes, transforms, mut model_matrices): Self::SystemData) {
        for (ent, root) in (&entities, &roots).join() {
            let mut stack: Vec<Mat4> = Vec::new();


            let transform: Mat4 = transforms
                .get(ent)
                .map(|x| *x)
                .unwrap_or(glm::identity::<f32, U4>().into())
                .into();
            stack.push(transform);

            if let Ok(entry) = model_matrices.entry(ent) {
                // Root node, no need to multiply
                entry.or_insert(ModelMatrix(stack[0]));
            }

            for child in root.children.iter() {
                TransformPropagation::propagate_transforms_rec(&mut stack, *child, &rgnodes, &transforms, &mut model_matrices);
            }
        }
    }
}

