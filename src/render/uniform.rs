use trekanten::mem::Uniform;

pub trait UniformBlock {
    const SET: u32;
    const BINDING: u32;
}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct PBRMaterialData {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub _padding: f32,
}

impl UniformBlock for PBRMaterialData {
    const SET: u32 = 1;
    const BINDING: u32 = 0;
}

impl Uniform for PBRMaterialData {}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct UnlitUniformData {
    pub color: [f32; 4],
}

impl UniformBlock for UnlitUniformData {
    const SET: u32 = 1;
    const BINDING: u32 = 0;
}
impl Uniform for UnlitUniformData {}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
pub struct PackedLight {
    pub pos: [f32; 4],         // position for point/spot light
    pub dir_cutoff: [f32; 4], // direction for spot/directional light. .w is the cos(cutoff_angle) of the spotlight
    pub color_range: [f32; 4], // color for all light types. .w is the range of point/spot lights
}

pub const MAX_NUM_LIGHTS: usize = 16;
#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
pub struct LightingData {
    pub punctual_lights: [PackedLight; MAX_NUM_LIGHTS],
    pub num_lights: u32,
}

impl UniformBlock for LightingData {
    const SET: u32 = 0;
    const BINDING: u32 = 1;
}
impl Uniform for LightingData {}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct Model {
    pub model: [[f32; 4]; 4],
    pub model_it: [[f32; 4]; 4],
}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct ViewData {
    pub view_proj: [[f32; 4]; 4],
    pub view_pos: [f32; 4],
}

impl UniformBlock for ViewData {
    const SET: u32 = 0;
    const BINDING: u32 = 0;
}
impl Uniform for ViewData {}
