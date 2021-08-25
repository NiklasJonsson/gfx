use std::convert::TryInto;

use trekanten::{Std140Compat, Uniform};

pub trait UniformBlock {
    const SET: u32;
    const BINDING: u32;
}

#[derive(Copy, Clone, Debug, Std140Compat)]
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

#[derive(Copy, Clone, Debug, Std140Compat)]
#[repr(C, packed)]
pub struct UnlitUniformData {
    pub color: [f32; 4],
}

impl UniformBlock for UnlitUniformData {
    const SET: u32 = 1;
    const BINDING: u32 = 0;
}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct PackedLight {
    pub pos: [f32; 4],         // position for point/spot light
    pub dir_cutoff: [f32; 4], // direction for spot/directional light. .w is the cos(cutoff_angle) of the spotlight
    pub color_range: [f32; 4], // color for all light types. .w is the range of point/spot lights
    pub shadow_idx: [u32; 4],
}

impl Default for PackedLight {
    fn default() -> Self {
        Self {
            pos: [0.0; 4],
            dir_cutoff: [0.0; 4],
            color_range: [0.0; 4],
            shadow_idx: [u32::MAX; 4],
        }
    }
}

pub type Mat4 = [f32; 16];

pub const MAX_NUM_LIGHTS: usize = 16;

#[derive(Copy, Clone, Debug, Default, Std140Compat)]
#[repr(C, packed)]
pub struct ShadowMatrices {
    pub matrices: [Mat4; MAX_NUM_LIGHTS],
    pub num_matrices: u32,
}
impl UniformBlock for ShadowMatrices {
    const SET: u32 = 0;
    const BINDING: u32 = 3;
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
pub struct LightingData {
    pub punctual_lights: [PackedLight; MAX_NUM_LIGHTS],
    pub ambient: [f32; 4],
    pub num_lights: u32,
}

impl UniformBlock for LightingData {
    const SET: u32 = 0;
    const BINDING: u32 = 1;
}

unsafe impl Uniform for LightingData {
    fn size() -> u16 {
        std::mem::size_of::<Self>()
            .try_into()
            .expect("uniform is too big")
    }
}

#[derive(Copy, Clone, Debug, Std140Compat)]
#[repr(C, packed)]
pub struct Model {
    pub model: Mat4,
    pub model_it: Mat4,
}

#[derive(Copy, Clone, Debug, Default, Std140Compat)]
#[repr(C, packed)]
pub struct ViewData {
    pub view_proj: Mat4,
    pub view_pos: [f32; 4],
}

impl UniformBlock for ViewData {
    const SET: u32 = 0;
    const BINDING: u32 = 0;
}
