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

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct UnlitUniformData {
    pub color: [f32; 4],
}

impl UniformBlock for UnlitUniformData {
    const SET: u32 = 1;
    const BINDING: u32 = 0;
}

#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
pub struct PunctualLight {
    pub pos: [f32; 4],
    pub color: [f32; 4],
}

pub const MAX_NUM_PUNCTUAL_LIGHTS: usize = 16;
pub const PUNCTUAL_LIGHTS_BITS: u32 = 0xF;
#[derive(Copy, Clone, Debug, Default)]
#[repr(C, packed)]
pub struct LightingData {
    pub punctual_lights: [PunctualLight; MAX_NUM_PUNCTUAL_LIGHTS],
    pub num_lights: u32, // bitmask
}

impl LightingData {
    pub fn set_num_punctual_lights(&mut self, n: u8) {
        self.num_lights = (self.num_lights & !PUNCTUAL_LIGHTS_BITS) | n as u32;
    }
}

impl UniformBlock for LightingData {
    const SET: u32 = 0;
    const BINDING: u32 = 1;
}

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

pub trait UniformBlock {
    const SET: u32;
    const BINDING: u32;
}
