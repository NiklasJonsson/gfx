use trekant::Std140;

#[derive(Copy, Clone, Debug, Std140, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, packed)]
pub struct PBRMaterialData {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub _padding: f32,
}

#[derive(Copy, Clone, Debug, Std140, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, packed)]
pub struct UnlitUniformData {
    pub color: [f32; 4],
}

pub const SHADOW_TYPE_DIRECTIONAL: u32 = 1;
pub const SHADOW_TYPE_SPOT: u32 = 2;
pub const SHADOW_TYPE_POINT: u32 = 3;

pub const SPOTLIGHT_SHADOW_MAP_COUNT: u32 = 16;
pub const DIRECTIONAL_SHADOW_MAP_COUNT: u32 = 1;
pub const POINTLIGHT_SHADOW_MAP_COUNT: u32 = 16;

#[derive(Copy, Clone, Debug, Std140, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, packed)]
pub struct PackedLight {
    pub pos: [f32; 4],         // position for point/spot light
    pub dir_cutoff: [f32; 4], // direction for spot/directional light. .w is the cos(cutoff_angle) of the spotlight
    pub color_range: [f32; 4], // color for all light types. .w is the range of point/spot lights
    pub shadow_info: [u32; 4], // x is shadow type, y is matrix index, z is texture index
}

impl Default for PackedLight {
    fn default() -> Self {
        Self {
            pos: [0.0; 4],
            dir_cutoff: [0.0; 4],
            color_range: [0.0; 4],
            shadow_info: [0; 4],
        }
    }
}

pub type Mat4 = [f32; 16];

pub const fn mat4_nan() -> Mat4 {
    [f32::NAN; 16]
}

pub const MAX_NUM_LIGHTS: usize = 16;

#[derive(Copy, Clone, Debug, Default, Std140, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, packed)]
pub struct LightingData {
    pub lights: [PackedLight; MAX_NUM_LIGHTS],
    pub ambient: [f32; 4],
    // v4 is needed for padding at the end. Use only the first value.
    pub num_lights: [u32; 4],
}

#[derive(Copy, Clone, Debug, Std140, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, packed)]
pub struct Model {
    pub model: Mat4,
    pub model_it: Mat4,
}

#[derive(Copy, Clone, Debug, Default, Std140, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, packed)]
pub struct ViewData {
    pub view_proj: Mat4,
    pub view_pos: [f32; 4],
}

#[derive(Copy, Clone, Debug, Default, Std140, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C, packed)]
pub struct PosOnlyViewData {
    pub view_proj: Mat4,
}

#[derive(Copy, Clone, Debug, Std140, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct ShadowLightInfo {
    pub view_proj: Mat4,
    pub pos: [f32; 4],
}

impl Default for ShadowLightInfo {
    fn default() -> Self {
        Self {
            view_proj: mat4_nan(),
            pos: [f32::NAN; 4],
        }
    }
}
