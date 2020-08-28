#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct PBRMaterialData {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub _padding: f32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct Transforms {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct LightingData {
    pub view_pos: [f32; 4],
    pub light_pos: [f32; 4],
}

#[derive(Copy, Clone, Debug)]
#[repr(C, packed)]
pub struct Model {
    pub model: [[f32; 4]; 4],
    pub model_it: [[f32; 4]; 4],
}
