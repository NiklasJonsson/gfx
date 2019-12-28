#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct PBRMaterialData {
    pub base_color_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_sacle: f32
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Transforms {
    pub view: [[f32; 4]; 4];
    pub proj: [[f32; 4]; 4];
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Model {
    pub model: [[f32; 4]; 4];
    pub model_it: [[f32; 4]; 4];
}

