use crate::texture::Texture;
use crate::uniform::UniformBuffer;
use crate::resource::Handle;

#[derive(Debug, Clone)]
pub struct TextureUse {
    pub handle: Handle<Texture>,
    pub coord_set: u32,
}

#[derive(Debug, Clone)]
pub struct NormalMap {
    pub tex: TextureUse,
    pub scale: f32,
}

#[derive(Debug, Clone)]
pub enum MaterialData {
    UniformColor {
        color: [f32; 4],
    },
    PBR {
        material_uniforms: Handle<UniformBuffer>,
        normal_map: Option<NormalMap>,
        base_color_texture: Option<TextureUse>,
        metallic_roughness_texture: Option<TextureUse>,
    },
}
