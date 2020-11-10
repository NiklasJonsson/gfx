use trekanten as trek;
use trekanten::uniform::UniformBuffer;
use trekanten::BufferHandle;
use trekanten::Handle;

use specs::Component;
use specs::DenseVecStorage;

#[derive(Debug, Clone)]
pub struct TextureUse {
    pub handle: Handle<trek::texture::Texture>,
    pub coord_set: u32,
}

#[derive(Debug, Clone, Component)]
pub enum Material {
    Unlit {
        color_uniform: BufferHandle<UniformBuffer>,
    },
    PBR {
        material_uniforms: BufferHandle<UniformBuffer>,
        normal_map: Option<TextureUse>,
        base_color_texture: Option<TextureUse>,
        metallic_roughness_texture: Option<TextureUse>,
        has_vertex_colors: bool,
    },
}
