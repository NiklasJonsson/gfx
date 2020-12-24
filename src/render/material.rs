use trekanten::texture;
use trekanten::uniform::UniformBuffer;
use trekanten::BufferHandle;
use trekanten::Handle;

use crate::ecs::prelude::*;
use ramneryd_derive::Inspect;

#[derive(Debug, Clone, Inspect)]
pub struct TextureUse {
    pub handle: Handle<texture::Texture>,
    pub coord_set: u32,
}

#[derive(Debug, Clone, Component)]
#[component(inspect)]
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
