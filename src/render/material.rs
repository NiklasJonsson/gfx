use trekanten::texture::Texture;
use trekanten::uniform::UniformBuffer;
use trekanten::BufferHandle;
use trekanten::Handle;

#[derive(Debug, Clone)]
pub struct TextureUse {
    pub handle: Handle<Texture>,
    pub coord_set: u32,
}

// TODO: Do we need to pass around the scale here?
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
        material_uniforms: BufferHandle<UniformBuffer>,
        normal_map: Option<NormalMap>,
        base_color_texture: Option<TextureUse>,
        metallic_roughness_texture: Option<TextureUse>,
        has_vertex_colors: bool,
    },
}
