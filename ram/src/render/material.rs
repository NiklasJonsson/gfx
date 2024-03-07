use trekant::{pipeline::PolygonMode, Texture, TextureDescriptor};
use trekant::{AsyncBufferHandle, BufferHandle, Handle};

use crate::math::Rgba;
use crate::render::Pending;

use crate::ecs::prelude::*;

use ram_derive::Visitable;
use trekant::resource::Async;

use super::GpuBuffer;

#[derive(Debug, Clone, Component, Visitable)]
pub struct Unlit {
    pub color: Rgba,
    pub polygon_mode: PolygonMode,
}

#[derive(Debug, Visitable)]
pub struct TextureUse2 {
    pub desc: TextureDescriptor<'static>,
    pub coord_set: u32,
}

#[derive(Debug, Component, Default, Visitable)]
pub struct PhysicallyBased {
    pub base_color_factor: Rgba,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_scale: f32,
    pub normal_map: Option<TextureUse2>,
    pub base_color_texture: Option<TextureUse2>,
    pub metallic_roughness_texture: Option<TextureUse2>,
    pub has_vertex_colors: bool,
}

#[derive(Debug, Clone, Visitable, PartialEq, Eq)]
pub struct TextureUse<T> {
    pub handle: Handle<T>,
    pub coord_set: u32,
}

#[derive(Debug, Component, Visitable)]
pub enum GpuMaterial {
    Unlit {
        color_uniform: BufferHandle,
        polygon_mode: PolygonMode,
    },
    PBR {
        material_uniforms: BufferHandle,
        normal_map: Option<TextureUse<Texture>>,
        base_color_texture: Option<TextureUse<Texture>>,
        metallic_roughness_texture: Option<TextureUse<Texture>>,
        has_vertex_colors: bool,
    },
}

#[derive(Debug, Component, Visitable)]
pub enum PendingMaterial {
    Unlit {
        color_uniform: GpuBuffer,
        polygon_mode: PolygonMode,
    },
    PBR {
        material_uniforms: GpuBuffer,
        normal_map: Option<Pending<TextureUse<Async<Texture>>, TextureUse<Texture>>>,
        base_color_texture: Option<Pending<TextureUse<Async<Texture>>, TextureUse<Texture>>>,
        metallic_roughness_texture:
            Option<Pending<TextureUse<Async<Texture>>, TextureUse<Texture>>>,
        has_vertex_colors: bool,
    },
}

impl PendingMaterial {
    pub fn is_done(&self) -> bool {
        match self {
            PendingMaterial::Unlit {
                color_uniform: GpuBuffer::Available(_),
                ..
            } => true,
            PendingMaterial::PBR {
                material_uniforms: GpuBuffer::Available(_),
                normal_map,
                base_color_texture,
                metallic_roughness_texture,
                ..
            } => {
                let is_done = |t: &Option<
                    Pending<TextureUse<Async<Texture>>, TextureUse<Texture>>,
                >|
                 -> bool {
                    match t {
                        Some(Pending::Available(_)) | None => true,
                        Some(Pending::Pending(_)) => false,
                    }
                };

                is_done(normal_map)
                    && is_done(base_color_texture)
                    && is_done(metallic_roughness_texture)
            }
            _ => false,
        }
    }

    pub fn finish(self) -> GpuMaterial {
        match self {
            PendingMaterial::Unlit {
                color_uniform: GpuBuffer::Available(color_uniform),
                polygon_mode,
            } => GpuMaterial::Unlit {
                color_uniform,
                polygon_mode,
            },
            PendingMaterial::PBR {
                material_uniforms: GpuBuffer::Available(material_uniforms),
                normal_map,
                base_color_texture,
                metallic_roughness_texture,
                has_vertex_colors,
            } => {
                let map_tex = |pend_tex: Pending<
                    TextureUse<Async<Texture>>,
                    TextureUse<Texture>,
                >|
                 -> Option<TextureUse<Texture>> {
                    if let Pending::Available(tex_use) = pend_tex {
                        Some(tex_use)
                    } else {
                        None
                    }
                };

                let normal_map = normal_map.and_then(map_tex);
                let base_color_texture = base_color_texture.and_then(map_tex);
                let metallic_roughness_texture = metallic_roughness_texture.and_then(map_tex);
                GpuMaterial::PBR {
                    material_uniforms,
                    normal_map,
                    base_color_texture,
                    metallic_roughness_texture,
                    has_vertex_colors,
                }
            }
            _ => unreachable!("Should be done by now"),
        }
    }
}
