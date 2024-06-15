use trekant::{pipeline::PolygonMode, Texture, TextureDescriptor};
use trekant::{BufferHandle, Handle};

use crate::math::Rgba;
use crate::render::Pending;

use crate::ecs::prelude::*;

use ram_derive::Visitable;
use trekant::resource::Async;

use super::GpuBuffer;

pub struct CpuTexture {
    data: image::RgbaImage,
    debug_name: String,
}

#[derive(Clone, PartialOrd, PartialEq, Eq, Ord, Hash, Debug)]
pub struct TextureHandle(resurs::Handle<CpuTexture>);

pub struct TextureAssetLoadError;

#[derive(Clone, PartialOrd, PartialEq, Eq, Ord, Hash, Debug)]
pub struct TextureAsset(std::path::PathBuf);

pub struct TextureAssetLoader {
    storage: resurs::CachedStorage<TextureAsset, CpuTexture>,
}

pub fn load_image(path: &std::path::Path) -> Result<image::RgbaImage, image::ImageError> {
    log::trace!("Trying to load image from {}", path.display());
    let image = image::open(path)?.to_rgba8();

    log::trace!(
        "Loaded RGBA image with dimensions: {:?}",
        image.dimensions()
    );

    Ok(image)
}

impl TextureAssetLoader {
    /// Load a texture from an asset.
    /// NOTE: This caches based on the filename so reloads of the same file will not get new content if it was changed in-between.
    pub fn load(
        &mut self,
        asset: TextureAsset,
        debug_name: &str,
    ) -> Result<TextureHandle, TextureAssetLoadError> {
        log::trace!("load_blocking texture for {}", asset.0.display());
        let mut cache_hit = true;
        let result: Result<Handle<CpuTexture>, image::ImageError> =
            self.storage.get_or_add(asset, |asset| {
                cache_hit = false;
                let image = load_image(&asset.0)?;
                Ok(CpuTexture {
                    data: image,
                    debug_name: debug_name.to_owned(),
                })
            });
        let handle = match result {
            Ok(h) => TextureHandle(h),
            // TODO: Map it
            Err(_) => return Err(TextureAssetLoadError),
        };
        if cache_hit {
            log::trace!("Hit cache");
        } else {
            log::trace!("Did not hit cache");
        }
        Ok(handle)
    }

    fn get(&self, handle: TextureHandle) -> Option<&CpuTexture> {
        self.storage.get(&handle.0)
    }
}

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
