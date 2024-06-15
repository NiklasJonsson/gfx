use trekant::{pipeline::PolygonMode, TextureDescriptor};
use trekant::{BufferHandle, Handle};

use crate::math::Rgba;
use crate::render::Pending;

use crate::ecs::prelude::*;

use ram_derive::Visitable;
use trekant::resource::Async;

use super::GpuBuffer;

pub struct HostTexture {
    data: image::RgbaImage,
    debug_name: String,
}

#[derive(Clone, PartialOrd, PartialEq, Eq, Ord, Hash, Debug, Visitable)]
pub struct HostTextureHandle {
    handle: resurs::Handle<HostTexture>,
    coord_set: u32,
}

#[derive(Clone, PartialOrd, PartialEq, Eq, Ord, Hash, Debug, Visitable)]
pub struct DeviceTextureHandle {
    handle: resurs::Handle<trekant::Texture>,
    coord_set: u32,
}

#[derive(Clone, PartialOrd, PartialEq, Eq, Ord, Hash, Debug, Visitable)]

pub struct AsyncDeviceTextureHandle {
    handle: resurs::Handle<Async<trekant::Texture>>,
    coord_set: u32,
}

pub struct TextureAssetLoadError;

#[derive(Clone, PartialOrd, PartialEq, Eq, Ord, Hash, Debug)]
pub struct TextureAsset(std::path::PathBuf);

#[derive(Default)]
pub struct TextureAssetLoader {
    storage: resurs::CachedStorage<TextureAsset, HostTexture>,
}

impl TextureAssetLoader {
    pub fn new() -> Self {
        Self::default()
    }
}

fn load_image(path: &std::path::Path) -> Result<image::RgbaImage, image::ImageError> {
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
    ) -> Result<Handle<HostTexture>, TextureAssetLoadError> {
        log::trace!("load_blocking texture for {}", asset.0.display());
        let mut cache_hit = true;
        let result: Result<Handle<HostTexture>, image::ImageError> =
            self.storage.get_or_add(asset, |asset| {
                cache_hit = false;
                let image = load_image(&asset.0)?;
                Ok(HostTexture {
                    data: image,
                    debug_name: debug_name.to_owned(),
                })
            });
        let handle = match result {
            Ok(h) => h,
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

    fn get(&self, handle: Handle<HostTexture>) -> Option<&HostTexture> {
        self.storage.get(&handle)
    }
}

#[derive(Debug, Clone, Component, Visitable)]
pub struct Unlit {
    pub color: Rgba,
    pub polygon_mode: PolygonMode,
}

#[derive(Debug, Component, Visitable)]
pub enum GpuMaterial {
    Unlit {
        color_uniform: BufferHandle,
        polygon_mode: PolygonMode,
    },
    PBR {
        material_uniforms: BufferHandle,
        normal_map: Option<DeviceTextureHandle>,
        base_color_texture: Option<DeviceTextureHandle>,
        metallic_roughness_texture: Option<DeviceTextureHandle>,
        has_vertex_colors: bool,
    },
}

#[derive(Debug, Visitable)]
enum PendingTextureUse {
    None,
    Pending(AsyncDeviceTextureHandle),
    Available(DeviceTextureHandle),
}

impl PendingTextureUse {
    fn is_done(&self) -> bool {
        match self {
            Self::Pending(_) => false,
            _ => true,
        }
    }
}

#[derive(Debug, Component, Visitable)]
pub enum PendingMaterial {
    Unlit {
        color_uniform: GpuBuffer,
        polygon_mode: PolygonMode,
    },
    PBR {
        material_uniforms: GpuBuffer,
        normal_map: PendingTextureUse,
        base_color_texture: PendingTextureUse,
        metallic_roughness_texture: PendingTextureUse,
        has_vertex_colors: bool,
    },
}

// TODO: Try finish instead of is_done/finish
impl PendingMaterial {
    pub fn is_done(&self) -> bool {
        match self {
            PendingMaterial::Unlit {
                color_uniform: GpuBuffer::Available(_),
                ..
            } => true,
            PendingMaterial::PBR {
                material_uniforms: GpuBuffer::Available(_),
                normal_map: PendingTextureUse::Available(_) | PendingTextureUse::None,
                base_color_texture: PendingTextureUse::Available(_) | PendingTextureUse::None,
                metallic_roughness_texture:
                    PendingTextureUse::Available(_) | PendingTextureUse::None,
                ..
            } => true,
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
                let map_tex = |pend_tex: PendingTextureUse| -> Option<DeviceTextureHandle> {
                    match pend_tex {
                        PendingTextureUse::None => None,
                        PendingTextureUse::Available(handle) => Some(handle),
                        PendingTextureUse::Pending(_) => {
                            panic!("Don't call this function if `is_done` returns false");
                        }
                    }
                };

                GpuMaterial::PBR {
                    material_uniforms,
                    normal_map: map_tex(normal_map),
                    base_color_texture: map_tex(base_color_texture),
                    metallic_roughness_texture: map_tex(metallic_roughness_texture),
                    has_vertex_colors,
                }
            }
            _ => unreachable!("Should be done by now"),
        }
    }
}

pub use pbr::PhysicallyBased;

mod pbr {
    use std::future::pending;

    use super::*;

    /// Attach this to an entity for it to have a lit material that is rendered with a
    /// "physically-based" model.
    #[derive(Debug, Component, Default, Visitable)]
    pub struct PhysicallyBased {
        pub base_color_factor: Rgba,
        pub metallic_factor: f32,
        pub roughness_factor: f32,
        pub normal_scale: f32,
        pub normal_map: Option<HostTextureHandle>,
        pub base_color_texture: Option<HostTextureHandle>,
        pub metallic_roughness_texture: Option<HostTextureHandle>,
        pub has_vertex_colors: bool,
    }

    #[derive(Debug, Component, Visitable)]
    pub struct Pending {
        material_uniforms: GpuBuffer,
        normal_map: PendingTextureUse,
        base_color_texture: PendingTextureUse,
        metallic_roughness_texture: PendingTextureUse,
        has_vertex_colors: bool,
    }

    #[derive(Debug, Component, Visitable)]
    pub struct Done {
        material_uniforms: BufferHandle,
        normal_map: Option<DeviceTextureHandle>,
        base_color_texture: Option<DeviceTextureHandle>,
        metallic_roughness_texture: Option<DeviceTextureHandle>,
        has_vertex_colors: bool,
    }

    pub struct Upload;

    impl<'a> System<'a> for Upload {
        type SystemData = (
            WriteExpect<'a, trekant::Loader>,
            WriteStorage<'a, PhysicallyBased>,
            WriteStorage<'a, PendingMaterial>,
            WriteStorage<'a, GpuMaterial>,
            Entities<'a>,
        );

        fn run(&mut self, data: Self::SystemData) {
            let (loader, materials, mut pending_materials, done_materials, entities) = data;

            // Physically based
            let mut ubuf_pbr = Vec::new();
            let mut textures
            for (pb_mat, _, _) in (&materials, !&pending_materials, !&done_materials).join() {
                ubuf_pbr.push(crate::render::uniform::PBRMaterialData {
                    base_color_factor: pb_mat.base_color_factor.into_array(),
                    metallic_factor: pb_mat.metallic_factor,
                    roughness_factor: pb_mat.roughness_factor,
                    normal_scale: pb_mat.normal_scale,
                    _padding: 0.0,
                });
            }

            let map_tex = |inp: &Option<material::TextureUse2>| -> Option<
                Pending<
                    material::TextureUse<resurs::Async<trekant::Texture>>,
                    material::TextureUse<trekant::Texture>,
                >,
            > {
                inp.as_ref().map(|tex| {
                    let handle = loader
                        .load_texture(tex.desc.clone())
                        .expect("Failed to load texture");
                    Pending::Pending(material::TextureUse {
                        coord_set: tex.coord_set,
                        handle,
                    })
                })
            };

            if !ubuf_pbr.is_empty() {
                let async_handle = loader
                    .load_buffer(trekant::BufferDescriptor::uniform_buffer(
                        ubuf_pbr,
                        trekant::BufferMutability::Immutable,
                        trekant::BufferLayout::MinBufferOffset,
                    ))
                    .expect("Failed to load uniform buffer");
                for (i, (ent, pb_mat, _)) in
                    (&entities, &physically_based_materials, !&gpu_materials)
                        .join()
                        .enumerate()
                {
                    if let StorageEntry::Vacant(entry) = pending_mats.entry(ent).unwrap() {
                        entry.insert(PendingMaterial::PBR {
                            material_uniforms: GpuBuffer::InFlight(AsyncBufferHandle::sub_buffer(
                                async_handle,
                                i as u32,
                                1,
                            )),
                            normal_map: map_tex(&pb_mat.normal_map),
                            base_color_texture: map_tex(&pb_mat.base_color_texture),
                            metallic_roughness_texture: map_tex(&pb_mat.metallic_roughness_texture),
                            has_vertex_colors: pb_mat.has_vertex_colors,
                        });
                    }
                }
            }
        }
    }
}

pub fn register_systems(builder: ExecutorBuilder) -> ExecutorBuilder {
    builder.with(pbr::Upload, "PBMaterialPipelineUpload", &[])
}
