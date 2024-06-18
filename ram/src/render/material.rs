use trekant::{pipeline::PolygonMode, TextureDescriptor};
use trekant::{BufferHandle, Handle};

use crate::math::Rgba;

use crate::ecs::prelude::*;

use ram_derive::Visitable;

use super::GpuBuffer;

pub struct HostTexture {
    data: Vec<u8>,
    format: trekant::Format,
    extent: trekant::Extent2D,
    debug_name: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Visitable)]
pub struct HostTextureHandle {
    handle: resurs::Handle<HostTexture>,
    coord_set: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Visitable)]
pub struct DeviceTextureHandle {
    handle: resurs::Handle<trekant::Texture>,
    coord_set: u32,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Visitable)]

pub struct PendingDeviceTextureHandle {
    handle: trekant::PendingTextureHandle,
    coord_set: u32,
}

pub struct TextureAssetLoadError;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TextureAsset {
    path: std::path::PathBuf,
    format: trekant::Format,
}

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
        log::trace!("load_blocking texture for {}", asset.path.display());
        let mut cache_hit = true;
        let result: Result<Handle<HostTexture>, image::ImageError> =
            self.storage.get_or_add(asset, |asset| {
                cache_hit = false;

                let image = load_image(&asset.path)?;
                let extent = trekant::Extent2D {
                    width: image.width(),
                    height: image.height(),
                };
                let raw_image_data = image.into_raw();
                Ok(HostTexture {
                    data: raw_image_data,
                    extent,
                    format: asset.format,
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
    Pending(PendingDeviceTextureHandle),
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
    use crate::visit::Visitable;

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

    const LOAD_ID: trekant::LoadId = trekant::LoadId("pbr-pipeline");

    pub struct StartLoad;

    impl StartLoad {
        pub const ID: &'static str = "PhysicallyBasedMaterialPipelineStartLoad";
    }

    impl<'a> System<'a> for StartLoad {
        type SystemData = (
            WriteExpect<'a, trekant::Loader>,
            WriteExpect<'a, TextureAssetLoader>,
            WriteStorage<'a, PhysicallyBased>,
            WriteStorage<'a, Pending>,
            WriteStorage<'a, Done>,
            Entities<'a>,
        );

        fn run(&mut self, data: Self::SystemData) {
            let (
                loader,
                texture_loader,
                materials,
                mut pending_materials,
                done_materials,
                entities,
            ) = data;

            // TODO: Consider rewriting this to collected:
            // 1. Entity
            // 2. PBRMaterialData
            // 3. Textures to be loaded
            // Iterate over these in the second loop
            let mut ubuf_pbr = Vec::new();
            for (pb_mat, _, _) in (&materials, !&pending_materials, !&done_materials).join() {
                ubuf_pbr.push(crate::render::uniform::PBRMaterialData {
                    base_color_factor: pb_mat.base_color_factor.into_array(),
                    metallic_factor: pb_mat.metallic_factor,
                    roughness_factor: pb_mat.roughness_factor,
                    normal_scale: pb_mat.normal_scale,
                    _padding: 0.0,
                });
            }

            let map_tex = |inp: &Option<HostTextureHandle>| -> PendingTextureUse {
                if let Some(tex) = inp {
                    let cpu_texture = texture_loader
                        .get(tex.handle)
                        .unwrap_or_else(|| panic!("Missing texture for handle {:?}", tex.handle));

                    // TODO: No clone. Use Arc instead.
                    let data = trekant::DescriptorData::from_vec(cpu_texture.data.clone());
                    let mipmaps = todo!("Figure out mipmaps");

                    let handle = loader
                        .load_texture(
                            TextureDescriptor::Raw {
                                data,
                                extent: cpu_texture.extent,
                                format: cpu_texture.format,
                                mipmaps,
                                ty: trekant::TextureType::Tex2D,
                            },
                            LOAD_ID,
                        )
                        .expect("Failed to load texture");
                    PendingTextureUse::Pending(PendingDeviceTextureHandle {
                        handle,
                        coord_set: tex.coord_set,
                    })
                } else {
                    PendingTextureUse::None
                }
            };

            if !ubuf_pbr.is_empty() {
                let full_buffer = loader
                    .load_buffer(
                        trekant::BufferDescriptor::uniform_buffer(
                            ubuf_pbr,
                            trekant::BufferMutability::Immutable,
                            trekant::BufferLayout::MinBufferOffset,
                        ),
                        LOAD_ID,
                    )
                    .expect("Failed to load uniform buffer");
                for (i, (ent, pb_mat, _)) in
                    (&entities, &materials, !&done_materials).join().enumerate()
                {
                    if let StorageEntry::Vacant(entry) = pending_materials.entry(ent).unwrap() {
                        entry.insert(Pending {
                            material_uniforms: GpuBuffer::InFlight(
                                trekant::PendingBufferHandle::sub_buffer(full_buffer, i as u32, 1),
                            ),
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

    #[derive(Default)]
    pub struct PBRPendingHandles {
        buffers: std::collections::HashMap<trekant::PendingBufferHandle, Vec<Entity>>,
        textures: std::collections::HashMap<trekant::PendingTextureHandle, Vec<Entity>>,
    }

    pub struct FinishLoad;

    impl FinishLoad {
        pub const ID: &'static str = "PhysicallyBasedMaterialPipelineFinishLoad";
    }
    impl<'a> System<'a> for FinishLoad {
        type SystemData = (
            WriteExpect<'a, trekant::Loader>,
            WriteStorage<'a, Pending>,
            WriteStorage<'a, Done>,
            Write<'a, PBRPendingHandles>,
        );

        fn run(&mut self, data: Self::SystemData) {
            let (loader, mut pending_materials, mut done_materials, mut handles) = data;

            for handle_mapping in loader.flush(LOAD_ID) {
                match handle_mapping {
                    trekant::HandleMapping::Buffer { old, new } => {
                        let entities = handles
                            .buffers
                            .remove(&old)
                            .expect("Got a buffer handle that was not requested");
                        for entity in entities {
                            let Some(pending) = pending_materials.get_mut(entity) else {
                                log::debug!("Entity {entity:?} had a pending buffer load for pbr uniform data but was destroyed while the buffer was loading");
                                continue;
                            };
                            assert!(old.same_base_buffer(
                                pending.material_uniforms.get_available().unwrap()
                            ));

                            pending.material_uniforms = GpuBuffer::Available(new);
                        }
                    }
                    trekant::HandleMapping::Texture { old, new } => {
                        let entities = handles
                            .textures
                            .remove(&old)
                            .expect("Got a texture handle that was not requested");
                        for entity in entities {
                            let Some(pending) = pending_materials.get_mut(entity) else {
                                log::debug!("Entity {entity:?} had a pending buffer load for pbr uniform data but was destroyed while the buffer was loading");
                                continue;
                            };

                            for tex in [
                                &mut pending.base_color_texture,
                                &mut pending.metallic_roughness_texture,
                                &mut pending.normal_map,
                            ] {
                                // TODO: Cleanup here
                                let mut correct = false;
                                if let PendingTextureUse::Pending(pending_use) = tex {
                                    correct = pending_use.handle == old;
                                }
                                if correct {
                                    *tex = PendingTextureUse::Available(DeviceTextureHandle {
                                        handle: new,
                                        coord_set: todo!(),
                                    });
                                }
                            }
                        }
                    }
                }
            }

            todo!("mipmaps");
        }
    }
}

pub use unlit::Unlit;

mod unlit {
    use trekant::{HandleMapping, PendingBufferHandle};

    use super::*;

    const LOAD_ID: trekant::LoadId = trekant::LoadId("unlit-pipeline");

    #[derive(Debug, Clone, Component, Visitable)]
    pub struct Unlit {
        pub color: Rgba,
        pub polygon_mode: PolygonMode,
    }

    #[derive(Debug, Component, Visitable)]
    pub struct Done {
        color_uniform: BufferHandle,
        polygon_mode: PolygonMode,
    }

    #[derive(Debug, Component, Visitable)]
    struct Pending {
        color_uniform: GpuBuffer,
        polygon_mode: PolygonMode,
    }

    pub struct StartLoad;

    #[derive(Default)]
    pub struct UnlitPendingHandles(std::collections::HashMap<PendingBufferHandle, Vec<Entity>>);

    impl StartLoad {
        pub const ID: &'static str = "UnlitMaterialPipelineStartLoad";
    }

    impl<'a> System<'a> for StartLoad {
        type SystemData = (
            WriteExpect<'a, trekant::Loader>,
            WriteStorage<'a, Unlit>,
            WriteStorage<'a, Pending>,
            WriteStorage<'a, Done>,
            Write<'a, UnlitPendingHandles>,
            Entities<'a>,
        );

        fn run(&mut self, data: Self::SystemData) {
            let (
                loader,
                materials,
                mut pending_materials,
                done_materials,
                mut pending_handles,
                entities,
            ) = data;

            let mut ubuf = Vec::new();
            for (unlit, _, _) in (&materials, !&done_materials, !&pending_materials).join() {
                ubuf.push(crate::render::uniform::UnlitUniformData {
                    color: unlit.color.into_array(),
                });
            }

            if !ubuf.is_empty() {
                let async_handle = loader
                    .load_buffer(
                        trekant::BufferDescriptor::uniform_buffer(
                            ubuf,
                            trekant::BufferMutability::Immutable,
                            trekant::BufferLayout::MinBufferOffset,
                        ),
                        LOAD_ID,
                    )
                    .expect("Failed to load uniform buffer");
                for (i, (ent, unlit, _)) in
                    (&entities, &materials, !&done_materials).join().enumerate()
                {
                    if let StorageEntry::Vacant(entry) = pending_materials.entry(ent).unwrap() {
                        pending_handles.0.entry(async_handle).or_default().push(ent);
                        entry.insert(Pending {
                            color_uniform: GpuBuffer::InFlight(
                                trekant::PendingBufferHandle::sub_buffer(async_handle, i as u32, 1),
                            ),
                            polygon_mode: unlit.polygon_mode,
                        });
                    }
                }
            }
        }
    }

    pub struct FinishLoad;

    impl FinishLoad {
        pub const ID: &'static str = "UnlitMaterialPipelineFinishLoad";
    }
    impl<'a> System<'a> for FinishLoad {
        type SystemData = (
            WriteExpect<'a, trekant::Loader>,
            WriteStorage<'a, Pending>,
            WriteStorage<'a, Done>,
            Write<'a, UnlitPendingHandles>,
        );

        fn run(&mut self, data: Self::SystemData) {
            let (loader, mut pending_materials, mut done_materials, mut handles) = data;

            for handle_mapping in loader.flush(LOAD_ID) {
                let HandleMapping::Buffer { old, new } = handle_mapping else {
                    panic!("Unlit pipeline has no textures");
                };
                let entities = handles
                    .0
                    .remove(&old)
                    .expect("Got a buffer handle that was not requested");
                for entity in entities {
                    if let Some(pending) = pending_materials.remove(entity) {
                        let result = done_materials.insert(
                            entity,
                            Done {
                                color_uniform: new,
                                polygon_mode: pending.polygon_mode,
                            },
                        );
                        if let Err(e) = result {
                            log::error!(
                                "Failed to finalize unlit material load for entity {entity:?}: {e}"
                            );
                        }
                    } else {
                        log::debug!("Entity {entity:?} had a pending buffer load for unlit uniform data but was destroyed while the buffer was loading");
                    }
                }
            }
        }
    }
}

pub fn register_systems(builder: ExecutorBuilder) -> ExecutorBuilder {
    builder
        .with(pbr::StartLoad, pbr::StartLoad::ID, &[])
        .with(unlit::StartLoad, unlit::StartLoad::ID, &[])
        .with(
            unlit::FinishLoad,
            unlit::FinishLoad::ID,
            &[unlit::StartLoad::ID],
        )
}
