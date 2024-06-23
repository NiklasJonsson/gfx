use trekant::{pipeline::PolygonMode, TextureDescriptor};
use trekant::{BufferHandle, Handle, Renderer};

use crate::math::Rgba;

use crate::ecs::prelude::*;

use ram_derive::Visitable;

use super::GpuBuffer;
use super::{PendingEntityBuffers, PendingEntityTextures};

pub struct HostTexture {
    data: Vec<u8>,
    format: trekant::Format,
    extent: trekant::Extent2D,
    debug_name: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Visitable)]
pub struct HostTextureHandle {
    pub handle: resurs::Handle<HostTexture>,
    pub coord_set: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Visitable)]
pub struct DeviceTextureHandle {
    pub handle: resurs::Handle<trekant::Texture>,
    pub coord_set: u32,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, Visitable)]

pub struct PendingDeviceTextureHandle {
    pub handle: trekant::PendingTextureHandle,
    pub coord_set: u32,
}

#[derive(Debug)]
pub struct TextureAssetLoadError;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TextureAsset {
    pub path: std::path::PathBuf,
    pub format: trekant::Format,
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
#[derive(Default, Debug)]
pub struct MipMapQueue(Vec<trekant::Handle<trekant::Texture>>);

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

#[derive(Debug, Visitable)]
enum PendingTextureUse {
    Pending(PendingDeviceTextureHandle),
    Available(DeviceTextureHandle),
}

impl PendingTextureUse {
    fn try_take(
        &mut self,
        old: trekant::PendingTextureHandle,
        new: trekant::Handle<trekant::Texture>,
    ) -> bool {
        match self {
            Self::Pending(pending) if pending.handle == old => {
                *self = Self::Available(DeviceTextureHandle {
                    handle: new,
                    coord_set: pending.coord_set,
                });
                true
            }
            _ => false,
        }
    }
}

pub use pbr::PhysicallyBased;

pub mod pbr {
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
        normal_map: Option<PendingTextureUse>,
        base_color_texture: Option<PendingTextureUse>,
        metallic_roughness_texture: Option<PendingTextureUse>,
        has_vertex_colors: bool,
    }

    impl Pending {
        fn try_finish(&self) -> Option<Done> {
            let material_uniforms = self.material_uniforms.get_available()?;
            // TODO: Clean this up
            let normal_map = match self.normal_map {
                None => None,
                Some(PendingTextureUse::Available(x)) => Some(x),
                Some(PendingTextureUse::Pending(_)) => return None,
            };
            let base_color_texture = match self.base_color_texture {
                None => None,
                Some(PendingTextureUse::Available(x)) => Some(x),
                Some(PendingTextureUse::Pending(_)) => return None,
            };
            let metallic_roughness_texture = match self.metallic_roughness_texture {
                None => None,
                Some(PendingTextureUse::Available(x)) => Some(x),
                Some(PendingTextureUse::Pending(_)) => return None,
            };

            Some(Done {
                material_uniforms,
                normal_map,
                base_color_texture,
                metallic_roughness_texture,
                has_vertex_colors: self.has_vertex_colors,
            })
        }
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

    pub struct PBRMaterialLoad;

    impl PBRMaterialLoad {
        pub const ID: &'static str = "PBRMaterialLoad";
    }

    impl<'a> System<'a> for PBRMaterialLoad {
        type SystemData = (
            WriteExpect<'a, trekant::Loader>,
            WriteExpect<'a, TextureAssetLoader>,
            WriteStorage<'a, PhysicallyBased>,
            WriteStorage<'a, Pending>,
            WriteStorage<'a, Done>,
            Write<'a, PBRPendingHandles>,
            Write<'a, MipMapQueue>,
            Entities<'a>,
        );

        fn run(&mut self, data: Self::SystemData) {
            let (
                loader,
                texture_loader,
                materials,
                mut pending_materials,
                mut done_materials,
                mut handles,
                mut mipmaps,
                entities,
            ) = data;

            // TODO: Consider rewriting this to collect:
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

            let map_tex = |tex: HostTextureHandle,
                           handles: &mut PBRPendingHandles,
                           entity: Entity|
             -> PendingTextureUse {
                let cpu_texture = texture_loader
                    .get(tex.handle)
                    .unwrap_or_else(|| panic!("Missing texture for handle {:?}", tex.handle));

                // TODO: No clone. Use Arc instead.
                let data = trekant::DescriptorData::from_vec(cpu_texture.data.clone());
                let handle = loader
                    .load_texture(
                        TextureDescriptor::Raw {
                            data,
                            extent: cpu_texture.extent,
                            format: cpu_texture.format,
                            mipmaps: trekant::MipMaps::None,
                            ty: trekant::TextureType::Tex2D,
                        },
                        LOAD_ID,
                    )
                    .expect("Failed to load texture");
                handles.textures.push(handle, entity);

                PendingTextureUse::Pending(PendingDeviceTextureHandle {
                    handle,
                    coord_set: tex.coord_set,
                })
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
                        handles.buffers.push(full_buffer, ent);
                        entry.insert(Pending {
                            material_uniforms: GpuBuffer::Pending(
                                trekant::PendingBufferHandle::sub_buffer(full_buffer, i as u32, 1),
                            ),
                            normal_map: pb_mat.normal_map.map(|t| map_tex(t, &mut *handles, ent)),
                            base_color_texture: pb_mat
                                .base_color_texture
                                .map(|t| map_tex(t, &mut *handles, ent)),
                            metallic_roughness_texture: pb_mat
                                .metallic_roughness_texture
                                .map(|t| map_tex(t, &mut *handles, ent)),
                            has_vertex_colors: pb_mat.has_vertex_colors,
                        });
                    }
                }
            }

            for handle_mapping in loader.flush(LOAD_ID) {
                match handle_mapping {
                    trekant::HandleMapping::Buffer { old, new } => {
                        for entity in handles.buffers.flush(old) {
                            let Some(pending) = pending_materials.get_mut(entity) else {
                                log::debug!("Entity {entity:?} had a pending buffer load for pbr uniform data but was destroyed while the buffer was loading");
                                continue;
                            };

                            let success = pending.material_uniforms.try_take(old, new);
                            assert!(success, "There is only one buffer that this could match");
                        }
                    }
                    trekant::HandleMapping::Texture { old, new } => {
                        for entity in handles.textures.flush(old) {
                            let Some(pending) = pending_materials.get_mut(entity) else {
                                log::debug!("Entity {entity:?} had a pending buffer load for pbr uniform data but was destroyed while the buffer was loading");
                                continue;
                            };

                            let mut any_success = false;
                            for tex in [
                                &mut pending.base_color_texture,
                                &mut pending.metallic_roughness_texture,
                                &mut pending.normal_map,
                            ]
                            .into_iter()
                            .flatten()
                            {
                                let success = tex.try_take(old, new);
                                any_success = any_success || success;
                                if success {
                                    mipmaps.0.push(new);
                                }
                            }
                            assert!(any_success, "Entity had a queued texture load but none of its PBR textures matched the handle");
                        }
                    }
                }
            }

            let mut done_entities = Vec::new();
            for (entity, pending) in (&entities, &pending_materials).join() {
                if let Some(done) = pending.try_finish() {
                    done_materials.insert(entity, done).unwrap();
                    done_entities.push(entity);
                    log::debug!("Entity {entity:?} is done");
                }
            }

            for entity in done_entities {
                pending_materials.remove(entity).unwrap();
            }
        }
    }

    #[derive(Default)]
    pub struct PBRPendingHandles {
        buffers: PendingEntityBuffers,
        textures: PendingEntityTextures,
    }

    pub fn create_pipeline_resource_set(
        renderer: &mut trekant::Renderer,
        mat: &Done,
    ) -> Handle<trekant::PipelineResourceSet> {
        let mut desc_set_builder = trekant::PipelineResourceSet::builder(renderer);

        desc_set_builder = desc_set_builder.add_buffer(
            mat.material_uniforms,
            0,
            trekant::pipeline::ShaderStage::FRAGMENT,
        );

        if let Some(bct) = &mat.base_color_texture {
            desc_set_builder = desc_set_builder.add_texture(
                bct.handle,
                1,
                trekant::pipeline::ShaderStage::FRAGMENT,
                false,
            );
        }

        if let Some(mrt) = &mat.metallic_roughness_texture {
            desc_set_builder = desc_set_builder.add_texture(
                mrt.handle,
                2,
                trekant::pipeline::ShaderStage::FRAGMENT,
                false,
            );
        }

        if let Some(nm) = &mat.normal_map {
            desc_set_builder = desc_set_builder.add_texture(
                nm.handle,
                3,
                trekant::pipeline::ShaderStage::FRAGMENT,
                false,
            );
        }

        desc_set_builder.build()
    }

    pub fn get_pipeline(
        renderer: &mut trekant::Renderer,
        vertex_format: &trekant::VertexFormat,
        shader_compiler: &crate::render::ShaderCompiler,
        shader_cache: &mut crate::render::ShaderCache,
        render_pass: Handle<trekant::RenderPass>,
        mat: &Done,
    ) -> Result<Handle<trekant::GraphicsPipeline>, crate::render::MaterialError> {
        // TODO: Normal map does not infer tangents at all times
        let has_nm = mat.normal_map.is_some();
        let has_bc = mat.base_color_texture.is_some();
        let has_mr = mat.metallic_roughness_texture.is_some();
        let def = crate::render::shader::pbr_gltf::ShaderDefinition {
            has_tex_coords: has_nm || has_bc || has_mr,
            has_vertex_colors: mat.has_vertex_colors,
            has_tangents: has_nm,
            has_base_color_texture: has_bc,
            has_metallic_roughness_texture: has_mr,
            has_normal_map: has_nm,
        };

        let (vert, frag) =
            crate::render::shader::pbr_gltf::compile(shader_compiler, shader_cache, &def)?;

        let vert = trekant::ShaderDescriptor {
            debug_name: Some("pbr-vert".to_owned()),
            spirv_code: vert.data(),
        };
        let frag = trekant::ShaderDescriptor {
            debug_name: Some("pbr-frag".to_owned()),
            spirv_code: frag.data(),
        };
        let desc = trekant::GraphicsPipelineDescriptor::builder()
            .vert(vert)
            .frag(frag)
            .vertex_format(vertex_format.clone())
            .polygon_mode(trekant::pipeline::PolygonMode::Fill)
            .build()?;

        Ok(renderer.create_gfx_pipeline(desc, &render_pass)?)
    }
}

pub use unlit::Unlit;

pub mod unlit {
    use super::*;
    use trekant::HandleMapping;

    const LOAD_ID: trekant::LoadId = trekant::LoadId("unlit-pipeline");

    #[derive(Debug, Clone, Component, Visitable)]
    pub struct Unlit {
        pub color: Rgba,
        pub polygon_mode: PolygonMode,
    }

    #[derive(Debug, Component, Visitable)]
    pub struct Done {
        pub color_uniform: BufferHandle,
        pub polygon_mode: PolygonMode,
    }

    #[derive(Debug, Component, Visitable)]
    pub struct Pending {
        color_uniform: GpuBuffer,
        polygon_mode: PolygonMode,
    }

    pub struct UnlitMaterialLoad;

    #[derive(Default)]
    pub struct UnlitPendingHandles(crate::render::PendingEntityBuffers);

    impl UnlitMaterialLoad {
        pub const ID: &'static str = "UnlitMaterialLoad";
    }

    impl<'a> System<'a> for UnlitMaterialLoad {
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
                mut done_materials,
                mut handles,
                entities,
            ) = data;

            let mut ubuf = Vec::new();
            for (unlit, _, _) in (&materials, !&done_materials, !&pending_materials).join() {
                ubuf.push(crate::render::uniform::UnlitUniformData {
                    color: unlit.color.into_array(),
                });
            }

            if !ubuf.is_empty() {
                let full_buffer = loader
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
                        handles.0.push(full_buffer, ent);
                        entry.insert(Pending {
                            color_uniform: GpuBuffer::Pending(
                                trekant::PendingBufferHandle::sub_buffer(full_buffer, i as u32, 1),
                            ),
                            polygon_mode: unlit.polygon_mode,
                        });
                    }
                }
            }

            for handle_mapping in loader.flush(LOAD_ID) {
                let HandleMapping::Buffer { old, new } = handle_mapping else {
                    panic!("Unlit pipeline has no textures");
                };
                for entity in handles.0.flush(old) {
                    let Some(pending) = pending_materials.remove(entity) else {
                        log::debug!("Entity {entity:?} had a pending buffer load for unlit uniform data but was destroyed while the buffer was loading");
                        continue;
                    };
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
                }
            }
        }
    }

    pub fn create_pipeline_resource_set(
        renderer: &mut Renderer,
        mat: &Done,
    ) -> Handle<trekant::PipelineResourceSet> {
        trekant::PipelineResourceSet::builder(renderer)
            .add_buffer(
                mat.color_uniform,
                0,
                trekant::pipeline::ShaderStage::FRAGMENT,
            )
            .build()
    }

    pub fn get_pipeline(
        renderer: &mut trekant::Renderer,
        vertex_format: &trekant::VertexFormat,
        shader_compiler: &crate::render::ShaderCompiler,
        shader_cache: &mut crate::render::ShaderCache,
        render_pass: Handle<trekant::RenderPass>,
        polygon_mode: PolygonMode,
    ) -> Result<Handle<trekant::GraphicsPipeline>, crate::render::MaterialError> {
        use crate::render::shader;
        use crate::render::shader_path;

        let vert = shader_compiler.compile(
            &shader_path(&["unlit", "vert.glsl"]),
            &shader::Defines::empty(),
            shader::ShaderType::Vertex,
            Some(shader_cache),
        )?;
        let frag = shader_compiler.compile(
            &shader_path(&["unlit", "frag.glsl"]),
            &shader::Defines::empty(),
            shader::ShaderType::Fragment,
            Some(shader_cache),
        )?;

        let vert = trekant::ShaderDescriptor {
            debug_name: Some("unlit-vert".to_owned()),
            spirv_code: vert.data(),
        };
        let frag = trekant::ShaderDescriptor {
            debug_name: Some("unlit-frag".to_owned()),
            spirv_code: frag.data(),
        };
        let desc = trekant::GraphicsPipelineDescriptor::builder()
            .vert(vert)
            .frag(frag)
            .vertex_format(vertex_format.clone())
            .culling(trekant::pipeline::TriangleCulling::None)
            .polygon_mode(polygon_mode)
            .build()?;
        Ok(renderer.create_gfx_pipeline(desc, &render_pass)?)
    }
}

fn create_renderables_pbr(renderer: &mut Renderer, world: &World) {
    let materials = world.read_storage::<pbr::Done>();
    let meshes = world.read_storage::<super::Mesh>();
    let frame_resources = world.read_resource::<super::FrameResources>();
    let shader_compiler = world.read_resource::<super::ShaderCompiler>();
    let shader_cache = world.read_resource::<super::GlobalShaderCache>();
    let mut renderables = world.write_storage::<super::RenderableMaterial>();
    let mut shader_cache = shader_cache.0.lock().unwrap();
    let entities = world.entities();

    for (ent, mesh, mat, _) in (&entities, &meshes, &materials, !&renderables.mask().clone()).join()
    {
        log::trace!("No Renderable found, creating new");
        log::trace!("Creating renderable: {:?}", mat);
        let material_descriptor_set = pbr::create_pipeline_resource_set(renderer, mat);
        let gfx_pipeline = pbr::get_pipeline(
            renderer,
            mesh.cpu_vertex_buffer.format(),
            &shader_compiler,
            &mut shader_cache,
            frame_resources.main_render_pass,
            mat,
        )
        .expect("Failed to create renderable material for PBR");

        renderables
            .insert(
                ent,
                super::RenderableMaterial {
                    gfx_pipeline,
                    material_descriptor_set,
                },
            )
            .unwrap();
    }
}

#[profiling::function]
fn create_renderables_unlit(renderer: &mut Renderer, world: &World) {
    let materials = world.read_storage::<unlit::Done>();
    let meshes = world.read_storage::<super::Mesh>();
    let frame_resources = world.read_resource::<super::FrameResources>();
    let shader_compiler = world.read_resource::<super::ShaderCompiler>();
    let shader_cache = world.read_resource::<super::GlobalShaderCache>();
    let mut renderables = world.write_storage::<super::RenderableMaterial>();
    let mut shader_cache = shader_cache.0.lock().unwrap();
    let entities = world.entities();

    for (ent, mesh, mat, _) in (&entities, &meshes, &materials, !&renderables.mask().clone()).join()
    {
        log::trace!("No Renderable found, creating new");
        log::trace!("Creating renderable: {:?}", mat);
        let material_descriptor_set = unlit::create_pipeline_resource_set(renderer, mat);
        let gfx_pipeline = unlit::get_pipeline(
            renderer,
            mesh.cpu_vertex_buffer.format(),
            &shader_compiler,
            &mut shader_cache,
            frame_resources.main_render_pass,
            mat.polygon_mode,
        )
        .expect("Failed to create pipeline for unlit material");

        let rend = super::RenderableMaterial {
            gfx_pipeline,
            material_descriptor_set,
        };
        renderables.insert(ent, rend).unwrap();
    }
}

pub fn pre_frame(world: &World, renderer: &mut Renderer) {
    // HACK: Mipmap generation needs to run before renderable creation,
    // because the vk image view that is used for the descriptor set
    // is replaced when the mipmap is loaded.
    // TODO: Handle this better.
    {
        let mut mipmaps = world.write_resource::<MipMapQueue>();
        renderer
            .generate_mipmaps(&mipmaps.0)
            .expect("Failed to generate mipmaps");
        mipmaps.0.clear();
    }
    create_renderables_pbr(renderer, world);
    create_renderables_unlit(renderer, world);
}

pub fn register_systems(builder: ExecutorBuilder) -> ExecutorBuilder {
    builder
        .with(unlit::UnlitMaterialLoad, unlit::UnlitMaterialLoad::ID, &[])
        .with(pbr::PBRMaterialLoad, pbr::PBRMaterialLoad::ID, &[])
}
