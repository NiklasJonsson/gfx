use ash::vk;

use thiserror::Error;

use crate::backend;

use crate::buffer::{BufferHandle, BufferType};
use crate::pipeline::ShaderStage;
use crate::resource::{BufferedStorage, Handle};
use crate::texture::Texture;
use crate::Renderer;
use backend::device::{Device, HasVkDevice, VkDeviceHandle};

use crate::common::MAX_FRAMES_IN_FLIGHT;

#[derive(Debug, Error)]
pub enum PipelineResourceError {
    #[error("Failed to create pipeline resource pool: {0}")]
    PoolCreation(vk::Result),
    #[error("Failed to allocate pipeline resource set: {0}")]
    SetAllocation(vk::Result),
}

struct PipelineResourcePool {
    vk_device: VkDeviceHandle,
    vk_descriptor_pool: vk::DescriptorPool,
    // These only exist here so that we can remove them all in the constructor
    vk_descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    // TODO: Keep track of buffers/images as well
    max_allocatable_sets: u32,
    n_allocated_sets: u32,
}

impl std::ops::Drop for PipelineResourcePool {
    fn drop(&mut self) {
        unsafe {
            self.vk_device
                .destroy_descriptor_pool(self.vk_descriptor_pool, None);
        }

        for dset_layout in self.vk_descriptor_set_layouts.iter() {
            unsafe {
                self.vk_device
                    .destroy_descriptor_set_layout(*dset_layout, None);
            }
        }
    }
}

impl PipelineResourcePool {
    // TODO: Accept layout here to compute individual descriptor counts
    fn new(device: &Device) -> Result<Self, PipelineResourceError> {
        let max_allocatable_sets = 1024 * MAX_FRAMES_IN_FLIGHT as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: max_allocatable_sets * 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: max_allocatable_sets * 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: max_allocatable_sets * 2,
            },
        ];

        let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(max_allocatable_sets);

        let vk_descriptor_pool = unsafe {
            device
                .vk_device()
                .create_descriptor_pool(&pool_create_info, None)
                .map_err(PipelineResourceError::PoolCreation)?
        };

        Ok(Self {
            vk_device: device.vk_device(),
            vk_descriptor_pool,
            vk_descriptor_set_layouts: Vec::new(),
            max_allocatable_sets,
            n_allocated_sets: 0,
        })
    }

    fn alloc(
        &mut self,
        layout: vk::DescriptorSetLayout,
        count: u32,
    ) -> Result<Vec<PipelineResourceSet>, PipelineResourceError> {
        assert!(
            self.n_allocated_sets + count < self.max_allocatable_sets,
            "Out of descriptor sets, time to implement dynamic creation of pools!"
        );
        let layouts = vec![layout; count as usize];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.vk_descriptor_pool)
            .set_layouts(&layouts);

        let desc_sets: Vec<PipelineResourceSet> = unsafe {
            self.vk_device
                .allocate_descriptor_sets(&info)
                .map_err(PipelineResourceError::SetAllocation)?
                .into_iter()
                .map(PipelineResourceSet::new)
                .collect()
        };

        self.n_allocated_sets += count;
        self.vk_descriptor_set_layouts.push(layout);

        Ok(desc_sets)
    }
}

pub struct PipelineResourceSetBuilder<'a> {
    renderer: &'a mut Renderer,
    bindings: Vec<(vk::DescriptorSetLayoutBinding, usize)>,
    buffer_infos: Vec<[vk::DescriptorBufferInfo; MAX_FRAMES_IN_FLIGHT]>,
    image_infos: Vec<vk::DescriptorImageInfo>,
}

impl<'a> PipelineResourceSetBuilder<'a> {
    fn new(renderer: &'a mut Renderer) -> Self {
        log::trace!("Starting to build pipeline resource set");
        Self {
            renderer,
            bindings: Vec::new(),
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
        }
    }
}

impl<'a> PipelineResourceSetBuilder<'a> {
    fn add_binding(
        &mut self,
        ty: vk::DescriptorType,
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
        count: u32,
    ) {
        let idx = if ty == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
            self.image_infos.len()
        } else {
            assert!([
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER
            ]
            .contains(&ty));
            self.buffer_infos.len()
        };

        self.bindings.push((
            vk::DescriptorSetLayoutBinding {
                binding,
                descriptor_type: ty,
                descriptor_count: count,
                stage_flags,
                ..Default::default()
            },
            idx,
        ));

        log::trace!("Added PRS binding {:?}", self.bindings.last().unwrap());
    }

    pub fn add_buffer(mut self, buf_h: BufferHandle, binding: u32, stage: ShaderStage) -> Self {
        let (buffer_type, buf0, buf1, stride0, stride1) = {
            let ubufs = &self.renderer.resources.buffers;

            let (buf0, buf1) = ubufs.get_all(buf_h).expect("Failed to get buffer");

            assert!(
                buf1.is_some() || buf_h.mutability() == crate::buffer::BufferMutability::Immutable
            );
            let buf1 = buf1.unwrap_or(buf0);
            (
                buf0.buffer_type(),
                buf0.vk_buffer(),
                buf1.vk_buffer(),
                buf0.stride(),
                buf1.stride(),
            )
        };

        let vk_desc_ty = match buffer_type {
            BufferType::Uniform(_) => vk::DescriptorType::UNIFORM_BUFFER,
            BufferType::Storage(_) => vk::DescriptorType::STORAGE_BUFFER,
            _ => panic!("Invalid buffer type, needs to be either uniform or storage."),
        };

        self.add_binding(vk_desc_ty, binding, vk::ShaderStageFlags::from(stage), 1);

        // TODO: This should check mutability of buffer
        // VMA allocator creates vk::Buffer from the device memory + offset so the offset from the buffer handle is enough here
        self.buffer_infos.push([
            vk::DescriptorBufferInfo {
                buffer: buf0,
                offset: buf_h.offset() as u64 * stride0 as u64,
                range: buf_h.len() as u64 * stride0 as u64,
            },
            vk::DescriptorBufferInfo {
                buffer: buf1,
                offset: buf_h.offset() as u64 * stride1 as u64,
                range: buf_h.len() as u64 * stride1 as u64,
            },
        ]);

        log::trace!(
            "Added PRS buffer info {:?}",
            self.buffer_infos.last().unwrap()
        );
        self
    }

    pub fn add_texture(
        self,
        tex_h: Handle<Texture>,
        binding: u32,
        stage: ShaderStage,
        is_depth: bool,
    ) -> Self {
        self.add_textures([(tex_h, is_depth)].into_iter(), binding, stage)
    }

    pub fn add_textures<I>(mut self, itr: I, binding: u32, stage: ShaderStage) -> Self
    where
        I: Iterator<Item = (Handle<Texture>, bool)> + ExactSizeIterator,
    {
        self.add_binding(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            binding,
            vk::ShaderStageFlags::from(stage),
            itr.len() as u32,
        );

        for (tex_handle, is_depth) in itr {
            let tex = self
                .renderer
                .get_texture(&tex_handle)
                .expect("Failed to get texture");

            let image_view = tex.full_image_view().vk_image_view();
            let sampler = tex.vk_sampler();

            let image_layout = if is_depth {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };
            let desc = vk::DescriptorImageInfo {
                image_layout,
                image_view,
                sampler,
            };
            log::trace!("Added PRS texture info {:?}", desc);
            self.image_infos.push(desc);
        }

        self
    }

    pub fn build(self) -> Handle<PipelineResourceSet> {
        let bindings_only = self.bindings.iter().map(|(x, _)| *x).collect::<Vec<_>>();
        let (handle, sets) = self.renderer.allocate_descriptor_sets(&bindings_only);
        let mut writes = Vec::new();

        for (bind_idx, (bind, info_idx)) in self.bindings.into_iter().enumerate() {
            for (set_idx, set) in sets.iter().enumerate() {
                let write = vk::WriteDescriptorSet {
                    dst_set: set.vk_descriptor_set,
                    dst_binding: bind_idx as u32,
                    descriptor_count: bind.descriptor_count,
                    descriptor_type: bind.descriptor_type,
                    ..Default::default()
                };
                if bind.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER {
                    assert!(self.image_infos.len() >= (info_idx + bind.descriptor_count as usize));
                    writes.push(vk::WriteDescriptorSet {
                        p_image_info: &self.image_infos[info_idx] as *const vk::DescriptorImageInfo,
                        ..write
                    });
                } else {
                    assert!([
                        vk::DescriptorType::STORAGE_BUFFER,
                        vk::DescriptorType::UNIFORM_BUFFER
                    ]
                    .contains(&bind.descriptor_type));
                    writes.push(vk::WriteDescriptorSet {
                        p_buffer_info: &self.buffer_infos[info_idx][set_idx]
                            as *const vk::DescriptorBufferInfo,
                        ..write
                    });
                }
            }
        }

        self.renderer.update_descriptor_sets(&writes);

        log::trace!("Built descriptor set with handle: {handle:?}");

        handle
    }
}

pub struct PipelineResourceSet {
    vk_descriptor_set: vk::DescriptorSet,
}

impl PipelineResourceSet {
    fn new(vk_descriptor_set: vk::DescriptorSet) -> Self {
        Self { vk_descriptor_set }
    }

    pub fn builder(renderer: &mut crate::Renderer) -> PipelineResourceSetBuilder {
        PipelineResourceSetBuilder::new(renderer)
    }

    pub fn vk_descriptor_set(&self) -> &vk::DescriptorSet {
        &self.vk_descriptor_set
    }
}

pub struct PipelineResourceSetStorage {
    vk_device: VkDeviceHandle,
    descriptor_pool: PipelineResourcePool,
    storage: BufferedStorage<PipelineResourceSet>,
}

impl PipelineResourceSetStorage {
    pub fn new(device: &Device) -> Result<Self, PipelineResourceError> {
        Ok(Self {
            vk_device: device.vk_device(),
            descriptor_pool: PipelineResourcePool::new(device)?,
            storage: Default::default(),
        })
    }

    pub fn alloc(
        &mut self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<(Handle<PipelineResourceSet>, &[PipelineResourceSet; 2]), PipelineResourceError>
    {
        // TODO: We should not create a descriptor set layout everytime we allocate. Hash bindings instead?
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
        let dset_layout = unsafe {
            self.vk_device
                .create_descriptor_set_layout(&info, None)
                .expect("Failed to create desc set layout")
        };

        let mut desc_sets = self
            .descriptor_pool
            .alloc(dset_layout, MAX_FRAMES_IN_FLIGHT as u32)?;
        let set0 = desc_sets.remove(0);
        let set1 = desc_sets.remove(0);
        let handle = self.storage.add([set0, set1]);
        let sets = self
            .storage
            .get_all(&handle)
            .expect("Descriptor sets that we just added are missing...");

        Ok((handle, sets))
    }

    pub fn get(
        &self,
        h: &Handle<PipelineResourceSet>,
        frame_idx: usize,
    ) -> Option<&PipelineResourceSet> {
        self.storage.get(h, frame_idx)
    }
}
