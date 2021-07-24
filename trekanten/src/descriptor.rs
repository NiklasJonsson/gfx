use ash::vk;

use ash::version::DeviceV1_0;

use thiserror::Error;

use crate::backend;

use crate::buffer::BufferHandle;
use crate::buffer::UniformBuffer;
use crate::pipeline::ShaderStage;
use crate::resource::{BufferedStorage, Handle};
use crate::texture::Texture;
use crate::Renderer;
use backend::device::{Device, HasVkDevice, VkDeviceHandle};

use crate::common::MAX_FRAMES_IN_FLIGHT;

#[derive(Debug, Error)]
pub enum DescriptorError {
    #[error("Failed to create descriptor pool: {0}")]
    PoolCreation(vk::Result),
    #[error("Failed to allocate descriptor set: {0}")]
    SetAllocation(vk::Result),
}

struct DescriptorPool {
    vk_device: VkDeviceHandle,
    vk_descriptor_pool: vk::DescriptorPool,
    // These only exist here so that we can remove them all in the constructor
    vk_descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    // TODO: Keep track of buffers/images as well
    max_allocatable_sets: u32,
    n_allocated_sets: u32,
}

impl std::ops::Drop for DescriptorPool {
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

impl DescriptorPool {
    // TODO: Accept layout here to compute individual descriptor counts
    fn new(device: &Device) -> Result<Self, DescriptorError> {
        let max_allocatable_sets = 128 * MAX_FRAMES_IN_FLIGHT as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: max_allocatable_sets * 2 as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: max_allocatable_sets * 2 as u32,
            },
        ];

        let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(max_allocatable_sets);

        let vk_descriptor_pool = unsafe {
            device
                .vk_device()
                .create_descriptor_pool(&pool_create_info, None)
                .map_err(DescriptorError::PoolCreation)?
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
    ) -> Result<Vec<DescriptorSet>, DescriptorError> {
        assert!(
            self.n_allocated_sets + count < self.max_allocatable_sets,
            "Out of descriptor sets, time to implement dynamic creation of pools!"
        );
        let layouts = vec![layout; count as usize];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.vk_descriptor_pool)
            .set_layouts(&layouts);

        let desc_sets: Vec<DescriptorSet> = unsafe {
            self.vk_device
                .allocate_descriptor_sets(&info)
                .map_err(DescriptorError::SetAllocation)?
                .into_iter()
                .map(DescriptorSet::new)
                .collect()
        };

        self.n_allocated_sets += count;
        self.vk_descriptor_set_layouts.push(layout);

        Ok(desc_sets)
    }
}

pub struct DescriptorSetBuilder<'a> {
    renderer: &'a mut Renderer,
    bindings: Vec<(vk::DescriptorSetLayoutBinding, usize)>,
    buffer_infos: Vec<[vk::DescriptorBufferInfo; MAX_FRAMES_IN_FLIGHT]>,
    image_infos: Vec<vk::DescriptorImageInfo>,
}

impl<'a> DescriptorSetBuilder<'a> {
    fn new(renderer: &'a mut Renderer) -> Self {
        Self {
            renderer,
            bindings: Vec::new(),
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
        }
    }
}

impl<'a> DescriptorSetBuilder<'a> {
    fn add_binding(
        &mut self,
        ty: vk::DescriptorType,
        binding: u32,
        stage_flags: vk::ShaderStageFlags,
        count: u32,
    ) {
        let idx = if let vk::DescriptorType::UNIFORM_BUFFER = ty {
            self.buffer_infos.len()
        } else {
            self.image_infos.len()
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

        log::trace!(
            "Added descriptor binding {:?}",
            self.bindings.last().unwrap()
        );
    }

    pub fn add_buffer(
        mut self,
        buf_h: &BufferHandle<UniformBuffer>,
        binding: u32,
        stage: ShaderStage,
    ) -> Self {
        let (buf0, buf1, stride0, stride1) = {
            let ubufs = &self.renderer.resources.uniform_buffers;

            let (buf0, buf1) = ubufs.get_all(&buf_h).expect("Failed to get buffer");

            assert!(
                buf1.is_some() || buf_h.mutability() == crate::buffer::BufferMutability::Immutable
            );
            let buf1 = buf1.unwrap_or(buf0);
            (
                *buf0.vk_buffer(),
                *buf1.vk_buffer(),
                buf0.stride(),
                buf1.stride(),
            )
        };

        self.add_binding(
            vk::DescriptorType::UNIFORM_BUFFER,
            binding,
            vk::ShaderStageFlags::from(stage),
            1,
        );

        // TODO: This should check mutability of buffer
        // VMA allocator creates vk::Buffer from the device memory + offset so the offset from the buffer handle is enough here
        self.buffer_infos.push([
            vk::DescriptorBufferInfo {
                buffer: buf0,
                offset: buf_h.idx() as u64 * stride0 as u64,
                range: buf_h.n_elems() as u64 * stride0 as u64,
            },
            vk::DescriptorBufferInfo {
                buffer: buf1,
                offset: buf_h.idx() as u64 * stride1 as u64,
                range: buf_h.n_elems() as u64 * stride1 as u64,
            },
        ]);

        log::trace!("Added buffer info {:?}", self.buffer_infos.last().unwrap());
        self
    }

    pub fn add_texture(
        mut self,
        tex_h: &Handle<Texture>,
        binding: u32,
        stage: ShaderStage,
        is_depth: bool,
    ) -> Self {
        let tex = self
            .renderer
            .get_texture(tex_h)
            .expect("Failed to get texture");

        let image_view = *tex.image_view().vk_image_view();
        let sampler = *tex.vk_sampler();

        self.add_binding(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            binding,
            vk::ShaderStageFlags::from(stage),
            1,
        );

        let image_layout = if is_depth {
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        self.image_infos.push(vk::DescriptorImageInfo {
            image_layout,
            image_view,
            sampler,
        });
        log::trace!("Added texture info {:?}", self.image_infos.last().unwrap());

        self
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

            let image_view = *tex.image_view().vk_image_view();
            let sampler = *tex.vk_sampler();

            let image_layout = if is_depth {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            } else {
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            };
            self.image_infos.push(vk::DescriptorImageInfo {
                image_layout,
                image_view,
                sampler,
            });
        }

        self
    }

    pub fn build(self) -> Handle<DescriptorSet> {
        let bindings_only = self.bindings.iter().map(|(x, _)| *x).collect::<Vec<_>>();
        let (handle, sets) = self.renderer.allocate_descriptor_sets(&bindings_only);
        let mut writes = Vec::new();

        for (bind_idx, (bind, info_idx)) in self.bindings.into_iter().enumerate() {
            for (set_idx, set) in sets.iter().enumerate() {
                if bind.descriptor_type == vk::DescriptorType::UNIFORM_BUFFER {
                    writes.push(set.write_buffer(
                        &self.buffer_infos[info_idx][set_idx],
                        bind_idx as u32,
                        1,
                    ));
                } else {
                    assert!(bind.descriptor_type == vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
                    assert!(self.image_infos.len() >= (info_idx + bind.descriptor_count as usize));
                    let write = vk::WriteDescriptorSet {
                        dst_set: set.vk_descriptor_set,
                        dst_binding: bind_idx as u32,
                        descriptor_count: bind.descriptor_count,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &self.image_infos[info_idx] as *const vk::DescriptorImageInfo,
                        ..Default::default()
                    };
                    writes.push(write);
                }
            }
        }

        self.renderer.update_descriptor_sets(&writes);

        handle
    }
}

// TODO: Rename? (to avoid DescriptorSetDescriptor)
pub struct DescriptorSet {
    vk_descriptor_set: vk::DescriptorSet,
}

impl DescriptorSet {
    fn new(vk_descriptor_set: vk::DescriptorSet) -> Self {
        Self { vk_descriptor_set }
    }

    pub fn builder(renderer: &mut crate::Renderer) -> DescriptorSetBuilder {
        DescriptorSetBuilder::new(renderer)
    }

    fn write_buffer(
        &self,
        buffer: &vk::DescriptorBufferInfo,
        dst_binding: u32,
        count: u32,
    ) -> vk::WriteDescriptorSet {
        vk::WriteDescriptorSet {
            dst_set: self.vk_descriptor_set,
            dst_binding,
            descriptor_count: count,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info: buffer as *const vk::DescriptorBufferInfo,
            ..Default::default()
        }
    }

    pub fn vk_descriptor_set(&self) -> &vk::DescriptorSet {
        &self.vk_descriptor_set
    }
}

pub struct DescriptorSetDescriptor {
    pub layout: vk::DescriptorSetLayout,
}

pub struct DescriptorSets {
    vk_device: VkDeviceHandle,
    descriptor_pool: DescriptorPool,
    storage: BufferedStorage<DescriptorSet>,
}

impl DescriptorSets {
    pub fn new(device: &Device) -> Result<Self, DescriptorError> {
        Ok(Self {
            vk_device: device.vk_device(),
            descriptor_pool: DescriptorPool::new(device)?,
            storage: Default::default(),
        })
    }

    pub fn alloc(
        &mut self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> Result<(Handle<DescriptorSet>, &[DescriptorSet; 2]), DescriptorError> {
        // TODO: We should not create a descriptor set layout everytime we allocate. Hash bindings instead?
        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
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

    pub fn get(&self, h: &Handle<DescriptorSet>, frame_idx: usize) -> Option<&DescriptorSet> {
        self.storage.get(h, frame_idx)
    }
}
