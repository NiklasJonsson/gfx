use ash::vk;

use ash::version::DeviceV1_0;

use thiserror::Error;

use crate::device::Device;
use crate::device::HasVkDevice;
use crate::device::VkDeviceHandle;
use crate::resource::{BufferedStorage, Handle};
use crate::texture::Texture;
use crate::uniform::UniformBuffer;

use crate::common::MAX_FRAMES_IN_FLIGHT;

#[derive(Debug, Error)]
pub enum DescriptorError {
    #[error("Failed to allocate descriptor set: {0}")]
    PoolCreation(vk::Result),
    #[error("Failed to allocate descriptor set: {0}")]
    SetAllocation(vk::Result),
}

struct DescriptorPool {
    vk_device: VkDeviceHandle,
    vk_descriptor_pool: vk::DescriptorPool,
    n_allocated: usize,
}

impl std::ops::Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.vk_device
                .destroy_descriptor_pool(self.vk_descriptor_pool, None);
        }
    }
}

impl DescriptorPool {
    fn new(device: &Device) -> Result<Self, DescriptorError> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
            },
        ];

        let pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

        let vk_descriptor_pool = unsafe {
            device
                .vk_device()
                .create_descriptor_pool(&pool_create_info, None)
                .map_err(DescriptorError::PoolCreation)?
        };

        Ok(Self {
            vk_device: device.vk_device(),
            vk_descriptor_pool,
            n_allocated: 0,
        })
    }

    fn alloc(
        &mut self,
        layout: &vk::DescriptorSetLayout,
        count: usize,
    ) -> Result<Vec<DescriptorSet>, DescriptorError> {
        let layouts = vec![*layout; count];
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

        self.n_allocated += count;

        Ok(desc_sets)
    }
}

pub struct DescriptorSetBuilder {

}

impl DescriptorSetBuilder {
    pub fn add_buffer(self, _: &Handle<UniformBuffer>) -> Self {
        unimplemented!()
    }

    pub fn add_texture(self, _: &Handle<Texture>) -> Self {
        unimplemented!()
    }

    pub fn build(self) -> Handle<DescriptorSet> {
        unimplemented!()
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

    pub fn builder(_renderer: &crate::Renderer) -> DescriptorSetBuilder {
        unimplemented!()
    }

    fn bind_resources(
        &self,
        vk_device: &VkDeviceHandle,
        buffer: &UniformBuffer,
        texture: &Texture,
    ) {
        let buffer_info = vk::DescriptorBufferInfo {
            buffer: *buffer.vk_buffer(),
            offset: 0,
            range: buffer.elem_size() as u64,
        };
        let buffer_infos = [buffer_info];

        // TODO: Use the values from the layout
        let buffer_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.vk_descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_infos)
            .build();

        let image_info = vk::DescriptorImageInfo {
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image_view: *texture.vk_image_view(),
            sampler: *texture.vk_sampler(),
        };
        let image_infos = [image_info];

        // TODO: Use the values from the layout
        let image_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.vk_descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos)
            .build();

        let writes = [buffer_write, image_write];

        unsafe {
            vk_device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn vk_descriptor_set(&self) -> &vk::DescriptorSet {
        &self.vk_descriptor_set
    }
}

pub struct DescriptorSetDescriptor<'a> {
    pub layout: vk::DescriptorSetLayout,
    pub uniform_buffers: &'a [UniformBuffer; MAX_FRAMES_IN_FLIGHT],
    pub texture: &'a Texture,
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

    pub fn create<'a>(
        &mut self,
        descriptor: DescriptorSetDescriptor<'a>,
    ) -> Result<Handle<DescriptorSet>, DescriptorError> {
        let mut desc_sets = self
            .descriptor_pool
            .alloc(&descriptor.layout, MAX_FRAMES_IN_FLIGHT)?;
        let set0 = desc_sets.remove(0);
        let set1 = desc_sets.remove(0);

        for (i, s) in [&set0, &set1].iter().enumerate() {
            s.bind_resources(
                &self.vk_device,
                &descriptor.uniform_buffers[i],
                descriptor.texture,
            );
        }

        Ok(self.storage.add([set0, set1]))
    }

    pub fn get(&self, h: &Handle<DescriptorSet>, frame_idx: usize) -> Option<&DescriptorSet> {
        self.storage.get(h, frame_idx)
    }
}
