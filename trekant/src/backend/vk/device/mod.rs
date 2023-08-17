use ash::vk;

use vma::Allocator;

use std::sync::Arc;

use super::instance::Instance;
use super::queue::{Queue, QueueFamily};
use super::surface::Surface;
use crate::util::lifetime::LifetimeToken;

mod device_selection;
mod error;

pub use error::DeviceError;

pub type VkDevice = ash::Device;
pub type VkDeviceHandle = Arc<VkDevice>;
pub type AllocatorHandle = Arc<Allocator>;

pub trait HasVkDevice {
    fn vk_device(&self) -> VkDeviceHandle;
}

impl HasVkDevice for VkDeviceHandle {
    fn vk_device(&self) -> VkDeviceHandle {
        VkDeviceHandle::clone(self)
    }
}

struct PhysicalDeviceProperties {
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    vk_device_properties: vk::PhysicalDeviceProperties,
    depth_buffer_format: vk::Format,
    _supported_msaa_sample_counts: vk::SampleCountFlags,
    max_supported_msaa_sample_count: vk::SampleCountFlags,
}

struct QueueInfo {
    graphics: Queue,
    present: Queue,
    transfer: Option<Queue>, // Ownership can be requested
}

// Use this to handle drop-order. Could have been done with unsafe/ManuallyDrop but this seems the easiest
// This needs to be after the allocator in the declaration order inside Device to ensure the allocator is destroyed before this.
struct InnerDevice {
    vk_device: VkDeviceHandle,
}

impl std::ops::Drop for InnerDevice {
    fn drop(&mut self) {
        // TODO: Change to weak
        if !VkDeviceHandle::strong_count(&self.vk_device) == 1 {
            log::error!(
                "References to inner vk device still existing but Device is being destroyed!"
            );
        }
        unsafe { self.vk_device.destroy_device(None) };
    }
}

pub struct Device {
    allocator: AllocatorHandle,
    queue_info: QueueInfo,
    vk_phys_device: vk::PhysicalDevice,

    physical_device_properties: PhysicalDeviceProperties,
    inner_device: InnerDevice,
    _parent_lifetime_token: LifetimeToken<Instance>,
}

impl HasVkDevice for Device {
    fn vk_device(&self) -> VkDeviceHandle {
        VkDeviceHandle::clone(&self.inner_device.vk_device)
    }
}

fn find_supported_format(
    instance: &Instance,
    vk_phys_device: &vk::PhysicalDevice,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Option<vk::Format> {
    for can in candidates {
        let props = unsafe {
            instance
                .vk_instance()
                .get_physical_device_format_properties(*vk_phys_device, *can)
        };

        let linear_match =
            tiling == vk::ImageTiling::LINEAR && props.linear_tiling_features.contains(features);
        let optimal_match =
            tiling == vk::ImageTiling::OPTIMAL && props.optimal_tiling_features.contains(features);
        if linear_match || optimal_match {
            return Some(*can);
        }
    }

    None
}

fn find_depth_format(
    instance: &Instance,
    vk_phys_device: &vk::PhysicalDevice,
) -> Option<vk::Format> {
    let cands = [
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];
    find_supported_format(
        instance,
        vk_phys_device,
        &cands,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

fn get_max_supported_msaa(flags: vk::SampleCountFlags) -> vk::SampleCountFlags {
    for &count in [
        vk::SampleCountFlags::TYPE_64,
        vk::SampleCountFlags::TYPE_32,
        vk::SampleCountFlags::TYPE_16,
        vk::SampleCountFlags::TYPE_8,
        vk::SampleCountFlags::TYPE_4,
        vk::SampleCountFlags::TYPE_2,
    ]
    .iter()
    {
        if flags.contains(count) {
            return count;
        }
    }

    vk::SampleCountFlags::TYPE_1
}

impl Device {
    pub fn new(instance: &Instance, surface: &Surface) -> Result<Self, DeviceError> {
        let (vk_device, vk_phys_device, queue_families) =
            device_selection::device_selection(instance, surface)?;

        let device_selection::QueueFamilies {
            graphics: graphics_fam,
            present: present_fam,
            transfer: transfer_fam,
        } = queue_families;

        let vk_device = VkDeviceHandle::new(vk_device);
        let create_queue = |qsel: device_selection::QueueSelection| -> Queue {
            let vk_queue =
                unsafe { vk_device.get_device_queue(qsel.family.index, qsel.queue_index) };

            Queue::new(VkDeviceHandle::clone(&vk_device), qsel.family, vk_queue)
        };

        let queue_info = QueueInfo {
            graphics: create_queue(graphics_fam),
            present: create_queue(present_fam),
            transfer: Some(create_queue(transfer_fam)),
        };

        let physical_device_properties = unsafe {
            let memory_properties = instance
                .vk_instance()
                .get_physical_device_memory_properties(vk_phys_device);

            let depth_buffer_format = find_depth_format(instance, &vk_phys_device)
                .expect("Missing depth buffer format, this device should not have been created");

            let vk_props = instance
                .vk_instance()
                .get_physical_device_properties(vk_phys_device);

            let _supported_msaa_sample_counts = vk_props.limits.framebuffer_color_sample_counts
                & vk_props.limits.framebuffer_depth_sample_counts;
            let max_supported_msaa_sample_count =
                get_max_supported_msaa(_supported_msaa_sample_counts);

            PhysicalDeviceProperties {
                memory_properties,
                vk_device_properties: vk_props,
                depth_buffer_format,
                _supported_msaa_sample_counts,
                max_supported_msaa_sample_count,
            }
        };

        let allocator = AllocatorHandle::new(Allocator::new(vma::AllocatorCreateInfo::new(
            instance.vk_instance(),
            &vk_device,
            vk_phys_device,
        ))?);

        let inner_device = InnerDevice { vk_device };

        Ok(Self {
            inner_device,
            allocator,
            vk_phys_device,
            queue_info,
            _parent_lifetime_token: instance.lifetime_token(),
            physical_device_properties,
        })
    }

    pub fn graphics_queue_family(&self) -> &QueueFamily {
        self.queue_info.graphics.family()
    }

    pub fn present_queue_family(&self) -> &QueueFamily {
        self.queue_info.present.family()
    }

    pub fn graphics_queue(&self) -> &Queue {
        &self.queue_info.graphics
    }

    pub fn present_queue(&self) -> &Queue {
        &self.queue_info.present
    }

    pub fn take_transfer_queue(&mut self) -> Option<Queue> {
        self.queue_info.transfer.take()
    }

    pub fn wait_idle(&self) -> Result<(), DeviceError> {
        unsafe {
            self.inner_device
                .vk_device
                .device_wait_idle()
                .map_err(DeviceError::WaitIdle)?;
        }

        Ok(())
    }

    pub fn vk_phys_device(&self) -> &vk::PhysicalDevice {
        &self.vk_phys_device
    }

    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.physical_device_properties.memory_properties
    }

    // TODO: Use util::Format here
    pub fn depth_buffer_format(&self) -> vk::Format {
        self.physical_device_properties.depth_buffer_format
    }

    pub fn max_msaa_sample_count(&self) -> vk::SampleCountFlags {
        self.physical_device_properties
            .max_supported_msaa_sample_count
    }

    pub fn uniform_buffer_offset_alignment(&self) -> u64 {
        self.physical_device_properties
            .vk_device_properties
            .limits
            .min_uniform_buffer_offset_alignment
    }

    pub fn allocator(&self) -> AllocatorHandle {
        AllocatorHandle::clone(&self.allocator)
    }
}
