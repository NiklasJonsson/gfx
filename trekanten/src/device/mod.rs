use ash::version::DeviceV1_0;
use ash::version::InstanceV1_0;
use ash::vk;

use vk_mem::Allocator;

use std::rc::Rc;

use crate::instance::Instance;
use crate::queue::Queue;
use crate::queue::QueueFamilies;
use crate::queue::QueueFamily;
use crate::surface::Surface;
use crate::util::lifetime::LifetimeToken;

mod device_selection;
mod error;

pub use error::DeviceError;

pub type VkDevice = ash::Device;
pub type VkDeviceHandle = Rc<VkDevice>;
pub type AllocatorHandle = Rc<Allocator>;

pub trait HasVkDevice {
    fn vk_device(&self) -> VkDeviceHandle;
}

impl HasVkDevice for VkDeviceHandle {
    fn vk_device(&self) -> VkDeviceHandle {
        Rc::clone(&self)
    }
}

struct PhysicalDeviceProperties {
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    depth_buffer_format: vk::Format,
    _supported_msaa_sample_counts: vk::SampleCountFlags,
    max_supported_msaa_sample_count: vk::SampleCountFlags,
}

struct QueueInfo {
    queue_families: QueueFamilies,
    graphics_queue: Queue,
    present_queue: Queue,
}

// Use this to handle drop-order. Could have been done with unsafe/ManuallyDrop but this seems the easiest
// This needs to be after the allocator in the declaration order inside Device to ensure the allocator is destroyed before this.
struct InnerDevice {
    vk_device: VkDeviceHandle,
}

impl std::ops::Drop for InnerDevice {
    fn drop(&mut self) {
        // TODO: Change to weak
        if !Rc::strong_count(&self.vk_device) == 1 {
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
        Rc::clone(&self.inner_device.vk_device)
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

        let (gfx, present) = unsafe {
            (
                vk_device.get_device_queue(queue_families.graphics.index, 0),
                vk_device.get_device_queue(queue_families.present.index, 0),
            )
        };

        let vk_device = Rc::new(vk_device);

        let graphics_queue = Queue::new(Rc::clone(&vk_device), gfx);
        let present_queue = Queue::new(Rc::clone(&vk_device), present);

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
                depth_buffer_format,
                _supported_msaa_sample_counts,
                max_supported_msaa_sample_count,
            }
        };

        let queue_info = QueueInfo {
            queue_families,
            graphics_queue,
            present_queue,
        };

        let allocator = Rc::new(Allocator::new(&vk_mem::AllocatorCreateInfo {
            physical_device: vk_phys_device,
            device: (*vk_device).clone(),
            instance: instance.vk_instance().clone(),
            ..Default::default()
        })?);

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
        &self.queue_info.queue_families.graphics
    }

    pub fn util_queue_family(&self) -> &QueueFamily {
        &self.queue_info.queue_families.graphics
    }

    pub fn present_queue_family(&self) -> &QueueFamily {
        &self.queue_info.queue_families.present
    }

    pub fn graphics_queue(&self) -> &Queue {
        &self.queue_info.graphics_queue
    }

    pub fn util_queue(&self) -> &Queue {
        &self.queue_info.graphics_queue
    }

    pub fn present_queue(&self) -> &Queue {
        &self.queue_info.present_queue
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

    pub fn allocator(&self) -> AllocatorHandle {
        Rc::clone(&self.allocator)
    }
}
