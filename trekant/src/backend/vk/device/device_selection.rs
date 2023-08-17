use ash::vk;

use std::ffi::CStr;
use std::ffi::CString;

use std::convert::{TryFrom, TryInto};

use crate::backend;

use backend::instance::Instance;
use backend::queue::QueueFamily;
use backend::surface::Surface;

use crate::util;

use super::error::DeviceCreationError;

fn log_physical_devices(instance: &Instance, devices: &[ash::vk::PhysicalDevice]) {
    for device in devices.iter() {
        log::info!("Found device: {:?}", device);
        let props = unsafe {
            instance
                .vk_instance()
                .get_physical_device_properties(*device)
        };
        log::debug!("Properties: {:#?}", props);
    }
}

fn log_device(instance: &Instance, device: &vk::PhysicalDevice) {
    log::info!("Vk device: {:?}", device);

    let props = unsafe {
        instance
            .vk_instance()
            .get_physical_device_properties(*device)
    };
    log::info!("Properties:");
    log::info!("\tvendor_id: {:?}", props.vendor_id);
    log::info!("\tdevice_id: {:?}", props.device_id);
    log::info!("\tdevice_type: {:?}", props.device_type);
    log::info!("\tdevice_name: {:?}", unsafe {
        CStr::from_ptr(props.device_name.as_ptr())
    });
}

fn required_device_extensions() -> Vec<CString> {
    vec![ash::extensions::khr::Swapchain::name().to_owned()]
}

#[derive(Clone, Debug)]
pub struct QueueSelection {
    pub family: QueueFamily,
    pub queue_index: u32,
}

#[derive(Clone, Debug)]
struct QueueFamiliesQuery {
    graphics: Option<QueueSelection>,
    present: Option<QueueSelection>,
    transfer: Option<QueueSelection>,
}

#[derive(Clone, Debug)]
pub struct QueueFamilies {
    pub graphics: QueueSelection,
    pub present: QueueSelection,
    pub transfer: QueueSelection,
}

impl TryFrom<QueueFamiliesQuery> for QueueFamilies {
    type Error = DeviceCreationError;
    fn try_from(v: QueueFamiliesQuery) -> Result<Self, Self::Error> {
        let present = v.present.ok_or(DeviceCreationError::UnsuitableDevice(
            DeviceSuitability::MissingPresentQueue,
        ))?;
        let graphics = v.graphics.ok_or(DeviceCreationError::UnsuitableDevice(
            DeviceSuitability::MissingGraphicsQueue,
        ))?;
        let transfer = v.transfer.ok_or(DeviceCreationError::UnsuitableDevice(
            DeviceSuitability::MissingTransferQueue,
        ))?;
        assert!(
            present.family.index != graphics.family.index
                || present.queue_index == graphics.queue_index
        );
        assert!(
            transfer.family.index != graphics.family.index
                || transfer.queue_index != graphics.queue_index
        );
        Ok(QueueFamilies {
            graphics,
            present,
            transfer,
        })
    }
}

fn find_queue_families(
    instance: &Instance,
    device: &vk::PhysicalDevice,
    surface: &Surface,
) -> Result<QueueFamiliesQuery, DeviceCreationError> {
    log::trace!("Checking queues for:");
    log_device(instance, device);

    let queue_fam_props = unsafe {
        instance
            .vk_instance()
            .get_physical_device_queue_family_properties(*device)
    };

    log::trace!("Found {} queues", queue_fam_props.len());
    for queue in queue_fam_props.iter() {
        log::trace!("{:#?}", queue);
    }

    let mut families = QueueFamiliesQuery {
        graphics: None,
        present: None,
        transfer: None,
    };

    families.graphics = queue_fam_props.iter().enumerate().find_map(|(i, fam)| {
        assert!(i <= u32::MAX as usize);
        if fam.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            Some(QueueSelection {
                family: QueueFamily {
                    props: *fam,
                    index: i as u32,
                },
                queue_index: 0,
            })
        } else {
            None
        }
    });

    // According to vulkan tutorial, "drawing and presentation" is more performant on the same
    // queue
    if let Some(gfx_index) = families.graphics.as_ref().map(|s| s.family.index) {
        for (i, fam) in queue_fam_props.iter().enumerate() {
            // present queue
            // If this is another family than graphics, we want the first queue.
            // If this is the same family as graphics, we use the same queue as we don't multithread
            // the access to the queue for drawing/presenting
            let same_as_gfx = gfx_index as usize == i;
            if surface.is_supported_by(device, i as u32)?
                && (same_as_gfx || families.present.is_none())
            {
                families.present = Some(QueueSelection {
                    family: QueueFamily {
                        props: *fam,
                        index: i as u32,
                    },
                    queue_index: 0,
                });
            }

            // transfer queue
            // We want the "solo" transfer queue family if possible, otherwise the same as graphics.
            // We want a different queue though, as it is used to submit independently from different threads
            let may_use_gfx_family = families.transfer.is_none() && fam.queue_count >= 2;
            if fam.queue_flags.contains(vk::QueueFlags::TRANSFER)
                && (!same_as_gfx || may_use_gfx_family)
            {
                let queue_index = if same_as_gfx && may_use_gfx_family {
                    1
                } else {
                    0
                };
                families.transfer = Some(QueueSelection {
                    family: QueueFamily {
                        props: *fam,
                        index: i as u32,
                    },
                    queue_index,
                });
            }
        }
    }

    Ok(families)
}

fn device_supports_extensions<T: AsRef<CStr>>(
    instance: &Instance,
    device: &vk::PhysicalDevice,
    required_extensions: &[T],
) -> Result<bool, DeviceCreationError> {
    let available = unsafe {
        instance
            .vk_instance()
            .enumerate_device_extension_properties(*device)
            .map_err(|e| DeviceCreationError::InternalVulkan(e, "Device exentension query"))?
    };

    for r in required_extensions.iter() {
        let mut found = false;
        for avail in available.iter() {
            let a = unsafe { CStr::from_ptr(avail.extension_name.as_ptr()) };
            if r.as_ref() == a {
                found = true;
            }
        }

        if !found {
            return Ok(false);
        }
    }

    Ok(true)
}

// TODO: Improve granularity of MissingRequiredExtensions
#[derive(Debug, Clone, Copy)]
pub enum DeviceSuitability {
    Suitable,
    MissingRequiredExtensions,
    MissingRequiredFeatures,
    MissingGraphicsQueue,
    MissingPresentQueue,
    MissingTransferQueue,
    MissingDepthFormat,
    UnsuitableSwapchainFormat,
    UnsuitableSwapchainPresentMode,
    MissingMipmapGenerationSupport,
}

impl DeviceSuitability {
    pub fn is_suitable(&self) -> bool {
        matches!(self, DeviceSuitability::Suitable)
    }
}

impl std::fmt::Display for DeviceSuitability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

fn required_device_features() -> vk::PhysicalDeviceFeatures {
    vk::PhysicalDeviceFeatures {
        sampler_anisotropy: vk::TRUE,
        fill_mode_non_solid: vk::TRUE,
        ..Default::default()
    }
}

// TODO: ash does not support struct eq for features :(
fn device_supports_features(
    instance: &Instance,
    phys_device: &vk::PhysicalDevice,
    _features: &vk::PhysicalDeviceFeatures,
) -> bool {
    let supported = unsafe {
        instance
            .vk_instance()
            .get_physical_device_features(*phys_device)
    };

    supported.sampler_anisotropy == vk::TRUE && supported.fill_mode_non_solid == vk::TRUE
}

fn device_supports_mipmap_generation(
    instance: &Instance,
    vk_phys_device: &vk::PhysicalDevice,
) -> bool {
    super::find_supported_format(
        instance,
        vk_phys_device,
        &[vk::Format::R8G8B8A8_SRGB],
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR,
    )
    .is_some()
}

fn check_device_suitability(
    instance: &Instance,
    device: &vk::PhysicalDevice,
    surface: &Surface,
) -> Result<DeviceSuitability, DeviceCreationError> {
    if !device_supports_extensions(instance, device, &required_device_extensions())? {
        return Ok(DeviceSuitability::MissingRequiredExtensions);
    }

    if !device_supports_features(instance, device, &required_device_features()) {
        return Ok(DeviceSuitability::MissingRequiredFeatures);
    }

    if !device_supports_mipmap_generation(instance, device) {
        return Ok(DeviceSuitability::MissingMipmapGenerationSupport);
    }

    if super::find_depth_format(instance, device).is_none() {
        return Ok(DeviceSuitability::MissingDepthFormat);
    }

    let fams = find_queue_families(instance, device, surface)?;

    if fams.graphics.is_none() {
        return Ok(DeviceSuitability::MissingGraphicsQueue);
    }

    if fams.present.is_none() {
        return Ok(DeviceSuitability::MissingPresentQueue);
    }

    if fams.transfer.is_none() {
        return Ok(DeviceSuitability::MissingTransferQueue);
    }

    let swapchain_query = surface.query_swapchain_support(device)?;

    if swapchain_query.formats.is_empty() {
        return Ok(DeviceSuitability::UnsuitableSwapchainFormat);
    }

    if swapchain_query.present_modes.is_empty() {
        return Ok(DeviceSuitability::UnsuitableSwapchainPresentMode);
    }

    Ok(DeviceSuitability::Suitable)
}

fn score_device(
    instance: &Instance,
    device: &vk::PhysicalDevice,
    surface: &Surface,
) -> Result<u32, DeviceCreationError> {
    let device_props = unsafe {
        instance
            .vk_instance()
            .get_physical_device_properties(*device)
    };

    let mut score = 0;

    if device_props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        score += 100;
    }

    if check_device_suitability(instance, device, surface)?.is_suitable() {
        score += 1000;
    }

    Ok(score)
}

fn log_queue_selection(sel: &QueueSelection) {
    let fam = &sel.family;
    log::trace!("\tfam_index: {}", fam.index);
    log::trace!("\tflags: {:?}", fam.props.queue_flags);
    log::trace!("\tqueue_count: {}", fam.props.queue_count);
    log::trace!("\tqueue_index: {}", sel.queue_index);
}

fn log_queue_families(qfams: &QueueFamilies) {
    log::trace!("Graphics:");
    log_queue_selection(&qfams.graphics);
    log::trace!("Present:");
    log_queue_selection(&qfams.present);
    log::trace!("Transfer");
    log_queue_selection(&qfams.transfer);
}

pub fn device_selection(
    instance: &Instance,
    surface: &Surface,
) -> Result<(ash::Device, vk::PhysicalDevice, QueueFamilies), DeviceCreationError> {
    let physical_devices = unsafe {
        instance
            .vk_instance()
            .enumerate_physical_devices()
            .map_err(|e| DeviceCreationError::InternalVulkan(e, "Physical device enumeration"))?
    };

    if physical_devices.is_empty() {
        return Err(DeviceCreationError::MissingPhysicalDevice);
    }

    log_physical_devices(instance, &physical_devices);
    let suitability_checks = physical_devices
        .iter()
        .map(|d| check_device_suitability(instance, d, surface))
        .collect::<Result<Vec<DeviceSuitability>, DeviceCreationError>>()?;

    if !suitability_checks.iter().any(|c| c.is_suitable()) {
        return Err(DeviceCreationError::UnsuitableDevice(suitability_checks[0]));
    }

    // The collect() creates a Result<Vec<_>>, using the first Err it finds in the vector (if any). Then ?
    // does an early return if it is Err.
    let mut scored: Vec<(u32, vk::PhysicalDevice)> = physical_devices
        .iter()
        .map(|d| score_device(instance, d, surface).map(|s| (s, *d)))
        .collect::<Result<Vec<_>, DeviceCreationError>>()?;

    // Note that switched args. Higher score should be earlier
    scored.sort_by(|a, b| b.0.cmp(&a.0));

    let vk_phys_device = scored[0].1;
    log::info!("Choosing device:");
    log_device(instance, &vk_phys_device);

    let queue_families_query = find_queue_families(instance, &vk_phys_device, surface)?;

    let queue_families: QueueFamilies = queue_families_query
        .try_into()
        .expect("This device should not have been chosen!");

    log::trace!("Choosing queue families:");
    log_queue_families(&queue_families);
    let mut fam_indices = [
        queue_families.graphics.family.index,
        queue_families.present.family.index,
        queue_families.transfer.family.index,
    ];
    fam_indices.sort_unstable();
    let mut queue_infos: Vec<vk::DeviceQueueCreateInfo> = Vec::with_capacity(3);
    let prio_1 = [1.0];
    let prio_2 = [1.0, 1.0];
    let prio_3 = [1.0, 1.0, 1.0];
    let prios: [&[f32]; 3] = [&prio_1, &prio_2, &prio_3];
    for &fi in &fam_indices {
        if queue_infos.is_empty() || fi != queue_infos.last().unwrap().queue_family_index {
            queue_infos.push(vk::DeviceQueueCreateInfo {
                queue_family_index: fi,
                p_queue_priorities: prio_1.as_ptr(),
                queue_count: 1,
                ..Default::default()
            });
        } else {
            let prev = queue_infos.last_mut().unwrap();
            assert_eq!(fi, prev.queue_family_index);
            prev.queue_count += 1;
            prev.p_queue_priorities = prios[prev.queue_count as usize - 1].as_ptr();
        }
    }

    let extensions = required_device_extensions();
    let extensions_ptrs = util::ffi::vec_cstring_to_raw(extensions);

    let features = required_device_features();

    let device_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extensions_ptrs)
        .enabled_features(&features);

    let vk_device = unsafe {
        instance
            .vk_instance()
            .create_device(vk_phys_device, &device_info, None)
            .map_err(DeviceCreationError::Creation)?
    };

    let _owned_extensions = util::ffi::vec_cstring_from_raw(extensions_ptrs);

    Ok((vk_device, vk_phys_device, queue_families))
}
