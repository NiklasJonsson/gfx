use ash::extensions::khr::Swapchain as SwapchainLoader;
use ash::vk;

use thiserror::Error;

use crate::device::VkDeviceHandle;

use crate::color_buffer::ColorBuffer;
use crate::depth_buffer::DepthBuffer;
use crate::device::Device;
use crate::device::HasVkDevice;
use crate::framebuffer::{Framebuffer, FramebufferError};
use crate::image::{ImageView, ImageViewError};
use crate::instance::Instance;
use crate::queue::Queue;
use crate::render_pass::RenderPass;
use crate::surface::{Surface, SurfaceError};
use crate::sync::Semaphore;
use crate::util;

#[derive(Clone, Debug, Error)]
pub enum SwapchainError {
    #[error("Vulkan object creation failed: {1} {0}")]
    VulkanObjectCreation(vk::Result, &'static str),
    #[error("Image view creation failed: {0}")]
    ImageView(#[from] ImageViewError),
    #[error("Framebuffer creation failed: {0}")]
    Framebuffer(#[from] FramebufferError),
    #[error("Failed to acquire next image {0}")]
    AcquireNextImage(vk::Result),
    #[error("Failed to enqueue present command {0}")]
    EnqueuePresent(vk::Result),
    #[error("Swapchain surface issue {0}")]
    Surface(#[from] SurfaceError),
    #[error("Swapchain out of date")]
    OutOfDate,
}
#[derive(Debug, Clone, Copy)]
pub enum SwapchainStatus {
    Optimal,
    SubOptimal,
}

#[derive(Debug, Clone, Copy)]
pub struct SwapchainInfo {
    pub format: vk::Format,
    pub extent: util::Extent2D,
}

pub struct Swapchain {
    loader: SwapchainLoader,
    handle: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<ImageView>,
    info: SwapchainInfo,
    vk_device: VkDeviceHandle,
}

impl std::ops::Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_swapchain(self.handle, None) };
    }
}

fn choose_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    for f in formats.iter() {
        if f.format == vk::Format::B8G8R8A8_SRGB
            && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return *f;
        }
    }

    formats[0]
}

fn choose_swapchain_surface_present_mode(pmodes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    for pm in pmodes.iter() {
        if *pm == vk::PresentModeKHR::MAILBOX {
            return *pm;
        }
    }

    // Always available according to spec
    vk::PresentModeKHR::FIFO
}

fn choose_swapchain_extent(
    capabilites: &vk::SurfaceCapabilitiesKHR,
    extent: &util::Extent2D,
) -> vk::Extent2D {
    if capabilites.current_extent.width != u32::MAX {
        capabilites.current_extent
    } else {
        vk::Extent2D {
            width: util::clamp(
                extent.width,
                capabilites.min_image_extent.width,
                capabilites.max_image_extent.width,
            ),
            height: util::clamp(
                extent.height,
                capabilites.min_image_extent.height,
                capabilites.max_image_extent.height,
            ),
        }
    }
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        surface: &Surface,
        extent: &util::Extent2D,
        old: Option<&Self>,
    ) -> Result<Self, SwapchainError> {
        let query = surface.query_swapchain_support(device.vk_phys_device())?;
        log::trace!("Creating swapchain");
        log::trace!("Available: {:#?}", query);
        let format = choose_swapchain_surface_format(&query.formats);
        let present_mode = choose_swapchain_surface_present_mode(&query.present_modes);
        let extent = choose_swapchain_extent(&query.capabilites, extent);

        let mut image_count = query.capabilites.min_image_count + 1;
        // Zero means no max
        if query.capabilites.max_image_count > 0 && image_count > query.capabilites.max_image_count
        {
            image_count = query.capabilites.max_image_count;
        }

        let mut builder = vk::SwapchainCreateInfoKHR::builder()
            .surface(*surface.vk_handle())
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

        let indices = [
            device.graphics_queue_family().index,
            device.present_queue_family().index,
        ];
        if indices[0] != indices[1] {
            // TODO: CONCURRENT is suboptimal but easier
            builder = builder
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&indices);
        } else {
            builder = builder
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&[]); // optional
        }

        let old_handle = old.map(|v| v.handle).unwrap_or_else(vk::SwapchainKHR::null);

        let info = builder
            .pre_transform(query.capabilites.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_handle)
            .build();

        log::trace!("Creating swapchain with info: {:#?}", info);
        let vk_device = device.vk_device();
        let loader = ash::extensions::khr::Swapchain::new(instance.vk_instance(), &*vk_device);

        let handle = unsafe {
            loader
                .create_swapchain(&info, None)
                .map_err(|e| SwapchainError::VulkanObjectCreation(e, "Swapchain"))?
        };

        let images = unsafe {
            loader
                .get_swapchain_images(handle)
                .map_err(|e| SwapchainError::VulkanObjectCreation(e, "Image"))?
        };

        let vk::SwapchainCreateInfoKHR {
            image_format,
            image_extent,
            ..
        } = info;

        let light_info = SwapchainInfo {
            format: image_format,
            extent: image_extent.into(),
        };

        let util_format = util::Format::from(image_format);
        let mip_levels = 1;

        let image_views = images
            .iter()
            .map(|img| {
                ImageView::new(
                    device,
                    img,
                    util_format,
                    vk::ImageAspectFlags::COLOR,
                    mip_levels,
                )
            })
            .collect::<Result<Vec<_>, ImageViewError>>()?;

        Ok(Self {
            loader,
            handle,
            images,
            image_views,
            info: light_info,
            vk_device: device.vk_device(),
        })
    }

    pub fn info(&self) -> &SwapchainInfo {
        &self.info
    }

    // TODO: Does this really belong here?
    pub fn create_framebuffers_for(
        &self,
        render_pass: &RenderPass,
        depth_buffer: &DepthBuffer,
        color_buffer: &ColorBuffer,
    ) -> Result<Vec<Framebuffer>, SwapchainError> {
        self.image_views
            .iter()
            .map(|iv| {
                let views = [color_buffer.image_view(), depth_buffer.image_view(), iv];
                Framebuffer::new(&self.vk_device, &views, render_pass, &self.info.extent)
            })
            .collect::<Result<Vec<_>, FramebufferError>>()
            .map_err(SwapchainError::Framebuffer)
    }

    pub fn acquire_next_image(&self, sem: Option<&Semaphore>) -> Result<u32, SwapchainError> {
        let s = sem
            .map(|x| *x.vk_semaphore())
            .unwrap_or_else(vk::Semaphore::null);
        let f = vk::Fence::null();
        let result = unsafe { self.loader.acquire_next_image(self.handle, u64::MAX, s, f) };

        let (idx, sub_optimal) = result.map_err(|e| {
            if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                SwapchainError::OutOfDate
            } else {
                SwapchainError::AcquireNextImage(e)
            }
        })?;

        if sub_optimal {
            log::warn!("Suboptimal swapchain!");
        }

        Ok(idx)
    }

    pub fn vk_swapchain(&self) -> &vk::SwapchainKHR {
        &self.handle
    }

    pub fn enqueue_present(
        &self,
        queue: &Queue,
        info: vk::PresentInfoKHR,
    ) -> Result<SwapchainStatus, SwapchainError> {
        let present_result = unsafe { self.loader.queue_present(*queue.vk_queue(), &info) };

        let sub_optimal = present_result.map_err(|e| {
            if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                SwapchainError::OutOfDate
            } else {
                SwapchainError::EnqueuePresent(e)
            }
        })?;

        if sub_optimal {
            Ok(SwapchainStatus::SubOptimal)
        } else {
            Ok(SwapchainStatus::Optimal)
        }
    }

    pub fn num_images(&self) -> usize {
        assert_eq!(self.images.len(), self.image_views.len());
        self.images.len()
    }
}
