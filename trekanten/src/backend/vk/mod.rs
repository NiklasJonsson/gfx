pub mod color_buffer;
pub mod depth_buffer;

pub mod command;
pub mod device;
pub mod framebuffer;
pub mod image;
pub mod instance;
pub mod queue;
pub mod render_pass;
pub mod surface;
pub mod swapchain;
pub mod sync;
pub mod validation_layers;

pub fn n_to_sample_count(n: u8) -> ash::vk::SampleCountFlags {
    match n {
        1 => ash::vk::SampleCountFlags::TYPE_1,
        2 => ash::vk::SampleCountFlags::TYPE_2,
        4 => ash::vk::SampleCountFlags::TYPE_4,
        8 => ash::vk::SampleCountFlags::TYPE_8,
        16 => ash::vk::SampleCountFlags::TYPE_16,
        32 => ash::vk::SampleCountFlags::TYPE_32,
        64 => ash::vk::SampleCountFlags::TYPE_64,
        x => unreachable!("{} is not a valid mssa count", x),
    }
}
