pub mod buffer;
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
pub mod util;
pub mod validation_layers;

pub use device::AllocatorHandle;

use command::CommandError;
use queue::QueueError;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("Failed to create buffer {0}")]
    BufferCreation(ash::vk::Result),
    #[error("Failed to create image {0}")]
    ImageCreation(ash::vk::Result),
    #[error("command error during copy {0}")]
    CopyCommand(#[from] CommandError),
    #[error("queue submission failed {0}")]
    CopySubmit(#[from] QueueError),
    #[error("memory mapping failed {0}")]
    MemoryMapping(ash::vk::Result),
    #[error("realloc failed {0}")]
    Realloc(ash::vk::Result),
    #[error("memory binding failed {0}")]
    MemoryBinding(ash::vk::Result),
}

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
