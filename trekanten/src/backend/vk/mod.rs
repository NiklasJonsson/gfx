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
    BufferCreation(vk_mem::Error),
    #[error("Failed to create image {0}")]
    ImageCreation(vk_mem::Error),
    #[error("command error during copy {0}")]
    CopyCommand(#[from] CommandError),
    #[error("queue submission failed {0}")]
    CopySubmit(#[from] QueueError),
    #[error("memory mapping failed {0}")]
    MemoryMapping(vk_mem::Error),
    #[error("realloc failed {0}")]
    Realloc(vk_mem::Error),
    #[error("memory binding failed {0}")]
    MemoryBinding(vk_mem::Error),
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
