use thiserror::Error;

use crate::command::CommandError;
use crate::queue::QueueError;

mod buffer;
mod buffer_storage;
mod image;

pub use self::image::*;
pub use buffer::*;
pub use buffer_storage::*;

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
}
