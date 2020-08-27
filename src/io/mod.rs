#[cfg(target_os = "unix")]
use winit::platform::unix::EventLoopExtUnix;

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopExtWindows;

pub mod input;
pub mod windowing;

pub type VkSurface = vulkano::swapchain::Surface<winit::window::Window>;
pub type EventQueue = crossbeam::queue::SegQueue<windowing::Event>;
pub type WindowingError = vulkano_win::CreationError;
