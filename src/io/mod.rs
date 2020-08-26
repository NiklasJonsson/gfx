#[cfg(target_os = "unix")]
use winit::platform::unix::EventLoopExtUnix;

#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopExtWindows;

use winit::event::DeviceEvent;
use winit::event_loop::ControlFlow as WinCFlow;
use winit::window::WindowBuilder;

use vulkano::instance::Instance;
use vulkano_win::VkSurfaceBuild;

use std::sync::Arc;

pub mod input;
pub mod windowing;

pub type VkSurface = vulkano::swapchain::Surface<winit::window::Window>;
pub type EventQueue = crossbeam::queue::SegQueue<windowing::Event>;
pub type WindowingError = vulkano_win::CreationError;
