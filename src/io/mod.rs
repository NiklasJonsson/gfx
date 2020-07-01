pub mod windowing;

use vulkano::instance::Instance;
use winit::event_loop::ControlFlow as WinCFlow;
use winit::window::WindowBuilder;

use winit::platform::unix::EventLoopExtUnix;

use vulkano_win::VkSurfaceBuild;

use std::sync::Arc;

enum EventLoopControl {
    Done(windowing::Event),
    Continue,
}

pub type VkSurface = vulkano::swapchain::Surface<winit::window::Window>;

pub type EventQueue = Arc<crossbeam::queue::SegQueue<windowing::Event>>;

pub type WindowingError = vulkano_win::CreationError;

pub fn init_windowing_and_input_thread(
    vk_instance: Arc<Instance>,
) -> Result<(Arc<VkSurface>, EventQueue), WindowingError> {
    let (sender, receiver) = std::sync::mpsc::channel();
    let event_queue = Arc::new(crossbeam::queue::SegQueue::new());

    let event_queue_2 = Arc::clone(&event_queue);

    // Spawn off thread that handles windowing/events
    std::thread::spawn(move || {
        let event_queue = event_queue_2;
        let event_loop = winit::event_loop::EventLoop::new_any_thread();

        let vk_surface_result =
            WindowBuilder::new().build_vk_surface(&event_loop, Arc::clone(&vk_instance));

        // We pass ownership with the channel so save this
        let mut is_err = vk_surface_result.is_err();
        is_err |= sender.send(vk_surface_result).is_err();

        if is_err {
            return;
        }

        let mut event_manager = windowing::EventManager::new();

        event_loop.run(move |winit_event, _, control_flow| {
            // Since this is a separate thread, it is fine to wait
            *control_flow = WinCFlow::Wait;

            match event_manager.collect_event(winit_event) {
                EventLoopControl::Done(event) => {
                    log::debug!("Sending event on queue: {:?}", event);
                    event_queue.push(event)
                }
                EventLoopControl::Continue => (),
            }
        });
    });

    let vk_surface_result = receiver.recv().expect("Failed to receive");
    vk_surface_result.map(|r| (r, event_queue))
}
