extern crate vulkano;
extern crate vulkano_win;
extern crate winit; // For required instance extensions

use vulkano::instance;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::device::{Device, Queue, DeviceExtensions, Features};

use winit::dpi::LogicalSize;
use winit::{Event, EventsLoop, Window, WindowEvent};

use std::sync::Arc;

struct App {
    events_loop: EventsLoop,
    vk_instance: Arc<Instance>,
    window: Window,
}

impl App {
    fn main_loop(&mut self) {
        println!("main_loop()");

        let mut quit = false;

        while !quit {
            self.events_loop.poll_events(|event| match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => quit = true,
                _ => (),
            });
        }
    }

    fn run(&mut self) {
        println!("run()");
        self.main_loop();
    }

    fn setup_window() -> (EventsLoop, Window) {
        let events_loop = winit::EventsLoop::new();
        let window = winit::Window::new(&events_loop).unwrap();
        return (events_loop, window);
    }

    fn setup_vk_instance() -> Arc<Instance> {
        let available_extensions =
            InstanceExtensions::supported_by_core().expect("can't get supported extensions");
        let required_extensions = vulkano_win::required_extensions();

        if available_extensions.intersection(&required_extensions) != required_extensions {
            println!("Can't create a window, not all extensions supported.");
        }

        if instance::layers_list().unwrap().len() == 0 {
            println!("No layers!");
        }

        for layer in instance::layers_list().unwrap() {
            println!("Layer: {}", layer.name());
        }

        let vk_instance = Instance::new(None, &required_extensions, None)
            .expect("Could not create vulkan instance");

        return vk_instance;
    }

    fn pick_physical_device(vk_instance: &Arc<Instance>) -> PhysicalDevice {
        let ph_dev = PhysicalDevice::enumerate(vk_instance)
            .find(|ph_dev| true )
            .expect("No device available");

        println!("Chose physical device, {}, which is a {:?}", ph_dev.name(), ph_dev.ty());

        return ph_dev;
    }

    fn create_logical_device(physical_device: PhysicalDevice) -> (Arc<Device>,
                                                                  Arc<Queue>) {
        println!("{:?}", physical_device.queue_families());
        let queue_family = physical_device
            .queue_families()
            .find(|q| { q.supports_graphics() })
            .expect("Could not find suitable queue");

        let (device, mut queues) = Device::new(
            physical_device,
            /* features */ &Features::none(),
            /* extensions */ &DeviceExtensions::none(),
            [(queue_family, 1.0)].iter().cloned())
            .expect("Failed to create device");

        let queue = queues.next().expect("Device queues not created");

        return (device, queue);
    }

    fn new() -> App {
        println!("new()");
        let (events_loop, window) = App::setup_window();
        let vk_instance = App::setup_vk_instance();
        let physical_device = App::pick_physical_device(&vk_instance);
        let (vk_device, graphics_queue) = App::create_logical_device(physical_device);

        return App {
            events_loop,
            vk_instance,
            window,
        };
    }
}

fn main() {
    let mut app = App::new();
    app.run();
}
