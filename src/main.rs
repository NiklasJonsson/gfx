extern crate vulkano;
extern crate vulkano_win;
extern crate winit;

use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::image::ImageUsage;
use vulkano::instance;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::swapchain::{
    ColorSpace, CompositeAlpha, PresentMode, Surface, SurfaceTransform, Swapchain,
};
use vulkano::sync::SharingMode;

use vulkano_win::VkSurfaceBuild;

use winit::dpi::LogicalSize;
use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent};

use std::sync::Arc;

struct App {
    events_loop: EventsLoop,
    vk_instance: Arc<Instance>,
    vk_surface: Arc<Surface<Window>>,
}

impl App {
    fn main_loop(&mut self) {
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
        self.main_loop();
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

    fn setup_surface(vk_instance: &Arc<Instance>) -> (EventsLoop, Arc<Surface<Window>>) {
        let events_loop = EventsLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&events_loop, Arc::clone(vk_instance))
            .expect("Unable to create window/surface");

        return (events_loop, surface);
    }

    fn pick_physical_device<'a>(
        vk_instance: &'a Arc<Instance>,
        required_extensions: &DeviceExtensions,
    ) -> PhysicalDevice<'a> {
        // TODO: For proper device selection, this should also be done:
        //  - the available queues should be checked as well
        //  - swap chain support/adequacy
        let ph_dev = PhysicalDevice::enumerate(vk_instance)
            .find(|&ph_dev| {
                let supported_extensions = DeviceExtensions::supported_by_device(ph_dev);

                let required_supported =
                    required_extensions.intersection(&supported_extensions) == *required_extensions;

                return required_supported;
            })
            .expect("No device available");

        println!(
            "Chose physical device, {}, which is a {:?}",
            ph_dev.name(),
            ph_dev.ty()
        );

        return ph_dev;
    }

    fn print_queue_families(physical_device: PhysicalDevice) {
        println!(
            "Found {} queue families",
            physical_device.queue_families().len()
        );

        for qf in physical_device.queue_families() {
            println!(
                "Queue {}[{}]. graphics:{}, compute:{}",
                qf.id(),
                qf.queues_count(),
                qf.supports_graphics(),
                qf.supports_compute()
            );
        }
    }

    fn create_logical_device(
        physical_device: PhysicalDevice,
        surface: &Arc<Surface<Window>>,
        device_extensions: &DeviceExtensions,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        App::print_queue_families(physical_device);

        let graphics_queue_family = physical_device
            .queue_families()
            .find(|&q| q.supports_graphics())
            .expect("Could not find suitable queue");

        let presentation_queue_family = physical_device
            .queue_families()
            .find(|&q| surface.is_supported(q).unwrap_or(false))
            .expect("Could not find suitable queue");

        let q_families = [
            (graphics_queue_family, 1.0),
            (presentation_queue_family, 1.0),
        ];

        let (device, mut queues) = Device::new(
            physical_device,
            /* features */ &Features::none(),
            /* extensions */ device_extensions,
            q_families.iter().cloned(),
        )
        .expect("Failed to create device");

        let graphics_queue = queues.next().expect("Device queues not created");
        let presentation_queue = queues.next().expect("Presentation queue not created");

        return (device, graphics_queue, presentation_queue);
    }

    fn create_swap_chain(
        device: &Arc<Device>,
        surface: &Arc<Surface<Window>>,
        sharing_mode: SharingMode,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = device.physical_device();
        let capabilities = surface
            .capabilities(physical_device)
            .expect("Can't fetch surface capabilites");

        for f in &capabilities.supported_formats {
            println!("{:?}", f);
        }

        let format = capabilities
            .supported_formats
            .iter()
            .find(|&format| format == &(Format::B8G8R8A8Unorm, ColorSpace::SrgbNonLinear))
            .expect("Unable to find proper format in surface capabilities");

        let present_mode = capabilities
            .present_modes
            .iter()
            .find(|&mode| mode == PresentMode::Mailbox)
            .unwrap_or(PresentMode::Fifo);

        // Setup swap chain dimensions to that of the window
        let dimensions = surface
            .window()
            .get_inner_size()
            .map(|dims| {
                let dim_scaled: (u32, u32) =
                    dims.to_physical(surface.window().get_hidpi_factor()).into();
                return [dim_scaled.0, dim_scaled.1];
            })
            .expect("Was not able to setup swapchain dimensions, is window open?");

        // Add 1 to try to get triple buffering
        let triple_buffering_img_count = capabilities.min_image_count;
        let img_count = match capabilities.max_image_count {
            Some(max) => std::cmp::min(max, triple_buffering_img_count),
            None => triple_buffering_img_count,
        };

        // The imageArrayLayers, likely. Is not 1 when rendering stereoscoping 3D
        let layers = 1;
        let usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };

        let alpha = CompositeAlpha::Opaque;

        return Swapchain::new(
            Arc::clone(device),
            Arc::clone(surface),
            img_count,
            format.0,
            dimensions,
            layers,
            usage,
            sharing_mode,
            SurfaceTransform::Identity,
            alpha,
            present_mode,
            /* clipped */ true,
            /* Old swapchain */ None,
        )
        .expect("Failed to create swap chain");
    }

    fn new() -> App {
        let vk_instance = App::setup_vk_instance();
        let (events_loop, vk_surface) = App::setup_surface(&vk_instance);
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        let physical_device = App::pick_physical_device(&vk_instance, &device_extensions);
        let (vk_device, graphics_queue, presentation_queue) =
            App::create_logical_device(physical_device, &vk_surface, &device_extensions);
        let sharing_mode: SharingMode;
        let g_fid = graphics_queue.family().id();
        let p_fid = presentation_queue.family().id();
        if g_fid == p_fid {
            sharing_mode = SharingMode::Exclusive(g_fid);
        } else {
            sharing_mode = SharingMode::Concurrent(vec![g_fid, p_fid]);
        }

        let (mut swapchain, images) = App::create_swap_chain(&vk_device, &vk_surface, sharing_mode);

        return App {
            events_loop,
            vk_instance,
            vk_surface,
        };
    }
}

fn main() {
    let mut app = App::new();
    app.run();
}
