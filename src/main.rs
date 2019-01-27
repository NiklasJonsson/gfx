#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate num_derive;

extern crate log;
extern crate env_logger;

extern crate image;
extern crate nalgebra_glm as glm;
extern crate tobj;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use image::ImageDecoder;

use log::{info};

use vulkano::buffer::{
    BufferAccess, BufferUsage, CpuAccessibleBuffer, ImmutableBuffer, TypedBufferAccess,
};
use vulkano::command_buffer::{
    AutoCommandBuffer, AutoCommandBufferBuilder, CommandBufferExecFuture, DynamicState,
};
use vulkano::descriptor::{
    descriptor_set::PersistentDescriptorSet, DescriptorSet, PipelineLayoutAbstract,
};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{
    attachment::AttachmentImage, immutable::ImmutableImage, swapchain::SwapchainImage, Dimensions,
    ImageAccess, ImageUsage, ImageViewAccess,
};
use vulkano::instance;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, CompositeAlpha, PresentMode, Surface, SurfaceTransform, Swapchain,
};

use vulkano::sampler::Sampler;
use vulkano::sync::{FenceSignalFuture, FlushError, GpuFuture, NowFuture, SharingMode};

use vulkano_win::VkSurfaceBuild;

use winit::{Event, EventsLoop, VirtualKeyCode, Window, WindowBuilder, WindowEvent};

use std::collections::HashSet;

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::prelude::*;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod camera;
mod input;
use crate::camera::{Camera, CameraController};
use crate::input::InputManager;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn new(position: [f32; 3], color: [f32; 3], tex_coords: [f32; 2]) -> Vertex {
        Vertex {
            position,
            color,
            tex_coords,
        }
    }
}

impl_vertex!(Vertex, position, color, tex_coords);

// Extra trait specialization for GpuFuture, intended for storing NowFuture or FenceSignalFuture
trait WaitableFuture : GpuFuture {
    fn wait_for(&self, timeout: Option<Duration>) -> Result<(), FlushError>;
}

impl WaitableFuture for NowFuture {
    fn wait_for(&self, _timeout: Option<Duration>) -> Result<(), FlushError> {
        Ok(())
    }
}

impl<F: GpuFuture> WaitableFuture for FenceSignalFuture<F> {
    fn wait_for(&self, timeout: Option<Duration>) -> Result<(), FlushError> {
        self.wait(timeout)
    }
}

struct App {
    events_loop: EventsLoop,
    vk_instance: Arc<Instance>,
    vk_surface: Arc<Surface<Window>>,
    vk_device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    presentation_queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Window>>,
    swapchain_images: Vec<Arc<SwapchainImage<Window>>>,
    vertex_buffer: Arc<BufferAccess + Send + Sync>,
    index_buffer: Arc<TypedBufferAccess<Content = [u32]> + Send + Sync>,
    image_buffer: Arc<ImageViewAccess + Send + Sync>,
    sampler: Arc<Sampler>,
    mvp_ubo_buffers: Vec<Arc<CpuAccessibleBuffer<vs::ty::MVPUniformBufferObject>>>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    g_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    command_buffers: Vec<Arc<AutoCommandBuffer>>,
    frame_completions: Vec<Box<WaitableFuture>>,
    camera: Rc<RefCell<Camera>>,
}

impl App {
    fn update_mvp(&mut self, idx: usize) {
        let mut content = self.mvp_ubo_buffers[idx].write().unwrap();
        content.view = glm::look_at(
            self.camera.borrow().get_pos(),
            &glm::vec3(0.0, 0.0, 0.0),
            self.camera.borrow().get_up(),
        )
        .into();
    }

    fn draw_frame(&mut self) {
        let (img_idx, swapchain_img_acquired) =
            match swapchain::acquire_next_image(Arc::clone(&self.swapchain), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swap_chain();
                    return;
                }
                Err(e) => panic!("Can't acquire next image from swapchain: \t{}", e),
            };

        // tmp_future is only used as an intermediary while we submit this frame
        let tmp_future = Box::new(vulkano::sync::now(Arc::clone(&self.vk_device)));
        let prev_frame_completed =
            std::mem::replace(&mut self.frame_completions[img_idx], tmp_future);

        // Wait for previous frame before we update MVP buffer
        prev_frame_completed.wait_for(None).unwrap();

        // This writes to the uniform buffer for the comming frame
        self.update_mvp(img_idx);

        let drawn_and_presented = swapchain_img_acquired
            .then_execute(
                Arc::clone(&self.graphics_queue),
                Arc::clone(&self.command_buffers[img_idx]),
            )
            .expect("Unable to execute command buffer")
            // Use presentation queue + semaphore when vulkano supports it
            .then_swapchain_present(
                Arc::clone(&self.graphics_queue),
                Arc::clone(&self.swapchain),
                img_idx,
            )
            .then_signal_fence_and_flush();
        let drawn_and_presented = match drawn_and_presented {
            Ok(r) => r,
            Err(FlushError::OutOfDate) => {
                self.recreate_swap_chain();
                return;
            }
            Err(e) => panic!(
                "Can't write to the swapchain image (idx: {}):\n\t{}",
                img_idx, e
            ),
        };

        self.frame_completions[img_idx] = Box::new(drawn_and_presented);
    }

    fn main_loop(&mut self) {
        // TODO: EventManager handling all events, passing appropriate ones down to InputManager
        let mut input_manager = InputManager::new();
        let camera_controller = CameraController::new(&mut input_manager, &self.camera);
        loop {
            let mut quit = false;
            self.events_loop.poll_events(|event| {
                // event is either WindowEvent or DeviceEvent, ignore the latter
                if let Event::WindowEvent {
                    event: window_event,
                    ..
                } = event
                {
                    match window_event {
                        WindowEvent::CloseRequested => quit = true,
                        WindowEvent::KeyboardInput {
                            device_id: _device_id,
                            input,
                        } => {
                            if let Some(key) = input.virtual_keycode {
                                input_manager.handle_button_input(key);
                            }
                        }
                        _ => (),
                    }
                }
            });

            if quit {
                break;
            }

            input_manager.dispatch();

            self.draw_frame();
        }
    }

    fn run(&mut self) {
        self.main_loop();
    }

    fn choose_validation_layers() -> Vec<String> {
        info!("Choosing vulkan validation layers.");

        let requested = vec!["VK_LAYER_LUNARG_standard_validation", "VK_LAYER_LUNARG_monitor"];

        info!("Requested layers:");
        for req in &requested {
            info!("\t{}", req);
        }

        info!("Available layers:");
        for avail in instance::layers_list().expect("Can't query validation layers") {
            info!("\t{}", avail.name());
        }

        let chosen = instance::layers_list()
            .unwrap()
            .map(|l| String::from(l.name()))
            .filter(|name| requested.iter().find(|&req| req == name).is_some())
            .collect::<Vec<String>>();

        info!("Chosen layers:");
        for choice in &chosen {
            info!("\t{}", choice);
        }

        return chosen;
    }

    fn setup_vk_instance() -> Arc<Instance> {
        let available_extensions =
            InstanceExtensions::supported_by_core().expect("can't get supported extensions");
        let required_extensions = vulkano_win::required_extensions();

        if available_extensions.intersection(&required_extensions) != required_extensions {
            println!("Can't create a window, not all extensions supported.");
        }

        let layers = Self::choose_validation_layers();


        let vk_instance = Instance::new(None, &required_extensions, layers.iter().map(|s| s.as_str()))
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

        return ph_dev;
    }

    fn create_logical_device(
        physical_device: PhysicalDevice,
        surface: &Arc<Surface<Window>>,
        device_extensions: &DeviceExtensions,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let graphics_queue_family = physical_device
            .queue_families()
            .find(|&q| q.supports_graphics())
            .expect("Could not find suitable queue");

        let presentation_queue_family = physical_device
            .queue_families()
            .find(|&q| surface.is_supported(q).unwrap_or(false))
            .expect("Could not find suitable queue");

        // TODO: This should not be necessary, it's a bug in vulkano
        let q_families = [graphics_queue_family, presentation_queue_family];
        use std::iter::FromIterator;
        let unique_queue_families: HashSet<u32> =
            HashSet::from_iter(q_families.iter().map(|qf| qf.id()));

        let queue_priority = 1.0;
        let q_families = unique_queue_families.iter().map(|&i| {
            (
                physical_device.queue_family_by_id(i).unwrap(),
                queue_priority,
            )
        });

        let (device, mut queues) = Device::new(
            physical_device,
            /* features */ &Features::none(),
            /* extensions */ device_extensions,
            q_families,
        )
        .expect("Failed to create device");

        let graphics_queue = queues.next().expect("Device queues not created");
        let presentation_queue = queues.next().unwrap_or(Arc::clone(&graphics_queue));

        return (device, graphics_queue, presentation_queue);
    }

    fn create_swap_chain(
        device: &Arc<Device>,
        surface: &Arc<Surface<Window>>,
        queue_family_ids: &[u32; 2],
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
            .find(|&format| format == &(Format::B8G8R8A8Srgb, ColorSpace::SrgbNonLinear))
            .expect("Unable to find proper format in surface capabilities");

        let present_mode = capabilities
            .present_modes
            .iter()
            .find(|&mode| mode == PresentMode::Mailbox)
            .unwrap_or(PresentMode::Fifo);

        // Setup swap chain dimensions to that of the window
        let dimensions = get_physical_window_dims(surface.window());

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

        let sharing_mode: SharingMode =
            match queue_family_ids.iter().all(|&x| x == queue_family_ids[0]) {
                true => SharingMode::Exclusive(queue_family_ids[0]),
                false => SharingMode::Concurrent(queue_family_ids.to_vec()),
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
            None,
        )
        .expect("Failed to create swap chain");
    }

    /*

    fn create_vertex_data() -> (Vec<Vertex>, Vec<u32>) {
        let vertices = vec![
            Vertex::new(
                [-0.5, -0.5, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0],
            ),
            Vertex::new(
                [0.5, -0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0],
            ),
            Vertex::new(
                [0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0],
            ),
            Vertex::new(
                [-0.5, 0.5, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0],
            ),
            Vertex::new(
                [-0.5, -0.5, -0.5],
                [1.0, 0.0, 0.0],
                [1.0, 0.0],
            ),
            Vertex::new(
                [0.5, -0.5, -0.5],
                [0.0, 1.0, 0.0],
                [0.0, 0.0],
            ),
            Vertex::new(
                [0.5, 0.5, -0.5],
                [0.0, 0.0, 1.0],
                [0.0, 1.0],
            ),
            Vertex::new(
                [-0.5, 0.5, -0.5],
                [1.0, 1.0, 1.0],
                [1.0, 1.0],
            )
        ];

        let indices: Vec<u32> = vec![0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4];

        return (vertices, indices);
    }

    */

    fn debug_obj(path: &str) {
        let data = tobj::load_obj(&Path::new(path));
        assert!(data.is_ok());
        let (models, materials) = data.unwrap();

        println!("# of models: {}", models.len());
        println!("# of materials: {}", materials.len());
        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;
            println!("model[{}].name = \'{}\'", i, m.name);
            println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

            println!("Size of model[{}].indices: {}", i, mesh.indices.len());
            /*
            for f in 0..mesh.indices.len() / 3 {
                println!("    idx[{}] = {}, {}, {}.", f, mesh.indices[3 * f],
                         mesh.indices[3 * f + 1], mesh.indices[3 * f + 2]);
            }
            */

            // Normals and texture coordinates are also loaded, but not printed in this example
            println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);
            assert!(mesh.positions.len() % 3 == 0);
            /*
            for v in 0..mesh.positions.len() / 3 {
                println!("    v[{}] = ({}, {}, {})", v, mesh.positions[3 * v],
                mesh.positions[3 * v + 1], mesh.positions[3 * v + 2]);
            }
            */

            for (i, m) in materials.iter().enumerate() {
                println!("material[{}].name = \'{}\'", i, m.name);
                println!(
                    "    material.Ka = ({}, {}, {})",
                    m.ambient[0], m.ambient[1], m.ambient[2]
                );
                println!(
                    "    material.Kd = ({}, {}, {})",
                    m.diffuse[0], m.diffuse[1], m.diffuse[2]
                );
                println!(
                    "    material.Ks = ({}, {}, {})",
                    m.specular[0], m.specular[1], m.specular[2]
                );
                println!("    material.Ns = {}", m.shininess);
                println!("    material.d = {}", m.dissolve);
                println!("    material.map_Ka = {}", m.ambient_texture);
                println!("    material.map_Kd = {}", m.diffuse_texture);
                println!("    material.map_Ks = {}", m.specular_texture);
                println!("    material.map_Ns = {}", m.normal_texture);
                println!("    material.map_d = {}", m.dissolve_texture);
                for (k, v) in &m.unknown_param {
                    println!("    material.{} = {}", k, v);
                }
            }
        }
    }

    fn load_obj(path: &str) -> (Vec<Vertex>, Vec<u32>) {
        let data = tobj::load_obj(&Path::new(path)).unwrap();;
        let tex_coords = data.0[0].mesh.texcoords.chunks_exact(2);
        let vertices = data.0[0]
            .mesh
            .positions
            .chunks_exact(3)
            .zip(tex_coords)
            .map(|(pos, tx_cs)| {
                Vertex::new(
                    [pos[0], pos[1], pos[2]],
                    [1.0, 1.0, 1.0],
                    [tx_cs[0], 1.0 - tx_cs[1]],
                )
            })
            .collect::<Vec<_>>();

        let indices = data.0[0].mesh.indices.to_owned();

        return (vertices, indices);
    }

    fn load_image() -> Vec<u8> {
        // TODO: Try to use JPEGDecoder + BufReader for better error handling
        let image = image::load_from_memory_with_format(
            include_bytes!("../textures/chalet.jpg"),
            image::ImageFormat::JPEG,
        )
        .unwrap()
        .to_rgba();

        let data = image.into_raw().clone();
        return data;
    }

    // TODO: Try to refactor here
    fn create_and_submit_vertex_buffer(
        queue: &Arc<Queue>,
        vertex_data: Vec<Vertex>,
    ) -> (
        Arc<BufferAccess + Send + Sync>,
        CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
    ) {
        let (buf, fut) = ImmutableBuffer::from_iter(
            vertex_data.iter().cloned(),
            BufferUsage::vertex_buffer(),
            Arc::clone(queue),
        )
        .expect("Could not create vertex buffer");

        return (buf, fut);
    }

    fn create_and_submit_index_buffer(
        queue: &Arc<Queue>,
        index_data: Vec<u32>,
    ) -> (
        Arc<TypedBufferAccess<Content = [u32]> + Send + Sync>,
        CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
    ) {
        let (buf, fut) = ImmutableBuffer::from_iter(
            index_data.iter().cloned(),
            BufferUsage::index_buffer(),
            Arc::clone(queue),
        )
        .expect("Could not create index buffer");

        return (buf, fut);
    }

    fn create_and_submit_texture_image(
        queue: &Arc<Queue>,
        image: &[u8],
    ) -> (
        Arc<ImageViewAccess + Send + Sync>,
        CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
    ) {
        let owned_copy = image.to_owned();
        // Destruct and combine again to more easilyinfer types
        let (buf, fut) = ImmutableImage::from_iter(
            owned_copy.into_iter(),
            Dimensions::Dim2d {
                width: 800,
                height: 600,
            },
            Format::R8G8B8A8Srgb,
            Arc::clone(queue),
        )
        .expect("Unable to create vertex buffer");

        return (buf, fut);
    }
    fn create_render_pass(
        device: &Arc<Device>,
        format: Format,
    ) -> Arc<RenderPassAbstract + Send + Sync> {
        Arc::new(
            single_pass_renderpass!(Arc::clone(device),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: format,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    // TODO: Choose this based on availability
                    format: Format::D32Sfloat,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            })
            .unwrap(),
        )
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        swapchain_dimensions: [u32; 2],
    ) -> Arc<GraphicsPipelineAbstract + Send + Sync> {
        let vs = vs::Shader::load(Arc::clone(device)).expect("Vertex shader compilation failed");
        let fs = fs::Shader::load(Arc::clone(device)).expect("Fragment shader compilation failed");

        let dims = [
            swapchain_dimensions[0] as f32,
            swapchain_dimensions[1] as f32,
        ];

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: dims,
            depth_range: 0.0..1.0,
        };

        let pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                // How to interpret the vertex input
                .triangle_list()
                .vertex_shader(vs.main_entry_point(), ())
                // Whether to support special indices in in the vertex buffer to split triangles
                .primitive_restart(false)
                .viewports([viewport].iter().cloned())
                .depth_stencil_simple_depth()
                .fragment_shader(fs.main_entry_point(), ())
                .blend_alpha_blending()
                .polygon_mode_fill()
                .line_width(1.0)
                .cull_mode_back()
                .front_face_counter_clockwise()
                .render_pass(Subpass::from(Arc::clone(render_pass), 0).unwrap())
                .build(Arc::clone(device))
                .expect("Could not create graphics pipeline"),
        );

        return pipeline;
    }

    fn create_framebuffers(
        device: &Arc<Device>,
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        sc_images: &[Arc<SwapchainImage<Window>>],
    ) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        sc_images
            .iter()
            .map(|image| {
                let depth_buffer = AttachmentImage::transient(
                    Arc::clone(device),
                    SwapchainImage::dimensions(&image),
                    render_pass.attachment_desc(1).unwrap().format, // Depth format
                )
                .unwrap();
                Arc::new(
                    Framebuffer::start(Arc::clone(&render_pass))
                        .add(Arc::clone(image))
                        .unwrap()
                        .add(depth_buffer)
                        .unwrap()
                        .build()
                        .unwrap(),
                ) as Arc<FramebufferAbstract + Send + Sync>
            })
            .collect::<Vec<_>>()
    }

    fn create_mvp_ubo(aspect_ratio: f32) -> vs::ty::MVPUniformBufferObject {
        let mut proj = glm::perspective(aspect_ratio, std::f32::consts::FRAC_PI_2, 0.1, 10.0);

        // glm::perspective is based on opengl left-handed coordinate system, vulkan has the y-axis
        // inverted (right-handed upside-down).
        proj[(1, 1)] *= -1.0;

        let mvp_ubo = vs::ty::MVPUniformBufferObject {
            model: glm::Mat4::identity().into(),
            view: glm::look_at(
                &glm::vec3(0.0, 0.0, -1.0),
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 1.0, 0.0),
            )
            .into(),
            proj: proj.into(),
        };

        return mvp_ubo;
    }

    fn create_mvp_ubo_buffers(
        device: &Arc<Device>,
        mvp_ubos: &[vs::ty::MVPUniformBufferObject],
    ) -> Vec<Arc<CpuAccessibleBuffer<vs::ty::MVPUniformBufferObject>>> {
        mvp_ubos
            .iter()
            .map(|&mvp_ubo| {
                CpuAccessibleBuffer::from_data(
                    Arc::clone(device),
                    BufferUsage::uniform_buffer(),
                    mvp_ubo,
                )
                .expect("Unable to create buffer for MVP UBO")
            })
            .collect::<Vec<_>>()
    }

    fn create_dsets(
        pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
        buffers: &[Arc<CpuAccessibleBuffer<vs::ty::MVPUniformBufferObject>>],
        image: &Arc<ImageViewAccess + Send + Sync>,
        sampler: &Arc<Sampler>,
    ) -> Vec<Arc<DescriptorSet + Send + Sync>> {
        buffers
            .iter()
            .map(|buffer| {
                Arc::new(
                    PersistentDescriptorSet::start(Arc::clone(pipeline), 0)
                        .add_buffer(Arc::clone(buffer))
                        .unwrap()
                        .add_sampled_image(Arc::clone(image), Arc::clone(sampler))
                        .unwrap()
                        .build()
                        .expect("Failed to create persistent descriptor set for mvp ubo"),
                ) as Arc<DescriptorSet + Send + Sync>
            })
            .collect::<Vec<_>>()
    }

    fn create_command_buffers(
        device: &Arc<Device>,
        queue_family: QueueFamily,
        vertex_buffer: &Arc<BufferAccess + Send + Sync>,
        index_buffer: &Arc<TypedBufferAccess<Content = [u32]> + Send + Sync>,
        descriptor_sets: &[Arc<DescriptorSet + Send + Sync>],
        pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
        framebuffers: &[Arc<FramebufferAbstract + Send + Sync>],
    ) -> Vec<Arc<AutoCommandBuffer>> {
        framebuffers
            .iter()
            .enumerate()
            .map(|(i, fb)| {
                let clear_color = vec![[0.0, 0.0, 0.0, 1.0].into(), 1.0f32.into()];

                Arc::new(
                    AutoCommandBufferBuilder::primary_simultaneous_use(
                        Arc::clone(device),
                        queue_family,
                    )
                    .expect("Failed to create command buffer builder")
                    .begin_render_pass(Arc::clone(fb), false, clear_color)
                    .expect("Failed after begin render pass")
                    .draw_indexed(
                        Arc::clone(pipeline),
                        &DynamicState::none(),
                        vec![Arc::clone(vertex_buffer)],
                        Arc::clone(index_buffer),
                        Arc::clone(&descriptor_sets[i]),
                        (),
                    )
                    .expect("Failed after draw_indexed")
                    .end_render_pass()
                    .unwrap()
                    .build()
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>()
    }

    fn create_swapchain_dependent_objects(
        device: &Arc<Device>,
        swapchain: &Arc<Swapchain<Window>>,
        images: &[Arc<SwapchainImage<Window>>],
        mvp_bufs: &[Arc<CpuAccessibleBuffer<vs::ty::MVPUniformBufferObject>>],
        vertex_buffer: &Arc<BufferAccess + Send + Sync>,
        index_buffer: &Arc<TypedBufferAccess<Content = [u32]> + Send + Sync>,
        image: &Arc<ImageViewAccess + Send + Sync>,
        sampler: &Arc<Sampler>,
        graphics_queue_family: QueueFamily,
    ) -> (
        Arc<RenderPassAbstract + Send + Sync>,
        Arc<GraphicsPipelineAbstract + Send + Sync>,
        Vec<Arc<FramebufferAbstract + Send + Sync>>,
        Vec<Arc<AutoCommandBuffer>>,
    ) {
        let render_pass = Self::create_render_pass(device, swapchain.format());
        let g_pipeline =
            Self::create_graphics_pipeline(device, &render_pass, swapchain.dimensions());
        let framebuffers = Self::create_framebuffers(device, &render_pass, images);

        let dsets = Self::create_dsets(&g_pipeline, mvp_bufs, image, sampler);

        let cmd_bufs = Self::create_command_buffers(
            device,
            graphics_queue_family,
            vertex_buffer,
            index_buffer,
            dsets.as_slice(),
            &g_pipeline,
            &framebuffers,
        );

        return (render_pass, g_pipeline, framebuffers, cmd_bufs);
    }

    fn recreate_swap_chain(&mut self) {
        // Setup swap chain dimensions to that of the window
        let dimensions = get_physical_window_dims(self.vk_surface.window());

        let (swapchain, swapchain_images) = self
            .swapchain
            .recreate_with_dimension(dimensions)
            .expect("Unable to recreated swap chain");

        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;

        for idx in 0..self.frame_completions.len() {
            let now = Box::new(vulkano::sync::now(Arc::clone(&self.vk_device)));
            let prev = std::mem::replace(&mut self.frame_completions[idx], now);
            prev.wait_for(None).unwrap();
        }

        let (render_pass, g_pipeline, framebuffers, cmd_bufs) =
            App::create_swapchain_dependent_objects(
                &self.vk_device,
                &self.swapchain,
                self.swapchain_images.as_slice(),
                self.mvp_ubo_buffers.as_slice(),
                &self.vertex_buffer,
                &self.index_buffer,
                &self.image_buffer,
                &self.sampler,
                self.graphics_queue.family(),
            );

        self.render_pass = render_pass;
        self.g_pipeline = g_pipeline;
        self.framebuffers = framebuffers;
        self.command_buffers = cmd_bufs;
    }

    fn new() -> Self {
        let vk_instance = Self::setup_vk_instance();
        let (events_loop, vk_surface) = Self::setup_surface(&vk_instance);
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        let physical_device = Self::pick_physical_device(&vk_instance, &device_extensions);
        let (vk_device, graphics_queue, presentation_queue) =
            Self::create_logical_device(physical_device, &vk_surface, &device_extensions);

        /*
        let (vertex_data, index_data) = Self::create_vertex_data();
        */
        Self::debug_obj("models/chalet.obj");
        let (vertex_data, index_data) = Self::load_obj("models/chalet.obj");

        // TODO: Use transfer queue here
        let (vertex_buffer, vertex_data_copied) =
            Self::create_and_submit_vertex_buffer(&graphics_queue, vertex_data);
        let (index_buffer, index_data_copied) =
            Self::create_and_submit_index_buffer(&graphics_queue, index_data);
        let data_copied = vertex_data_copied
            .join(index_data_copied)
            .then_signal_fence_and_flush()
            .expect("Unable to signal fence and flush vertex + index data copy command");

        let image_data = Self::load_image();
        let (image_buffer, texture_data_copied) =
            Self::create_and_submit_texture_image(&graphics_queue, image_data.as_slice());

        let data_copied = data_copied
            .join(texture_data_copied)
            .then_signal_fence_and_flush()
            .expect("Unable to signal fence and flush texture data copy command");

        let sampler = Sampler::simple_repeat_linear(Arc::clone(&vk_device));

        let (swapchain, images) = Self::create_swap_chain(
            &vk_device,
            &vk_surface,
            &[
                graphics_queue.family().id(),
                presentation_queue.family().id(),
            ],
        );

        let n_frames = images.len();

        let dims = get_physical_window_dims(vk_surface.window());
        let aspect_ratio = dims[0] as f32 / dims[1] as f32;
        let mvp_ubos = (0..n_frames)
            .map(|_| Self::create_mvp_ubo(aspect_ratio))
            .collect::<Vec<_>>();

        let mvp_bufs = Self::create_mvp_ubo_buffers(&vk_device, mvp_ubos.as_slice());

        let (render_pass, g_pipeline, framebuffers, cmd_bufs) =
            App::create_swapchain_dependent_objects(
                &vk_device,
                &swapchain,
                images.as_slice(),
                mvp_bufs.as_slice(),
                &vertex_buffer,
                &index_buffer,
                &image_buffer,
                &sampler,
                graphics_queue.family(),
            );

        let frame_completions = init_frame_completions(&vk_device, n_frames);

        data_copied
            .wait(None)
            .expect("Transfer of application constant data failed");

        let camera = Rc::new(RefCell::new(Camera::new([2.0, 2.0, 2.0], [0.0, 0.0, 0.0])));

        return App {
            events_loop,
            vk_instance,
            vk_surface,
            vk_device,
            graphics_queue,
            presentation_queue,
            swapchain,
            swapchain_images: images,
            vertex_buffer,
            index_buffer,
            image_buffer,
            sampler,
            mvp_ubo_buffers: mvp_bufs,
            render_pass,
            framebuffers,
            g_pipeline,
            command_buffers: cmd_bufs,
            frame_completions,
            camera,
        };
    }
}

fn main() {
    env_logger::init();
    let mut app = App::new();
    app.run();
}

fn init_frame_completions(device: &Arc<Device>, n_frames: usize) -> Vec<Box<WaitableFuture>> {
    (0..n_frames)
        .map(|_| Box::new(vulkano::sync::now(Arc::clone(device))) as Box<WaitableFuture>)
        .collect::<Vec<_>>()
}

fn get_physical_window_dims(window: &Window) -> [u32; 2] {
    window
        .get_inner_size()
        .map(|dims| {
            let dims: (u32, u32) = dims.to_physical(window.get_hidpi_factor()).into();
            return [dims.0, dims.1];
        })
        .expect("Was not able to read window dimensions, is it open?")
}

mod vs {
    vulkano_shaders::shader! {
    ty: "vertex",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform MVPUniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 tex_coords;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 frag_tex_coords;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
    frag_color = color;
    frag_tex_coords = tex_coords;
}
"
    }
}

mod fs {
    vulkano_shaders::shader! {
    ty: "fragment",
        src: "
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec2 frag_tex_coords;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, frag_tex_coords);
}
"
    }
}
