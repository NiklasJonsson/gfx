#[macro_use]
extern crate vulkano;
extern crate nalgebra_glm as glm;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use vulkano::buffer::{
    BufferAccess, BufferUsage, CpuAccessibleBuffer, ImmutableBuffer, TypedBufferAccess,
};
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::{
    descriptor_set::PersistentDescriptorSet, DescriptorSet, PipelineLayoutAbstract,
};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{swapchain::SwapchainImage, ImageUsage};
use vulkano::instance;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, CompositeAlpha, PresentMode, Surface, SurfaceTransform, Swapchain,
};
use vulkano::sync::{FlushError, GpuFuture, SharingMode};

use vulkano_win::VkSurfaceBuild;

use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent};

use std::sync::Arc;
use std::time::Instant;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

impl_vertex!(Vertex, position, color);

struct App {
    rotation_start: Instant,
    events_loop: EventsLoop,
    vk_instance: Arc<Instance>,
    vk_surface: Arc<Surface<Window>>,
    vk_device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    presentation_queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Window>>,
    swapchain_images: Vec<Arc<SwapchainImage<Window>>>,
    vertex_buffer: Arc<BufferAccess + Send + Sync>,
    index_buffer: Arc<TypedBufferAccess<Content = [u16]> + Send + Sync>,
    // TODO: Use ringbuffer
    mvp_ubo_buffers: Vec<Arc<CpuAccessibleBuffer<vs::ty::MVPUniformBufferObject>>>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    g_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    command_buffers: Vec<Arc<AutoCommandBuffer>>,
}

impl App {
    fn update_mvp(&mut self, idx: usize) {
        let diff = Instant::now() - self.rotation_start;
        let diff = diff.as_secs() as f32 + diff.subsec_millis() as f32 / 1000.0;
        let mut content = self.mvp_ubo_buffers[idx].write().unwrap();
        content.model =
            glm::rotate_z(&glm::Mat4::identity(), std::f32::consts::FRAC_PI_2 * diff).into();
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

        self.update_mvp(img_idx);
        let drawn_and_presented = swapchain_img_acquired
            .then_execute(
                Arc::clone(&self.graphics_queue),
                Arc::clone(&self.command_buffers[img_idx]),
            )
            .expect("Then execute")
            // TODO: This should be done on the presentation queue but it seems Vulkano does not
            // support this.
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

        drawn_and_presented.wait(None).unwrap();
    }

    fn main_loop(&mut self) {
        loop {
            let mut quit = false;
            self.events_loop.poll_events(|event| match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    quit = true;
                }
                _ => (),
            });

            if quit {
                break;
            }

            self.draw_frame();
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
        Self::print_queue_families(physical_device);

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
            .find(|&format| format == &(Format::B8G8R8A8Unorm, ColorSpace::SrgbNonLinear))
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

    fn create_vertex_data() -> (Vec<Vertex>, Vec<u16>) {
        let vertices = vec![
            Vertex {
                position: [-0.5, -0.5],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.5, -0.5],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                position: [0.5, 0.5],
                color: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-0.5, 0.5],
                color: [1.0, 1.0, 1.0],
            },
        ];

        let indices: Vec<u16> = vec![0, 1, 2, 2, 3, 0];

        return (vertices, indices);
    }

    // Create a vertex buffer and send it to the device on buffer_copy_queue
    fn create_and_submit_vertex_buffer(
        buffer_copy_queue: &Arc<Queue>,
        vertex_data: (Vec<Vertex>, Vec<u16>),
    ) -> (
        Arc<BufferAccess + Send + Sync>,
        Arc<TypedBufferAccess<Content = [u16]> + Send + Sync>,
    ) {
        let (vertex_buffer, vertex_data_copied) = ImmutableBuffer::from_iter(
            vertex_data.0.iter().cloned(),
            BufferUsage::vertex_buffer(),
            Arc::clone(buffer_copy_queue),
        )
        .expect("Could not create vertex buffer");

        let (index_buffer, index_data_copied) = ImmutableBuffer::from_iter(
            vertex_data.1.iter().cloned(),
            BufferUsage::index_buffer(),
            Arc::clone(buffer_copy_queue),
        )
        .expect("Could not create index buffer");

        vertex_data_copied
            .join(index_data_copied)
            .flush()
            .expect("Could not send index/vertex buffer to device");

        return (vertex_buffer, index_buffer);
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
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
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
                .fragment_shader(fs.main_entry_point(), ())
                .depth_clamp(false)
                .polygon_mode_fill()
                .line_width(1.0)
                .cull_mode_back()
                .front_face_clockwise()
                .render_pass(Subpass::from(Arc::clone(render_pass), 0).unwrap())
                .build(Arc::clone(device))
                .expect("Could not create graphics pipeline"),
        );

        return pipeline;
    }

    fn create_framebuffers(
        render_pass: &Arc<RenderPassAbstract + Send + Sync>,
        sc_images: &[Arc<SwapchainImage<Window>>],
    ) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
        sc_images
            .iter()
            .map(|image| {
                Arc::new(
                    Framebuffer::start(Arc::clone(&render_pass))
                        .add(Arc::clone(image))
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
                &glm::vec3(0.0, 0.0, -2.0),
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

    fn create_dsets_for_buffers(
        pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
        buffers: &[Arc<CpuAccessibleBuffer<vs::ty::MVPUniformBufferObject>>],
    ) -> Vec<Arc<DescriptorSet + Send + Sync>> {
        buffers
            .iter()
            .map(|buffer| {
                Arc::new(
                    PersistentDescriptorSet::start(Arc::clone(pipeline), 0)
                        .add_buffer(Arc::clone(buffer))
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
        index_buffer: &Arc<TypedBufferAccess<Content = [u16]> + Send + Sync>,
        descriptor_sets: &[Arc<DescriptorSet + Send + Sync>],
        pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
        framebuffers: &[Arc<FramebufferAbstract + Send + Sync>],
    ) -> Vec<Arc<AutoCommandBuffer>> {
        framebuffers
            .iter()
            .enumerate()
            .map(|(i, fb)| {
                let clear_color = vec![[0.0, 0.0, 0.0, 1.0].into()];

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
        index_buffer: &Arc<TypedBufferAccess<Content = [u16]> + Send + Sync>,
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
        let framebuffers = Self::create_framebuffers(&render_pass, images);

        let dsets = Self::create_dsets_for_buffers(&g_pipeline, mvp_bufs);

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

        let (render_pass, g_pipeline, framebuffers, cmd_bufs) =
            App::create_swapchain_dependent_objects(
                &self.vk_device,
                &self.swapchain,
                self.swapchain_images.as_slice(),
                self.mvp_ubo_buffers.as_slice(),
                &self.vertex_buffer,
                &self.index_buffer,
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

        let vertex_data = Self::create_vertex_data();
        let (vertex_buffer, index_buffer) =
            Self::create_and_submit_vertex_buffer(&graphics_queue, vertex_data);

        let (swapchain, images) = Self::create_swap_chain(
            &vk_device,
            &vk_surface,
            &[
                graphics_queue.family().id(),
                presentation_queue.family().id(),
            ],
        );

        let dims = get_physical_window_dims(vk_surface.window());
        let aspect_ratio = dims[0] as f32 / dims[1] as f32;
        let mvp_ubos = vec![
            Self::create_mvp_ubo(aspect_ratio),
            Self::create_mvp_ubo(aspect_ratio),
            Self::create_mvp_ubo(aspect_ratio),
        ];
        let mvp_bufs = Self::create_mvp_ubo_buffers(&vk_device, mvp_ubos.as_slice());

        let (render_pass, g_pipeline, framebuffers, cmd_bufs) =
            App::create_swapchain_dependent_objects(
                &vk_device,
                &swapchain,
                images.as_slice(),
                mvp_bufs.as_slice(),
                &vertex_buffer,
                &index_buffer,
                graphics_queue.family(),
            );

        let rotation_start = Instant::now();

        return App {
            rotation_start,
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
            mvp_ubo_buffers: mvp_bufs,
            render_pass,
            framebuffers,
            g_pipeline,
            command_buffers: cmd_bufs,
        };
    }
}

fn main() {
    let mut app = App::new();
    app.run();
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

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 0.0, 1.0);
    fragColor = color;
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

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
"
    }
}
