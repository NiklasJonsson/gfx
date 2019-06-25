use image::RgbaImage;
use vulkano::buffer::{
    cpu_pool::CpuBufferPoolSubbuffer, BufferAccess, BufferUsage, CpuBufferPool, ImmutableBuffer,
    TypedBufferAccess,
};
use vulkano::command_buffer::{
    AutoCommandBuffer, AutoCommandBufferBuilder, CommandBufferExecFuture, DynamicState,
};
use vulkano::descriptor::{descriptor_set::PersistentDescriptorSet, DescriptorSet};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{
    attachment::AttachmentImage, immutable::ImmutableImage, swapchain::SwapchainImage, Dimensions,
    ImageUsage, ImageViewAccess,
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

use winit::Window;

use nalgebra_glm as glm;

use specs::prelude::*;

use std::collections::HashSet;
use std::sync::Arc;

use crate::asset::*;
use crate::asset;
use crate::camera::*;
use crate::common::*;

use std::time::Duration;

mod shader;

#[derive(Debug, Default)]
pub struct ActiveCamera(Option<Entity>);

impl ActiveCamera {
    pub fn empty() -> Self {
        ActiveCamera(None)
    }

    pub fn with_entity(entity: Entity) -> Self {
        ActiveCamera(Some(entity))
    }
}

pub struct VKManager {
    vk_instance: Arc<Instance>,
    vk_surface: Arc<Surface<Window>>,
    vk_device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    presentation_queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Window>>,
    swapchain_images: Vec<Arc<SwapchainImage<Window>>>,
    transforms_buf: CpuBufferPool<shader::vs_pbr::ty::Transforms>,
    lighting_data_buf: CpuBufferPool<shader::fs_pbr::ty::LightingData>,
    framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    frame_completions: Vec<Option<Box<GpuFuture>>>,
    multisample_count: u32,
    current_sc_index: usize,
}

impl VKManager {
    fn recreate_swap_chain(&mut self) {
        // Setup swap chain dimensions to that of the window
        let dimensions = get_physical_window_dims(self.vk_surface.window());

        let (swapchain, swapchain_images) = self
            .swapchain
            .recreate_with_dimension(dimensions)
            .expect("Unable to recreated swap chain");

        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;

        self.frame_completions = vec![None, None, None];

        let (render_pass, framebuffers) = create_swapchain_dependent_objects(
            &self.vk_device,
            &self.swapchain,
            self.swapchain_images.as_slice(),
            self.multisample_count,
        );

        self.render_pass = render_pass;
        self.framebuffers = framebuffers;
    }

    pub fn create(vk_instance: Arc<Instance>, vk_surface: Arc<Surface<Window>>) -> Self {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        let physical_device = pick_physical_device(&vk_instance, device_extensions);
        let (vk_device, graphics_queue, presentation_queue) =
            create_logical_device(physical_device, &vk_surface, device_extensions);

        let (swapchain, images) = create_swapchain(
            &vk_device,
            &vk_surface,
            [
                graphics_queue.family().id(),
                presentation_queue.family().id(),
            ],
        );

        let multisample_count = query_max_sample_count(&physical_device);

        let (render_pass, framebuffers) = create_swapchain_dependent_objects(
            &vk_device,
            &swapchain,
            images.as_slice(),
            multisample_count,
        );

        let frame_completions = vec![None, None, None];

        let transforms_buf = CpuBufferPool::uniform_buffer(Arc::clone(&vk_device));
        let lighting_data_buf = CpuBufferPool::uniform_buffer(Arc::clone(&vk_device));

        VKManager {
            vk_instance,
            vk_surface,
            vk_device,
            graphics_queue,
            presentation_queue,
            swapchain,
            swapchain_images: images,
            framebuffers,
            render_pass,
            multisample_count,
            current_sc_index: 0,
            frame_completions,
            transforms_buf,
            lighting_data_buf,
        }
    }

    fn send_data_to_gpu(&self, primitive: &Primitive) -> Renderable {

        // TODO: Use transfer queue here
        let (vertex_buffer, vertex_data_copied) =
            create_and_submit_vertex_buffer(&self.graphics_queue, primitive.vertex_data.to_owned());
        let (index_buffer, index_data_copied) =
            create_and_submit_index_buffer(&self.graphics_queue, primitive.index_data.to_owned());

        let data_copied = vertex_data_copied.join(index_data_copied);

        assert!(primitive.texture_data.is_none());

        /* TODO: Re-enable textures
        let image_buffer = texture_data_opt.map(|texture_data| {
            let (image_buffer, texture_data_copied) =
                create_and_submit_texture_image(&self.graphics_queue, texture_data);
            data_copied = data_copied.join(texture_data_copied);
            image_buffer
        });
        */

        data_copied
            .then_signal_fence_and_flush()
            .expect("Unable to signal fence and flush for prepare for rendering data copy command")
            .wait(None).unwrap();
        let tex_access = None;

        /*
        let tex_access = image_buffer.map(|buf| {
                let sampler = Sampler::simple_repeat_linear(Arc::clone(&self.vk_device));
                TextureAccess{buf, sampler}});
        */

        // TODO: Choose shader based on model_opt/texture availability etc
        let g_pipeline = create_graphics_pipeline(
            &self.vk_device,
            &self.render_pass,
            self.swapchain.dimensions(),
        );

        let model = primitive.transform.unwrap_or_else(|| glm::identity());

        Renderable {
            vertex_buffer,
            index_buffer,
            tex_access,
            g_pipeline,
            model,
            material: PBRMaterial::ColorOnly{color: primitive.color.unwrap().into()},
        }
    }

    pub fn prepare_static_asset_for_rendering(&self, asset: Asset) -> Vec<Renderable> {
        asset.primitives.iter().map(|primitive| {
            self.send_data_to_gpu(primitive)
        }).collect::<Vec<_>>()
    }

    pub fn grab_cursor(&mut self, grab_cursor: bool) {
        self.vk_surface
            .window()
            .grab_cursor(grab_cursor)
            .expect("Unable to grab cursor");
        self.vk_surface.window().hide_cursor(grab_cursor);
    }

    pub fn prepare_frame(&mut self) {
        let (img_idx, swapchain_img_acquired) =
            match swapchain::acquire_next_image(Arc::clone(&self.swapchain), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swap_chain();
                    return;
                }
                Err(e) => panic!("Can't acquire next image from swapchain: \t{}", e),
            };

        self.current_sc_index = img_idx;
        let prev_frame = std::mem::replace(&mut self.frame_completions[img_idx], None);

        self.frame_completions[img_idx] = Some(match prev_frame {
            None => Box::new(swapchain_img_acquired),
            Some(mut prev) => {
                prev.cleanup_finished();
                Box::new(prev.join(swapchain_img_acquired))
            }
        });
    }

    // TODO: Can we migrate to System? Would be nice but the VKManager monolith is noth "Send +
    // Sync". We might get by by mutexing and such but do we really need it? Might be able to use
    // thread_local_system which would not require thread safe systems.
    pub fn draw_next_frame(&mut self, world: &mut World) {
        let active_camera = world.read_resource::<ActiveCamera>();
        let positions = world.read_storage::<Position>();
        let cam_rots = world.read_storage::<CameraRotationState>();
        let renderables = world.read_storage::<Renderable>();

        let camera_entity = active_camera.0.unwrap();

        let frame_idx = self.current_sc_index;

        let cam_pos = positions
            .get(camera_entity)
            .expect("Could not get position component for camera");

        let cam_rotation_state = cam_rots
            .get(camera_entity)
            .expect("Could not get rotation state for camera");

        // TODO: Camera system should write to ViewMatrixResource at the end of system
        // and we should read it here.
        let cam_ori = FreeFlyCameraController::get_orientation_from(cam_rotation_state);
        let view = get_view_matrix(cam_pos, &cam_ori);

        let dims = get_physical_window_dims(self.vk_surface.window());
        let aspect_ratio = dims[0] as f32 / dims[1] as f32;
        let proj = get_proj_matrix(aspect_ratio);

        let vp_buf = shader::vs_pbr::ty::Transforms {
            view: view.into(),
            proj: proj.into(),
        };

        let transforms_sub_buf = self
            .transforms_buf
            .next(vp_buf)
            .expect("Could not get next ring buffer sub buffer for view proj");


        let lighting_data = shader::fs_pbr::ty::LightingData {
            _dummy0: [0; 4], // Magic Vulkano alignment
            light_pos: [10.0f32, 10.0f32, 10.0f32].into(),
            view_pos: cam_pos.to_vec3().into(),
        };

        let lighting_sub_buf = self
            .lighting_data_buf
            .next(lighting_data)
            .expect("Could not get next ring buffer sub buffer for view proj");

        let prev_frame = std::mem::replace(&mut self.frame_completions[frame_idx], None).unwrap();

        let clear_color = vec![
            [0.0, 0.0, 0.0, 1.0].into(),
            vulkano::format::ClearValue::None,
            1.0f32.into(),
        ];

        // Render all the renderables as one render pass
        let builder = AutoCommandBufferBuilder::primary_one_time_submit(
            Arc::clone(&self.vk_device),
            self.graphics_queue.family(),
        )
        .expect("Failed to create command buffer builder")
        .begin_render_pass(
            Arc::clone(&self.framebuffers[frame_idx]),
            false,
            clear_color,
        )
        .expect("Failed after begin render pass");

        // TODO: Think about sending the uniform data to the GPU for each renderable.
        // 1. We want to have it per primtive/renderable.
        // 2. It might or might not be constant for a primitive
        let mut pbr_material_buf = CpuBufferPool::uniform_buffer(Arc::clone(&self.vk_device));
        let mut model_ubo_buf = CpuBufferPool::uniform_buffer(Arc::clone(&self.vk_device));

        let builder = renderables.join().fold(builder, |builder, renderable| {
            renderable.render_unto(builder, &transforms_sub_buf,
                                   &lighting_sub_buf, &pbr_material_buf,
                                   &model_ubo_buf)
        });

        let cmd_buf = builder
            .end_render_pass()
            .expect("Unable to end render pass")
            .build()
            .expect("Unable to build render pass");

        let presented = prev_frame
            .then_execute(Arc::clone(&self.graphics_queue), cmd_buf)
            .expect("unable to execute render cmd buf")
            .then_swapchain_present(
                Arc::clone(&self.graphics_queue),
                Arc::clone(&self.swapchain),
                frame_idx,
            )
            .then_signal_fence_and_flush();

        let rendered_and_presented = match presented {
            Ok(r) => r,
            Err(FlushError::OutOfDate) => {
                self.recreate_swap_chain();
                return;
            }
            Err(e) => panic!(
                "Can't write to the swapchain image (idx: {}):\n\t{}",
                frame_idx, e
            ),
        };

        self.frame_completions[frame_idx] = Some(Box::new(rendered_and_presented));
    }
}

fn get_proj_matrix(aspect_ratio: f32) -> glm::Mat4 {
    // TODO: Rewrite this
    let mut proj = glm::perspective(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 10.0);

    // glm::perspective is based on opengl left-handed coordinate system, vulkan has the y-axis
    // inverted (right-handed upside-down).
    proj[(1, 1)] *= -1.0;

    proj
}

fn create_framebuffers(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPassAbstract + Send + Sync>,
    sc_images: &[Arc<SwapchainImage<Window>>],
    multisample_count: u32,
    color_format: Format,
    depth_format: Format,
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    sc_images
        .iter()
        .map(|image| {
            let dims = SwapchainImage::dimensions(&image);
            let depth_buffer = AttachmentImage::transient_multisampled(
                Arc::clone(device),
                dims,
                multisample_count,
                depth_format,
            )
            .unwrap();
            let multisample_image = AttachmentImage::transient_multisampled(
                Arc::clone(device),
                dims,
                multisample_count,
                color_format,
            )
            .unwrap();

            Arc::new(
                Framebuffer::start(Arc::clone(&render_pass))
                    .add(multisample_image)
                    .unwrap()
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

fn pick_physical_device<'a>(
    vk_instance: &'a Arc<Instance>,
    required_extensions: DeviceExtensions,
) -> PhysicalDevice<'a> {
    // TODO: For proper device selection, this should also be done:
    //  - the available queues should be checked as well
    //  - swap chain support/adequacy
    PhysicalDevice::enumerate(vk_instance)
        .find(|&ph_dev| {
            // TODO: Subset instead
            let supported_extensions = DeviceExtensions::supported_by_device(ph_dev);

            required_extensions.intersection(&supported_extensions) == required_extensions
        })
        .expect("No suitable device available")
}

fn query_max_sample_count(physical_device: &PhysicalDevice) -> u32 {
    let color_samples = physical_device.limits().framebuffer_color_sample_counts();
    let depth_samples = physical_device.limits().framebuffer_depth_sample_counts();

    log::info!(
        "Physical device, framebuffer_color_sample_counts: {}",
        color_samples
    );
    log::info!(
        "Physical device, framebuffer_depth_sample_counts: {}",
        depth_samples
    );

    // TODO: Clean this up (port to vulkano?)
    let bits = vec![
        vk_sys::SAMPLE_COUNT_1_BIT,
        vk_sys::SAMPLE_COUNT_2_BIT,
        vk_sys::SAMPLE_COUNT_4_BIT,
        vk_sys::SAMPLE_COUNT_8_BIT,
        vk_sys::SAMPLE_COUNT_16_BIT,
        vk_sys::SAMPLE_COUNT_32_BIT,
        vk_sys::SAMPLE_COUNT_64_BIT,
    ];

    bits.iter()
        .rev()
        .find(|&&bit| (color_samples & bit != 0) && (depth_samples & bit != 0))
        .cloned()
        .unwrap_or(bits[0])
}

fn create_swapchain(
    device: &Arc<Device>,
    surface: &Arc<Surface<Window>>,
    queue_family_ids: [u32; 2],
) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    let physical_device = device.physical_device();
    let capabilities = surface
        .capabilities(physical_device)
        .expect("Can't fetch surface capabilites");

    log::info!("Surface capabilities, supported formats:");
    for f in &capabilities.supported_formats {
        log::info!("{:?}", f);
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

    let sharing_mode: SharingMode = if queue_family_ids.iter().all(|&x| x == queue_family_ids[0]) {
        SharingMode::Exclusive(queue_family_ids[0])
    } else {
        SharingMode::Concurrent(queue_family_ids.to_vec())
    };

    let alpha = CompositeAlpha::Opaque;

    Swapchain::new(
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
    .expect("Failed to create swap chain")
}

fn create_render_pass(
    device: &Arc<Device>,
    color_format: Format,
    depth_format: Format,
    multisample_count: u32,
) -> Arc<RenderPassAbstract + Send + Sync> {
    Arc::new(
        single_pass_renderpass!(Arc::clone(device),
        attachments: {
            msaa_color: {
                load: Clear,
                store: DontCare,
                format: color_format,
                samples: multisample_count,
            },
            color: {
                load: DontCare,
                store: Store,
                format: color_format,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: depth_format,
                samples: multisample_count,
            }
        },
        pass: {
            color: [msaa_color],
            depth_stencil: {depth},
            resolve: [color],
        })
        .unwrap(),
    )
}

fn create_swapchain_dependent_objects(
    device: &Arc<Device>,
    swapchain: &Arc<Swapchain<Window>>,
    images: &[Arc<SwapchainImage<Window>>],
    multisample_count: u32,
) -> (
    Arc<RenderPassAbstract + Send + Sync>,
    Vec<Arc<FramebufferAbstract + Send + Sync>>,
) {
    // TODO: If we can query the swapchain formats before creating it, we can decouple the
    // creation of render pass and swapchain.
    let color_format = swapchain.format();
    // TODO: Query for support for this
    let depth_format = Format::D32Sfloat;

    let render_pass = create_render_pass(device, color_format, depth_format, multisample_count);

    let framebuffers = create_framebuffers(
        device,
        &render_pass,
        images,
        multisample_count,
        color_format,
        depth_format,
    );
    (render_pass, framebuffers)
}

fn create_logical_device(
    physical_device: PhysicalDevice,
    surface: &Arc<Surface<Window>>,
    device_extensions: DeviceExtensions,
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
        HashSet::from_iter(q_families.iter().map(QueueFamily::id));

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
        /* extensions */ &device_extensions,
        q_families,
    )
    .expect("Failed to create device");

    let graphics_queue = queues.next().expect("Device queues not created");
    let presentation_queue = queues.next().unwrap_or_else(|| Arc::clone(&graphics_queue));

    (device, graphics_queue, presentation_queue)
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

    (buf, fut)
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

    (buf, fut)
}

fn create_and_submit_texture_image(
    queue: &Arc<Queue>,
    image: RgbaImage,
) -> (
    Arc<ImageViewAccess + Send + Sync>,
    CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
) {
    // TODO: Support mip maps
    let width = image.width();
    let height = image.height();
    let (buf, fut) = ImmutableImage::from_iter(
        image.into_raw().into_iter(),
        Dimensions::Dim2d { width, height },
        Format::R8G8B8A8Srgb,
        Arc::clone(queue),
    )
    .expect("Unable to create vertex buffer");

    (buf, fut)
}

fn create_graphics_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPassAbstract + Send + Sync>,
    swapchain_dimensions: [u32; 2],
) -> Arc<GraphicsPipelineAbstract + Send + Sync> {
    let vs = shader::vs_pbr::Shader::load(Arc::clone(device)).expect("Vertex shader compilation failed");
    let fs = shader::fs_pbr::Shader::load(Arc::clone(device)).expect("Fragment shader compilation failed");

    let dims = [
        swapchain_dimensions[0] as f32,
        swapchain_dimensions[1] as f32,
    ];

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: dims,
        depth_range: 0.0..1.0,
    };

    Arc::new(
        GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            // How to interpret the vertex input
            .triangle_list()
            .vertex_shader(vs.main_entry_point(), ())
            // Whether to support special indices in in the vertex buffer to split triangles
            .primitive_restart(false)
            .viewports([viewport].iter().cloned())
            .depth_clamp(false)
            .polygon_mode_fill()
            .line_width(1.0)
            .cull_mode_back()
            .depth_stencil_simple_depth()
            .fragment_shader(fs.main_entry_point(), ())
            .blend_pass_through()
            .front_face_counter_clockwise()
            .render_pass(Subpass::from(Arc::clone(render_pass), 0).unwrap())
            .build(Arc::clone(device))
            .expect("Could not create graphics pipeline"),
    )
}

/*
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
            let clear_color = vec![
                [0.0, 0.0, 0.0, 1.0].into(),
                vulkano::format::ClearValue::None,
                1.0f32.into(),
            ];

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
*/
/*
fn create_command_buffer_with_push_constants(
    device: &Arc<Device>,
    queue_family: QueueFamily,
    vertex_buffer: &Arc<BufferAccess + Send + Sync>,
    index_buffer: &Arc<TypedBufferAccess<Content = [u32]> + Send + Sync>,
    descriptor_set: &Arc<DescriptorSet + Send + Sync>,
    pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
    framebuffer: &Arc<FramebufferAbstract + Send + Sync>,
    model_matrix: glm::Mat4,
) -> Arc<AutoCommandBuffer> {
    let clear_color = vec![
        [0.0, 0.0, 0.0, 1.0].into(),
        vulkano::format::ClearValue::None,
        1.0f32.into(),
    ];

    let push_constant_model_matrix = shader::vs_pbr_static::ty::ModelMatrix {
        matrix: model_matrix.into(),
    };

    Arc::new(
        AutoCommandBufferBuilder::primary_one_time_submit(Arc::clone(device), queue_family)
            .expect("Failed to create command buffer builder")
            .begin_render_pass(Arc::clone(framebuffer), false, clear_color)
            .expect("Failed after begin render pass")
            .draw_indexed(
                Arc::clone(pipeline),
                &DynamicState::none(),
                vec![Arc::clone(vertex_buffer)],
                Arc::clone(index_buffer),
                Arc::clone(&descriptor_set),
                push_constant_model_matrix,
            )
            .expect("Failed after draw_indexed")
            .end_render_pass()
            .expect("Failed after end_render_pass")
            .build()
            .expect("Failed to build command buffer"),
    )
}
*/

impl_vertex!(Vertex, position, normal);

fn get_view_matrix(pos: &Position, ori: &CameraOrientation) -> glm::Mat4 {
    let dir = ori.direction;
    let up = ori.up;

    // Based on https://learnopengl.com/Getting-started/Camera
    // Which is based on Gram-Schmidt process:
    // https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

    // Reverse direction here as the camera will look in negative z
    let cam_dir = glm::normalize(&-dir);

    // We need a right vector
    let cam_right = glm::normalize(&glm::cross::<f32, glm::U3>(&up, &cam_dir));

    // The up vector is guaranteed to be be perpendicular to both direction and right
    // => create a new one
    let cam_up = glm::normalize(&glm::cross::<f32, glm::U3>(&cam_dir, &cam_right));

    let trns = glm::translate(&glm::identity(), &-(pos.to_vec3()));

    let view = glm::mat4(
        cam_right.x,
        cam_right.y,
        cam_right.z,
        0.0,
        cam_up.x,
        cam_up.y,
        cam_up.z,
        0.0,
        cam_dir.x,
        cam_dir.y,
        cam_dir.z,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    view * trns
}

/*
fn create_dsets<T, A: vulkano::mem::MemoryPool>(
    pipeline: &Arc<GraphicsPipelineAbstract + Send + Sync>,
    buffers: &[Arc<BufTy>],
    image: &Arc<ImageViewAccess + Send + Sync>,
    sampler: &Arc<Sampler>,
) -> Vec<Arc<DescriptorSet + Send + Sync>> {
    buffers
        .iter()
        .map(|buffer| create_descriptor_set(pipeline, buffer, image, sampler))
        .collect::<Vec<_>>()
}
*/

struct TextureAccess {
    image_buffer: Arc<ImageViewAccess + Send + Sync>,
    sampler: Arc<Sampler>
}

enum PBRMaterial {
    ColorOnly {color: [f32; 4]},
    NotPBR,
}

#[derive(Component)]
#[storage(VecStorage)]
pub struct Renderable {
    vertex_buffer: Arc<BufferAccess + Send + Sync>,
    index_buffer: Arc<TypedBufferAccess<Content = [u32]> + Send + Sync>,
    tex_access: Option<TextureAccess>,
    g_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    model: glm::Mat4,
    material: PBRMaterial,
}

impl Renderable {
    fn render_unto(
        &self,
        cmd_buf: AutoCommandBufferBuilder,
        vp_buf: &CpuBufferPoolSubbuffer<
            shader::vs_pbr::ty::Transforms,
            Arc<vulkano::memory::pool::StdMemoryPool>,
        >,
        light_buf: &CpuBufferPoolSubbuffer<
            shader::fs_pbr::ty::LightingData,
            Arc<vulkano::memory::pool::StdMemoryPool>,
        >,
        pbr_buf_pol: &CpuBufferPool<shader::fs_pbr::ty::PBRMaterialData>,
        model_ubo_buf: &CpuBufferPool<shader::vs_pbr::ty::Model>,
    ) -> AutoCommandBufferBuilder {

        let color = match self.material {
            PBRMaterial::ColorOnly {color: col} => col,
            PBRMaterial::NotPBR => panic!("Not implemented"),
        };

        let pbr_material = shader::fs_pbr::ty::PBRMaterialData {
             base_color_factor: color.into(),
             metallic_factor: 0.0f32,
             roughness_factor: 1.0f32,
        };

        let pbr_buf = pbr_buf_pol.next(pbr_material).expect("Could not allocate sub uni buf");

        let model_data = shader::vs_pbr::ty::Model {
            model: self.model.into(),
            model_it: glm::inverse_transpose(self.model).into(),
        };

        let model_buf = model_ubo_buf.next(model_data).expect("Could not allocated usub model buf");


        let descriptor_set = Arc::new(
            PersistentDescriptorSet::start(Arc::clone(&self.g_pipeline), 0)
                .add_buffer(vp_buf.clone())
                .expect("Could not add vp buf")
                .add_buffer(model_buf)
                .expect("Could not add model buf")
                .add_buffer(pbr_buf.clone())
                .expect("Could not add pbr buf")
                .add_buffer(light_buf.clone())
                .expect("Could not add light buf")
                .build()
                .expect("Failed to create persistent descriptor set for mvp ubo"),
        ) as Arc<DescriptorSet + Send + Sync>;

        /*
        let fs_descriptor_set = Arc::new(
            PersistentDescriptorSet::start(Arc::clone(&self.g_pipeline), 0)
                //.add_sampled_image(Arc::clone(&self.image_buffer), Arc::clone(&self.sampler))
                //.unwrap()
                .build()
                .expect("Failed to create persistent descriptor set for mvp ubo"),
        ) as Arc<DescriptorSet + Send + Sync>;

        let push_constant_model_matrix = shader::vs_pbr_static::ty::ModelMatrix {
            matrix: self.model.into(),
        };

        */
        cmd_buf
            .draw_indexed(
                Arc::clone(&self.g_pipeline),
                &DynamicState::none(),
                vec![Arc::clone(&self.vertex_buffer)],
                Arc::clone(&self.index_buffer),
                Arc::clone(&descriptor_set),
                (),
            )
            .expect("Failed after draw_indexed")
    }
}

fn print_validation_layers() {
    log::info!("Available vulkan validation layers:");
    for avail in instance::layers_list().expect("Can't query validation layers") {
        log::info!("\t{}", avail.name());
    }
}

pub fn get_vk_instance() -> Arc<Instance> {
    let available_extensions =
        InstanceExtensions::supported_by_core().expect("can't get supported extensions");
    let required_extensions = vulkano_win::required_extensions();

    // TODO: Subset
    if available_extensions.intersection(&required_extensions) != required_extensions {
        log::error!("Can't create a window, not all extensions supported.");
        unreachable!();
    }

    print_validation_layers();

    Instance::new(None, &required_extensions, None).expect("Could not create vulkan instance")
}

fn get_physical_window_dims(window: &Window) -> [u32; 2] {
    window
        .get_inner_size()
        .map(|dims| {
            let dims: (u32, u32) = dims.to_physical(window.get_hidpi_factor()).into();
            [dims.0, dims.1]
        })
        .expect("Was not able to read window dimensions, is it open?")
}
