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
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract};
use vulkano::image::{
    attachment::AttachmentImage, immutable::ImmutableImage, swapchain::SwapchainImage, Dimensions,
    ImageUsage, ImageViewAccess,
};
use vulkano::instance;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::swapchain;
use vulkano::swapchain::Surface;
use vulkano::swapchain::{
    AcquireError, ColorSpace, CompositeAlpha, PresentMode, SurfaceTransform, Swapchain,
};

use specs::world::EntitiesRes;
use vulkano::sampler::Sampler;
use vulkano::sync::{FlushError, GpuFuture, NowFuture, SharingMode};

use winit::Window;

use nalgebra_glm as glm;

use specs::prelude::*;
use specs::storage::StorageEntry;

use std::collections::HashSet;
use std::sync::Arc;

use crate::camera::*;
use crate::common::*;

use crate::settings::{RenderMode, RenderSettings};

mod pipeline;

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

// TODO: Move all vs_pbr::* uniform types to its own file
pub struct VKManager {
    vk_instance: Arc<Instance>,
    vk_surface: Arc<Surface<Window>>,
    vk_device: Arc<Device>,
    graphics_queue: Arc<Queue>,
    presentation_queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Window>>,
    swapchain_images: Vec<Arc<SwapchainImage<Window>>>,
    transforms_buf: CpuBufferPool<pipeline::vs_pbr::ty::Transforms>,
    lighting_data_buf: CpuBufferPool<pipeline::fs_pbr::ty::LightingData>,
    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    frame_completions: Vec<Option<Box<dyn GpuFuture>>>,
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

    fn create_renderable(&self, primitive: &GraphicsPrimitive, mode: RenderMode) -> Renderable {
        use vulkano::pipeline::input_assembly::PrimitiveTopology as PT;
        let (indices, vk_mode) = match mode {
            RenderMode::Wireframe => (&primitive.line_indices.data, PT::LineList),
            RenderMode::Opaque => (&primitive.triangle_indices.data, PT::TriangleList),
        };

        let g_pipeline = pipeline::create_graphics_pipeline(
            &self.vk_device,
            &self.render_pass,
            self.swapchain.dimensions(),
            vk_mode,
            &primitive.vertex_data,
            &primitive.material,
        );

        // TODO: Use transfer queue for sending to gpu
        let (vertex_buffer, vertex_data_copied) = match &primitive.vertex_data {
            VertexBuf::Base(vertices) => {
                create_and_submit_vertex_buffer::<VertexBase>(&self.graphics_queue, vertices.to_owned())
            }
            VertexBuf::UV(vertices) => {
                create_and_submit_vertex_buffer::<VertexUV>(&self.graphics_queue, vertices.to_owned())
            }
            VertexBuf::UVCol(vertices) => {
                create_and_submit_vertex_buffer::<VertexUVCol>(&self.graphics_queue, vertices.to_owned())
            }
        };

        let (index_buffer, index_data_copied) =
            create_and_submit_index_buffer(&self.graphics_queue, indices.to_owned());

        let data_copied = vertex_data_copied.join(index_data_copied);

        let (material_data_buf, base_color_tex, material_data_copied) =
            submit_material_uniform_data(
                &self.vk_device,
                &self.graphics_queue,
                &primitive.material,
            );

        data_copied
            .join(material_data_copied)
            .then_signal_fence_and_flush()
            .expect("Unable to signal fence and flush for prepare for rendering data copy command")
            .wait(None)
            .unwrap();

        Renderable {
            vertex_buffer,
            index_buffer,
            g_pipeline,
            material_data_buf,
            base_color_tex,
            mode,
        }
    }

    pub fn prepare_primitives_for_rendering(&self, world: &World) {
        let primitives = world.read_storage::<GraphicsPrimitive>();
        let mut renderables = world.write_storage::<Renderable>();
        let entities = world.read_resource::<EntitiesRes>();
        let render_settings = world.read_resource::<RenderSettings>();
        let render_mode = render_settings.render_mode;

        for (ent, prim) in (&entities, &primitives).join() {
            let entry = renderables.entry(ent).expect("Failed to get entry!");
            match entry {
                StorageEntry::Occupied(mut occ_entry) => {
                    if occ_entry.get().mode != render_mode {
                        log::trace!("Renderable did not match render mode, creating new");
                        let rend = self.create_renderable(prim, render_mode);
                        occ_entry.insert(rend);
                    } else {
                        log::trace!("Using existing renderable");
                    }
                }
                StorageEntry::Vacant(vac_entry) => {
                    log::trace!("No renderable found, creating new");
                    let rend = self.create_renderable(prim, render_mode);
                    vac_entry.insert(rend);
                }
            }
        }
    }

    pub fn take_cursor(&mut self) {
        self.grab_cursor(true);
    }

    pub fn release_cursor(&mut self) {
        self.grab_cursor(false);
    }

    fn grab_cursor(&mut self, grab_cursor: bool) {
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
        let model_matrices = world.read_storage::<ModelMatrix>();
        let entities = world.read_resource::<EntitiesRes>();

        let camera_entity = active_camera.0.unwrap();

        let frame_idx = self.current_sc_index;

        let cam_pos = positions
            .get(camera_entity)
            .expect("Could not get position component for camera");

        let cam_rotation_state = cam_rots
            .get(camera_entity)
            .expect("Could not get rotation state for camera");

        // TODO: Camera system should write to ViewMatrixResource at the end of system
        // and we should read it here. Or there should be a resource 'ActiveCamera' that
        // we read the values from.
        let view = FreeFlyCameraController::get_view_matrix_from(cam_pos, cam_rotation_state);

        log::trace!("View matrix: {:#?}", view);

        let dims = get_physical_window_dims(self.vk_surface.window());
        let aspect_ratio = dims[0] as f32 / dims[1] as f32;
        let proj = get_proj_matrix(aspect_ratio);

        let vp_buf = pipeline::vs_pbr::ty::Transforms {
            view: view.into(),
            proj: proj.into(),
        };

        let transforms_sub_buf = self
            .transforms_buf
            .next(vp_buf)
            .expect("Could not get next ring buffer sub buffer for view proj");

        let lighting_data = pipeline::fs_pbr::ty::LightingData {
            light_pos: [5.0f32, 5.0f32, 5.0f32],
            view_pos: cam_pos.to_vec3().into(),
            _dummy0: [0; 4], // Magic Vulkano alignment
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

        let model_ubo_buf = CpuBufferPool::uniform_buffer(Arc::clone(&self.vk_device));

        let builder =
            (&entities, &renderables)
                .join()
                .fold(builder, |builder, (ent, renderable)| {
                    renderable.record_draw_commands(
                        builder,
                        &transforms_sub_buf,
                        &lighting_sub_buf,
                        &model_ubo_buf,
                        model_matrices
                            .get(ent)
                            .copied()
                            .unwrap_or_else(ModelMatrix::identity),
                    )
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
    render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
    sc_images: &[Arc<SwapchainImage<Window>>],
    multisample_count: u32,
    color_format: Format,
    depth_format: Format,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
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
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
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
) -> Arc<dyn RenderPassAbstract + Send + Sync> {
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
    Arc<dyn RenderPassAbstract + Send + Sync>,
    Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
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
fn create_and_submit_vertex_buffer<T>(
    queue: &Arc<Queue>,
    vertex_data: Vec<T>,
) -> (
    Arc<dyn BufferAccess + Send + Sync>,
    CommandBufferExecFuture<NowFuture, AutoCommandBuffer>,
)
where
    T: vulkano::pipeline::vertex::Vertex + std::clone::Clone,
{
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
    Arc<dyn TypedBufferAccess<Content = [u32]> + Send + Sync>,
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
    Arc<dyn ImageViewAccess + Send + Sync>,
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

fn submit_material_uniform_data(
    vk_device: &Arc<Device>,
    queue: &Arc<Queue>,
    material: &Material,
) -> (
    Arc<dyn BufferAccess + Send + Sync>,
    Option<TextureAccess>,
    // TODO: This is boxed since we want to return different types of GPUFuture,
    // can we solve this another way?
    Box<dyn GpuFuture>,
) {
    if let Material::GlTFPBRMaterial {
        base_color_factor,
        metallic_factor,
        roughness_factor,
        base_color_texture,
    } = material
    {
        let data = pipeline::fs_pbr_base_color_texture::ty::PBRMaterialData {
            base_color_factor: *base_color_factor,
            metallic_factor: *metallic_factor,
            roughness_factor: *roughness_factor,
        };
        let (buf, mat_copied) =
            ImmutableBuffer::from_data(data, BufferUsage::uniform_buffer(), Arc::clone(queue))
                .expect("Could not create vertex buffer");

        let (tex_access, fut) = match base_color_texture {
            Some(texture) => {
                assert_eq!(texture.coord_set, 0, "Not implemented!");

                let (image_buffer, texture_data_copied) =
                    create_and_submit_texture_image(queue, texture.image.clone());

                let sampler = Sampler::simple_repeat_linear(Arc::clone(vk_device));

                (
                    Some(TextureAccess {
                        image_buffer,
                        sampler,
                    }),
                    Box::new(mat_copied.join(texture_data_copied)) as Box<dyn GpuFuture>,
                )
            }
            None => (None, Box::new(mat_copied) as Box<dyn GpuFuture>),
        };

        (buf, tex_access, fut)
    } else {
        // TODO: Support more material types
        unimplemented!()
    }
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

    let push_constant_model_matrix = pipeline::vs_pbr_static::ty::ModelMatrix {
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

struct TextureAccess {
    pub image_buffer: Arc<dyn ImageViewAccess + Send + Sync>,
    pub sampler: Arc<Sampler>,
}

// Renderable needs to have the sampler and the material data buf bound into
// its descriptor set. The descriptor set is created here because it needs to
// be recreated each time we pass in new data to the CpuBufferPool, which is done
// each draw call.
//
// Descriptor set organisation:
// 0: Draw call variant data,
//    0: Transforms (view/proj)
//    1: Model matrix
//    2: Lighting Data (light_pos, view_pos)
// 1: Constant/Material data
//    0: MaterialData
//    1: BaseColorTexture
// TODO:
// - Should Model matrix have its own set?
// - How to precompute the MVP matrices as much as possible?

#[derive(Component)]
#[storage(VecStorage)]
pub struct Renderable {
    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    index_buffer: Arc<dyn TypedBufferAccess<Content = [u32]> + Send + Sync>,
    g_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    material_data_buf: Arc<dyn BufferAccess + Send + Sync>,
    base_color_tex: Option<TextureAccess>,
    mode: RenderMode,
}

impl Renderable {
    fn record_draw_commands(
        &self,
        cmd_buf: AutoCommandBufferBuilder,
        vp_buf: &CpuBufferPoolSubbuffer<
            pipeline::vs_pbr::ty::Transforms,
            Arc<vulkano::memory::pool::StdMemoryPool>,
        >,
        light_buf: &CpuBufferPoolSubbuffer<
            pipeline::fs_pbr::ty::LightingData,
            Arc<vulkano::memory::pool::StdMemoryPool>,
        >,
        model_ubo_buf: &CpuBufferPool<pipeline::vs_pbr::ty::Model>,
        model_matrix: ModelMatrix,
    ) -> AutoCommandBufferBuilder {
        let model_data = pipeline::vs_pbr::ty::Model {
            model: model_matrix.into(),
            model_it: glm::inverse_transpose(model_matrix.into()).into(),
        };

        let model_buf = model_ubo_buf
            .next(model_data)
            .expect("Could not allocated usub model buf");

        let descriptor_set_0 = Arc::new(
            PersistentDescriptorSet::start(Arc::clone(&self.g_pipeline), 0)
                .add_buffer(vp_buf.clone())
                .expect("Could not add vp buf")
                .add_buffer(model_buf)
                .expect("Could not add model buf")
                .add_buffer(light_buf.clone())
                .expect("Could not add light buf")
                .build()
                .expect("Failed to create persistent descriptor set for mvp ubo"),
        ) as Arc<dyn DescriptorSet + Send + Sync>;

        // TODO: Create this earlier?
        let descriptor_set_1 = Arc::new(match &self.base_color_tex {
            Some(tex_access) => Arc::new(
                PersistentDescriptorSet::start(Arc::clone(&self.g_pipeline), 1)
                    .add_buffer(self.material_data_buf.clone())
                    .expect("Could not add material data buf")
                    .add_sampled_image(tex_access.image_buffer.clone(), tex_access.sampler.clone())
                    .expect("Could not add texture access!")
                    .build()
                    .expect("Failed to build desc set 1 for material data"),
            ) as Arc<dyn DescriptorSet + Send + Sync>,
            None => Arc::new(
                PersistentDescriptorSet::start(Arc::clone(&self.g_pipeline), 1)
                    .add_buffer(self.material_data_buf.clone())
                    .expect("Could not add material data buf")
                    .build()
                    .expect("Failed to build desc set 1 for material data"),
            ),
        });

        cmd_buf
            .draw_indexed(
                Arc::clone(&self.g_pipeline),
                &DynamicState::none(),
                vec![Arc::clone(&self.vertex_buffer)],
                Arc::clone(&self.index_buffer),
                (descriptor_set_0, descriptor_set_1),
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
