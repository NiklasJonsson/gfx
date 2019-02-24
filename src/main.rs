#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate num_derive;
extern crate vk_sys;

extern crate env_logger;
extern crate log;

extern crate image;
extern crate nalgebra_glm as glm;
extern crate specs;
#[macro_use]
extern crate specs_derive;
extern crate tobj;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use image::RgbaImage;

use log::{info, warn};

use vulkano::buffer::{
    BufferAccess, BufferUsage, CpuAccessibleBuffer, ImmutableBuffer, TypedBufferAccess,
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

use vulkano_win::VkSurfaceBuild;

use winit::{Event, EventsLoop, Window, WindowBuilder, WindowEvent};

use specs::prelude::*;

use std::collections::HashSet;

use std::path::Path;

use std::sync::Arc;
use std::time::Duration;

mod camera;
mod common;
mod input;

use self::common::*;
use self::input::{ActionId, InputContext, InputContextPriority, MappedInput};

// REFACTOR: The vulkan/rendering specific parts of this should go into its own module

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn new(position: [f32; 3], tex_coords: [f32; 2]) -> Vertex {
        Vertex {
            position,
            tex_coords,
        }
    }
}

impl_vertex!(Vertex, position, tex_coords);

// Extra trait specialization for GpuFuture, intended for storing NowFuture or FenceSignalFuture
trait WaitableFuture: GpuFuture {
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

type AppEvents = Vec<Event>;

// World resources
#[derive(Default, Debug)]
struct CurrentFrameWindowEvents(AppEvents);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum GameState {
    Paused,
    Running,
}

impl Default for GameState {
    fn default() -> Self {
        GameState::Running
    }
}

const GAME_STATE_SWITCH: ActionId = 0;

#[derive(Default, Component)]
#[storage(NullStorage)]
struct GameStateSwitcher;

impl<'a> System<'a> for GameStateSwitcher {
    type SystemData = (
        Write<'a, GameState>,
        WriteStorage<'a, InputContext>,
        WriteStorage<'a, MappedInput>,
        ReadStorage<'a, Self>,
    );

    fn run(&mut self, (mut state, mut contexts, mut inputs, _unique_id): Self::SystemData) {
        log::trace!("GameStateSwitcher: run");
        // TODO: Verify that we only get one mapped input? (and one context)

        for (inp, ctx) in (&mut inputs, &mut contexts).join() {
            use crate::GameState::*;
            if inp.actions.contains(&GAME_STATE_SWITCH) {
                *state = match *state {
                    Paused => Running,
                    Running => Paused,
                };

                ctx.set_consume_all(*state == Paused);
            }
            inp.clear();
        }
    }
}

struct ActiveCamera(Option<Entity>);

struct App {
    world: World,
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
    multisample_count: u32,
}

// TODO: Handle resized here as well
#[derive(Debug)]
enum AppAction {
    Quit,
    IgnoreInput,
    AcceptInput(AppEvents),
    HandleEvents(AppEvents),
}

impl AppAction {
    fn update_with(self, new: Self) -> Self {
        use AppAction::*;
        match (new, self) {
            (_, Quit) => Quit,
            (_, IgnoreInput) => IgnoreInput,
            (Quit, _) => Quit,
            (IgnoreInput, _) => IgnoreInput,
            (AcceptInput(_), _) => AcceptInput(Vec::new()),
            (HandleEvents(mut new_events), AcceptInput(mut old_events)) => {
                old_events.append(&mut new_events);
                AcceptInput(old_events)
            }
            (HandleEvents(mut new_events), HandleEvents(mut old_events)) => {
                old_events.append(&mut new_events);
                HandleEvents(old_events)
            }
        }
    }
}

struct EventManager {
    action: AppAction,
}

// TODO: We should not have "ignore-code" in both event manager and input manager
// Only stuff that is relevant to input manager should be forwarded.
// Create enum to represent what we want the input manager to receive
// But should this really be done here? Separate window/input handling?

impl EventManager {
    fn new() -> Self {
        Self {
            action: AppAction::HandleEvents(Vec::new()),
        }
    }

    fn update_action(&mut self, action: AppAction) {
        let cur = std::mem::replace(&mut self.action, AppAction::Quit);
        self.action = cur.update_with(action);
    }

    fn collect_event(&mut self, event: Event) {
        let action: Option<AppAction> = match &event {
            Event::WindowEvent {
                event: inner_event, ..
            } => match inner_event {
                WindowEvent::CloseRequested => {
                    log::info!("EventManager: Received CloseRequested window event");
                    Some(AppAction::Quit)
                }
                WindowEvent::Focused(focused) => Some(if !focused {
                    log::trace!("Window lost focus, ignoring input");
                    AppAction::IgnoreInput
                } else {
                    log::trace!("Window gained focus, accepting input");
                    AppAction::AcceptInput(Vec::new())
                }),
                _ => Some(AppAction::HandleEvents(Vec::new())),
            },
            Event::DeviceEvent {
                event: inner_event, ..
            } => {
                if let winit::DeviceEvent::MouseMotion { .. } = inner_event {
                    Some(AppAction::HandleEvents(Vec::new()))
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(action) = action {
            let action = match action {
                AppAction::AcceptInput(mut empty_vec) => {
                    empty_vec.push(event);
                    AppAction::AcceptInput(empty_vec)
                }
                AppAction::HandleEvents(mut empty_vec) => {
                    empty_vec.push(event);
                    AppAction::HandleEvents(empty_vec)
                }
                action => action,
            };
            self.update_action(action);
        }
    }

    fn resolve(&mut self) -> AppAction {
        std::mem::replace(&mut self.action, AppAction::HandleEvents(Vec::new()))
    }
}

impl App {
    fn update_mvp(&mut self, idx: usize) {
        let mut content = self.mvp_ubo_buffers[idx].write().unwrap();
        let active_cam_entity = (*self.world.read_resource::<ActiveCamera>()).0.unwrap();
        let pos_storage = self.world.read_storage::<Position>();
        let cam_pos = pos_storage.get(active_cam_entity).unwrap().to_vec3();

        let ori_storage = self
            .world
            .read_storage::<crate::camera::CameraOrientation>();
        let cam_ori = ori_storage.get(active_cam_entity).unwrap();

        let dir = cam_ori.direction;
        let up = cam_ori.up;

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

        let trns = glm::translate(&glm::identity(), &-(cam_pos));

        let view = glm::mat4(
            cam_right[0],
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

        let view = view * trns;

        content.view = view.into();
    }

    // FIXME: This lives here only because the lifetime parameters are a pain.
    // The whole App struct would need to be templated if this was included.
    // Maybe this can be solved in another way...
    fn init_dispatcher<'a, 'b>() -> Dispatcher<'a, 'b> {
        let builder = DispatcherBuilder::new();
        // Input needs to go before as camera depends on it
        let builder = input::register_systems(builder);
        let builder = camera::register_systems(builder);

        builder
            .with(
                GameStateSwitcher,
                "game_state_switcher",
                &[input::INPUT_MANAGER_SYSTEM_ID],
            )
            .build()
    }

    fn populate_world(&mut self) {
        let cam_entity = camera::init_camera(&mut self.world);
        {
            let mut active_camera = self.world.write_resource::<ActiveCamera>();
            *active_camera = ActiveCamera(Some(cam_entity));
        }

        // TODO: Clean up
        let escape_catcher = InputContext::start("EscapeCatcher")
            .with_description("Global top-level escape catcher for game state switcher")
            .with_action(winit::VirtualKeyCode::Escape, GAME_STATE_SWITCH)
            .with_priority(InputContextPriority::First)
            .build();
        let mapped_input = MappedInput::new();
        self.world
            .create_entity()
            .with(escape_catcher)
            .with(mapped_input)
            .with(GameStateSwitcher {})
            .build();
    }

    fn main_loop(&mut self) {
        let mut dispatcher = Self::init_dispatcher();

        // Register all component types
        dispatcher.setup(&mut self.world.res);

        // Setup camera
        self.populate_world();

        let mut event_manager = EventManager::new();

        loop {
            log::trace!("Polling events");

            self.events_loop
                .poll_events(|event| event_manager.collect_event(event));

            let free_cursor = *self.world.read_resource::<GameState>() == GameState::Paused;

            self.vk_surface
                .window()
                .grab_cursor(!free_cursor)
                .expect("Unable to grab cursor");
            self.vk_surface.window().hide_cursor(!free_cursor);

            match event_manager.resolve() {
                AppAction::Quit => {
                    return;
                }
                AppAction::IgnoreInput => {
                    continue;
                }
                // TODO: Can we merge these? AcceptEvents
                // TODO: Atleast we can use the | "or" pattern here
                AppAction::AcceptInput(events) => {
                    let mut cur_events = self.world.write_resource::<CurrentFrameWindowEvents>();
                    *cur_events = CurrentFrameWindowEvents(events);
                }
                AppAction::HandleEvents(window_events) => {
                    let mut cur_events = self.world.write_resource::<CurrentFrameWindowEvents>();
                    *cur_events = CurrentFrameWindowEvents(window_events);
                }
            }

            log::trace!("Acquiring swapchain");
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

            // Wait for previous frame for this image before we update MVP buffer
            prev_frame_completed.wait_for(None).unwrap();

            log::trace!("Dispatching");
            // Run all ECS systems (blocking call)
            dispatcher.dispatch(&self.world.res);

            // This writes to the uniform buffer for the upcoming frame
            // TODO: Merge this into ECS
            log::trace!("Update MVP");
            self.update_mvp(img_idx);

            let drawn_and_presented = swapchain_img_acquired
                .then_execute(
                    Arc::clone(&self.graphics_queue),
                    Arc::clone(&self.command_buffers[img_idx]),
                )
                .expect("Unable to execute command buffer")
                // TODO: Use presentation queue + semaphore when vulkano supports it
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
    }

    fn run(&mut self) {
        self.main_loop();
    }

    fn print_validation_layers() {
        info!("Available vulkan validation layers:");
        for avail in instance::layers_list().expect("Can't query validation layers") {
            info!("\t{}", avail.name());
        }
    }

    fn setup_vk_instance() -> Arc<Instance> {
        let available_extensions =
            InstanceExtensions::supported_by_core().expect("can't get supported extensions");
        let required_extensions = vulkano_win::required_extensions();

        if available_extensions.intersection(&required_extensions) != required_extensions {
            log::error!("Can't create a window, not all extensions supported.");
            unreachable!();
        }

        Self::print_validation_layers();

        Instance::new(None, &required_extensions, None).expect("Could not create vulkan instance")
    }

    fn setup_surface(vk_instance: &Arc<Instance>) -> (EventsLoop, Arc<Surface<Window>>) {
        let events_loop = EventsLoop::new();
        let surface = WindowBuilder::new()
            .build_vk_surface(&events_loop, Arc::clone(vk_instance))
            .expect("Unable to create window/surface");

        (events_loop, surface)
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
            /* extensions */ &device_extensions,
            q_families,
        )
        .expect("Failed to create device");

        let graphics_queue = queues.next().expect("Device queues not created");
        let presentation_queue = queues.next().unwrap_or_else(|| Arc::clone(&graphics_queue));

        (device, graphics_queue, presentation_queue)
    }

    fn create_swap_chain(
        device: &Arc<Device>,
        surface: &Arc<Surface<Window>>,
        queue_family_ids: [u32; 2],
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = device.physical_device();
        let capabilities = surface
            .capabilities(physical_device)
            .expect("Can't fetch surface capabilites");

        info!("Surface capabilities, supported formats:");
        for f in &capabilities.supported_formats {
            info!("{:?}", f);
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
            if queue_family_ids.iter().all(|&x| x == queue_family_ids[0]) {
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

    fn debug_obj(models: &[tobj::Model], materials: &[tobj::Material]) {
        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;
            log::debug!("model[{}].name = \'{}\'", i, m.name);
            log::debug!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

            log::debug!("Size of model[{}].indices: {}", i, mesh.indices.len());
            /*
            for f in 0..mesh.indices.len() / 3 {
                log::debug!("    idx[{}] = {}, {}, {}.", f, mesh.indices[3 * f],
                         mesh.indices[3 * f + 1], mesh.indices[3 * f + 2]);
            }
            */

            // Normals and texture coordinates are also loaded, but not printed in this example
            log::debug!("model[{}].vertices: {}", i, mesh.positions.len() / 3);
            assert!(mesh.positions.len() % 3 == 0);
            /*
            for v in 0..mesh.positions.len() / 3 {
                log::debug!("    v[{}] = ({}, {}, {})", v, mesh.positions[3 * v],
                mesh.positions[3 * v + 1], mesh.positions[3 * v + 2]);
            }
            */

            for (i, m) in materials.iter().enumerate() {
                log::debug!("material[{}].name = \'{}\'", i, m.name);
                log::debug!(
                    "    material.Ka = ({}, {}, {})",
                    m.ambient[0],
                    m.ambient[1],
                    m.ambient[2]
                );
                log::debug!(
                    "    material.Kd = ({}, {}, {})",
                    m.diffuse[0],
                    m.diffuse[1],
                    m.diffuse[2]
                );
                log::debug!(
                    "    material.Ks = ({}, {}, {})",
                    m.specular[0],
                    m.specular[1],
                    m.specular[2]
                );
                log::debug!("    material.Ns = {}", m.shininess);
                log::debug!("    material.d = {}", m.dissolve);
                log::debug!("    material.map_Ka = {}", m.ambient_texture);
                log::debug!("    material.map_Kd = {}", m.diffuse_texture);
                log::debug!("    material.map_Ks = {}", m.specular_texture);
                log::debug!("    material.map_Ns = {}", m.normal_texture);
                log::debug!("    material.map_d = {}", m.dissolve_texture);
                for (k, v) in &m.unknown_param {
                    log::debug!("    material.{} = {}", k, v);
                }
            }
        }
    }

    fn load_obj(path: &str) -> (Vec<Vertex>, Vec<u32>) {
        info!("Loading models from {}", path);
        // TODO: "Vertex dedup" instead of storing duplicates of
        // vertices, we should re-use old ones if they are identical.

        let (models, materials) = tobj::load_obj(&Path::new(path)).unwrap();
        info!(
            "Found {} models and {} materials",
            models.len(),
            materials.len()
        );
        warn!("Ignoring materials and models other than model[0]");
        Self::debug_obj(models.as_slice(), materials.as_slice());

        let tex_coords = models[0].mesh.texcoords.chunks_exact(2);
        let vertices = models[0]
            .mesh
            .positions
            .chunks_exact(3)
            .zip(tex_coords)
            .map(|(pos, tx_cs)| Vertex::new([pos[0], pos[1], pos[2]], [tx_cs[0], 1.0 - tx_cs[1]]))
            .collect::<Vec<_>>();

        let indices = models[0].mesh.indices.to_owned();

        (vertices, indices)
    }

    fn load_image(path: &str) -> image::RgbaImage {
        info!("Trying to load image from {}", path);
        let image = image::open(path).expect("Unable to load image").to_rgba();

        info!(
            "Loaded RGBA image with dimensions: {:?}",
            image.dimensions()
        );

        image
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

    fn create_mvp_ubo(aspect_ratio: f32) -> vs::ty::MVPUniformBufferObject {
        let mut proj = glm::perspective(aspect_ratio, std::f32::consts::FRAC_PI_4, 0.1, 10.0);

        // glm::perspective is based on opengl left-handed coordinate system, vulkan has the y-axis
        // inverted (right-handed upside-down).
        proj[(1, 1)] *= -1.0;

        // FIXME: This is due to the orientation of the chalet model from the vulkan tutorial
        let model = glm::rotate_x(&glm::Mat4::identity(), -std::f32::consts::FRAC_PI_2);
        let model = glm::rotate_y(&model, std::f32::consts::FRAC_1_PI);

        vs::ty::MVPUniformBufferObject {
            model: model.into(),
            view: glm::look_at(
                &glm::vec3(2.0, 2.0, 2.0),
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 1.0, 0.0),
            )
            .into(),
            proj: proj.into(),
        }
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
        multisample_count: u32,
    ) -> (
        Arc<RenderPassAbstract + Send + Sync>,
        Arc<GraphicsPipelineAbstract + Send + Sync>,
        Vec<Arc<FramebufferAbstract + Send + Sync>>,
        Vec<Arc<AutoCommandBuffer>>,
    ) {
        let color_format = swapchain.format();
        // TODO: Query for support for this
        let depth_format = Format::D32Sfloat;
        let render_pass =
            Self::create_render_pass(device, color_format, depth_format, multisample_count);
        let g_pipeline =
            Self::create_graphics_pipeline(device, &render_pass, swapchain.dimensions());
        let framebuffers = Self::create_framebuffers(
            device,
            &render_pass,
            images,
            multisample_count,
            color_format,
            depth_format,
        );

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

        (render_pass, g_pipeline, framebuffers, cmd_bufs)
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
                self.multisample_count,
            );

        self.render_pass = render_pass;
        self.g_pipeline = g_pipeline;
        self.framebuffers = framebuffers;
        self.command_buffers = cmd_bufs;
    }

    fn query_max_sample_count(physical_device: &PhysicalDevice) -> u32 {
        let color_samples = physical_device.limits().framebuffer_color_sample_counts();
        let depth_samples = physical_device.limits().framebuffer_depth_sample_counts();

        info!(
            "Physical device, framebuffer_color_sample_counts: {}",
            color_samples
        );
        info!(
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

    fn init_world() -> World {
        let mut world = World::new();

        world.add_resource(CurrentFrameWindowEvents(Vec::new()));
        world.add_resource(ActiveCamera(None));
        world.add_resource(GameState::Running);

        input::add_resources(&mut world);

        world
    }

    fn new() -> Self {
        let vk_instance = Self::setup_vk_instance();
        let (events_loop, vk_surface) = Self::setup_surface(&vk_instance);
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        let physical_device = Self::pick_physical_device(&vk_instance, device_extensions);

        let multisample_count = Self::query_max_sample_count(&physical_device);
        let (vk_device, graphics_queue, presentation_queue) =
            Self::create_logical_device(physical_device, &vk_surface, device_extensions);

        let world = Self::init_world();

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

        let image = Self::load_image("textures/chalet.jpg");
        let (image_buffer, texture_data_copied) =
            Self::create_and_submit_texture_image(&graphics_queue, image);

        let data_copied = data_copied
            .join(texture_data_copied)
            .then_signal_fence_and_flush()
            .expect("Unable to signal fence and flush texture data copy command");

        let sampler = Sampler::simple_repeat_linear(Arc::clone(&vk_device));

        let (swapchain, images) = Self::create_swap_chain(
            &vk_device,
            &vk_surface,
            [
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
                multisample_count,
            );

        let frame_completions = init_frame_completions(&vk_device, n_frames);

        data_copied
            .wait(None)
            .expect("Transfer of application constant data failed");

        App {
            world,
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
            multisample_count,
        }
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
            [dims.0, dims.1]
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
layout(location = 2) in vec2 tex_coords;

layout(location = 1) out vec2 frag_tex_coords;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);
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

layout(location = 1) in vec2 frag_tex_coords;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texSampler, frag_tex_coords);
}
"
    }
}
