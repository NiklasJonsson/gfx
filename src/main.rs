use specs::prelude::*;

use std::time::Instant;

use std::path::PathBuf;
use std::sync::Arc;

mod arg_parse;
mod asset;
mod camera;
mod common;
mod game_state;
mod io;
mod render;
mod settings;

use self::asset::AssetDescriptor;
use self::common::*;
use self::render::*;
use crate::arg_parse::Args;

use self::game_state::GameState;

use io::windowing::Event;

#[derive(Debug, PartialEq, Eq)]
enum AppState {
    Focused,
    Unfocused,
}

struct App {
    world: World,
    event_queue: Arc<io::EventQueue>,
    renderer: trekanten::Renderer,
    state: AppState,
}

impl App {
    // FIXME: This lives here only because the lifetime parameters are a pain.
    // The whole App struct would need to be templated if this was included.
    // Maybe this can be solved in another way...
    // TODO: Add a dispatcher with <'static, 'static> to app (or two)
    /// Return two dispatchers, one for systems that run regardless if the game is paused
    /// or not and one that is is not run when paused.
    fn init_dispatchers<'a, 'b>() -> (Dispatcher<'a, 'b>, Dispatcher<'a, 'b>) {
        let control_builder = DispatcherBuilder::new();
        // Input needs to go before as most systems depends on it
        let control_builder = io::input::register_systems(control_builder);
        let control_builder = game_state::register_systems(control_builder);

        let control = control_builder.build();

        let engine_builder = DispatcherBuilder::new();
        let engine_builder = camera::register_systems(engine_builder);
        let engine_builder = settings::register_systems(engine_builder);

        let engine = engine_builder
            .with_barrier()
            .with(
                render_graph::TransformPropagation,
                render_graph::TRANSFORM_PROPAGATION_SYSTEM_ID,
                &[],
            )
            .with(
                render_graph::RenderedBoundingBoxes,
                render_graph::RENDERED_BOUNDING_BOXES_SYSTEM_ID,
                &[render_graph::TRANSFORM_PROPAGATION_SYSTEM_ID],
            )
            .build();

        (control, engine)
    }

    fn setup_resources(&mut self) {
        self.world
            .insert(io::input::CurrentFrameExternalInputs(Vec::new()));
        self.world.insert(ActiveCamera::empty());
        self.world.insert(DeltaTime::zero());
        self.world.insert(render::texture::Textures::default());
    }

    pub fn get_entity_with_marker<C>(w: &World) -> Entity
    where
        C: specs::Component,
    {
        let markers = w.read_storage::<C>();
        let entities = w.read_resource::<specs::world::EntitiesRes>();

        let mut joined = (&entities, &markers).join();
        let item = joined.next();
        assert!(
            joined.next().is_none(),
            "Expected only one entity with marker component!"
        );
        let (ent, _) = item.expect("Expected an entity!");

        ent
    }

    // TODO: Move this
    pub fn entity_has_component<C>(w: &World, e: Entity) -> bool
    where
        C: specs::Component,
    {
        w.read_storage::<C>().get(e).is_some()
    }

    fn populate_world(&mut self, args: &Args) {
        self.setup_resources();

        let cam_entity = Self::get_entity_with_marker::<camera::Camera>(&self.world);
        *self.world.write_resource::<ActiveCamera>() = ActiveCamera::with_entity(cam_entity);

        let desc = AssetDescriptor::Gltf {
            path: args.gltf_path.to_owned(),
        };

        let loaded_asset = asset::load_asset_into(&mut self.world, desc);
        if let Some(path) = &args.scene_out_file {
            // TODO: Base this into print_graph_to_dot?
            match std::fs::File::create(path) {
                Ok(file) => {
                    let result = render_graph::print_graph_to_dot(
                        &self.world,
                        loaded_asset.scene_roots.iter().cloned(),
                        file,
                    );
                    if let Err(e) = result {
                        log::warn!("Failed to write scene graph to file: {}", e);
                    }
                }
                Err(e) => log::warn!(
                    "Unable to write scene graph to {}, because {}",
                    path.display(),
                    e
                ),
            }
        }

        // REFACTOR: Flatten this when support for && and if-let is on stable
        if args.use_scene_camera {
            if let Some(transform) = loaded_asset.camera {
                camera::Camera::set_camera_state(&mut self.world, cam_entity, &transform);
            }
        }

        /* Uncomment for runtime shaders
        let match_material = |mat: &Material| {
            if let Material::GlTFPBR {
                normal_map: Some(_),
                base_color_texture: Some(_),
                metallic_roughness_texture: Some(_),
                ..
            } = mat
            {
                true
            } else {
                false
            }
        };

        for root in loaded_asset.scene_roots.iter() {
            runtime_shaders_for_material(
                &self.world,
                *root,
                "src/render/shaders/pbr_gltf_vert.spv",
                "src/render/shaders/pbr_gltf_frag.spv",
                match_material,
            )
        }
        */
    }

    fn next_event(&self) -> Option<Event> {
        self.event_queue.pop().ok()
    }

    fn post_frame(&mut self) {
        self.world.maintain();

        let mut cur_inputs = self
            .world
            .write_resource::<io::input::CurrentFrameExternalInputs>();
        cur_inputs.0.clear();
    }

    fn run(&mut self, args: arg_parse::Args) {
        let (mut control_systems, mut engine_systems) = Self::init_dispatchers();

        // Register all component types
        self.world.register::<Renderable>();
        self.world.register::<Mesh>();
        self.world.register::<Material>();
        self.world.register::<render_graph::RenderGraphNode>();
        self.world.register::<render_graph::RenderGraphRoot>();
        self.world.register::<render_graph::RenderGraphChild>();
        self.world.register::<camera::Camera>();
        control_systems.setup(&mut self.world);
        engine_systems.setup(&mut self.world);

        // Setup world objects, e.g. camera and model from cmdline
        self.populate_world(&args);

        // Collects events and resolves to AppAction
        let _start_time = Instant::now();
        let mut prev_frame = Instant::now();

        let mut frame_count = 0;

        // Main loop is structured like:
        // 1. Poll events
        // 2. Resolve events
        // 3. Grab/release cursor
        // 4. Acquire swapchain image
        // 5. Wait for previous frame
        // 6. Run logic systems
        // 7. Render
        loop {
            // Update global delta time
            let now = Instant::now();
            let diff = now - prev_frame;
            prev_frame = now;

            *self.world.write_resource::<DeltaTime>() = diff.into();

            match self.next_event() {
                Some(Event::Quit) => return,
                Some(Event::Focus) => self.state = AppState::Focused,
                Some(Event::Unfocus) => {
                    self.state = AppState::Unfocused;
                    *self.world.write_resource::<GameState>() = GameState::Paused;
                }
                Some(Event::Input(input)) => {
                    let mut cur_inputs = self
                        .world
                        .write_resource::<io::input::CurrentFrameExternalInputs>();
                    *cur_inputs = io::input::CurrentFrameExternalInputs(input);
                }
                None => (),
            }

            let running = *self.world.read_resource::<GameState>() == GameState::Running;
            let focused = self.state == AppState::Focused;

            if running {
                assert!(focused, "Can't be running but not be in focus!");
                self.renderer.take_cursor();
            } else {
                self.renderer.release_cursor();
            }

            if !focused {
                continue;
            }

            // Run input manager and escape catcher here
            control_systems.dispatch(&self.world);

            if let GameState::Paused = *self.world.read_resource::<GameState>() {
                continue;
            }

            render::draw_frame(&mut self.world, &mut self.renderer);
            self.post_frame();

            frame_count += 1;
            if let Some(n_frames) = args.run_n_frames {
                assert!(frame_count <= n_frames);
                if frame_count == n_frames {
                    break;
                }
            }
        }
    }

    fn new(renderer: trekanten::Renderer, event_queue: Arc<io::EventQueue>) -> Self {
        App {
            world: World::new(),
            renderer,
            event_queue,
            state: AppState::Focused,
        }
    }
}

fn window_extents(window: &winit::window::Window) -> trekanten::util::Extent2D {
    let winit::dpi::PhysicalSize { width, height } = window.inner_size();
    trekanten::util::Extent2D { width, height }
}

fn main() {
    env_logger::init();
    let args = match arg_parse::parse() {
        None => return,
        Some(args) => args,
    };

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop).expect("Failed to create window");

    let event_queue = Arc::new(io::EventQueue::new());
    let event_queue2 = Arc::clone(&event_queue);

    // Thread runs the app while main takes the event loop
    std::thread::spawn(move || {
        let renderer = match trekanten::Renderer::new(&window, window_extents(&window)) {
            Err(e) => {
                println!("Failed to create renderer: {}", e);
                return;
            },
            Ok(x) => x,
        };
        let mut app = App::new(renderer, event_queue2);
        app.run(args);
    });

    let mut event_manager = io::windowing::EventManager::new();
    event_loop.run(move |winit_event, _, control_flow| {
        // Since this is a separate thread, it is fine to wait
        *control_flow = winit::event_loop::ControlFlow::Wait;

        match event_manager.collect_event(winit_event) {
            io::windowing::EventLoopControl::Done(event) => {
                log::debug!("Sending event on queue: {:?}", event);
                event_queue.push(event)
            }
            io::windowing::EventLoopControl::Continue => (),
        }
    });
}
/*
pub fn init_windowing_and_input_thread(
    vk_instance: Arc<Instance>,
) -> Result<(Arc<VkSurface>, EventQueue), WindowingError> {

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
                windowing::EventLoopControl::Done(event) => {
                    log::debug!("Sending event on queue: {:?}", event);
                    event_queue.push(event)
                }
                windowing::EventLoopControl::Continue => (),
            }
        });
    });

    let vk_surface_result = receiver.recv().expect("Failed to receive");
    vk_surface_result.map(|r| (r, event_queue))
}
*/
