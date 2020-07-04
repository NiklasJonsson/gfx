use specs::prelude::*;

use std::time::Instant;

use std::path::PathBuf;
use std::sync::Arc;

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

use self::game_state::GameState;

use io::windowing::Event;

#[derive(Debug, PartialEq, Eq)]
enum AppState {
    Focused,
    Unfocused,
}

struct App {
    world: World,
    event_queue: io::EventQueue,
    vk_manager: VKManager,
    state: AppState,
}

struct Args {
    gltf_path: PathBuf,
    use_scene_camera: bool,
    run_n_frames: Option<usize>,
    scene_out_file: Option<PathBuf>,
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

    fn run(&mut self, args: Args) {
        let (mut control_systems, mut engine_systems) = Self::init_dispatchers();

        // Register all component types
        self.world.register::<Renderable>();
        self.world.register::<PolygonMesh>();
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
                self.vk_manager.take_cursor();
            } else {
                self.vk_manager.release_cursor();
            }

            if !focused {
                continue;
            }

            // Run input manager and escape catcher here
            control_systems.dispatch(&self.world);

            if let GameState::Paused = *self.world.read_resource::<GameState>() {
                continue;
            }

            // Acquires next swapchain frame and waits for previous work to the upcoming framebuffer to be finished.
            self.vk_manager.prepare_frame();

            // Run all ECS systems (blocking call)
            engine_systems.dispatch(&self.world);

            // Send data to GPU
            // TODO: Merge this with prepare_primitives?
            self.vk_manager
                .prepare_primitives_for_rendering(&self.world);

            // Run render systems, this is done after the dispatch call to enforce serialization
            self.vk_manager.draw_next_frame(&mut self.world);

            self.world.maintain();

            frame_count += 1;
            if let Some(n_frames) = args.run_n_frames {
                assert!(frame_count <= n_frames);
                if frame_count == n_frames {
                    break;
                }
            }
        }
    }

    fn new() -> Self {
        let vk_instance = render::get_vk_instance();

        let (vk_surface, event_queue) =
            io::init_windowing_and_input_thread(Arc::clone(&vk_instance))
                .expect("Failed to init windowing!");

        let world = World::new();

        let vk_manager = VKManager::create(vk_instance, vk_surface);

        let state = AppState::Focused;

        App {
            world,
            vk_manager,
            event_queue,
            state,
        }
    }
}

fn main() {
    env_logger::init();
    let matches = clap::App::new("ramneryd")
        .version("0.1.0")
        .about("Vulkan renderer")
        .arg(
            clap::Arg::with_name("view-gltf")
                .short("-i")
                .long("view-gltf")
                .value_name("GLTF-FILE")
                .help("Reads a gltf file and renders it.")
                .takes_value(true)
                .required(true),
        )
        // TODO: This can only be used if we are passing a scene from the command line
        .arg(
            clap::Arg::with_name("use-scene-camera")
                .long("use-scene-camera")
                .help("Use the camera encoded in e.g. a gltf scene"),
        )
        .arg(
            clap::Arg::with_name("scene-out-file")
                .value_name("FILE")
                .long("scene-graph-to-file")
                .takes_value(true)
                .help(
                    "Write the scene graph of the loaded scene in dot-format to the supplied file",
                ),
        )
        .arg(
            clap::Arg::with_name("run-n-frames")
                .long("run-n-frames")
                .value_name("N")
                .takes_value(true)
                .help("Run only N frames"),
        )
        .get_matches();

    let path = matches.value_of("view-gltf").expect("This is required!");
    let path_buf = PathBuf::from(path);
    if !path_buf.exists() {
        println!("No such path: {}!", path_buf.as_path().display());
        return;
    }

    let use_scene_camera = matches.is_present("use-scene-camera");

    let run_n_frames = if let Some(s) = matches.value_of("run-n-frames") {
        match s.parse::<usize>() {
            Ok(n) => Some(n),
            Err(e) => {
                println!("Invalid value for run-n-frames: {}", e);
                return;
            }
        }
    } else {
        None
    };

    let scene_out_file = matches.value_of("scene-out-file").map(|p| PathBuf::from(p));

    let args = Args {
        gltf_path: path_buf,
        use_scene_camera,
        run_n_frames,
        scene_out_file,
    };

    let mut app = App::new();

    app.run(args);
}
