use specs::prelude::*;
use std::sync::Arc;

mod arg_parse;
mod asset;
mod camera;
mod common;
mod game_state;
mod io;
mod render;
mod settings;
mod time;
mod transform_graph;

use arg_parse::Args;
use common::*;
use time::DeltaTime;

use game_state::GameState;

use io::windowing::Event;

#[derive(Debug, PartialEq, Eq)]
enum AppState {
    Focused,
    Unfocused,
}

#[derive(Debug, PartialEq, Eq)]
enum AppAction {
    Quit,
    SkipFrame,
    ContinueFrame,
}

struct App {
    world: World,
    window: winit::window::Window,
    event_queue: Arc<io::EventQueue>,
    renderer: trekanten::Renderer,
    state: AppState,
    control_systems: Dispatcher<'static, 'static>,
    engine_systems: Dispatcher<'static, 'static>,
    frame_count: usize,
    timer: time::Timer,
}

impl App {
    pub fn take_cursor(&mut self) {
        self.cursor_grab(true);
    }

    pub fn release_cursor(&mut self) {
        self.cursor_grab(false);
    }

    fn cursor_grab(&mut self, cursor_grab: bool) {
        self.window
            .set_cursor_grab(cursor_grab)
            .expect("Unable to grab cursor");
        self.window.set_cursor_visible(!cursor_grab);
    }
}

impl App {
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
                transform_graph::TransformPropagation,
                transform_graph::TRANSFORM_PROPAGATION_SYSTEM_ID,
                &[],
            )
            /* TODO: TREKANTEN
            .with(
                transform_graph::RenderedBoundingBoxes,
                transform_graph::RENDERED_BOUNDING_BOXES_SYSTEM_ID,
                &[transform_graph::TRANSFORM_PROPAGATION_SYSTEM_ID],
            )
            */
            .build();

        (control, engine)
    }

    fn setup_resources(&mut self) {
        self.world
            .insert(io::input::CurrentFrameExternalInputs(Vec::new()));
        self.world.insert(render::ActiveCamera::empty());
        self.world.insert(DeltaTime::zero());
        render::setup_resources(&mut self.world, &mut self.renderer);
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
        *self.world.write_resource::<render::ActiveCamera>() =
            render::ActiveCamera::with_entity(cam_entity);

        let loaded_asset =
            asset::gltf::load_asset(&mut self.world, &mut self.renderer, &args.gltf_path);
        if let Some(path) = &args.scene_out_file {
            // TODO: Base this into print_graph_to_dot?
            match std::fs::File::create(path) {
                Ok(file) => {
                    let result = transform_graph::print_graph_to_dot(
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

        if let (Some(transform), true) = (loaded_asset.camera, args.use_scene_camera) {
            camera::Camera::set_camera_state(&mut self.world, cam_entity, &transform);
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

    fn post_frame(&mut self, args: &arg_parse::Args) -> AppAction {
        self.world.maintain();

        let mut cur_inputs = self
            .world
            .write_resource::<io::input::CurrentFrameExternalInputs>();
        cur_inputs.0.clear();

        self.frame_count += 1;
        if let Some(n_frames) = args.run_n_frames {
            assert!(self.frame_count <= n_frames);
            if self.frame_count == n_frames {
                return AppAction::Quit;
            }
        }

        AppAction::ContinueFrame
    }

    fn pre_frame(&mut self) -> AppAction {
        self.timer.tick();
        *self.world.write_resource::<DeltaTime>() = self.timer.delta();

        match self.next_event() {
            Some(Event::Quit) => return AppAction::Quit,
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
            self.take_cursor();
        } else {
            self.release_cursor();
        }

        if focused {
            AppAction::ContinueFrame
        } else {
            AppAction::SkipFrame
        }
    }

    fn run(&mut self, args: arg_parse::Args) {
        // Setup world objects, e.g. camera and model from cmdline
        self.populate_world(&args);
        self.timer.start();

        loop {
            match self.pre_frame() {
                AppAction::Quit => return,
                AppAction::SkipFrame => continue,
                AppAction::ContinueFrame => (),
            }

            self.control_systems.dispatch(&self.world);
            let state = *self.world.read_resource::<GameState>();
            if let GameState::Running = state {
                self.engine_systems.dispatch(&self.world);
                render::draw_frame(&mut self.world, &mut self.renderer);
            }

            if let AppAction::Quit = self.post_frame(&args) {
                return;
            }
        }
    }

    fn new(
        renderer: trekanten::Renderer,
        window: winit::window::Window,
        event_queue: Arc<io::EventQueue>,
    ) -> Self {
        let mut world = World::new();
        let (mut control_systems, mut engine_systems) = Self::init_dispatchers();
        asset::gltf::register_components(&mut world);
        render::register_components(&mut world);
        control_systems.setup(&mut world);
        engine_systems.setup(&mut world);

        App {
            world,
            window,
            renderer,
            event_queue,
            state: AppState::Focused,
            control_systems,
            engine_systems,
            frame_count: 0,
            timer: time::Timer::default(),
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

    let (send, recv) = std::sync::mpsc::channel();

    // Thread runs the app while main takes the event loop
    // As we don't keep the join handle, this is detached from us. Still, it will be destroyed when we exit as we are the main thread.
    std::thread::spawn(move || {
        match trekanten::Renderer::new(&window, window_extents(&window)) {
            Ok(renderer) => {
                let mut app = App::new(renderer, window, event_queue2);
                app.run(args);
            }
            Err(e) => log::error!("Failed to create renderer: {}", e),
        }
        if let Err(e) = send.send(io::Command::Quit) {
            log::error!("Failed to send quit command to event thread: {}", e);
        }
        log::info!("Runner thread exiting");
    });

    let mut event_manager = io::windowing::EventManager::new();
    event_loop.run(move |winit_event, _, control_flow| {
        io::windowing::event_thread_work(
            &mut event_manager,
            Arc::clone(&event_queue),
            &recv,
            winit_event,
            control_flow,
        );
    });
}
