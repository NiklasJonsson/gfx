use specs::prelude::*;
use std::sync::Arc;

mod arg_parse;
mod asset;
mod camera;
mod ecs;
mod editor;
mod game_state;
mod graph;
mod io;
mod math;
mod render;
mod settings;
mod time;

use arg_parse::Args;
use time::DeltaTime;

use game_state::GameState;

use io::event::Event;

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
    event_queue: Arc<io::EventQueue>,
    renderer: trekanten::Renderer,
    ui: render::ui::UIContext,
    state: AppState,
    control_systems: Dispatcher<'static, 'static>,
    engine_systems: Dispatcher<'static, 'static>,
    frame_count: usize,
    timer: time::Timer,
}

impl App {
    fn take_cursor(&mut self) {
        self.cursor_grab(true);
    }

    fn release_cursor(&mut self) {
        self.cursor_grab(false);
    }

    fn cursor_grab(&mut self, cursor_grab: bool) {
        self.world
            .write_resource::<io::MainWindow>()
            .cursor_grab(cursor_grab);
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
        let engine_builder = render::register_systems(engine_builder);

        let engine = engine_builder
            .with_barrier()
            .with(
                graph::TransformPropagation,
                graph::TRANSFORM_PROPAGATION_SYSTEM_ID,
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
        self.world.insert(render::ActiveCamera::empty());
        self.world.insert(DeltaTime::zero());
    }

    fn populate_world(&mut self, args: &Args) {
        self.setup_resources();

        let cam_entity = ecs::get_singleton_entity::<camera::Camera>(&self.world);
        *self.world.write_resource::<render::ActiveCamera>() =
            render::ActiveCamera::with_entity(cam_entity);

        let loaded_asset =
            asset::gltf::load_asset(&mut self.world, &mut self.renderer, &args.gltf_path);

        if let (Some(transform), true) = (loaded_asset.camera, args.use_scene_camera) {
            camera::Camera::set_camera_state(&mut self.world, cam_entity, &transform);
        }
    }

    fn next_event(&self) -> Option<Event> {
        // TODO:
        // * Can we move this to the event thread
        // * Merge mouse deltas?
        let mut all_inputs = Vec::with_capacity(self.event_queue.len());
        while let Ok(event) = self.event_queue.pop() {
            match event {
                Event::Input(mut inputs) => all_inputs.append(&mut inputs),
                e => return Some(e),
            }
        }

        if all_inputs.is_empty() {
            None
        } else {
            Some(Event::Input(all_inputs))
        }
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
            // TODO: Don't ignore resizes
            None | Some(Event::Resize(_)) => (),
        }

        let running = *self.world.read_resource::<GameState>() == GameState::Running;
        let focused = self.state == AppState::Focused;

        if running {
            assert!(focused, "Can't be running but not be in focus!");
            self.take_cursor();
        } else {
            self.release_cursor();
        }

        if !focused {
            return AppAction::SkipFrame;
        }

        self.ui.pre_frame(&self.world);

        AppAction::ContinueFrame
    }

    fn run(&mut self, args: arg_parse::Args) {
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
            }
            render::draw_frame(&mut self.world, &mut self.ui, &mut self.renderer);

            if let AppAction::Quit = self.post_frame(&args) {
                return;
            }
        }
    }

    fn new(
        mut renderer: trekanten::Renderer,
        window: winit::window::Window,
        event_queue: Arc<io::EventQueue>,
    ) -> Self {
        let mut world = World::new();
        render::setup_resources(&mut world, &mut renderer);

        let (mut control_systems, mut engine_systems) = Self::init_dispatchers();
        asset::gltf::register_components(&mut world);
        render::register_components(&mut world);
        control_systems.setup(&mut world);
        engine_systems.setup(&mut world);
        io::setup(&mut world, window);

        let ui = render::ui::UIContext::new(&mut renderer, &mut world);

        App {
            world,
            renderer,
            ui,
            event_queue,
            state: AppState::Focused,
            control_systems,
            engine_systems,
            frame_count: 0,
            timer: time::Timer::default(),
        }
    }
}

fn main() {
    env_logger::init();
    let args = match arg_parse::parse() {
        None => return,
        Some(args) => args,
    };

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_maximized(true)
        .build(&event_loop)
        .expect("Failed to create window");

    let event_queue = Arc::new(io::EventQueue::new());
    let event_queue2 = Arc::clone(&event_queue);

    let (send, recv) = std::sync::mpsc::channel();

    // Thread runs the app while main takes the event loop
    // As we don't keep the join handle, this is detached from us. Still, it will be destroyed when we exit as we are the main thread.
    std::thread::spawn(move || {
        match trekanten::Renderer::new(&window, io::window_extents(&window)) {
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

    let mut event_manager = io::event::EventManager::new();
    event_loop.run(move |winit_event, _, control_flow| {
        io::event::event_thread_work(
            &mut event_manager,
            Arc::clone(&event_queue),
            &recv,
            winit_event,
            control_flow,
        );
    });
}
