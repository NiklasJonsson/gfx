use std::sync::Arc;

mod arg_parse;
mod asset;
mod camera;
mod common;
mod ecs;
mod editor;
mod game_state;
mod graph;
mod io;
mod math;
mod profile;
mod render;
mod time;

use arg_parse::Args;
use time::DeltaTime;

use game_state::GameState;

use ecs::prelude::*;
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
    control_systems: ecs::Executor<'static, 'static>,
    engine_systems: ecs::Executor<'static, 'static>,
    frame_count: usize,
    timer: time::Timer,
}

/* Unused for now
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
*/

impl App {
    fn init_dispatchers<'a, 'b>() -> (Executor<'a, 'b>, Executor<'a, 'b>) {
        let control_builder = ExecutorBuilder::new();
        // Input needs to go before as most systems depends on it
        let control_builder = io::input::register_systems(control_builder);
        let control_builder = game_state::register_systems(control_builder);

        let control = control_builder.build();

        let engine_builder = ExecutorBuilder::new();
        let engine_builder = asset::register_systems(engine_builder);
        let engine_builder = camera::register_systems(engine_builder);
        let engine_builder = render::register_systems(engine_builder);

        let engine = engine_builder
            .with_barrier()
            .with(
                graph::TransformPropagation,
                graph::TransformPropagation::ID,
                &[],
            )
            .build();

        (control, engine)
    }

    fn setup_resources(&mut self) {
        self.world.insert(DeltaTime::zero());
    }

    fn populate_world(&mut self, args: &Args) {
        self.setup_resources();
        asset::gltf::load_asset(&mut self.world, &args.gltf_path);
        self.world
            .create_entity()
            .with(render::light::Light::Point {
                color: math::Vec3 {
                    x: 1.0,
                    y: 0.5,
                    z: 1.0,
                },
                range: 5.0,
            })
            .with(math::Transform::pos(math::Vec3 {
                x: 2.0,
                y: 1.0,
                z: 3.0,
            }))
            .build();
        self.world
            .create_entity()
            .with(render::light::Light::Directional {
                color: math::Vec3 {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                },
            })
            .with(math::Transform::identity())
            .build();
    }

    fn next_event(&self) -> Option<Event> {
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

    #[profiling::function]
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

    #[profiling::function]
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

        let focused = self.state == AppState::Focused;

        if !focused {
            return AppAction::SkipFrame;
        }

        self.ui.pre_frame(&self.world);

        AppAction::ContinueFrame
    }

    #[profiling::function]
    fn run(&mut self, args: arg_parse::Args) {
        self.populate_world(&args);
        self.timer.start();

        loop {
            profiling::scope!("main_loop");
            match self.pre_frame() {
                AppAction::Quit => return,
                AppAction::SkipFrame => continue,
                AppAction::ContinueFrame => (),
            }

            self.control_systems.execute(&self.world);
            let state = *self.world.read_resource::<GameState>();
            if let GameState::Running = state {
                self.engine_systems.execute(&self.world);
            }
            render::draw_frame(&mut self.world, &mut self.ui, &mut self.renderer);

            if let AppAction::Quit = self.post_frame(&args) {
                return;
            }

            profiling::finish_frame!();
        }
    }

    #[profiling::function]
    fn new(
        mut renderer: trekanten::Renderer,
        window: winit::window::Window,
        event_queue: Arc<io::EventQueue>,
    ) -> Self {
        let mut world = World::new();
        ecs::meta::register_all_components(&mut world);

        render::setup_resources(&mut world, &mut renderer);

        let (mut control_systems, mut engine_systems) = Self::init_dispatchers();
        control_systems.setup(&mut world);
        engine_systems.setup(&mut world);
        io::setup(&mut world, window);
        profile::setup(&mut world);

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

    #[cfg(feature = "profile-with-puffin")]
    profiling::puffin::set_scopes_on(true);

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
    std::thread::Builder::new()
        .name("ramneryd::engine".to_string())
        .spawn(move || {
            profiling::register_thread!("ramneryd::engine");
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
        })
        .expect("Failed to start engine thread");

    profiling::register_thread!("ramneryd::input");
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
