use std::sync::Arc;

#[macro_use]
mod macros;
pub mod asset;
mod camera;
pub mod common;
pub mod ecs;
mod editor;
mod game_state;
mod graph;
mod io;
pub mod math;
pub mod render;
mod time;

use time::Time;

use game_state::GameState;

use ecs::prelude::*;
use io::event::Event;

#[derive(Debug, PartialEq, Eq)]
enum State {
    Focused,
    Unfocused,
}

#[derive(Debug, PartialEq, Eq)]
enum Action {
    Quit,
    SkipFrame,
    ContinueFrame,
}

struct Engine {
    world: World,
    event_queue: Arc<io::EventQueue>,
    ui: render::ui::UIContext,
    state: State,
    control_systems: ecs::Executor<'static, 'static>,
    engine_systems: ecs::Executor<'static, 'static>,
    renderer: trekanten::Renderer,
}

/* Unused for now
impl Engine {
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

impl Engine {
    fn init_dispatchers<'a, 'b>() -> (Executor<'a, 'b>, Executor<'a, 'b>) {
        let control_builder = ExecutorBuilder::new();
        // Input needs to go before as most systems depends on it
        let control = register_module_systems!(control_builder, io::input, game_state).build();

        let engine_builder = ExecutorBuilder::new();
        let engine = register_module_systems!(engine_builder, asset, camera, render)
            .with_barrier()
            .with(
                graph::TransformPropagation,
                graph::TransformPropagation::ID,
                &[],
            )
            .build();

        (control, engine)
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
    fn post_frame(&mut self) {
        self.world.maintain();
        io::post_frame(&mut self.world);
    }

    #[profiling::function]
    fn pre_frame(&mut self) -> Action {
        self.world.write_resource::<Time>().tick();

        match self.next_event() {
            Some(Event::Quit) => return Action::Quit,
            Some(Event::Focus) => self.state = State::Focused,
            Some(Event::Unfocus) => {
                self.state = State::Unfocused;
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

        let focused = self.state == State::Focused;

        if !focused {
            return Action::SkipFrame;
        }

        self.ui.pre_frame(&self.world);

        Action::ContinueFrame
    }

    #[profiling::function]
    fn run(&mut self) {
        loop {
            profiling::scope!("main_loop");
            match self.pre_frame() {
                Action::Quit => return,
                Action::SkipFrame => continue,
                Action::ContinueFrame => (),
            }

            self.control_systems.execute(&self.world);
            let state = *self.world.read_resource::<GameState>();
            if let GameState::Running = state {
                self.engine_systems.execute(&self.world);
            }
            render::draw_frame(&mut self.world, &mut self.ui, &mut self.renderer);

            self.post_frame();
            profiling::finish_frame!();
        }
    }
}

#[allow(unused_variables)]
pub trait Module: Send {
    fn init(&mut self, world: &mut World) {}
}

pub struct Modules(pub Vec<Box<dyn Module>>);

pub fn run(modules: Modules) -> ! {
    env_logger::init();

    #[cfg(feature = "profile-with-puffin")]
    profiling::puffin::set_scopes_on(true);

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_maximized(true)
        .build(&event_loop)
        .expect("Failed to create window");

    let event_queue_recv = Arc::new(io::EventQueue::new());
    let event_queue_send = Arc::clone(&event_queue_recv);
    let mut renderer = trekanten::Renderer::new(&window, io::window_extents(&window))
        .expect("Failed to create renderer");
    let (send, recv) = std::sync::mpsc::channel();

    // Thread runs the app while main takes the event loop
    // As we don't keep the join handle, this is detached from us. Still, it will be destroyed when we exit as we are the main thread.
    std::thread::Builder::new()
        .name("ramneryd::engine".to_string())
        .spawn(move || {
            profiling::register_thread!("ramneryd::engine");

            let mut world = World::new();
            let (mut control_systems, mut engine_systems) = Engine::init_dispatchers();

            ecs::meta::register_all_components(&mut world);

            world.insert(Time::default());
            ecs::serde::setup_resources(&mut world);

            control_systems.setup(&mut world);
            engine_systems.setup(&mut world);
            io::setup(&mut world, window);
            render::setup_resources(&mut world, &mut renderer);
            let ui_modules = vec![editor::ui_module()];
            let ui = render::ui::UIContext::new(&mut renderer, &mut world, ui_modules);

            for mut m in modules.0.into_iter() {
                m.init(&mut world);
            }

            Engine {
                world,
                ui,
                event_queue: event_queue_recv,
                state: State::Focused,
                control_systems,
                engine_systems,
                renderer,
            }
            .run();

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
            Arc::clone(&event_queue_send),
            &recv,
            winit_event,
            control_flow,
        );
    });
}
