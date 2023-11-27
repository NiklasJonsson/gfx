#![allow(clippy::type_complexity)]

// This thing lets us refer to our own crate by its name instead of self. This is useful for making
// derives that refer to this crate work both in this crate and outside. See
// https://github.com/rust-lang/rust/pull/55275
// for more details.
extern crate self as ram;

use std::sync::Arc;

#[macro_use]
mod macros;
pub mod asset;
mod camera;
pub mod common;
pub mod ecs;
mod graph;
mod io;
pub mod math;
mod recipe;
pub mod render;
mod time;
mod ui;
mod visit;

pub use render::debug::DebugRenderer;
pub use render::shader::ShaderCompiler;
pub use time::Time;

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
    ui: render::imgui::UIContext,
    state: State,
    systems: ecs::Executor,
    renderer: trekant::Renderer,
    #[allow(dead_code)]
    spec: Init,
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
    fn finalize_executor(builder: ExecutorBuilder) -> Executor {
        register_module_systems!(builder, render)
            .with_barrier()
            .with(
                graph::TransformPropagation,
                graph::TransformPropagation::ID,
                &[],
            )
            .build()
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
    }

    #[profiling::function]
    fn pre_frame(&mut self) -> Action {
        self.world.write_resource::<Time>().tick();

        match self.next_event() {
            Some(Event::Quit) => return Action::Quit,
            Some(Event::Focus) => self.state = State::Focused,
            Some(Event::Unfocus) => {
                self.state = State::Unfocused;
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

            self.systems.execute(&self.world);
            render::draw_frame(&mut self.world, &mut self.ui, &mut self.renderer);
            self.post_frame();

            profiling::finish_frame!();
        }
    }
}

pub struct ModuleLoader {
    pub world: World,
    executor_builder: ExecutorBuilder,
}

impl ModuleLoader {
    pub fn add_system<S>(&mut self, s: S, id: &str, deps: &[&str])
    where
        S: for<'a> System<'a> + Send + 'static,
    {
        S::SystemData::setup(&mut self.world);
        self.executor_builder.add(s, id, deps);
    }

    pub fn add_barrier(&mut self) {
        self.executor_builder.add_barrier();
    }
}

#[allow(unused_variables)]
pub trait Module: Send {
    fn load(&mut self, loader: &mut ModuleLoader);
}

#[derive(Default)]
struct Modules(Vec<Box<dyn Module>>);

fn default_modules() -> Modules {
    Modules(vec![
        Box::new(io::InputModule),
        Box::new(camera::FPSCamera),
        Box::new(camera::DefaultCamera),
        Box::new(asset::GltfModule),
        Box::new(asset::RsfModule),
    ])
}

pub struct Init {
    modules: Modules,
}

impl Init {
    pub fn with_module(mut self, module: impl Module + 'static) -> Self {
        self.add_module(module);

        self
    }

    pub fn add_module(&mut self, module: impl Module + 'static) {
        self.modules.0.push(Box::new(module));
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn run(self) -> ! {
        run(self);
    }
}

impl Default for Init {
    fn default() -> Self {
        Self {
            modules: default_modules(),
        }
    }
}

fn run(mut spec: Init) -> ! {
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
    let mut renderer = trekant::Renderer::new(&window, io::window_extents(&window))
        .expect("Failed to create renderer");
    let (send, recv) = std::sync::mpsc::channel();

    // Thread runs the app while main takes the event loop
    // As we don't keep the join handle, this is detached from us. Still, it will be destroyed when we exit as we are the main thread.
    std::thread::Builder::new()
        .name("ram::engine".to_string())
        .spawn(move || {
            profiling::register_thread!("ram::engine");

            let mut world = World::new();
            world.insert(Time::default());
            world.insert(crate::io::MainWindow::new(window));
            math::register_components(&mut world);
            render::register_components(&mut world);

            let mut loader = ModuleLoader {
                world,
                executor_builder: ExecutorBuilder::new(),
            };
            for m in spec.modules.0.iter_mut() {
                m.load(&mut loader);
            }

            let ModuleLoader {
                mut world,
                executor_builder,
            } = loader;

            let mut systems = Engine::finalize_executor(executor_builder);
            systems.setup(&mut world);

            ecs::serde::setup_resources(&mut world);
            render::setup_resources(&mut world, &mut renderer);

            let ui_modules = vec![ui::ui_module()];
            let ui = render::imgui::UIContext::new(&mut renderer, &mut world, ui_modules);

            Engine {
                world,
                ui,
                event_queue: event_queue_recv,
                state: State::Focused,
                systems,
                renderer,
                spec,
            }
            .run();

            if let Err(e) = send.send(io::Command::Quit) {
                log::error!("Failed to send quit command to event thread: {}", e);
            }
            log::info!("Runner thread exiting");
        })
        .expect("Failed to start engine thread");

    profiling::register_thread!("ram::input");
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
