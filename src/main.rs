#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate num_derive;
#[macro_use]
extern crate specs_derive;
extern crate nalgebra_glm as glm;

use vulkano_win::VkSurfaceBuild;

use winit::{Event, EventsLoop, WindowBuilder, WindowEvent};

use specs::prelude::*;

use std::sync::Arc;

mod asset;
mod camera;
mod common;
mod input;
mod render;

use self::asset::AssetDescriptor;
use self::render::*;

use self::input::{ActionId, InputContext, InputContextPriority, MappedInput};

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

// Use NullStorage and ReadStorage<'a, Self> to only access this system's MappedInput and
// InputContext. Only one game object will have GameStateSwitcher as null storage, the one holding
// the right mapped input and input context.
impl<'a> System<'a> for GameStateSwitcher {
    type SystemData = (
        Write<'a, GameState>,
        WriteStorage<'a, InputContext>,
        WriteStorage<'a, MappedInput>,
        ReadStorage<'a, Self>,
    );

    fn run(&mut self, (mut state, mut contexts, mut inputs, unique_id): Self::SystemData) {
        log::trace!("GameStateSwitcher: run");
        // TODO: Verify that we only get one mapped input? (and one context)
        // We don't need to join!!!
        //
        // use
        // let camera = camera.get(active_cam.0).unwrap();
        // let view_matrix = transform.get(active_cam.0).unwrap().invert();

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

struct App {
    world: World,
    events_loop: EventsLoop,
    vk_manager: VKManager,
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
            (AcceptInput(vec), _) => AcceptInput(vec),
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

        // This is needed since we unpack the event above and then want to store the complete event
        // here. Instead, it might be better to define an enum of events that the input manager
        // should react to.

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
            *active_camera = ActiveCamera::with_entity(cam_entity);
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

        let desc = AssetDescriptor {
            data_file: "models/chalet.obj".to_string(),
            texture_file: "textures/chalet.jpg".to_string(),
        };

        let chalet_asset = asset::load_asset(desc);
        let chalet_renderable = self.vk_manager.prepare_asset_for_rendering(chalet_asset);

        self.world
            .create_entity()
            .with(chalet_renderable)
            .build();
    }

    fn main_loop(&mut self) {
        let mut dispatcher = Self::init_dispatcher();

        // Register all component types
        self.world.register::<Renderable>();
        dispatcher.setup(&mut self.world.res);

        // Setup world objects, e.g. camera and chalet model
        self.populate_world();

        // Collects events and resolves to AppAction
        let mut event_manager = EventManager::new();

        // Main loop is structured like:
        // 1. Poll events
        // 2. Resolve events
        // 3. Grab/release cursor
        // 4. Acquire swapchain image
        // 5. Wait for previous frame
        // 6. Run logic systems
        // 7. Render
        loop {
            self.events_loop
                .poll_events(|event| event_manager.collect_event(event));

            match event_manager.resolve() {
                AppAction::Quit => {
                    return;
                }
                AppAction::IgnoreInput => {
                    continue;
                }
                AppAction::AcceptInput(events) | AppAction::HandleEvents(events) => {
                    let mut cur_events = self.world.write_resource::<CurrentFrameWindowEvents>();
                    *cur_events = CurrentFrameWindowEvents(events);
                }
            }

            let grab_cursor = *self.world.read_resource::<GameState>() == GameState::Running;

            self.vk_manager.grab_cursor(grab_cursor);

            // Acquires next swapchain frame and waits for previous work to the upcoming framebuffer to be finished.

            self.vk_manager.prepare_frame();

            // Run all ECS systems (blocking call)
            dispatcher.dispatch(&self.world.res);

            // Run render systems, this is done after the dispatch call to enforce serialization
            self.vk_manager.draw_next_frame(&mut self.world);
        }
    }

    fn run(&mut self) {
        self.main_loop();
    }

    fn new() -> Self {
        let vk_instance = render::get_vk_instance();

        let events_loop = EventsLoop::new();
        let vk_surface = WindowBuilder::new()
            .build_vk_surface(&events_loop, Arc::clone(&vk_instance))
            .expect("Unable to create window/surface");

        let mut world = World::new();

        world.add_resource(CurrentFrameWindowEvents(Vec::new()));
        world.add_resource(ActiveCamera::empty());
        world.add_resource(GameState::Running);

        input::add_resources(&mut world);

        let vk_manager = VKManager::create(vk_instance, vk_surface);

        App {
            world,
            events_loop,
            vk_manager,
        }
    }
}

fn main() {
    env_logger::init();
    let mut app = App::new();
    app.run();
}
