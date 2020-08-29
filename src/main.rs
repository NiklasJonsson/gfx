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

use arg_parse::Args;
use common::*;

use game_state::GameState;

use io::windowing::Event;

#[derive(Debug, PartialEq, Eq)]
enum AppState {
    Focused,
    Unfocused,
}

struct App {
    world: World,
    window: winit::window::Window,
    event_queue: Arc<io::EventQueue>,
    renderer: trekanten::Renderer,
    state: AppState,
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
            /* TODO: TREKANTEN
            .with(
                render_graph::RenderedBoundingBoxes,
                render_graph::RENDERED_BOUNDING_BOXES_SYSTEM_ID,
                &[render_graph::TRANSFORM_PROPAGATION_SYSTEM_ID],
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

        // TODO: Move
        use render::uniform::LightingData;
        use render::uniform::Transforms;
        use trekanten::descriptor::DescriptorSet;
        use trekanten::pipeline::GraphicsPipelineDescriptor;
        use trekanten::resource::ResourceManager;
        use trekanten::uniform::UniformBuffer;
        use trekanten::uniform::UniformBufferDescriptor;
        use trekanten::util;
        use trekanten::vertex::VertexFormat;
        use trekanten::BufferHandle;

        let desc = UniformBufferDescriptor::uninitialized::<LightingData>(1);
        let light_buffer = self.renderer.create_resource(desc).expect("FAIL");
        let light_buffer =
            BufferHandle::<UniformBuffer>::from_typed_buffer::<LightingData>(light_buffer, 0, 1);

        let desc = UniformBufferDescriptor::uninitialized::<Transforms>(1);
        let transforms_buffer = self.renderer.create_resource(desc).expect("FAIL");
        let transforms_buffer =
            BufferHandle::<UniformBuffer>::from_typed_buffer::<Transforms>(transforms_buffer, 0, 1);

        // TODO: Put these in the same set
        let transforms_set =
            DescriptorSet::builder(&mut self.renderer, trekanten::pipeline::ShaderStage::Vertex)
                .add_buffer(&transforms_buffer, 0)
                .build();

        let light_set = DescriptorSet::builder(
            &mut self.renderer,
            trekanten::pipeline::ShaderStage::Fragment,
        )
        .add_buffer(&light_buffer, 0)
        .build();

        let vertex_format = VertexFormat::builder()
            .add_attribute(util::Format::FLOAT3)
            .add_attribute(util::Format::FLOAT3)
            .build();
        let desc = GraphicsPipelineDescriptor {
            vert: PathBuf::from("vs_pbr_base.spv"),
            frag: PathBuf::from("fs_pbr_base.spv"),
            vertex_format,
        };

        let dummy_pipeline = self.renderer.create_resource(desc).expect("FAIL");

        self.world.insert(render::FrameData {
            light_buffer,
            light_set,
            transforms_buffer,
            transforms_set,
            dummy_pipeline,
        });
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
        self.world.register::<render::RenderableMaterial>();
        self.world.register::<render::Mesh>();
        self.world.register::<Material>();
        self.world.register::<render_graph::RenderGraphNode>();
        self.world.register::<render_graph::RenderGraphRoot>();
        self.world.register::<render_graph::RenderGraphChild>();
        self.world.register::<camera::Camera>();
        asset::gltf::register_components(&mut self.world);
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
                self.take_cursor();
            } else {
                self.release_cursor();
            }

            if !focused {
                continue;
            }

            // Run input manager and escape catcher here
            control_systems.dispatch(&self.world);

            if let GameState::Paused = *self.world.read_resource::<GameState>() {
                continue;
            }

            engine_systems.dispatch(&self.world);

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

    fn new(
        renderer: trekanten::Renderer,
        window: winit::window::Window,
        event_queue: Arc<io::EventQueue>,
    ) -> Self {
        App {
            world: World::new(),
            window,
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
    // As we don't keep the join handle, this is detached from us. Still, it will be destroyed when we exit as we are the main thread.
    std::thread::spawn(move || {
        match trekanten::Renderer::new(&window, window_extents(&window)) {
            Ok(renderer) => {
                let mut app = App::new(renderer, window, event_queue2);
                app.run(args);
            }
            Err(e) => log::error!("Failed to create renderer: {}", e),
        }
        log::info!("Runner thread exiting");
    });

    let mut event_manager = io::windowing::EventManager::new();
    event_loop.run(move |winit_event, _, control_flow| {
        // Since this is a separate thread, it is fine to wait
        *control_flow = winit::event_loop::ControlFlow::Wait;

        match event_manager.collect_event(winit_event) {
            io::windowing::EventLoopControl::SendEvent(event) => {
                log::debug!("Sending event on queue: {:?}", event);
                event_queue.push(event)
            }
            io::windowing::EventLoopControl::Continue => (),
            io::windowing::EventLoopControl::Quit => {
                log::info!("Event loop thread received quit");
                log::info!("Sending {:?} on event queue", io::windowing::Event::Quit);
                event_queue.push(io::windowing::Event::Quit);
                log::info!("Event loop thread exiting") * control_flow =
                    winit::event_loop::ControlFlow::Exit;
            }
        }
    });
}
