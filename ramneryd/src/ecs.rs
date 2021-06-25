pub type World = specs::World;
pub use ramneryd_derive::Component;

pub mod prelude {
    pub use specs::prelude::ResourceId;
    pub use specs::SystemData;
    pub use specs::{DenseVecStorage, HashMapStorage, NullStorage, VecStorage};
    pub use specs::{Entities, Entity};
    pub use specs::{Read, ReadExpect, ReadStorage, Write, WriteExpect, WriteStorage};

    pub use specs::{Builder as _, Join as _, SystemData as _, WorldExt};

    pub use super::Component;
    pub use specs::world::Component;

    pub use specs::storage::StorageEntry;

    pub use super::{Executor, ExecutorBuilder, System, World};
}

pub mod serde {
    use super::prelude::*;
    use specs::saveload::{SimpleMarker, SimpleMarkerAllocator};

    #[derive(Debug)]
    pub enum Error {
        Ron(ron::Error),
    }

    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match *self {
                Self::Ron(ref e) => write!(f, "{}", e),
            }
        }
    }

    impl From<ron::Error> for Error {
        fn from(x: ron::Error) -> Self {
            Self::Ron(x)
        }
    }

    // This cannot be called.
    impl From<std::convert::Infallible> for Error {
        fn from(_: std::convert::Infallible) -> Self {
            unreachable!()
        }
    }

    impl From<specs::error::NoError> for Error {
        fn from(_: specs::error::NoError) -> Self {
            unreachable!()
        }
    }

    pub struct DoSerialize;
    pub type Marker = SimpleMarker<DoSerialize>;
    pub type MarkerAllocator = SimpleMarkerAllocator<DoSerialize>;

    #[derive(SystemData)]
    pub struct Data<'a> {
        pub entities: Entities<'a>,
        pub markers: WriteStorage<'a, Marker>,
        pub allocator: Write<'a, MarkerAllocator>,
        pub transforms: WriteStorage<'a, crate::math::Transform>,
        pub lights: WriteStorage<'a, crate::render::light::Light>,
        pub names: WriteStorage<'a, crate::common::Name>,
    }

    pub fn setup_resources(world: &mut World) {
        world.insert(MarkerAllocator::new())
    }
}

pub type Entity = specs::Entity;

pub fn find_singleton_entity<C>(w: &World) -> Option<Entity>
where
    C: specs::Component,
{
    use specs::Join as _;
    use specs::WorldExt as _;
    let markers = w.read_storage::<C>();
    let entities = w.read_resource::<specs::world::EntitiesRes>();

    let mut joined = (&entities, &markers).join();
    let item = joined.next();

    item.map(|(ent, _)| ent)
}

pub fn get_singleton_entity<C>(w: &World) -> Entity
where
    C: specs::Component,
{
    find_singleton_entity::<C>(w).expect("Expected an entity!")
}

pub trait System<'a> {
    type SystemData: specs::SystemData<'a>;

    fn run(&mut self, data: Self::SystemData);
    fn setup(&mut self, _world: &mut specs::World) {}
}

// Too many lifetimes below, not sure how they work. Mostly taken from specs impl.
// It does compile and run though :)

pub struct SpecsSystem<S>
where
    for<'a> S: System<'a> + Sync,
{
    s: S,
}

impl<S> SpecsSystem<S>
where
    for<'a> S: System<'a> + Sync,
{
    pub fn new(s: S) -> Self {
        Self { s }
    }
}

impl<'a, S> specs::System<'a> for SpecsSystem<S>
where
    for<'b> S: System<'b> + Sync,
{
    type SystemData = <S as System<'a>>::SystemData;

    fn run(&mut self, data: <S as System<'a>>::SystemData) {
        log::trace!("Running {}", std::any::type_name::<S>());
        profiling::scope!(std::any::type_name::<S>());
        self.s.run(data);
    }

    fn setup(&mut self, world: &mut World) {
        use specs::SystemData as _;
        Self::SystemData::setup(world);
        <S as System>::setup(&mut self.s, world);
    }
}

pub struct Executor<'a, 'b> {
    dispatcher: specs::Dispatcher<'a, 'b>,
}

impl<'a, 'b> Executor<'a, 'b> {
    pub fn execute(&mut self, world: &specs::World) {
        self.dispatcher.dispatch(world);
    }

    pub fn setup(&mut self, world: &mut specs::World) {
        self.dispatcher.setup(world);
    }
}

pub struct ExecutorBuilder<'a, 'b> {
    builder: specs::DispatcherBuilder<'a, 'b>,
}

impl<'a, 'b> ExecutorBuilder<'a, 'b> {
    pub fn with<S>(mut self, s: S, id: &str, deps: &[&str]) -> Self
    where
        S: for<'c> System<'c> + Send + 'a + Sync,
    {
        self.builder.add(SpecsSystem::new(s), id, deps);
        self
    }

    pub fn build(self) -> Executor<'a, 'b> {
        Executor {
            dispatcher: self.builder.build(),
        }
    }

    pub fn with_barrier(mut self) -> ExecutorBuilder<'a, 'b> {
        self.builder.add_barrier();
        self
    }

    pub fn new() -> Self {
        Self {
            builder: specs::DispatcherBuilder::new(),
        }
    }
}

pub mod meta {
    use linkme::distributed_slice;

    #[distributed_slice]
    pub static ALL_COMPONENTS: [Component] = [..];

    pub struct Component {
        pub name: &'static str,
        pub size: usize,
        pub has: fn(world: &super::World, ent: super::Entity) -> bool,
        pub register: fn(world: &mut super::World),

        pub inspect: Option<fn(world: &mut super::World, ent: super::Entity, ui: &imgui::Ui<'_>)>,
    }

    pub fn register_all_components(world: &mut super::World) {
        use specs::WorldExt;
        world.register::<super::serde::Marker>();
        for comp in ALL_COMPONENTS {
            (comp.register)(world);
        }
    }
}
