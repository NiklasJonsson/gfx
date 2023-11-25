/// specs ecs wrapper to Wrap or alias specs types to not leak specs namespaces throughout code base

// TODO: Improvements to specs:
// - Drain filter for storage
// - Remove lifetime args for dispatcher and dispatcher builder
// - Write<Resource> requires Sync which it doesn't need
// - Decouple system component auto setup and setting up entities for that system.
// - Async? Would be cool for interacting with other external parts
// - Remove the WorldExt trait
// - read_component should return a specific component for an entity. read_storage should return the storage. read_resource should return the resource...

pub type World = specs::World;
pub use ram_derive::Component;
use specs::WorldExt;

pub trait WorldUtil {
    fn has_component<C>(&self, e: specs::Entity) -> bool
    where
        C: specs::Component;
}

pub mod prelude {
    pub use specs::prelude::ResourceId;
    pub use specs::prelude::SystemData as _;
    pub use specs::SystemData;
    pub use specs::{world::EntitiesRes, Entities, Entity};
    pub use specs::{DenseVecStorage, HashMapStorage, NullStorage, VecStorage};
    pub use specs::{Read, ReadExpect, ReadStorage, Write, WriteExpect, WriteStorage};

    pub use super::WorldUtil as _;
    pub use specs::{Builder as _, Join as _, SystemData as _, WorldExt};

    pub use super::Component;
    pub use specs::world::Component;

    pub use specs::storage::StorageEntry;

    pub use super::{Executor, ExecutorBuilder, System, World};
}

impl WorldUtil for specs::World {
    fn has_component<C>(&self, e: specs::Entity) -> bool
    where
        C: specs::Component,
    {
        self.read_storage::<C>().get(e).is_some()
    }
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

    pub fn from_ron_str<'a>(ron: &str, serde_data: &mut Data<'a>) -> Result<(), Error> {
        let Data {
            entities,
            markers,
            allocator,
            transforms,
            lights,
            names,
        } = serde_data;
        let mut ron_deserializer = ron::Deserializer::from_str(ron)?;
        specs::saveload::DeserializeComponents::<crate::ecs::serde::Error, _>::deserialize(
            &mut (transforms, lights, names),
            entities,
            markers,
            allocator,
            &mut ron_deserializer,
        )?;
        markers.clear();

        Ok(())
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
    assert!(joined.next().is_none());

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
    fn setup(&mut self, world: &mut specs::World) {
        use specs::SystemData as _;
        Self::SystemData::setup(world);
    }
}

pub struct SpecsSystem<S>
where
    for<'a> S: System<'a>,
{
    s: S,
}

impl<S> SpecsSystem<S>
where
    for<'a> S: System<'a>,
{
    pub fn new(s: S) -> Self {
        Self { s }
    }
}

impl<'a, S> specs::System<'a> for SpecsSystem<S>
where
    for<'b> S: System<'b>,
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

pub struct Executor {
    dispatcher: specs::Dispatcher<'static, 'static>,
}

impl Executor {
    pub fn execute(&mut self, world: &specs::World) {
        self.dispatcher.dispatch(world);
    }

    pub fn setup(&mut self, world: &mut specs::World) {
        self.dispatcher.setup(world);
    }
}

pub struct ExecutorBuilder {
    builder: specs::DispatcherBuilder<'static, 'static>,
}

impl ExecutorBuilder {
    pub fn with<S>(mut self, s: S, id: &str, deps: &[&str]) -> Self
    where
        S: for<'a> System<'a> + Send + 'static,
    {
        self.builder.add(SpecsSystem::new(s), id, deps);
        self
    }

    pub fn add<S>(&mut self, s: S, id: &str, deps: &[&str])
    where
        S: for<'a> System<'a> + Send + 'static,
    {
        self.builder.add(SpecsSystem::new(s), id, deps);
    }

    pub fn build(self) -> Executor {
        Executor {
            dispatcher: self.builder.build(),
        }
    }

    pub fn with_barrier(mut self) -> ExecutorBuilder {
        self.builder.add_barrier();
        self
    }

    pub fn add_barrier(&mut self) {
        self.builder.add_barrier();
    }

    pub fn new() -> Self {
        Self {
            builder: specs::DispatcherBuilder::new(),
        }
    }
}

impl Default for ExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
