use crate::ecs::prelude::*;
use trekanten::loader::ResourceLoader;
use trekanten::Loader;

#[derive(Component)]
#[component(inspect)]
pub struct GpuMesh {
    pub mesh: trekanten::mesh::Mesh,
    pub polygon_mode: trekanten::pipeline::PolygonMode,
}

impl std::ops::Deref for GpuMesh {
    type Target = trekanten::mesh::Mesh;
    fn deref(&self) -> &Self::Target {
        &self.mesh
    }
}

impl Into<GpuMesh> for trekanten::mesh::Mesh {
    fn into(self) -> GpuMesh {
        GpuMesh {
            mesh: self,
            polygon_mode: trekanten::pipeline::PolygonMode::Fill,
        }
    }
}

// Option is here to handle WriteStorage::remove() not returning the contents
// TODO: Replace Option
#[derive(Component)]
#[component(inspect)]
pub struct PendingMesh(pub Option<GpuMesh>);

impl From<GpuMesh> for PendingMesh {
    fn from(m: GpuMesh) -> Self {
        Self(Some(m))
    }
}

impl PendingMesh {
    pub fn new<T: Into<GpuMesh>>(t: T) -> Self {
        Self(Some(t.into()))
    }
}

struct ResolvePending;
impl ResolvePending {
    const ID: &'static str = "ResolvePendingMesh";
}

impl<'a> System<'a> for ResolvePending {
    type SystemData = (
        WriteStorage<'a, GpuMesh>,
        WriteStorage<'a, PendingMesh>,
        Entities<'a>,
        ReadExpect<'a, Loader>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut materials, mut pending, entities, loader) = data;

        let mut done = specs::BitSet::new();
        for (ent, pend) in (&entities, &pending).join() {
            let d = loader
                .is_done(&pend.0.as_ref().unwrap().vertex_buffer)
                .expect("bad handle")
                && loader
                    .is_done(&pend.0.as_ref().unwrap().index_buffer)
                    .expect("bad handle");
            if d {
                done.add(ent.id());
            }
        }

        for (ent, _) in (&entities, done).join() {
            let mat = pending
                .get_mut(ent)
                .expect("bad bitset")
                .0
                .take()
                .expect("This should be available here");

            materials.insert(ent, mat).unwrap();
            pending.remove(ent);
        }
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(ResolvePending, ResolvePending::ID, &[])
}
