use crate::ecs;
use crate::ecs::prelude::*;

use std::path::{Path, PathBuf};

#[derive(Default, Component)]
struct LoadRsfAsset {
    path: PathBuf,
}

#[derive(Default, Component)]
#[component(inspect)]
pub struct RsfAsset {
    path: PathBuf,
}

pub fn load_asset(world: &mut World, path: &Path) {
    world
        .create_entity()
        .with(LoadRsfAsset {
            path: PathBuf::from(path),
        })
        .build();
}

struct RsfLoader;

impl RsfLoader {
    pub const ID: &'static str = "RsfLoader";
}
#[derive(SystemData)]
struct LoaderData<'a> {
    load_assets: WriteStorage<'a, LoadRsfAsset>,
    serde_data: crate::ecs::serde::Data<'a>,
}

impl<'a> System<'a> for RsfLoader {
    type SystemData = LoaderData<'a>;
    fn run(&mut self, data: Self::SystemData) {
        let LoaderData {
            mut serde_data,
            mut load_assets,
        } = data;

        for asset in (&load_assets).join() {
            use specs::saveload::DeserializeComponents;
            let contents = std::fs::read_to_string(&asset.path)
                .expect("Something went wrong reading the file");

            let ecs::serde::Data {
                entities,
                markers,
                allocator,
                transforms,
                lights,
                names,
            } = &mut serde_data;
            match ron::Deserializer::from_str(&contents) {
                Ok(mut d) => DeserializeComponents::<crate::ecs::serde::Error, _>::deserialize(
                    &mut (transforms, lights, names),
                    &entities,
                    markers,
                    allocator,
                    &mut d,
                )
                .expect("Failed to deserialize rsf"),
                Err(e) => log::error!(
                    "Failed to deserialize ron file {} due to {}",
                    asset.path.display(),
                    e
                ),
            }
        }
        load_assets.clear();
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(RsfLoader, RsfLoader::ID, &[])
}
