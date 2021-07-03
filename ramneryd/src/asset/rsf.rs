use crate::ecs::prelude::*;

use ramneryd_derive::Visitable;
use std::path::{Path, PathBuf};

#[derive(Default, Component)]
struct LoadRsfAsset {
    path: PathBuf,
}

#[derive(Default, Component, Visitable)]
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
            let contents = std::fs::read_to_string(&asset.path)
                .expect("Something went wrong reading the file");

            crate::ecs::serde::from_ron_str(&contents, &mut serde_data)
                .expect("Failed to deserialize rsf");
        }
        load_assets.clear();
    }
}

pub fn register_systems<'a, 'b>(builder: ExecutorBuilder<'a, 'b>) -> ExecutorBuilder<'a, 'b> {
    builder.with(RsfLoader, RsfLoader::ID, &[])
}
