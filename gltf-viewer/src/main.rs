use ramneryd::ecs::prelude::*;
use ramneryd::{Module, Modules};

use structopt::StructOpt;

use std::path::PathBuf;

#[derive(Debug, StructOpt)]
#[structopt(name = "gltf-viewer", about = "view a gltf file")]
struct GltfViewer {
    #[structopt(parse(from_os_str))]
    file: PathBuf,
}

impl Module for GltfViewer {
    fn init(&mut self, world: &mut World) {
        use ramneryd::{math::{Transform, Vec3, Quat, Rgb}, render::Light};

        ramneryd::asset::gltf::load_asset(world, &self.file);
        world.create_entity().with(Light::Directional {
            color: Rgb { r: 1.0, g: 1.0, b: 1.0},
        })
        .with(Transform {
            position: Vec3 {x: 0.0, y: 100.0, z: 0.0 },
            rotation: Quat::rotation_from_to_3d(Light::DEFAULT_FACING, Vec3 { x: 0.0, y: -1.0, z: 0.0}),
            ..Default::default()
        })
        .build();

        world.create_entity().with(Light::Ambient {
            strength: 0.05,
            color: Rgb { r: 1.0, g: 1.0, b: 1.0},
        })
        .with(Transform {
            position: Vec3 {x: 0.0, y: 100.0, z: 0.0 },
            ..Default::default()
        })
        .build();
    }
}

fn main() {
    let viewer = Box::new(GltfViewer::from_args());
    let modules = Modules(vec![viewer]);
    ramneryd::run(modules);
}
