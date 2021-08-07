use ramneryd::math;
use ramneryd::math::Rgba;
use ramneryd::render;

use ramneryd::common::Name;
use ramneryd::ecs::prelude::*;
use ramneryd::Module;

use structopt::StructOpt;

use std::path::PathBuf;

#[derive(Debug, StructOpt)]
#[structopt(name = "editor", about = "ramneryd editor")]
struct EditorArgs {
    #[structopt(parse(from_os_str), name = "gltf-file", long)]
    gltf_files: Vec<PathBuf>,
    #[structopt(parse(from_os_str), name = "rsf-file", long)]
    rsf_files: Vec<PathBuf>,
}

impl Module for EditorArgs {
    fn init(&mut self, world: &mut World) {
        self.gltf_files
            .iter()
            .for_each(|f| ramneryd::asset::gltf::load_asset(world, f));
        self.rsf_files
            .iter()
            .for_each(|f| ramneryd::asset::rsf::load_asset(world, f));
    }
}

struct Spawn {
    spawn_plane: bool,
    spawn_cube: bool,
}

impl Module for Spawn {
    fn init(&mut self, world: &mut World) {
        let plane_side = 1000.0;
        let plane_height = 1.0;
        if self.spawn_plane {
            world
                .create_entity()
                .with(Name::from("Plane"))
                .with(math::Transform::pos(0.0, -plane_height / 2.0, 0.0))
                .with(render::Shape::Box {
                    width: plane_side,
                    height: plane_height,
                    depth: plane_side,
                })
                .with(render::material::PhysicallyBased {
                    base_color_factor: Rgba {
                        r: 0.3,
                        g: 0.3,
                        b: 0.3,
                        a: 1.0,
                    },
                    metallic_factor: 0.0,
                    roughness_factor: 0.7,
                    ..Default::default()
                })
                .build();
        }

        if self.spawn_cube {
            world
                .create_entity()
                .with(Name::from("Cube"))
                .with(math::Transform::pos(0.0, 3.0, 0.0))
                .with(render::Shape::Box {
                    width: 1.0,
                    height: 1.0,
                    depth: 1.0,
                })
                .with(render::material::PhysicallyBased {
                    base_color_factor: Rgba {
                        r: 0.3,
                        g: 0.6,
                        b: 0.3,
                        a: 1.0,
                    },
                    metallic_factor: 0.0,
                    roughness_factor: 0.7,
                    ..Default::default()
                })
                .build();
        }
    }
}

fn main() {
    ramneryd::EngineSpec::builder()
        .with_module(EditorArgs::from_args())
        .with_module(Spawn {
            spawn_plane: false,
            spawn_cube: false,
        })
        .build()
        .run();
}
