use ramneryd::ecs::prelude::*;
use ramneryd::Module;

use structopt::StructOpt;

use std::path::PathBuf;

#[derive(Debug, StructOpt)]
#[structopt(name = "gltf-viewer", about = "view a gltf file")]
struct GltfViewer {
    #[structopt(parse(from_os_str))]
    file: PathBuf,
}

impl Module for GltfViewer {
    fn load(&mut self, world: &mut World) {
        use ramneryd::{
            math::{Quat, Rgb, Transform, Vec3},
            render::Light,
        };

        ramneryd::asset::gltf::load_asset(world, &self.file);

        if false {
            world
                .create_entity()
                .with(Light::Directional {
                    color: Rgb {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                    },
                })
                .with(Transform {
                    position: Vec3 {
                        x: 0.0,
                        y: 100.0,
                        z: 0.0,
                    },
                    rotation: Quat::rotation_from_to_3d(
                        Light::DEFAULT_FACING,
                        Vec3 {
                            x: 0.0,
                            y: -1.0,
                            z: 0.0,
                        },
                    ),
                    ..Default::default()
                })
                .build();
        }

        world
            .create_entity()
            .with(Light::Ambient {
                strength: 0.01,
                color: Rgb {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                },
            })
            .with(Transform {
                position: Vec3 {
                    x: 0.0,
                    y: 100.0,
                    z: 0.0,
                },
                ..Default::default()
            })
            .build();

        let spot = Light::Spot {
            color: Rgb {
                r: 1.0,
                g: 1.0,
                b: 1.0,
            },
            angle: std::f32::consts::FRAC_PI_4,
            range: 5.0,
        };

        for i in 0..5 {
            world
                .create_entity()
                .with(spot.clone())
                .with(Transform {
                    position: Vec3 {
                        x: -i as f32,
                        y: 3.0,
                        z: 0.0,
                    },
                    ..Default::default()
                })
                .build();
        }
    }
}

fn main() {
    ramneryd::Init::new()
        .with_module(GltfViewer::from_args())
        .run();
}
