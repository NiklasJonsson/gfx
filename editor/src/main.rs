use ramneryd::ecs::prelude::*;
use ramneryd::{Module, Modules};

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

fn main() {
    let viewer = Box::new(EditorArgs::from_args());
    let modules = Modules(vec![viewer]);
    ramneryd::run(modules);
}
