// build.rs

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn find_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut ret = Vec::new();
    for entry in fs::read_dir(dir).expect("Failed to read dir") {
        let entry = entry.expect("Failed to read dir entry");
        let path = entry.path();
        if path.is_dir() {
            ret.extend(find_ext(&path, ext));
        } else {
            ret.push(path.to_path_buf());
        }
    }
    ret
}

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("builtin-shaders");
    if !dest_path.is_dir() {
        if dest_path.exists() {
            std::fs::remove_file(&dest_path).expect(&format!(
                "Failed to remove file (it should be a dir) {}",
                dest_path.display()
            ));
        }
        std::fs::create_dir_all(&dest_path).expect("Failed to create dirs for shaders");
    }
    let shader_sources = Path::new("src/render/shaders/");
    for path in find_ext(&shader_sources, "glsl").into_iter() {
        let dest = dest_path.join(
            path.strip_prefix(&shader_sources)
                .expect("Failed to relative dir"),
        );
        std::fs::create_dir_all(&dest.parent().expect("No parent"))
            .expect("Failed to created dirs");
        eprintln!("{}", dest.display());
        std::fs::copy(&path, &dest).expect("Failed to copy shader source");
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}
