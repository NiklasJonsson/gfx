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
            std::fs::remove_file(&dest_path).unwrap_or_else(|err| {
                panic!(
                    "Failed to remove file {} (it should be a dir) due to {}",
                    dest_path.display(),
                    err
                )
            });
        }
        std::fs::create_dir_all(&dest_path).expect("Failed to create dirs for shaders");
    }
    let shader_sources = Path::new("src/render/shaders/");
    for path in find_ext(&shader_sources, "glsl").into_iter() {
        let dest = dest_path.join(
            path.strip_prefix(&shader_sources)
                .expect("Failed to relative dir"),
        );
        eprintln!("dest: {}", dest.display());
        let dest_dir = dest
            .parent()
            .expect("Expected parent dir for destination path");
        if dest_dir.is_file() {
            std::fs::remove_file(&dest_dir).unwrap_or_else(|e| {
                panic!(
                    "{} is a file and can't be replaced by dir {}",
                    dest_dir.display(),
                    e
                )
            });
        }
        assert!(
            !dest_dir.exists() || dest_dir.is_dir(),
            "{} exists but is not a dir",
            dest_dir.display()
        );
        match std::fs::create_dir_all(&dest_dir) {
            Ok(_) => (),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => (),
            Err(e) => panic!("failed to create destination directory for shaders {}", e),
        }
        std::fs::copy(&path, &dest).expect("Failed to copy shader source");
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}
