use ramneryd::render::shader::{Defines, ShaderLocation, ShaderType};

use std::fs::DirEntry;
use std::path::PathBuf;

fn main() {
    let mut shaders = Vec::new();
    let mut queue = Vec::new();

    let cwd = std::env::current_dir().expect("Failed to read CWD");

    println!("WORKING DIR: {d}", d = cwd.display());

    let queue_dirs = |q: &mut Vec<DirEntry>, path| {
        let new = std::fs::read_dir(path)
            .expect("Failed to read shader dir")
            .map(|e| e.expect("Failed to read entry"));
        q.extend(new);
    };

    queue_dirs(&mut queue, PathBuf::from("ramneryd/src/render/shaders/"));

    while let Some(entry) = queue.pop() {
        let path = entry.path();
        if path.is_file() {
            let filename = path
                .file_name()
                .expect("File is missing filename")
                .to_str()
                .expect("Failed to convert OsStr to str");
            if filename.ends_with("frag.glsl") {
                shaders.push((path, ShaderType::Fragment));
            } else if path.ends_with("vert.glsl") {
                shaders.push((path, ShaderType::Vertex));
            }
        } else if path.is_dir() {
            queue_dirs(&mut queue, path);
        }
    }

    let shader_compiler =
        ramneryd::ShaderCompiler::new().expect("Failed to create shader compiler");

    let count = shaders.len();
    println!("Found {count} shaders");
    for (idx, (shader, ty)) in shaders.into_iter().enumerate() {
        let display = shader.display();
        println!("[{i}/{count}] Checking {display}", i = idx + 1);
        let loc = ShaderLocation::abs(&shader);
        let result = shader_compiler.compile(&loc, &Defines::empty(), ty);

        match result {
            Err(e) => {
                println!("");
                println!("ERROR: Failed to compile shader {display}");
                println!("{e}");
                println!("");
            }
            Ok(_) => (),
        }
    }
}
