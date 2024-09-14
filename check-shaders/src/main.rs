use ram::render::shader::{Defines, ShaderLocation, ShaderType};

use std::fs::DirEntry;

fn main() {
    let mut shaders = Vec::new();
    let mut queue = Vec::new();

    let cwd = std::env::current_dir().expect("Failed to read CWD");
    let include_path = cwd.join("ram/src/render/shaders/include");

    println!("WORKING DIR: {d}", d = cwd.display());

    let queue_subdirs = |q: &mut Vec<DirEntry>, path| {
        let new = std::fs::read_dir(path)
            .expect("Failed to read shader dir")
            .map(|e| e.expect("Failed to read entry"));
        q.extend(new);
    };

    let mut start = cwd.clone();
    start.push("ram/src/render/shaders/");
    queue_subdirs(&mut queue, start);

    while let Some(entry) = queue.pop() {
        let path = entry.path();
        if path.is_file() {
            let filename = path
                .file_name()
                .expect("File is missing filename")
                .to_str()
                .expect("Failed to convert OsStr to str");
            if path.starts_with(&include_path) {
                println!("Skipping {} as it is a shared header", path.display());
                continue;
            }
            if filename.ends_with("frag.glsl") {
                shaders.push((path, ShaderType::Fragment));
            } else if filename.ends_with("vert.glsl") {
                shaders.push((path, ShaderType::Vertex));
            } else if filename.ends_with(".glsl") {
                println!(
                    "WARNING: {} ends in glsl but can't infer shader type. Skipping.",
                    path.display()
                );
            }
        } else if path.is_dir() {
            queue_subdirs(&mut queue, path);
        }
    }

    let mut shader_compiler = ram::ShaderCompiler::new().expect("Failed to create shader compiler");
    shader_compiler.add_include_path(include_path);

    let count = shaders.len();
    let mut err_count = 0;
    println!("Found {count} shaders");
    for (idx, (shader, ty)) in shaders.into_iter().enumerate() {
        let display = shader.display();
        println!("[{i}/{count}] Checking {display}", i = idx + 1);
        let loc = ShaderLocation::abs(&shader);
        let result = shader_compiler.compile(&loc, &Defines::empty(), ty);

        if let Err(e) = result {
            println!();
            println!("ERROR: Failed to compile shader {display}");
            println!("{e}");
            println!();
            err_count += 1;
        }
    }
    println!("{} shaders failed to compile", err_count);
}
