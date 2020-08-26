use std::path::PathBuf;

pub struct Args {
    pub gltf_path: PathBuf,
    pub use_scene_camera: bool,
    pub run_n_frames: Option<usize>,
    pub scene_out_file: Option<PathBuf>,
}

pub fn parse() -> Option<Args> {
let matches = clap::App::new("ramneryd")
        .version("0.1.0")
        .about("Vulkan renderer")
        .arg(
            clap::Arg::with_name("view-gltf")
                .short("-i")
                .long("view-gltf")
                .value_name("GLTF-FILE")
                .help("Reads a gltf file and renders it.")
                .takes_value(true)
                .required(true),
        )
        // TODO: This can only be used if we are passing a scene from the command line
        .arg(
            clap::Arg::with_name("use-scene-camera")
                .long("use-scene-camera")
                .help("Use the camera encoded in e.g. a gltf scene"),
        )
        .arg(
            clap::Arg::with_name("scene-out-file")
                .value_name("FILE")
                .long("scene-graph-to-file")
                .takes_value(true)
                .help(
                    "Write the scene graph of the loaded scene in dot-format to the supplied file",
                ),
        )
        .arg(
            clap::Arg::with_name("run-n-frames")
                .long("run-n-frames")
                .value_name("N")
                .takes_value(true)
                .help("Run only N frames"),
        )
        .get_matches();

    let path = matches.value_of("view-gltf").expect("This is required!");
    let path_buf = PathBuf::from(path);
    if !path_buf.exists() {
        println!("No such path: {}!", path_buf.as_path().display());
        return None;
    }

    let use_scene_camera = matches.is_present("use-scene-camera");

    let run_n_frames = if let Some(s) = matches.value_of("run-n-frames") {
        match s.parse::<usize>() {
            Ok(n) => Some(n),
            Err(e) => {
                println!("Invalid value for run-n-frames: {}", e);
                return None;
            }
        }
    } else {
        None
    };

    let scene_out_file = matches.value_of("scene-out-file").map(PathBuf::from);

    Some (Args {
        gltf_path: path_buf,
        use_scene_camera,
        run_n_frames,
        scene_out_file,
    })
}