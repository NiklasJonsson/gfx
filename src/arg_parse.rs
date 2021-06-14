use std::path::PathBuf;

pub struct Args {
    pub gltf_path: Option<PathBuf>,
    pub rsf_paths: Vec<PathBuf>,
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
                .takes_value(true),
        )
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
        .arg(
            clap::Arg::with_name("rsf-file")
                .long("rsf-file")
                .takes_value(true)
                .multiple(true)
                .number_of_values(1),
        )
        .get_matches();

    let gltf_path = matches.value_of("view-gltf").map(|x| PathBuf::from(x));

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
    let rsf_paths = matches
        .values_of("rsf-file")
        .map(|x| x.map(PathBuf::from).collect())
        .unwrap_or_default();

    Some(Args {
        rsf_paths,
        gltf_path,
        use_scene_camera,
        run_n_frames,
        scene_out_file,
    })
}
