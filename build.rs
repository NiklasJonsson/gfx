use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

fn print_change_detection_files() {
    let dir_contents = std::fs::read_dir("src/render/shaders").expect("Failed to read shader dir");
    for dir_entry in dir_contents {
        let path = dir_entry.expect("Failed to read file").path();
        let is_glsl = path.extension().map(|x| x == "glsl").unwrap_or(false);
        if is_glsl {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

// TODO: we can reduce the number of combinations if we handle vertex and fragment separataley (vertex needs less)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ShaderDefinition {
    has_tex_coords: bool,
    has_vertex_colors: bool,
    has_tangents: bool,
    has_base_color_texture: bool,
    has_metallic_roughness_texture: bool,
    has_normal_map: bool,
}

impl ShaderDefinition {
    fn empty() -> Self {
        Self {
            has_tex_coords: false,
            has_vertex_colors: false,
            has_tangents: false,
            has_base_color_texture: false,
            has_metallic_roughness_texture: false,
            has_normal_map: false,
        }
    }
    fn iter(&self) -> impl Iterator<Item = bool> {
        use std::iter::once;
        once(self.has_tex_coords)
            .chain(once(self.has_vertex_colors))
            .chain(once(self.has_tangents))
            .chain(once(self.has_base_color_texture))
            .chain(once(self.has_metallic_roughness_texture))
            .chain(once(self.has_normal_map))
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut bool> {
        use std::iter::once;
        once(&mut self.has_tex_coords)
            .chain(once(&mut self.has_vertex_colors))
            .chain(once(&mut self.has_tangents))
            .chain(once(&mut self.has_base_color_texture))
            .chain(once(&mut self.has_metallic_roughness_texture))
            .chain(once(&mut self.has_normal_map))
    }

    fn n_members() -> u64 {
        6
    }

    fn max_bits_value() -> u64 {
        (1 << Self::n_members()) - 1
    }

    fn from_bits(bits: u64) -> Self {
        assert!(bits <= Self::max_bits_value());
        let mut ret = Self::empty();
        for (i, flag) in ret.iter_mut().enumerate() {
            if (bits & (1 << i)) == (1 << i) as u64 {
                *flag = true;
            }
        }

        ret
    }

    /*
    fn as_bits(&self) -> u64 {
        let mut ret = 0;
        for (i, c) in self.iter().enumerate() {
            if c {
                ret |= 1 << i;
            }
        }

        ret
    }
    */

    fn defines(&self) -> Defines {
        let mut defines = Defines { vals: Vec::new() };

        let mut attribute_count = 2; // Positions and normals are assumed to exist

        let all_defines = [
            ("HAS_TEX_COORDS", vec!["TEX_COORDS_LOC"]),
            ("HAS_VERTEX_COLOR", vec!["VCOL_LOC"]),
            ("HAS_TANGENTS", vec!["TAN_LOC", "BITAN_LOC"]),
            ("HAS_BASE_COLOR_TEXTURE", vec![]),
            ("HAS_METALLIC_ROUGHNESS_TEXTURE", vec![]),
            ("HAS_NORMAL_MAP", vec![]),
        ];

        for (_cond, (has_define, loc_defines)) in self
            .iter()
            .zip(all_defines.iter())
            .filter(|(cond, _define)| *cond)
        {
            defines.push((String::from(*has_define), String::from("1")));
            for &loc_define in loc_defines.iter() {
                defines.push((String::from(loc_define), format!("{}", attribute_count)));
                attribute_count += 1;
            }
        }

        defines
    }

    fn filename(&self) -> String {
        let names = ["uv", "vcol", "tan", "bc", "mr", "nm"];
        let it = self
            .iter()
            .zip(names.iter())
            .filter(|&(cond, _name)| cond)
            .map(|(_cond, name)| name);
        itertools::join(it, "_")
    }

    fn vert_filename(&self) -> String {
        String::from("pbr_gltf_vert_") + &self.filename() + ".spv"
    }

    fn frag_filename(&self) -> String {
        String::from("pbr_gltf_frag_") + &self.filename() + ".spv"
    }

    fn is_valid(&self) -> bool {
        let uses_tex = self.has_normal_map
            || self.has_base_color_texture
            || self.has_metallic_roughness_texture;
        if uses_tex && !self.has_tex_coords {
            return false;
        }

        if self.has_normal_map && !self.has_tangents {
            return false;
        }

        true
    }
}

struct Defines {
    vals: Vec<(String, String)>,
}

impl Defines {
    fn push(&mut self, v: (String, String)) {
        self.vals.push(v);
    }

    fn iter(&self) -> impl Iterator<Item = &(String, String)> {
        self.vals.iter()
    }
}

fn base_generated_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("Failed to get variable")).join("generated")
}

fn base_debug_path() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("Failed to get variable")).join("debug")
}

fn write_to_file(path: &Path, data: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).expect("Failed to create dirs");
    let mut file = std::fs::File::create(path).expect("Failed to open dst file");
    file.write_all(data).expect("Failed to write dst file");
}

fn get_shader_binary_path(fname: &str) -> PathBuf {
    base_generated_path()
        .join("precompiled_shaders")
        .join(fname)
}

fn write_shader_binary(result_fname: &str, data: &[u8]) {
    write_to_file(&get_shader_binary_path(result_fname), data);
}

fn write_preprocessed_shader(result_fname: &str, data: &str) {
    let path = base_debug_path()
        .join("preprocessed_shaders")
        .join(result_fname);

    write_to_file(&path, data.as_bytes());
}

fn compile(compiler: &mut shaderc::Compiler, shader: &str, defines: &Defines, result_fname: &str) {
    let mut options = shaderc::CompileOptions::new().expect("Failed to create compiler options");
    for d in defines.iter() {
        options.add_macro_definition(&d.0, Some(&d.1));
    }

    let stage = if shader.contains("vert") {
        shaderc::ShaderKind::Vertex
    } else {
        assert!(shader.contains("frag"));
        shaderc::ShaderKind::Fragment
    };

    let path = PathBuf::new()
        .join("src")
        .join("render")
        .join("shaders")
        .join(shader);
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Failed to read shader source: {}", shader));

    let text_result = compiler
        .preprocess(&source, result_fname, "main", Some(&options))
        .expect("Failed to preprocess");
    write_preprocessed_shader(result_fname, &text_result.as_text());

    let binary_result = compiler
        .compile_into_spirv(&source, stage, result_fname, "main", Some(&options))
        .expect("Failed to compile spirv");

    write_shader_binary(result_fname, binary_result.as_binary_u8());
}

fn all_combinations() -> Vec<ShaderDefinition> {
    (0..ShaderDefinition::max_bits_value())
        .map(ShaderDefinition::from_bits)
        .filter(|def| def.is_valid())
        .collect::<Vec<_>>()
}

// TODO: Use quote lib?
fn write_hashmap(combinations: &[ShaderDefinition], kind: shaderc::ShaderKind) -> String {
    let func_name = match kind {
        shaderc::ShaderKind::Vertex => "vert_shader_mapping",
        shaderc::ShaderKind::Fragment => "frag_shader_mapping",
        _ => unimplemented!("Unsupported shader type"),
    };

    let mut out = String::from("pub fn ");
    out.push_str(func_name);
    out.push_str(
        "() -> ::std::collections::HashMap<ShaderDefinition, &'static str> {
    let mut map = ::std::collections::HashMap::new();\n",
    );

    for comb in combinations.iter() {
        let fname = match kind {
            shaderc::ShaderKind::Vertex => comb.vert_filename(),
            shaderc::ShaderKind::Fragment => comb.frag_filename(),
            _ => unimplemented!("Unsupported shader type"),
        };
        let fname = get_shader_binary_path(&fname);
        out.push_str(
            format!(
                "\tmap.insert(\n\t{:#?},\n\tr#\"{}\"#);\n",
                comb,
                fname.display()
            )
            .as_str(),
        );
    }

    out.push_str("\n\tmap\n}\n\n");

    out
}

fn generate_mapping(combinations: &[ShaderDefinition]) {
    let mut out = format!("// Generated {} combinations\n", combinations.len());
    out.push_str(&write_hashmap(combinations, shaderc::ShaderKind::Vertex));
    out.push_str(&write_hashmap(combinations, shaderc::ShaderKind::Fragment));

    let path = base_generated_path()
        .join("code")
        .join("gltf_pbr_shader_mapping.rs");

    write_to_file(&path, out.as_bytes());
}

fn compile_all_combinations() {
    let mut compiler = shaderc::Compiler::new().expect("Failed to create shaderc");

    let all = all_combinations();
    for comb in all.iter() {
        let defines = comb.defines();
        // TODO: Fix name for preprocessed shader debug output
        let vert_fname = comb.vert_filename();
        let frag_fname = comb.frag_filename();
        compile(&mut compiler, "pbr_gltf_vert.glsl", &defines, &vert_fname);
        compile(&mut compiler, "pbr_gltf_frag.glsl", &defines, &frag_fname);
    }

    generate_mapping(&all);
}

fn main() {
    print_change_detection_files();
    compile_all_combinations();
}
