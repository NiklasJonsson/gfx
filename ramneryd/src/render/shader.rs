use std::path::Path;
use std::path::PathBuf;
use thiserror::Error;

use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug, Clone, Default)]
pub struct Defines {
    vals: Vec<(String, String)>,
}

impl Defines {
    pub fn push(&mut self, v: (String, String)) {
        self.vals.push(v);
    }

    pub fn iter(&self) -> impl Iterator<Item = &(String, String)> {
        self.vals.iter()
    }

    pub fn empty() -> Self {
        Self { vals: Vec::new() }
    }
}

pub mod pbr_gltf {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
    pub struct ShaderDefinition {
        pub has_tex_coords: bool,
        pub has_vertex_colors: bool,
        pub has_tangents: bool,
        pub has_base_color_texture: bool,
        pub has_metallic_roughness_texture: bool,
        pub has_normal_map: bool,
    }

    impl ShaderDefinition {
        const fn empty() -> Self {
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

        fn defines(&self) -> Defines {
            let mut defines = Defines::default();

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

    pub fn compile(
        compiler: &ShaderCompiler,
        def: &ShaderDefinition,
    ) -> Result<(SpvBinary, SpvBinary), CompilerError> {
        assert!(def.is_valid());
        let defines = def.defines();
        let vert = compiler.compile(
            &defines,
            Path::new("render/shaders/pbr/vert.glsl"),
            ShaderType::Vertex,
        )?;
        let frag = compiler.compile(
            &defines,
            Path::new("render/shaders/pbr/frag.glsl"),
            ShaderType::Fragment,
        )?;

        Ok((vert, frag))
    }

    pub fn compile_default(
        compiler: &ShaderCompiler,
    ) -> Result<(SpvBinary, SpvBinary), CompilerError> {
        compile(compiler, &ShaderDefinition::empty())
    }
}

pub struct SpvBinary {
    data: Vec<u32>,
    _ty: ShaderType,
}

impl SpvBinary {
    pub fn data(self) -> Vec<u32> {
        self.data
    }
}

pub struct ShaderCompiler {
    compiler: Arc<Mutex<shaderc::Compiler>>,
}

unsafe impl Send for ShaderCompiler {}
unsafe impl Sync for ShaderCompiler {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Vertex,
    Fragment,
}

#[cfg(windows)]
const SHADER_PATH: &str = concat!(env!("OUT_DIR"), "\\builtin-shaders");

#[cfg(not(windows))]
const SHADER_PATH: &str = concat!(env!("OUT_DIR"), "/builtin-shaders");

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("Failed to initialize")]
    Init,
    #[error("Failed to read shader: {0}")]
    IO(#[from] std::io::Error),
    #[error("Failed to compile shader {rel_path} (resolved to {path}), due to: {source}", path=.full_path.display())]
    ShaderC {
        source: shaderc::Error,
        rel_path: String,
        full_path: PathBuf,
    },
    #[error("Compiler mutex has been posioned")]
    Sync,
}

fn log_compilation(defines: &Defines, rel_path: &Path, ty: ShaderType) {
    log::trace!("Compiling {} as {:?}", rel_path.display(), ty);
    log::trace!("With defines:");
    for d in defines.iter() {
        log::trace!("{} = {}", d.0, d.1);
    }
}

fn include_callback(
    include_request: &str,
    ty: shaderc::IncludeType,
    include_source: &str,
    _depth: usize,
) -> shaderc::IncludeCallbackResult {
    let path = PathBuf::from(SHADER_PATH).join("render/shaderlib/include");
    let dir = std::fs::read_dir(path).expect("Failed to read shaderlib dir");

    if ty == shaderc::IncludeType::Relative {
        return Err(format!("Tried to '#include \"{include_request}\" in {include_source} but *relative* imports are not supported (yet)!"));
    }

    for entry in dir {
        match entry {
            Ok(entry) if entry.file_name() == include_request => {
                let path = entry.path();
                let content = match std::fs::read_to_string(&path) {
                    Ok(c) => c,
                    Err(e) => {
                        return Err(format!(
                            "Failed to read from {p} due to {e}",
                            p = path.display()
                        ))
                    }
                };

                let display_path = format!("shaderlib/include/{include_request}");
                return Ok(shaderc::ResolvedInclude {
                    resolved_name: display_path,
                    content,
                });
            }
            Err(e) => log::debug!("Failed to read shaderlib dir entry with error: {e}"),
            _ => (), // Ignore non-matching files
        }
    }

    Err(format!(
        "Failed to find include '{include_request}', requested from {include_source}"
    ))
}

impl ShaderCompiler {
    pub fn new() -> Result<Self, CompilerError> {
        let compiler = Arc::new(Mutex::new(
            shaderc::Compiler::new().ok_or(CompilerError::Init)?,
        ));
        Ok(Self { compiler })
    }

    pub fn compile<P: AsRef<Path>>(
        &self,
        defines: &Defines,
        rel_path: P,
        ty: ShaderType,
    ) -> Result<SpvBinary, CompilerError> {
        let rel_path = rel_path.as_ref();
        let mut options =
            shaderc::CompileOptions::new().expect("Failed to create compiler options");
        for d in defines.iter() {
            options.add_macro_definition(&d.0, Some(&d.1));
        }

        options.set_include_callback(include_callback);

        log_compilation(defines, rel_path, ty);

        let stage = match ty {
            ShaderType::Fragment => shaderc::ShaderKind::Fragment,
            ShaderType::Vertex => shaderc::ShaderKind::Vertex,
        };

        let path = PathBuf::from(SHADER_PATH).join(rel_path);

        let source = std::fs::read_to_string(&path)?;

        let binary_result = self
            .compiler
            .lock()
            .map_err(|_| CompilerError::Sync)?
            .compile_into_spirv(
                &source,
                stage,
                rel_path.to_str().expect("Bad shader path"),
                "main",
                Some(&options),
            );

        match binary_result {
            Err(e) => Err(CompilerError::ShaderC {
                source: e,
                rel_path: rel_path.display().to_string(),
                full_path: path,
            }),
            Ok(bin) => Ok(SpvBinary {
                _ty: ty,
                data: Vec::from(bin.as_binary()),
            }),
        }
    }
}
