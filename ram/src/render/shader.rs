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
        // TODO: This is not pretty
        use_builtin: bool,
    ) -> Result<(SpvBinary, SpvBinary), CompilerError> {
        assert!(def.is_valid());
        let defines = def.defines();

        let shader_location = |path| {
            if use_builtin {
                ShaderLocation::builtin(path)
            } else {
                let path = PathBuf::from("ram/src").join(path);
                ShaderLocation::CwdRelative(path)
            }
        };

        let vert = compiler.compile(
            &shader_location("render/shaders/pbr/vert.glsl"),
            &defines,
            ShaderType::Vertex,
        )?;
        let frag = compiler.compile(
            &shader_location("render/shaders/pbr/frag.glsl"),
            &defines,
            ShaderType::Fragment,
        )?;

        Ok((vert, frag))
    }

    pub fn compile_default(
        compiler: &ShaderCompiler,
    ) -> Result<(SpvBinary, SpvBinary), CompilerError> {
        compile(compiler, &ShaderDefinition::empty(), true)
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
    #[error("Failed to compile shader {loc} (resolved to {path}), due to: {source}", path=.abspath.display())]
    ShaderC {
        source: shaderc::Error,
        loc: ShaderLocation,
        abspath: PathBuf,
    },
    #[error("Compiler mutex has been posioned")]
    Sync,
}

fn log_compilation(defines: &Defines, loc: &ShaderLocation, ty: ShaderType) {
    log::trace!("Compiling {loc} as {ty:?}");
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

#[derive(Clone, Debug)]
pub enum ShaderLocation {
    /// Compile one of the builtin shaders. The path here is relative to `ram/src`.
    Builtin(PathBuf),
    /// Compile a shader at an absolute path.
    Absolute(PathBuf),
    /// Compile a shader at a path relative to current working directory when calling `compile`.
    CwdRelative(PathBuf),
}

impl ShaderLocation {
    pub fn builtin<P>(p: P) -> Self
    where
        P: Into<PathBuf>,
    {
        Self::Builtin(p.into())
    }

    pub fn abs<P>(p: P) -> Self
    where
        P: Into<PathBuf>,
    {
        let pathbuf: PathBuf = p.into();
        assert!(
            pathbuf.is_absolute(),
            "Tried to create a shader location referring to an absolute path but {p} is not absolute",
            p = pathbuf.display()
        );
        Self::Absolute(pathbuf)
    }

    pub fn cwd<P>(p: P) -> Self
    where
        P: Into<PathBuf>,
    {
        let pathbuf: PathBuf = p.into();
        Self::CwdRelative(pathbuf)
    }
}

impl std::fmt::Display for ShaderLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Absolute(p) => write!(f, "{p}", p = p.display()),
            Self::Builtin(p) => write!(f, "<builtin>/{p}", p = p.display()),
            Self::CwdRelative(p) => write!(f, "$CWD/{p}", p = p.display()),
        }
    }
}

fn find_shader(loc: &ShaderLocation) -> Result<PathBuf, std::io::Error> {
    let out = match loc {
        ShaderLocation::Absolute(path) => path.clone(),
        ShaderLocation::Builtin(path) => PathBuf::from(SHADER_PATH).join(path),
        ShaderLocation::CwdRelative(path) => std::env::current_dir()?.join(path),
    };

    Ok(out)
}

impl ShaderCompiler {
    pub fn new() -> Result<Self, CompilerError> {
        let compiler = Arc::new(Mutex::new(
            shaderc::Compiler::new().ok_or(CompilerError::Init)?,
        ));
        Ok(Self { compiler })
    }

    pub fn compile(
        &self,
        shader: &ShaderLocation,
        defines: &Defines,
        ty: ShaderType,
    ) -> Result<SpvBinary, CompilerError> {
        let mut options =
            shaderc::CompileOptions::new().expect("Failed to create compiler options");
        for d in defines.iter() {
            options.add_macro_definition(&d.0, Some(&d.1));
        }

        options.set_include_callback(include_callback);

        log_compilation(defines, shader, ty);

        let stage = match ty {
            ShaderType::Fragment => shaderc::ShaderKind::Fragment,
            ShaderType::Vertex => shaderc::ShaderKind::Vertex,
        };

        let abspath = find_shader(shader)?;

        if !abspath.is_file() {
            log::error!(
                "Tried to compile shader file that doesn't exist: {p}",
                p = abspath.display()
            );
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Failed to find shader {p}", p = abspath.display()),
            ))?;
        }

        let source = std::fs::read_to_string(&abspath)?;

        let binary_result = self
            .compiler
            .lock()
            .map_err(|_| CompilerError::Sync)?
            .compile_into_spirv(&source, stage, &shader.to_string(), "main", Some(&options));

        match binary_result {
            Err(e) => Err(CompilerError::ShaderC {
                source: e,
                loc: shader.clone(),
                abspath,
            }),
            Ok(bin) => Ok(SpvBinary {
                _ty: ty,
                data: Vec::from(bin.as_binary()),
            }),
        }
    }
}
