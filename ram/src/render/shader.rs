use std::path::{Path, PathBuf};
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
            &ShaderLocation::search(PathBuf::from_iter([
                "render",
                "shaders",
                "pbr",
                "vert.glsl",
            ])),
            &defines,
            ShaderType::Vertex,
        )?;
        let frag = compiler.compile(
            &ShaderLocation::search(PathBuf::from_iter([
                "render",
                "shaders",
                "pbr",
                "frag.glsl",
            ])),
            &defines,
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
    shader_paths: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
}

unsafe impl Send for ShaderCompiler {}
unsafe impl Sync for ShaderCompiler {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Vertex,
    Fragment,
}

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("Failed to initialize")]
    Init,
    #[error("Failed to read shader '{}'. Error: {}", path.display(), err)]
    FailedToRead { path: PathBuf, err: std::io::Error },
    #[error("Failed to find shader '{p}'", p = path.display())]
    NotFound { path: PathBuf },

    #[error("Failed to compile shader {loc} (resolved to {p}), due to: {source}", p=path.display())]
    ShaderC {
        source: shaderc::Error,
        loc: ShaderLocation,
        path: PathBuf,
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

enum FindFileError {
    NotFound(PathBuf),
    Found(PathBuf, std::io::Error),
}

struct FoundFile(PathBuf, String);

fn find_file_in(
    file_relpath: &Path,
    search_directories: &[PathBuf],
) -> Result<FoundFile, FindFileError> {
    for search_directory in search_directories {
        log::debug!(
            "Searching for '{}' in '{}'",
            file_relpath.display(),
            search_directory.display()
        );
        let abspath = search_directory.join(file_relpath);
        if abspath.is_file() {
            let abspath = std::fs::canonicalize(abspath).unwrap();

            match std::fs::read_to_string(&abspath) {
                Ok(c) => return Ok(FoundFile(abspath, c)),
                Err(e) => return Err(FindFileError::Found(abspath, e)),
            }
        }
    }
    log::debug!("Failed to find {}", file_relpath.display());
    Err(FindFileError::NotFound(file_relpath.to_path_buf()))
}

fn include_callback(
    search_paths: &[PathBuf],
    include_request: &str,
    ty: shaderc::IncludeType,
    include_source: &str,
    _depth: usize,
) -> shaderc::IncludeCallbackResult {
    if ty == shaderc::IncludeType::Relative {
        return Err(format!("Tried to '#include \"{include_request}\" in {include_source} but *relative* imports are not supported (yet)!"));
    }

    match find_file_in(Path::new(include_request), search_paths) {
        Ok(FoundFile(path, content)) => {
            log::debug!(
                "Resolved include {include_request} to {}",
                std::fs::canonicalize(path)
                    .expect("Failed to canonicalize file path")
                    .display()
            );

            let display_path = format!("include/{include_request}");
            Ok(shaderc::ResolvedInclude {
                resolved_name: display_path,
                content,
            })
        }
        Err(FindFileError::Found(path, e)) => Err(format!(
            "Found include file at '{}' but couldn't read it due to {e}",
            path.display()
        )),
        Err(FindFileError::NotFound(path)) => Err(format!(
            "Failed to find include '{p}', requested from {include_source}",
            p = path.display()
        )),
    }
}

#[derive(Clone, Debug)]
enum ShaderLocationContents {
    Absolute(PathBuf),
    /// Search relative to one of the shader search paths in the shader compiler.
    Search(PathBuf),
}

#[derive(Clone, Debug)]
pub struct ShaderLocation {
    contents: ShaderLocationContents,
}

impl ShaderLocation {
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
        Self {
            contents: ShaderLocationContents::Absolute(pathbuf),
        }
    }

    /// Search relative to one of the shader search paths in the shader compiler.
    pub fn search<P>(p: P) -> Self
    where
        P: Into<PathBuf>,
    {
        let pathbuf: PathBuf = p.into();
        Self {
            contents: ShaderLocationContents::Search(pathbuf),
        }
    }
}

impl std::fmt::Display for ShaderLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.contents {
            ShaderLocationContents::Absolute(p) => write!(f, "{p}", p = p.display()),
            ShaderLocationContents::Search(p) => write!(f, "<SHADER_PATH>/{p}", p = p.display()),
        }
    }
}

fn read_shader(search_paths: &[PathBuf], loc: &ShaderLocation) -> Result<FoundFile, FindFileError> {
    match &loc.contents {
        ShaderLocationContents::Absolute(path) => {
            assert!(
                path.is_absolute(),
                "Expected {} to be absolute",
                path.display()
            );
            let contents =
                std::fs::read_to_string(path).map_err(|e| FindFileError::Found(path.clone(), e))?;
            Ok(FoundFile(path.clone(), contents))
        }
        ShaderLocationContents::Search(path) => find_file_in(path, search_paths),
    }
}

impl ShaderCompiler {
    pub fn new() -> Result<Self, CompilerError> {
        let compiler = Arc::new(Mutex::new(
            shaderc::Compiler::new().ok_or(CompilerError::Init)?,
        ));
        let shader_paths = Vec::new();
        let include_paths = Vec::new();
        Ok(Self {
            compiler,
            shader_paths,
            include_paths,
        })
    }

    pub fn add_shader_path<P>(&mut self, path: P)
    where
        P: Into<PathBuf>,
    {
        self.shader_paths.insert(0, path.into());
    }

    pub fn add_include_path<P>(&mut self, path: P)
    where
        P: Into<PathBuf>,
    {
        self.include_paths.insert(0, path.into());
    }

    pub fn compile(
        &self,
        shader: &ShaderLocation,
        defines: &Defines,
        ty: ShaderType,
    ) -> Result<SpvBinary, CompilerError> {
        let (path, source) = match read_shader(&self.shader_paths, shader) {
            Ok(FoundFile(path, contents)) => {
                log::debug!("Resolved {shader} to {}", path.display());
                (path, contents)
            }
            Err(FindFileError::Found(path, err)) => {
                return Err(CompilerError::FailedToRead { path, err });
            }
            Err(FindFileError::NotFound(path)) => {
                return Err(CompilerError::NotFound { path });
            }
        };

        log_compilation(defines, shader, ty);

        let stage = match ty {
            ShaderType::Fragment => shaderc::ShaderKind::Fragment,
            ShaderType::Vertex => shaderc::ShaderKind::Vertex,
        };

        let mut options =
            shaderc::CompileOptions::new().expect("Failed to create compiler options");
        for d in defines.iter() {
            options.add_macro_definition(&d.0, Some(&d.1));
        }

        let callback = |include_request: &str,
                        ty: shaderc::IncludeType,
                        include_source: &str,
                        depth: usize|
         -> shaderc::IncludeCallbackResult {
            include_callback(
                &self.include_paths,
                include_request,
                ty,
                include_source,
                depth,
            )
        };

        options.set_include_callback(callback);

        let binary_result = self
            .compiler
            .lock()
            .map_err(|_| CompilerError::Sync)?
            .compile_into_spirv(&source, stage, &shader.to_string(), "main", Some(&options));

        match binary_result {
            Err(e) => Err(CompilerError::ShaderC {
                source: e,
                loc: shader.clone(),
                path,
            }),
            Ok(bin) => Ok(SpvBinary {
                _ty: ty,
                data: Vec::from(bin.as_binary()),
            }),
        }
    }
}
