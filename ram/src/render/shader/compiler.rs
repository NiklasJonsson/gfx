use thiserror::Error;

use std::borrow::Borrow;
use std::path::{Path, PathBuf};

use super::ShaderAbsPath;

#[derive(Debug, Clone, Default)]
pub struct Defines {
    vals: Vec<(String, String)>,
}

impl Defines {
    pub fn push(&mut self, v: (String, String)) {
        self.vals.push(v);
    }

    pub fn empty() -> Self {
        Self { vals: Vec::new() }
    }
}

impl<'a> IntoIterator for &'a Defines {
    type IntoIter = std::slice::Iter<'a, (String, String)>;
    type Item = &'a (String, String);
    fn into_iter(self) -> Self::IntoIter {
        self.vals.as_slice().iter()
    }
}

#[derive(Clone)]
pub struct SpvBinary {
    data: Vec<u32>,
}

impl SpvBinary {
    pub fn data(self) -> Vec<u32> {
        self.data
    }
}

pub struct ShaderCompiler {
    compiler: shaderc::Compiler,
    shader_paths: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
}

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
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct ShaderCacheKey {
    source: String,
}

impl ShaderCacheKey {
    fn new(source: String) -> Self {
        Self { source }
    }
}

#[derive(Default)]
pub struct ShaderCache {
    cache: std::collections::HashMap<ShaderCacheKey, SpvBinary>,
}

impl ShaderCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: ShaderCacheKey, binary: SpvBinary) {
        self.cache.insert(key, binary);
    }
}

fn log_compilation(defines: &Defines, loc: &ShaderLocation, ty: ShaderType) {
    log::trace!("Compiling {loc} as {ty:?}");
    log::trace!("With defines:");
    for d in defines {
        log::trace!("{} = {}", d.0, d.1);
    }
}

#[derive(Debug)]
pub struct FileNotFound {
    pub path: std::path::PathBuf,
}

impl std::fmt::Display for FileNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to find shader file '{}'", self.path.display())
    }
}

fn find_file_in(
    file_relpath: &Path,
    search_directories: &[PathBuf],
) -> Result<std::path::PathBuf, FileNotFound> {
    for search_directory in search_directories {
        log::debug!(
            "Searching for '{}' in '{}'",
            file_relpath.display(),
            search_directory.display()
        );
        let abspath = search_directory.join(file_relpath);
        if abspath.is_file() {
            let abspath = std::fs::canonicalize(abspath).unwrap();
            return Ok(abspath);
        }
    }
    log::debug!("Failed to find {}", file_relpath.display());
    Err(FileNotFound {
        path: file_relpath.to_path_buf(),
    })
}

fn include_callback(
    search_directories: &[PathBuf],
    include_request: &str,
    ty: shaderc::IncludeType,
    include_source: &str,
    _depth: usize,
) -> shaderc::IncludeCallbackResult {
    if ty == shaderc::IncludeType::Relative {
        return Err(format!("Tried to '#include \"{include_request}\" in {include_source} but *relative* imports are not supported (yet)!"));
    }

    match find_file_in(Path::new(include_request), search_directories) {
        Ok(path) => {
            log::debug!(
                "Resolved include {include_request} to {}",
                std::fs::canonicalize(&path)
                    .expect("Failed to canonicalize file path")
                    .display()
            );

            let content = match std::fs::read_to_string(&path) {
                Ok(content) => content,
                Err(e) => {
                    return Err(format!(
                        "Found include file at '{}' but couldn't read it due to {e}",
                        path.display()
                    ));
                }
            };
            let display_path = format!("include/{include_request}");
            Ok(shaderc::ResolvedInclude {
                resolved_name: display_path,
                content,
            })
        }
        Err(FileNotFound { path }) => Err(format!(
            "Failed to find include '{p}', requested from {include_source}",
            p = path.display()
        )),
    }
}

// TODO: no pub
#[derive(Clone, Debug)]
pub enum ShaderLocationContents {
    Absolute(PathBuf),
    // TODO: Remove
    Absolute2(ShaderAbsPath),
    /// Search relative to one of the shader search paths in the shader compiler.
    Search(PathBuf),
}

#[derive(Clone, Debug)]
pub struct ShaderLocation {
    // TODO: Remove pub
    pub contents: ShaderLocationContents,
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

impl From<super::ShaderAbsPath> for ShaderLocation {
    fn from(value: super::ShaderAbsPath) -> Self {
        Self {
            contents: ShaderLocationContents::Absolute2(value),
        }
    }
}

impl std::fmt::Display for ShaderLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.contents {
            ShaderLocationContents::Absolute(p) => write!(f, "{p}", p = p.display()),
            ShaderLocationContents::Absolute2(p) => write!(f, "{p}", p = p.display()),
            ShaderLocationContents::Search(p) => write!(f, "<SHADER_PATH>/{p}", p = p.display()),
        }
    }
}

fn find_shader(
    search_directories: &[PathBuf],
    loc: &ShaderLocation,
) -> Result<std::path::PathBuf, FileNotFound> {
    match &loc.contents {
        ShaderLocationContents::Absolute(path) => {
            assert!(
                path.is_absolute(),
                "Expected {} to be absolute",
                path.display()
            );
            Ok(path.clone())
        }
        ShaderLocationContents::Absolute2(path) => {
            let path: &Path = path.borrow();
            Ok(path.to_path_buf())
        }
        ShaderLocationContents::Search(path) => find_file_in(path, search_directories),
    }
}

struct FoundShader {
    path: PathBuf,
    contents: String,
}

#[derive(Debug)]
enum ReadShaderError {
    NotFound(PathBuf),
    Found(PathBuf, std::io::Error),
}

fn read_shader(
    search_directories: &[PathBuf],
    loc: &ShaderLocation,
) -> Result<FoundShader, ReadShaderError> {
    let path = match find_shader(search_directories, loc) {
        Ok(path) => path,
        Err(FileNotFound { path }) => return Err(ReadShaderError::NotFound(path)),
    };

    let contents =
        std::fs::read_to_string(&path).map_err(|e| ReadShaderError::Found(path.clone(), e))?;
    return Ok(FoundShader { path, contents });
}

impl ShaderCompiler {
    pub fn new() -> Result<Self, CompilerError> {
        let compiler = shaderc::Compiler::new().ok_or(CompilerError::Init)?;
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

    pub fn find(&self, loc: &ShaderLocation) -> Result<ShaderAbsPath, FileNotFound> {
        find_shader(&self.shader_paths, loc).map(ShaderAbsPath::from_abspath)
    }

    pub fn compile(
        &self,
        shader: &ShaderLocation,
        defines: &Defines,
        ty: ShaderType,
        cache: Option<&mut ShaderCache>,
    ) -> Result<SpvBinary, CompilerError> {
        let (path, source) = match read_shader(&self.shader_paths, shader) {
            Ok(FoundShader { path, contents }) => {
                log::debug!("Resolved {shader} to {}", path.display());
                (path, contents)
            }
            Err(ReadShaderError::Found(path, err)) => {
                return Err(CompilerError::FailedToRead { path, err });
            }
            Err(ReadShaderError::NotFound(path)) => {
                return Err(CompilerError::NotFound { path });
            }
        };

        log::debug!("Running preprocess for {shader}");

        let stage = match ty {
            ShaderType::Fragment => shaderc::ShaderKind::Fragment,
            ShaderType::Vertex => shaderc::ShaderKind::Vertex,
        };

        let mut options =
            shaderc::CompileOptions::new().expect("Failed to create compiler options");
        for d in defines {
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
        let compiler = &self.compiler;

        // Caching has to be done after preprocessing as there is no model of dependencies on headers
        // for a shader source file. Therefore, we cannot map a file path to a spirv binary. An include
        // might have changed so even if the file itself has not changed, we need to compile it.
        let source = match compiler.preprocess(&source, &shader.to_string(), "main", Some(&options))
        {
            Ok(artifact) => artifact.as_text(),
            Err(e) => {
                return Err(CompilerError::ShaderC {
                    source: e,
                    loc: shader.clone(),
                    path,
                });
            }
        };

        if let Some(cache) = &cache {
            // TODO: Can we use a non-owning cache key here?
            let key = ShaderCacheKey::new(source.clone());
            if let Some(binary) = cache.cache.get(&key) {
                log::debug!("Hit cache for {shader}");
                return Ok(binary.clone());
            }
        }

        log_compilation(defines, shader, ty);
        let binary_result =
            compiler.compile_into_spirv(&source, stage, &shader.to_string(), "main", None);

        let binary = match binary_result {
            Err(e) => {
                return Err(CompilerError::ShaderC {
                    source: e,
                    loc: shader.clone(),
                    path,
                });
            }
            Ok(bin) => SpvBinary {
                data: Vec::from(bin.as_binary()),
            },
        };

        if let Some(cache) = cache {
            let key = ShaderCacheKey::new(source.clone());
            cache.cache.insert(key, binary.clone());
        }
        Ok(binary)
    }

    pub fn compile2(
        &self,
        shader: &ShaderAbsPath,
        defines: &Defines,
        ty: ShaderType,
    ) -> Result<(SpvBinary, ShaderCacheKey), CompilerError> {
        todo!()
    }
}
