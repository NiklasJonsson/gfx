use thiserror::Error;

use std::borrow::Borrow;
use std::path::{Path, PathBuf};

use super::{
    Defines, ShaderAbsPath, ShaderLocation, ShaderLocationContents, ShaderType, SpvBinary,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShaderSource(pub String);

pub struct ShaderCompiler {
    compiler: shaderc::Compiler,
    shader_paths: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
}

#[derive(Debug, Error)]
pub enum CompilerError {
    #[error("Failed to initialize")]
    Init,
    #[error("Failed to read shader '{}'. Error: {}", path.display(), err)]
    FailedToRead { path: PathBuf, err: std::io::Error },
    #[error("Failed to find shader '{p}'", p = path.display())]
    NotFound { path: PathBuf },
    #[error("Failed to compile shader:\n{error}")]
    ShaderC { error: shaderc::Error },
}

pub type CompilerResult<T> = Result<T, CompilerError>;

fn log_compilation(defines: &Defines, path: &ShaderAbsPath, ty: ShaderType) {
    log::trace!("Compiling {p} as {ty:?}", p = path.display());
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

fn find_shader(
    search_directories: &[PathBuf],
    loc: &ShaderLocation,
) -> Result<std::path::PathBuf, FileNotFound> {
    match &loc.0 {
        ShaderLocationContents::Absolute(path) => {
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
        let result = find_shader(&self.shader_paths, loc).map(ShaderAbsPath::from_abspath);
        if let Ok(path) = &result {
            log::debug!("Resolved '{loc}' to '{p}'", p = path.display());
        }
        result
    }

    pub fn compile(
        &self,
        shader: &ShaderLocation,
        defines: &Defines,
        ty: ShaderType,
        cache: Option<&mut super::service::ShaderCache>,
    ) -> CompilerResult<SpvBinary> {
        todo!()
    }

    pub fn compile_source(
        &self,
        source: &ShaderSource,
        ty: ShaderType,
        input_file_name: &str,
    ) -> CompilerResult<SpvBinary> {
        let stage = match ty {
            ShaderType::Fragment => shaderc::ShaderKind::Fragment,
            ShaderType::Vertex => shaderc::ShaderKind::Vertex,
        };

        let binary_result =
            self.compiler
                .compile_into_spirv(&source.0, stage, input_file_name, "main", None);

        let binary = match binary_result {
            Err(e) => {
                return Err(CompilerError::ShaderC { error: e });
            }
            Ok(bin) => SpvBinary {
                data: Vec::from(bin.as_binary()),
            },
        };

        Ok(binary)
    }

    pub fn preprocess(
        &self,
        shader: &ShaderAbsPath,
        defines: &Defines,
    ) -> CompilerResult<ShaderSource> {
        let source = std::fs::read_to_string(shader).map_err(|e| CompilerError::FailedToRead {
            path: shader.to_path_buf(),
            err: e,
        })?;

        log::debug!("Running preprocess for {p}", p = shader.display());

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

        let path_as_str = shader.0.display().to_string();
        let source =
            match compiler.preprocess(&source, &path_as_str.to_string(), "main", Some(&options)) {
                Ok(artifact) => artifact.as_text(),
                Err(e) => {
                    return Err(CompilerError::ShaderC { error: e });
                }
            };
        Ok(ShaderSource(source))
    }
}
