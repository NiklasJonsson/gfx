use std::path::PathBuf;
use std::sync::Arc;

/// The absolute path to a shader.
///
/// This is intended to be used in places where the shader path needs to be passed around
/// a lot but references are unwanted, for example over thread boundaries. Internally,
/// it uses reference counting to make it cheap to clone.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ShaderAbsPath(Arc<std::path::Path>);

impl std::borrow::Borrow<std::path::Path> for ShaderAbsPath {
    fn borrow(&self) -> &std::path::Path {
        &self.0
    }
}

impl std::ops::Deref for ShaderAbsPath {
    type Target = std::path::Path;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<std::path::Path> for ShaderAbsPath {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl ShaderAbsPath {
    pub fn from_abspath(abspath: std::path::PathBuf) -> Self {
        assert!(
            abspath.is_absolute(),
            "Expected {p} to be an absolute path to a shader",
            p = abspath.display()
        );
        Self(Arc::from(abspath.as_path()))
    }
}

/// A shader location, either an absolute path or a path that is relative to the shader
/// search directories.
#[derive(Clone, Debug)]
pub enum ShaderLocation {
    /// Absolute path to a shader
    Absolute(ShaderAbsPath),
    /// Search relative to one of the shader search paths in the shader compiler.
    Search(PathBuf),
}

impl ShaderLocation {
    // Create a shader location that is absolute. This will panic if the path is not absolute.
    pub fn abs<P>(p: P) -> Self
    where
        P: Into<PathBuf>,
    {
        let path = ShaderAbsPath::from_abspath(p.into());
        Self::Absolute(path)
    }

    /// Search relative to one of the shader search paths in the shader compiler.
    pub fn search<P>(p: P) -> Self
    where
        P: Into<PathBuf>,
    {
        let pathbuf: PathBuf = p.into();
        Self::Search(pathbuf)
    }
}

impl std::fmt::Display for ShaderLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Self::Absolute(p) => write!(f, "{p}", p = p.display()),
            Self::Search(p) => write!(f, "<SHADER_PATH>/{p}", p = p.display()),
        }
    }
}
