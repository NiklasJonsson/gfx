mod compiler;
mod service;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::{atomic::AtomicU64, Arc, Mutex};

use trekant::{
    BlendState, DepthTest, GraphicsPipeline, GraphicsPipelineDescriptor, Handle, PipelineError,
    PolygonMode, PrimitiveTopology, RenderPass, Renderer, ShaderDescriptor, TriangleCulling,
    TriangleWinding, VertexFormat,
};

pub use compiler::{CompilerError, CompilerResult, ShaderCompiler};

use service::{CompiledShader, ShaderCompilationService, ShaderCompilationServiceConfig, UserId};

const ASYNC_USER_ID: UserId = UserId(0);

struct UserIdGenerator {
    counter: AtomicU64,
}

impl UserIdGenerator {
    fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }
    fn next(&self) -> UserId {
        let prev = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = if prev == std::u64::MAX { 1 } else { prev + 1 };
        UserId(id)
    }
}

#[derive(Clone, Debug)]
enum ShaderLocationContents {
    /// Absolute path to a shader
    Absolute(ShaderAbsPath),
    /// Search relative to one of the shader search paths in the shader compiler.
    Search(PathBuf),
}

#[derive(Clone, Debug)]
pub struct ShaderLocation(ShaderLocationContents);

impl ShaderLocation {
    pub fn abs<P>(p: P) -> Self
    where
        P: Into<PathBuf>,
    {
        let path = ShaderAbsPath::from_abspath(p.into());
        Self(ShaderLocationContents::Absolute(path))
    }

    /// Search relative to one of the shader search paths in the shader compiler.
    pub fn search<P>(p: P) -> Self
    where
        P: Into<PathBuf>,
    {
        let pathbuf: PathBuf = p.into();
        Self(ShaderLocationContents::Search(pathbuf))
    }
}

impl std::fmt::Display for ShaderLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            ShaderLocationContents::Absolute(p) => write!(f, "{p}", p = p.display()),
            ShaderLocationContents::Search(p) => write!(f, "<SHADER_PATH>/{p}", p = p.display()),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Vertex,
    Fragment,
}

/// The absolute path to a shader.
///
/// This is intended to be used in places where the shader path needs to be passed around
/// a lot but references are unwanted, for example over thread boundaries. Internally,
/// it uses reference counting to make it cheap to clone.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct ShaderAbsPath(Arc<std::path::Path>);

impl Borrow<std::path::Path> for ShaderAbsPath {
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
    fn from_abspath(abspath: std::path::PathBuf) -> Self {
        assert!(
            abspath.is_absolute(),
            "Expected {p} to be an absolute path to a shader",
            p = abspath.display()
        );
        Self(Arc::from(abspath.as_path()))
    }
}

fn file_notify_callback(service: &ShaderCompilationService, event: notify::Result<notify::Event>) {
    match event {
        Ok(event) if event.kind.is_create() || event.kind.is_modify() => {
            for path in event.paths {
                let loc = ShaderLocation::abs(path);
                if let Err(e) = service.queue(&loc, ASYNC_USER_ID) {
                    log::error!("Got event that {loc} was changed but failed to queue it for recompilation: {e}");
                }
            }
        }
        Ok(_) => (),
        Err(e) => {
            log::error!("Got error from file watcher: {e}");
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecreatePipeline {
    pub pipeline: Handle<GraphicsPipeline>,
    pub descriptor: GraphicsPipelineDescriptor,
    pub render_pass: Handle<RenderPass>,
}

pub struct Shader {
    pub loc: ShaderLocation,
    pub defines: Defines,
    pub debug_name: Option<String>,
}

pub struct Shaders {
    pub vert: Shader,
    pub frag: Option<Shader>,
}

#[derive(Default)]
pub struct PipelineSettings {
    pub vertex_format: VertexFormat,
    pub culling: TriangleCulling,
    pub winding: TriangleWinding,
    pub blend_state: BlendState,
    pub depth_testing: DepthTest,
    pub polygon_mode: PolygonMode,
    pub primitive_topology: PrimitiveTopology,
}

pub struct PipelineServiceConfig {
    pub live_recompile: bool,
    pub n_threads: std::num::NonZero<usize>,
}

#[derive(Debug)]
pub enum Error {
    Compilation(CompilerError),
    PipelineCreation(trekant::PipelineError),
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compilation(e) => write!(f, "Failed to compile: {}", e),
            Self::PipelineCreation(e) => write!(f, "Failed to create pipeline: {}", e),
        }
    }
}

struct PipelineInfo {
    vert: SpvBinary,
    frag: Option<SpvBinary>,
    settings: PipelineSettings,
    render_pass: Handle<RenderPass>,
    cur: Handle<GraphicsPipeline>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ShaderPermutation {
    path: ShaderAbsPath,
    compilation_info: Arc<service::ShaderCompilationInfo>,
}

#[derive(Default)]
struct PipelineServiceState {
    pipelines_infos: resurs::Storage<PipelineInfo>,
    shader_pipelines: HashMap<ShaderPermutation, Vec<Handle<PipelineInfo>>>,
    watcher: Option<Box<dyn notify::Watcher + Send>>,
}

pub struct PipelineService {
    shader_service: Arc<ShaderCompilationService>,
    state: Mutex<PipelineServiceState>,
    id_generator: UserIdGenerator,
}

// TODO testing:
// * Include paths lookup:
// 1. Setup the include paths as: [a/, b/].
// 2. Create two files, a/x/y and b/x/y, with different contents.
// 3. Create a pipeline with a relpath x/y.
// 4. Delete a/x/y
// 5. Queue recompile of x/y
// 6. It should use the contents of b/x/y and the pipeline should be replaced.
// *

/// A service to create graphics pipeline (descriptors).
///
/// It exposes a blocking API with `create()` which is intended to be used
/// the first time a pipeline is created. After that, the async API can be
/// used to recreate the pipeline in case the shaders have changed.
///
/// The async API is made up of two functions, queue_recompile and flush_recreate. The former
/// sends a signal to the service to recompile a particular shader and once that is
/// done, a graphics pipeline descriptor will be created and later flushed with
/// `flush_recreate` which is intended to be called every frame and the returned
/// GraphicsPipelineDescriptor can be used to recreate the graphics pipeline appropriately.
///
impl PipelineService {
    pub fn new(config: PipelineServiceConfig) -> Self {
        let shader_service = ShaderCompilationService::new(
            ShaderCompiler::new().unwrap(),
            ShaderCompilationServiceConfig {
                n_threads: config.n_threads,
            },
        );
        let shader_service = Arc::new(shader_service);

        let watcher = if config.live_recompile {
            let shader_service = Arc::clone(&shader_service);
            Some(Box::new(
                notify::recommended_watcher(move |event| {
                    file_notify_callback(&shader_service, event)
                })
                .expect("Failed to launch shader file watcher"),
            ) as Box<dyn notify::Watcher + Send>)
        } else {
            None
        };

        Self {
            shader_service,
            state: Mutex::new(PipelineServiceState {
                shader_pipelines: HashMap::new(),
                pipelines_infos: resurs::Storage::new(),
                watcher,
            }),
            id_generator: UserIdGenerator::new(),
        }
    }

    pub fn flush(&self, renderer: &mut Renderer) {
        // START HERE:
        // 2. Update the rest of 'ram'
        // 3. Test
        // 4. Cleanup TODOs in this code
        // 5. Consider renaming the module to `pipeline`?

        // This loop will contain both shaders that we got from the watcher and
        // shaders we got from manual queueing. Therefore, they have to only contain
        // ShaderAbsPath.

        let done_shaders: Vec<CompiledShader> = Vec::new();
        let mut state = self.state.lock().unwrap();
        for done_shader in done_shaders {
            let spv = match done_shader.result {
                Err(e) => {
                    log::error!(
                        "Failed to recompile shader '{}' due to:\n{}",
                        done_shader.path.display(),
                        e
                    );
                    continue;
                }
                Ok(spv) => spv,
            };

            let shader_type = done_shader.compilation_info.ty;
            let shader_permutation = ShaderPermutation {
                path: done_shader.path,
                compilation_info: done_shader.compilation_info,
            };

            // TODO perf: tmp alloc
            let pipelines: Vec<Handle<PipelineInfo>> = state
                .shader_pipelines
                .get(&shader_permutation)
                .unwrap()
                .clone();
            for pipeline_handle in pipelines {
                let pipeline_info = state
                    .pipelines_infos
                    .get_mut(&pipeline_handle)
                    .expect("Pipeline info was removed but hash map was not cleaned up");
                if shader_type == ShaderType::Vertex {
                    pipeline_info.vert = spv.clone();
                } else {
                    assert_eq!(shader_type, ShaderType::Fragment);
                    pipeline_info.frag = Some(spv.clone());
                }

                // TODO: Debug name
                let vert = ShaderDescriptor {
                    spirv_code: pipeline_info.vert.clone().data(),
                    debug_name: None,
                };

                let frag: Option<ShaderDescriptor> =
                    pipeline_info.frag.as_ref().map(|x| ShaderDescriptor {
                        spirv_code: x.clone().data(),
                        debug_name: None,
                    });
                let settings = &pipeline_info.settings;
                let descriptor = GraphicsPipelineDescriptor {
                    vert,
                    frag,
                    vertex_format: settings.vertex_format.clone(),
                    culling: settings.culling,
                    winding: settings.winding,
                    blend_state: settings.blend_state,
                    depth_testing: settings.depth_testing,
                    polygon_mode: settings.polygon_mode,
                    primitive_topology: settings.primitive_topology,
                };

                renderer
                    .recreate_gfx_pipeline(descriptor, pipeline_info.render_pass, pipeline_info.cur)
                    .unwrap();
            }
        }
    }

    pub fn queue_recompile(&self, shader: &ShaderLocation) {
        // TODO: Error handling
        self.shader_service.queue(shader, ASYNC_USER_ID).unwrap();
    }

    pub fn create(
        &self,
        shaders: Shaders,
        settings: PipelineSettings,
        render_pass: Handle<trekant::RenderPass>,
        renderer: &mut Renderer,
    ) -> Result<Handle<GraphicsPipeline>, Error> {
        // We want to:
        // 1. Compile vertex and fragment shaders in parallel, async.
        // 2. Register the shader path so that it can be recompiled.
        // 3. TODO: Store information so that the pipeline can be re-created
        // 4. Create the pipeline

        let shaders = [
            Some((shaders.vert, ShaderType::Vertex)),
            shaders.frag.map(|f| (f, ShaderType::Fragment)),
        ];
        let id = self.id_generator.next();
        // TODO: Loop doesn't look to great, try with a lambda
        for shader in shaders.into_iter().flatten() {
            let (Shader { loc, defines, .. }, ty) = shader;

            let abspath = self
                .shader_service
                .add_new_permutation(&loc, defines, ty, id)
                .map_err(|e| Error::Compilation(CompilerError::NotFound { path: e.path }))?;

            let mut state = self.state.lock().unwrap();
            if let Some(watcher) = &mut state.watcher {
                watcher.watch(&abspath, notify::RecursiveMode::NonRecursive);
            }
        }

        // TODO perf: tmp alloc
        let mut results = Vec::with_capacity(2);
        self.shader_service.wait(&[id], &mut results);

        let vert: SpvBinary = results.remove(0).result.map_err(Error::Compilation)?;
        let frag: Option<SpvBinary> = match results.pop() {
            Some(CompiledShader {
                result: Ok(spv), ..
            }) => Some(spv),
            Some(CompiledShader { result: Err(e), .. }) => return Err(Error::Compilation(e)),
            None => None,
        };

        // TODO: Forward debug name
        let vert = ShaderDescriptor {
            spirv_code: vert.data(),
            debug_name: None,
        };

        let frag: Option<ShaderDescriptor> = frag.map(|x| ShaderDescriptor {
            spirv_code: x.data(),
            debug_name: None,
        });
        let descriptor = GraphicsPipelineDescriptor {
            vert,
            frag,
            vertex_format: settings.vertex_format,
            culling: settings.culling,
            winding: settings.winding,
            blend_state: settings.blend_state,
            depth_testing: settings.depth_testing,
            polygon_mode: settings.polygon_mode,
            primitive_topology: settings.primitive_topology,
        };

        let pipeline = renderer
            .create_gfx_pipeline(descriptor, &render_pass)
            .map_err(Error::PipelineCreation)?;
        Ok(pipeline)
    }
}
