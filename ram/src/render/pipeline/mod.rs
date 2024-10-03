mod shader_compiler;
mod shader_service;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use trekant::{
    BlendState, DepthTest, GraphicsPipeline, GraphicsPipelineDescriptor, Handle, PipelineError,
    PolygonMode, PrimitiveTopology, RenderPass, Renderer, ShaderDescriptor, TriangleCulling,
    TriangleWinding, VertexFormat,
};

pub use shader_compiler::{CompilerError, CompilerResult, ShaderCompiler};

use shader_service::{
    CompiledShader, ShaderCompilationInfo, ShaderCompilationService, ShaderCompilationServiceConfig,
};

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

#[derive(Clone, Debug)]
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
pub struct ShaderAbsPath(Arc<std::path::Path>);

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
                if let Err(e) = service.queue(&loc) {
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

#[derive(Debug)]
pub struct Shader {
    pub loc: ShaderLocation,
    pub defines: Defines,
    pub debug_name: Option<String>,
}

#[derive(Debug)]
pub struct Shaders {
    pub vert: Shader,
    pub frag: Option<Shader>,
}

#[derive(Default, Debug, Clone)]
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
    pub shader_compiler: ShaderCompiler,
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

// TODO: Store the PipelineDescriptor instead?
#[derive(Clone)]
struct PipelineInfo {
    descriptor: GraphicsPipelineDescriptor,
    render_pass: Handle<RenderPass>,
    cur: Handle<GraphicsPipeline>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ShaderPermutation {
    path: ShaderAbsPath,
    compilation_info: Arc<shader_service::ShaderCompilationInfo>,
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
}

#[derive(Clone)]
pub struct ShaderStats {
    pub path: ShaderAbsPath,
}

#[derive(Clone)]
pub struct PipelineStats {
    pub vert: ShaderStats,
    pub frag: Option<ShaderStats>,
    // pub settings: PipelineSettings,
    pub handle: Handle<GraphicsPipeline>,
}

#[derive(Clone)]
pub struct Stats {
    pub pipelines: Vec<PipelineStats>,
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
            config.shader_compiler,
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
        }
    }

    pub fn flush(&self, renderer: &mut Renderer) {
        // This loop will contain both shaders that we got from the watcher and
        // shaders we got from manual queueing. Therefore, they have to only contain
        // ShaderAbsPath.

        // TODO perf: tmp alloc
        let mut done_shaders: Vec<CompiledShader> = Vec::new();
        self.shader_service.flush(&mut done_shaders);
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
                let pipeline_info = state.pipelines_infos.get_mut(&pipeline_handle).expect(
                    "Pipeline info was removed but shader permutation map was not cleaned up",
                );
                let shader_desc: &mut ShaderDescriptor;
                if shader_type == ShaderType::Vertex {
                    shader_desc = &mut pipeline_info.descriptor.vert;
                } else {
                    assert_eq!(shader_type, ShaderType::Fragment);
                    shader_desc = pipeline_info.descriptor.frag.as_mut().unwrap();
                }

                shader_desc.spirv_code = spv.clone().data();
                let new_handle = renderer
                    .recreate_gfx_pipeline(
                        pipeline_info.descriptor.clone(),
                        pipeline_info.render_pass,
                        pipeline_info.cur,
                    )
                    .unwrap();
                pipeline_info.cur = new_handle;
            }
        }
    }

    pub fn queue_recompile(&self, shader: &ShaderLocation) {
        // TODO: Error handling
        self.shader_service.queue(shader).unwrap();
    }

    pub fn stats(&self) -> Stats {
        let state = self.state.lock().unwrap();
        let mut stats = Stats {
            pipelines: Vec::with_capacity(state.shader_pipelines.len()),
        };

        let mut map: std::collections::HashMap<
            Handle<PipelineInfo>,
            (Option<ShaderPermutation>, Option<ShaderPermutation>),
        > = HashMap::new();

        crate::imdbg!(state.shader_pipelines.len());
        for (perm, handles) in &state.shader_pipelines {
            for handle in handles {
                if perm.compilation_info.ty == ShaderType::Vertex {
                    map.entry(*handle)
                        .and_modify(|entry| entry.0 = Some(perm.clone()))
                        .or_insert((Some(perm.clone()), None));
                } else {
                    map.entry(*handle)
                        .and_modify(|entry| entry.1 = Some(perm.clone()))
                        .or_insert((None, Some(perm.clone())));
                }
            }
        }

        for (handle, perms) in map {
            let info = state.pipelines_infos.get(&handle).unwrap();
            stats.pipelines.push(PipelineStats {
                vert: ShaderStats {
                    path: perms.0.unwrap().path.clone(),
                },
                frag: perms.1.map(|x| ShaderStats {
                    path: x.path.clone(),
                }),
                handle: info.cur,
            });
        }

        stats.pipelines.sort_by(|a, b| a.handle.cmp(&b.handle));
        stats
    }

    pub fn create(
        &self,
        shaders: Shaders,
        settings: PipelineSettings,
        render_pass: Handle<trekant::RenderPass>,
        renderer: &mut Renderer,
    ) -> Result<Handle<GraphicsPipeline>, Error> {
        log::debug!("Creating pipeline with {:?} {:?}", shaders, settings);
        // TODO: Caching?
        // We want to:
        // 1. Compile vertex and fragment shaders in parallel, async.
        // 2. Register the shader path so that it can be recompiled.
        // 4. Create the pipeline

        #[derive(Debug)]
        struct CompiledShaderInfo {
            path: ShaderAbsPath,
            spv: SpvBinary,
            ci: Arc<ShaderCompilationInfo>,
            debug_name: Option<String>,
        }

        let vert = &shaders.vert;
        let vert_csi: CompiledShaderInfo;
        let frag_csi: Option<CompiledShaderInfo>;
        if let Some(frag) = &shaders.frag {
            log::debug!("Compiling vert and frag shaders");
            let mut results = [None, None];
            self.shader_service.compile_shaders(
                &[
                    (
                        vert.loc.clone(),
                        Arc::new(ShaderCompilationInfo {
                            defines: vert.defines.clone(),
                            ty: ShaderType::Vertex,
                        }),
                    ),
                    (
                        frag.loc.clone(),
                        Arc::new(ShaderCompilationInfo {
                            defines: frag.defines.clone(),
                            ty: ShaderType::Fragment,
                        }),
                    ),
                ],
                &mut results,
            );
            let [compiled_vert, compiled_frag] = results.map(Option::unwrap);
            let vert_spv = compiled_vert.result.map_err(Error::Compilation)?;
            let frag_spv = compiled_frag.result.map_err(Error::Compilation)?;
            vert_csi = CompiledShaderInfo {
                path: compiled_vert.path,
                ci: compiled_vert.compilation_info,
                spv: vert_spv,
                debug_name: vert.debug_name.clone(),
            };
            frag_csi = Some(CompiledShaderInfo {
                path: compiled_frag.path,
                ci: compiled_frag.compilation_info,
                spv: frag_spv,
                debug_name: frag.debug_name.clone(),
            });
        } else {
            log::debug!("Compiling vert shader");
            let mut results = [None];
            self.shader_service.compile_shaders(
                &[(
                    vert.loc.clone(),
                    Arc::new(ShaderCompilationInfo {
                        defines: vert.defines.clone(),
                        ty: ShaderType::Vertex,
                    }),
                )],
                &mut results,
            );
            let [CompiledShader {
                path,
                compilation_info,
                result,
                ..
            }] = results.map(Option::unwrap);
            let spv = result.map_err(Error::Compilation)?;
            vert_csi = CompiledShaderInfo {
                path,
                ci: compilation_info,
                spv,
                debug_name: vert.debug_name.clone(),
            };
            frag_csi = None;
        }

        log::debug!("Done compiling. Got vert {vert_csi:?} and frag {frag_csi:?}");

        let vert = ShaderDescriptor {
            spirv_code: vert_csi.spv.clone().data(),
            debug_name: vert_csi.debug_name.clone(),
        };

        let frag: Option<ShaderDescriptor> = frag_csi.as_ref().map(|x| ShaderDescriptor {
            spirv_code: x.spv.clone().data(),
            debug_name: x.debug_name.clone(),
        });
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

        let pipeline = renderer
            .create_gfx_pipeline(descriptor.clone(), &render_pass)
            .map_err(Error::PipelineCreation)?;

        self.shader_service
            .register_permutation(vert_csi.path.clone(), vert_csi.ci.clone());
        if let Some(frag) = &frag_csi {
            self.shader_service
                .register_permutation(frag.path.clone(), frag.ci.clone());
        }

        {
            let mut state = self.state.lock().unwrap();
            if let Some(watcher) = &mut state.watcher {
                watcher
                    .watch(&vert_csi.path, notify::RecursiveMode::NonRecursive)
                    .unwrap_or_else(|_| {
                        panic!("Failed to watch file: {}", vert_csi.path.display())
                    });
                if let Some(frag_csi) = &frag_csi {
                    watcher
                        .watch(&frag_csi.path, notify::RecursiveMode::NonRecursive)
                        .unwrap_or_else(|_| {
                            panic!("Failed to watch file: {}", frag_csi.path.display())
                        });
                }
            }

            let pipeline_info = state.pipelines_infos.add(PipelineInfo {
                descriptor,
                render_pass,
                cur: pipeline,
            });

            state
                .shader_pipelines
                .entry(ShaderPermutation {
                    path: vert_csi.path.clone(),
                    compilation_info: vert_csi.ci.clone(),
                })
                .or_default()
                .push(pipeline_info);

            if let Some(frag_csi) = frag_csi {
                state
                    .shader_pipelines
                    .entry(ShaderPermutation {
                        path: frag_csi.path.clone(),
                        compilation_info: frag_csi.ci.clone(),
                    })
                    .or_default()
                    .push(pipeline_info);
            }
        }
        log::debug!("Done creating pipeline");

        Ok(pipeline)
    }
}
