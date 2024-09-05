mod compiler;
mod recompile;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::{collections::HashSet, hash::Hash};

use bytemuck::Contiguous;
use compiler::FileNotFound;
use trekant::{
    BlendState, DepthTest, GraphicsPipeline, GraphicsPipelineDescriptor, Handle, PipelineError,
    PolygonMode, PrimitiveTopology, RenderPass, Renderer, ShaderDescriptor, TriangleCulling,
    TriangleWinding, VertexFormat,
};

pub use compiler::{
    CompilerError, Defines, ShaderCache, ShaderCompiler, ShaderLocation, ShaderType, SpvBinary,
};

#[derive(Clone, Debug)]
pub struct PipelineId(u32);

pub struct Pipeline;

struct PipelineServiceState {}

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

impl ShaderAbsPath {
    fn from_abspath(abspath: std::path::PathBuf) -> Self {
        assert!(
            abspath.is_absolute(),
            "Expected {p} to be an absolute path to a shader",
            p = abspath.display()
        );
        Self(Arc::from(abspath.as_path()))
    }

    fn as_shader_location(&self) -> ShaderLocation {
        ShaderLocation {
            contents: compiler::ShaderLocationContents::Absolute2(self.clone()),
        }
    }
}

struct ShaderRecompileComms {
    new_shader_tx: Sender<ShaderAbsPath>,
    shader_done_rx: Receiver<(ShaderAbsPath, SpvBinary)>,
}

struct ShaderWatcher {
    files: HashSet<ShaderAbsPath>,
    register_new_file: Receiver<ShaderAbsPath>,
    file_modified_tx: Sender<ShaderAbsPath>,
}

impl ShaderWatcher {
    fn handle_event(&mut self, res: notify::Result<notify::Event>) {
        match res {
            Ok(event) if event.kind.is_modify() || event.kind.is_create() => {
                for path in event.paths {
                    assert!(path.is_absolute());
                    if let Some(shader_path) = self.files.get(path.as_path()) {
                        self.file_modified_tx.send(shader_path.clone());
                    }
                }
            }
            Err(e) => {
                log::error!("Error from file watcher: {e}");
            }
            _ => (),
        }
    }
}

type RecompilerResult = Result<SpvBinary, CompilerError>;

#[derive(Debug, Clone)]
struct ShaderCompilationInfo {
    defines: Defines,
    ty: ShaderType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShaderWorkItemState {
    None,
    Queued,
    InProgress,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShaderWorkInfo {
    state: ShaderWorkItemState,
    request_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RequestIdx(u64);

struct ThreadState {
    work_queue: Vec<(RequestIdx, ShaderAbsPath)>,
    done_shaders: HashMap<RequestIdx, Result<SpvBinary, CompilerError>>,
    shader_compilation_info: HashMap<ShaderAbsPath, ShaderCompilationInfo>,
    shader_cache: ShaderCache,
}

struct ThreadContext {
    compiler: ShaderCompiler,
    has_work: std::sync::Condvar,
    has_done: std::sync::Condvar,
    shared_state: Mutex<ThreadState>,
}

struct ShaderCompilationService {
    thread_context: Arc<ThreadContext>,
    threads: Vec<std::thread::JoinHandle<()>>,
    request_counter: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RequestId {
    start: RequestIdx,
    len: u64,
}

impl ShaderCompilationService {
    fn thread_work(ctx: &ThreadContext) -> RecompilerResult {
        loop {
            let path: ShaderAbsPath;
            let shader_info: ShaderCompilationInfo;
            let request_idx: RequestIdx;

            // Don't do compilation within the lock window
            {
                let state = ctx.shared_state.lock().expect("Mutex poison");
                let mut state = ctx.has_work.wait(state).expect("Mutex poison");
                if state.work_queue.is_empty() {
                    continue;
                }

                let (idx, item) = state.work_queue.remove(0);
                let Some(si) = state.shader_compilation_info.get(&item) else {
                    log::error!("Internal error: 'The shader {p} doesn't have any compilation info stored so it cannot be compiled.", p = item.display());
                    continue;
                };

                // TODO: This is a string allocation/copy clone. Consider changing to Arc<CompilationInfo> to make it read-only and cheap-copy.
                shader_info = si.clone();
                path = item;
                request_idx = idx;
            }

            let result = ctx
                .compiler
                .compile2(&path, &shader_info.defines, shader_info.ty);

            {
                let mut state = ctx.shared_state.lock().expect("Mutex poison");
                match result {
                    Ok((spv, key)) => {
                        state.shader_cache.insert(key, spv.clone());
                        state.done_shaders.insert(request_idx, Ok(spv));
                    }
                    Err(e) => {
                        state.done_shaders.insert(request_idx, Err(e));
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct RecompilerConfig {
    n_threads: std::num::NonZero<u8>,
}

impl ShaderCompilationService {
    pub fn new(shader_compiler: ShaderCompiler, config: RecompilerConfig) -> Self {
        let n_threads = config.n_threads.into_integer();
        let thread_context = Arc::new(ThreadContext {
            compiler: shader_compiler,
            has_work: Condvar::new(),
            has_done: Condvar::new(),
            shared_state: Mutex::new(ThreadState {
                work_queue: Vec::with_capacity(16),
                done_shaders: HashMap::default(),
                shader_compilation_info: HashMap::default(),
                shader_cache: ShaderCache::new(),
            }),
        });
        let mut r = Self {
            thread_context: Arc::clone(&thread_context),
            threads: Vec::with_capacity(n_threads as usize),
            request_counter: AtomicU64::new(0),
        };

        for i in 0..n_threads {
            let ctx = Arc::clone(&thread_context);
            r.threads.push(std::thread::spawn(move || {
                log::info!("Starting shader compilation thread {i}");
                ShaderCompilationService::thread_work(&ctx);
            }));
        }

        r
    }

    pub fn register(
        &self,
        loc: ShaderLocation,
        defines: Defines,
        ty: ShaderType,
    ) -> Result<ShaderAbsPath, compiler::FileNotFound> {
        // TODO: defer file search? But what to use on the caller side to identify the shader?
        let abspath = self.thread_context.compiler.find(&loc)?;
        let mut guard = self
            .thread_context
            .shared_state
            .lock()
            .expect("Mutex poison");
        guard
            .shader_compilation_info
            .insert(abspath.clone(), ShaderCompilationInfo { defines, ty });
        Ok(abspath)
    }

    pub fn queue(&self, paths: &[ShaderAbsPath]) -> RequestId {
        let len = paths
            .len()
            .try_into()
            .expect("The amount of shaders doesn't fit");
        let start = self
            .request_counter
            .fetch_add(len, std::sync::atomic::Ordering::Relaxed);
        let rid = RequestId {
            start: RequestIdx(start),
            len,
        };
        {
            let mut state = self
                .thread_context
                .shared_state
                .lock()
                .expect("Mutex poison");
            state.work_queue.reserve(paths.len());
            for (i, p) in paths.into_iter().enumerate() {
                state
                    .work_queue
                    .push((RequestIdx(start + i as u64), p.clone()));
            }
        }
        self.thread_context.has_work.notify_all();
        rid
    }

    pub fn queue_loc(&self, loc: &ShaderLocation) -> Result<RequestId, FileNotFound> {
        let path = self.thread_context.compiler.find(loc)?;
        let rid = self.queue(&[path]);
        Ok(rid)
    }

    pub fn wait(&self, requests: &[RequestId], results: &mut Vec<RecompilerResult>) {
        assert_eq!(requests.len(), results.len());
        for rid in requests.into_iter() {
            for ridx in rid.start.0..rid.start.0 + rid.len {
                let result = {
                    let mut state = self
                        .thread_context
                        .shared_state
                        .lock()
                        .expect("Mutex poison");
                    state.done_shaders.remove(&RequestIdx(ridx))
                };
                if let Some(result) = result {
                    results.push(result);
                }
            }
        }
    }
}

fn file_notify_callback(
    rc: &ShaderCompilationService,
    event: notify::Result<notify::Event>,
    buf: &mut Vec<ShaderAbsPath>,
) {
    match event {
        Ok(event) if event.kind.is_create() || event.kind.is_modify() => {
            buf.clear();
            for path in event.paths {
                let abspath = ShaderAbsPath::from_abspath(path);
                buf.push(abspath);
            }
            rc.queue(&buf);
        }
        Ok(_) => (),
        Err(e) => {
            log::error!("Got error from file watcher: {e}");
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecreatePipeline {
    pub id: PipelineId,
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
    pub n_threads: std::num::NonZero<u8>,
}

#[derive(Debug)]
pub enum Error {
    Compilation(CompilerError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

pub struct PipelineService {
    shader_service: Arc<ShaderCompilationService>,
    watcher: Option<Box<dyn notify::Watcher>>,
}

struct ShaderPromise {
    value: Arc<Mutex<Result<SpvBinary, Error>>>,
    arrived: Condvar,
}

impl PipelineService {
    pub fn new(config: PipelineServiceConfig) -> Self {
        let shader_service = ShaderCompilationService::new(
            ShaderCompiler::new().unwrap(),
            RecompilerConfig {
                n_threads: config.n_threads,
            },
        );
        let shader_service = Arc::new(shader_service);

        let watcher = if config.live_recompile {
            let mut buf = Vec::new();
            let rc = Arc::clone(&shader_service);
            Some(Box::new(
                notify::recommended_watcher(move |event| {
                    file_notify_callback(&rc, event, &mut buf)
                })
                .expect("Failed to launch shader file watcher"),
            ) as Box<dyn notify::Watcher>)
        } else {
            None
        };

        Self {
            shader_service,
            watcher,
        }
    }

    pub fn flush_recreate(&self) -> impl Iterator<Item = RecreatePipeline> {
        // For each shader that is done,
        // Fetch its graphics pipeline creation info
        // Recreate it and return
        todo!();
        std::iter::empty()
    }

    pub fn queue_recompile(&self, shader: &ShaderLocation) {
        // TODO: Error handling
        // TODO: Cache eviction
        self.shader_service
            .queue_loc(shader)
            .expect("Failed to find path");
    }

    pub fn create(
        &mut self,
        shaders: Shaders,
        settings: PipelineSettings,
    ) -> Result<GraphicsPipelineDescriptor, Error> {
        // We want to:
        // 1. Compile vertex and fragment shaders in parallel, async.
        // 2. Register the shader path so that it can be recompiled.
        // 3. Store information so that the pipeline can be re-created
        // 4. Create the pipeline

        // TODO perf: tmp alloc
        let mut shader_paths = Vec::new();
        let shaders = [
            Some((shaders.vert, ShaderType::Vertex)),
            shaders.frag.map(|f| (f, ShaderType::Fragment)),
        ];
        for shader in shaders.into_iter().flatten() {
            let (Shader { loc, defines, .. }, ty) = shader;

            let abspath = self
                .shader_service
                .register(loc, defines, ty)
                .map_err(|e| Error::Compilation(CompilerError::NotFound { path: e.path }))?;

            let rid = self.shader_service.queue(&[abspath.clone()]);

            if let Some(watcher) = &mut self.watcher {
                watcher.watch(&abspath, notify::RecursiveMode::NonRecursive);
            }

            shader_paths.push(rid);
        }

        // TODO perf: tmp alloc
        let mut results = Vec::with_capacity(shader_paths.len());
        self.shader_service.wait(&shader_paths, &mut results);

        for r in results {}
        todo!()

        // let (vert, frag): (ShaderDescriptor, Option<ShaderDescriptor>) = {
        //     let vert = {
        //         let vert = shaders.vert;
        //         let spv_binary = self
        //             .shader_compiler
        //             .compile(
        //                 &vert.loc,
        //                 &vert.defines,
        //                 vert.ty,
        //                 Some(&mut self.shader_cache),
        //             )
        //             .map_err(Error::Compilation)?;

        //     };

        //     let frag = if let Some(frag) = shaders.frag {
        //         let spv_binary = self
        //             .shader_compiler
        //             .compile(
        //                 &frag.loc,
        //                 &frag.defines,
        //                 frag.ty,
        //                 Some(&mut self.shader_cache),
        //             )
        //             .map_err(Error::Compilation)?;

        //         Some(ShaderDescriptor {
        //             spirv_code: spv_binary.data(),
        //             debug_name: frag.debug_name,
        //         })
        //     } else {
        //         None
        //     };

        //     (vert, frag)
        // };
        // let descriptor = GraphicsPipelineDescriptor {
        //     vert,
        //     frag,
        //     vertex_format: settings.vertex_format,
        //     culling: settings.culling,
        //     winding: settings.winding,
        //     blend_state: settings.blend_state,
        //     depth_testing: settings.depth_testing,
        //     polygon_mode: settings.polygon_mode,
        //     primitive_topology: settings.primitive_topology,
        // };

        // renderer
        //     .create_gfx_pipeline(descriptor, &render_pass)
        //     .map_err(Error::PipelineCreation)
    }
}

// TODO: Move to material.rs?
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
        cache: &mut ShaderCache,
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
            Some(cache),
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
            Some(cache),
        )?;

        Ok((vert, frag))
    }

    pub fn compile_default(
        compiler: &ShaderCompiler,
        cache: &mut ShaderCache,
    ) -> Result<(SpvBinary, SpvBinary), CompilerError> {
        compile(compiler, cache, &ShaderDefinition::empty())
    }
}
