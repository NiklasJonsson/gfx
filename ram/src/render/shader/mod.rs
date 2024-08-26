mod compiler;
mod recompile;

use std::borrow::Borrow;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::{collections::HashSet, hash::Hash};

use resurs::Handle;

use trekant::{
    BlendState, DepthTest, GraphicsPipeline, GraphicsPipelineDescriptor, PipelineError,
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

struct ShaderCompilationInfo {
    defines: Defines,
    ty: ShaderType,
}

struct ShaderRecompiler {
    compiler: ShaderCompiler,
    cache: ShaderCache,
    shader_info: HashMap<ShaderAbsPath, ShaderCompilationInfo>,
    file_modified_rx: Receiver<notify::Result<notify::Event>>,
    shader_done_tx: Sender<(ShaderAbsPath, SpvBinary)>,
    watcher: Option<Box<dyn notify::Watcher>>,
}

impl ShaderRecompiler {
    fn recompile(&mut self, abspath: ShaderAbsPath) {
        let Some(info) = self.shader_info.get(&abspath) else {
            log::error!("Got path '{p}' from file watcher as a modified shader but there is not compilatio info, so it cannot be recompiled.", p = abspath.display());
            return;
        };
        let loc: ShaderLocation = abspath.clone().into();
        let result = self
            .compiler
            .compile(&loc, &info.defines, info.ty, Some(&mut self.cache));
        match result {
            Ok(binary) => {
                if let Err(_) = self.shader_done_tx.send((abspath, binary)) {
                    log::info!("Shutting down shader recompiler thread, receiver has disconnected");
                    return;
                };
            }
            Err(e) => {
                log::error!("Failed to compile shader '{loc}':\n{e}");
                return;
            }
        }
    }

    fn run(&mut self) {
        match self.file_modified_rx.recv() {
            Ok(Ok(event)) if event.kind.is_create() || event.kind.is_modify() => {
                for path in event.paths {
                    let abspath = ShaderAbsPath::from_abspath(path);
                    self.recompile(abspath);
                }
            }
            Ok(Ok(_)) => (),
            Ok(Err(e)) => {
                log::error!("Got error from file watcher: {e}");
            }
            Err(_) => {
                log::info!("Shutting down shader recompiler thread, sender has disconnected");
                return;
            }
        }
    }

    fn new_shader(&mut self, abspath: ShaderAbsPath, defines: Defines, ty: ShaderType) {
        self.shader_info
            .insert(abspath.clone(), ShaderCompilationInfo { defines, ty });
        if let Some(watcher) = &mut self.watcher {
            watcher.watch(&abspath, notify::RecursiveMode::NonRecursive);
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
    pub ty: ShaderType,
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
}

#[derive(Debug)]
pub enum Error {
    Compilation(CompilerError),
    PipelineCreation(PipelineError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

pub struct PipelineService {
    recompiler: ShaderRecompiler,
    recompile_comms: ShaderRecompileComms,
    shader_compiler: Arc<ShaderCompiler>,
    shader_cache: ShaderCache,
}

impl PipelineService {
    pub fn new(config: PipelineServiceConfig) -> Self {
        let (new_shader_tx, new_shader_rx) = std::sync::mpsc::channel();
        let (shader_done_tx, shader_done_rx) = std::sync::mpsc::channel();
        let (file_modified_tx, file_modified_rx) =
            std::sync::mpsc::channel::<notify::Result<notify::Event>>();
        let recompile_comms = ShaderRecompileComms {
            new_shader_tx,
            shader_done_rx,
        };

        let recompiler = {
            let watcher = if config.live_recompile {
                Some(Box::new(
                    notify::recommended_watcher(file_modified_tx)
                        .expect("Failed to launch shader file watcher"),
                ) as Box<dyn notify::Watcher>)
            } else {
                None
            };

            let recompiler = ShaderRecompiler {
                compiler: ShaderCompiler::new().unwrap(),
                cache: ShaderCache::new(),
                shader_done_tx,
                shader_info: HashMap::new(),
                file_modified_rx,
                watcher,
            };
            recompiler
        };

        let shader_compiler = ShaderCompiler::new().expect("Failed to create shader compiler");
        Self {
            recompiler,
            recompile_comms,
            shader_compiler: Arc::new(shader_compiler),
            shader_cache: ShaderCache::new(),
        }
    }

    pub fn flush_recreate(&self) -> impl Iterator<Item = RecreatePipeline> {
        todo!();
        std::iter::empty()
    }

    pub fn recompile_shader(&self, id: PipelineId, shader: Shader) {
        todo!()
    }

    pub fn create(
        &mut self,
        shaders: Shaders,
        settings: PipelineSettings,
        render_pass: Handle<RenderPass>,
        renderer: &mut Renderer,
    ) -> Result<Handle<GraphicsPipeline>, Error> {
        let (vert, frag): (ShaderDescriptor, Option<ShaderDescriptor>) = {
            let vert = {
                let vert = shaders.vert;
                let spv_binary = self
                    .shader_compiler
                    .compile(
                        &vert.loc,
                        &vert.defines,
                        vert.ty,
                        Some(&mut self.shader_cache),
                    )
                    .map_err(Error::Compilation)?;

                ShaderDescriptor {
                    spirv_code: spv_binary.data(),
                    debug_name: vert.debug_name,
                }
            };

            let frag = if let Some(frag) = shaders.frag {
                let spv_binary = self
                    .shader_compiler
                    .compile(
                        &frag.loc,
                        &frag.defines,
                        frag.ty,
                        Some(&mut self.shader_cache),
                    )
                    .map_err(Error::Compilation)?;

                Some(ShaderDescriptor {
                    spirv_code: spv_binary.data(),
                    debug_name: frag.debug_name,
                })
            } else {
                None
            };

            (vert, frag)
        };
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

        renderer
            .create_gfx_pipeline(descriptor, &render_pass)
            .map_err(Error::PipelineCreation)
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
