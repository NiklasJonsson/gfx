use crate::render::uniform::PBRMaterialData;

use super::{Defines, ShaderAbsPath, ShaderLocation, ShaderType, SpvBinary};

use super::compiler::{CompilerError, CompilerResult, FileNotFound, ShaderCompiler, ShaderSource};

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderCachePolicy {
    ForceRecompile,
    AllowCache,
}

#[derive(Default)]
pub struct ShaderCache {
    cache: std::collections::HashMap<ShaderSource, SpvBinary>,
}

impl ShaderCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: ShaderSource, binary: SpvBinary) {
        self.cache.insert(key, binary);
    }

    pub fn get(&self, key: &ShaderSource) -> Option<&SpvBinary> {
        self.cache.get(key)
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UserId(pub u64);

#[derive(Debug, Clone)]
struct WorkItem {
    request_id: UserId,
    path: ShaderAbsPath,
    compilation_info: Arc<ShaderCompilationInfo>,
}

struct UserJobs {
    expected: usize,
    results: Vec<CompilerResult<SpvBinary>>,
}

impl UserJobs {
    pub fn is_done(&self) -> bool {
        self.expected == self.results.len()
    }

    pub fn take(&mut self, out: &mut Vec<CompilerResult<SpvBinary>>) {
        assert!(self.expected >= self.results.len());
        self.expected -= self.results.len();
        out.append(&mut self.results);
    }
}

struct SharedThreadState {
    work_queue: Vec<WorkItem>,
    jobs: HashMap<UserId, UserJobs>,
    shader_cache: ShaderCache,
}

struct ThreadContext {
    compiler: ShaderCompiler,
    has_work: std::sync::Condvar,
    has_done: std::sync::Condvar,
    shared_state: Mutex<SharedThreadState>,
}

struct ShaderCompilationServiceState {
    shader_compilation_info: HashMap<ShaderAbsPath, Vec<Arc<ShaderCompilationInfo>>>,
}

pub struct ShaderCompilationService {
    thread_context: Arc<ThreadContext>,
    threads: Vec<std::thread::JoinHandle<()>>,
    state: Mutex<ShaderCompilationServiceState>,
}

fn thread_work(ctx: &ThreadContext) {
    loop {
        let work_item: WorkItem;

        // Caching has to be done after preprocessing as there is no model of dependencies on headers
        // for a shader source file. Therefore, we cannot map a file path to a spirv binary. An include
        // might have changed so even if the file itself has not changed, we need to compile it.
        //
        // Minimize the lock windows in general. Definitely don't hold a lock while compiling.
        {
            let mut state = ctx.shared_state.lock().unwrap();
            while state.work_queue.is_empty() {
                state = ctx.has_work.wait(state).unwrap();
            }
            work_item = state.work_queue.remove(0);
        }

        let WorkItem {
            request_id,
            path,
            compilation_info,
        } = work_item;

        let result = ctx.compiler.preprocess(&path, &compilation_info.defines);

        let write_result = |state: &mut MutexGuard<'_, SharedThreadState>,
                            request_id: UserId,
                            result: CompilerResult<SpvBinary>| {
            let jobs = state.jobs.get_mut(&request_id).unwrap_or_else(|| {
                panic!("Finished a shader for user id {request_id:?} but there is no jobs entry")
            });
            jobs.results.push(result);
        };

        match result {
            Ok(src) => {
                let mut result = None;
                {
                    let state = ctx.shared_state.lock().unwrap();
                    if let Some(bin) = state.shader_cache.cache.get(&src) {
                        let spv = bin.clone();
                        result = Some(Ok(spv));
                    }
                }

                if result.is_none() {
                    result = Some(
                        ctx.compiler
                            .compile_source(&src, compilation_info.ty, "TODO"),
                    );
                }
                let mut state = ctx.shared_state.lock().unwrap();
                write_result(&mut state, request_id, result.unwrap());
            }
            Err(e) => {
                let mut state = ctx.shared_state.lock().unwrap();
                write_result(&mut state, request_id, Err(e));
            }
        }
        ctx.has_done.notify_all();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShaderCompilationServiceConfig {
    pub n_threads: std::num::NonZero<u8>,
}

impl ShaderCompilationService {
    pub fn new(shader_compiler: ShaderCompiler, config: ShaderCompilationServiceConfig) -> Self {
        let n_threads = config.n_threads.get();
        let thread_context = Arc::new(ThreadContext {
            compiler: shader_compiler,
            has_work: Condvar::new(),
            has_done: Condvar::new(),
            shared_state: Mutex::new(SharedThreadState {
                work_queue: Vec::with_capacity(16),
                jobs: HashMap::default(),
                shader_cache: ShaderCache::new(),
            }),
        });
        let mut r = Self {
            thread_context: Arc::clone(&thread_context),
            threads: Vec::with_capacity(n_threads as usize),
            state: Mutex::new(ShaderCompilationServiceState {
                shader_compilation_info: HashMap::default(),
            }),
        };

        for i in 0..n_threads {
            let ctx = Arc::clone(&thread_context);
            r.threads.push(std::thread::spawn(move || {
                log::info!("Starting shader compilation thread {i}");
                thread_work(&ctx);
            }));
        }

        r
    }

    pub fn add_new_permutation(
        &self,
        loc: &ShaderLocation,
        defines: Defines,
        ty: ShaderType,
        id: UserId,
    ) -> Result<ShaderAbsPath, FileNotFound> {
        let abspath = self.thread_context.compiler.find(&loc)?;
        let sci = Arc::new(ShaderCompilationInfo { defines, ty });

        {
            let mut state = self.state.lock().unwrap();
            state
                .shader_compilation_info
                .entry(abspath.clone())
                .or_default()
                .push(sci.clone());
        }

        {
            let mut thread_state = self.thread_context.shared_state.lock().unwrap();
            thread_state.work_queue.push(WorkItem {
                request_id: id,
                path: abspath.clone(),
                compilation_info: sci,
            });
        }
        Ok(abspath)
    }

    pub fn queue(&self, loc: &ShaderLocation, id: UserId) -> Result<(), FileNotFound> {
        let path = self.thread_context.compiler.find(loc)?;
        // TOOD perf: There are some temporary allocations here that are used
        // to avoid locking both the mutexes at once.
        let permutations: Vec<Arc<ShaderCompilationInfo>>;
        {
            let service_state = self.state.lock().unwrap();
            permutations = service_state
                .shader_compilation_info
                .get(&path)
                .expect("TODO")
                .clone();
        }

        let mut new_jobs: Vec<WorkItem> = permutations
            .into_iter()
            .map(|sci| WorkItem {
                request_id: id,
                path: path.clone(),
                compilation_info: Arc::clone(&sci),
            })
            .collect();

        {
            let mut thread_state = self.thread_context.shared_state.lock().unwrap();
            thread_state.work_queue.append(&mut new_jobs);
        }
        self.thread_context.has_work.notify_all();
        Ok(())
    }

    pub fn flush(&self, id: UserId, results: &mut Vec<CompilerResult<SpvBinary>>) {
        let mut state = self.thread_context.shared_state.lock().unwrap();

        if let Some(jobs) = state.jobs.get_mut(&id) {
            jobs.take(results);
        }
    }

    pub fn wait(&self, ids: &[UserId], results: &mut Vec<CompilerResult<SpvBinary>>) {
        for id in ids {
            let mut state = self.thread_context.shared_state.lock().unwrap();

            let mut jobs = state.jobs.get(id);
            while jobs.as_ref().map(|x| !x.is_done()).unwrap_or(true) {
                state = self.thread_context.has_done.wait(state).unwrap();
                jobs = state.jobs.get(id);
            }

            let mut jobs = state.jobs.remove(id).unwrap();
            jobs.take(results);
        }
    }
}
