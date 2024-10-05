use super::{Defines, ShaderAbsPath, ShaderLocation, ShaderType, SpvBinary};

use super::shader_compiler::{CompilerResult, FileNotFound, ShaderCompiler, ShaderSource};

use std::collections::HashMap;
use std::sync::{atomic::AtomicU64, Arc, Condvar, Mutex};

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

#[derive(Debug)]
pub struct CompiledShader {
    pub path: ShaderAbsPath,
    pub compilation_info: Arc<ShaderCompilationInfo>,
    pub result: CompilerResult<SpvBinary>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShaderCompilationInfo {
    pub defines: Defines,
    pub ty: ShaderType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum JobId {
    Async,
    Blocking(u64),
}

struct JobIdGenerator {
    counter: AtomicU64,
}

impl JobIdGenerator {
    fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }

    /// Take 'n' ids from the generator, return a range containing them.
    fn take(&self, n: u64) -> std::ops::Range<u64> {
        let start = self
            .counter
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        start..start + n
    }
}

#[derive(Debug, Clone)]
struct WorkItem {
    job_id: JobId,
    path: ShaderAbsPath,
    compilation_info: Arc<ShaderCompilationInfo>,
}

struct SharedThreadState {
    work_queue: Vec<WorkItem>,
    // TODO: Get rid of u64?
    sync_jobs: HashMap<u64, CompiledShader>,
    async_jobs: Vec<CompiledShader>,
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
    id_generator: JobIdGenerator,
    state: Mutex<ShaderCompilationServiceState>,
}

fn thread_work(ctx: &ThreadContext, thread_idx: usize) {
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
        log::debug!(
            "Shader compiler thread (SCT) {} picked up {:?}",
            thread_idx,
            work_item
        );

        let WorkItem {
            job_id,
            path,
            compilation_info,
        } = work_item;

        log::debug!("[SCT {thread_idx}] Preprocessing...");
        let preprocess_result = ctx.compiler.preprocess(&path, &compilation_info.defines);
        log::debug!("[SCT {thread_idx}] Preprocessing done");
        let compilation_result: CompilerResult<SpvBinary>;
        let mut cache_key: Option<ShaderSource> = None;
        match preprocess_result {
            Ok(src) => {
                let mut result = None;
                // Minimize lock window
                {
                    let state = ctx.shared_state.lock().unwrap();
                    if let Some(bin) = state.shader_cache.get(&src) {
                        log::debug!("[SCT {thread_idx}] hit the cache");
                        let spv = bin.clone();
                        result = Some(Ok(spv));
                    }
                }

                if result.is_none() {
                    log::debug!("[SCT {thread_idx}] did not hit the cache, compiling");
                    let name = path.display().to_string();
                    result = Some(
                        ctx.compiler
                            .compile_source(&src, compilation_info.ty, &name),
                    );
                    cache_key = Some(src);
                }
                compilation_result = result.unwrap();
            }
            Err(e) => {
                compilation_result = Err(e);
            }
        }

        {
            let mut state = ctx.shared_state.lock().unwrap();
            if let (Ok(spv), Some(src)) = (&compilation_result, cache_key) {
                log::debug!("[SCT {thread_idx}] Compilation succeeded, adding to cache");
                state.shader_cache.insert(src, spv.clone());
            }

            log::debug!("[SCT {thread_idx}] Compilation done: path: {p}, compilation info: {compilation_info:?}, result: {r}", p = path.display(), r = compilation_result.is_ok());
            let cs = CompiledShader {
                path,
                compilation_info: compilation_info.clone(),
                result: compilation_result,
            };
            match job_id {
                JobId::Async => {
                    log::debug!("[SCT {thread_idx}] Pushing to async job result queue");
                    state.async_jobs.push(cs);
                }
                JobId::Blocking(id) => {
                    log::debug!("[SCT {thread_idx}] Pushing to job-result queue for {}", id);
                    let prev = state.sync_jobs.insert(id, cs);
                    if let Some(prev) = prev {
                        panic!("[SCT {thread_idx}] Finished a shader for id {id:?} but there was already a job entry: {prev:?}")
                    }
                }
            }
        }
        ctx.has_done.notify_all();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShaderCompilationServiceConfig {
    pub n_threads: std::num::NonZero<usize>,
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
                sync_jobs: HashMap::default(),
                shader_cache: ShaderCache::new(),
                async_jobs: Vec::with_capacity(16),
            }),
        });
        let mut r = Self {
            thread_context: Arc::clone(&thread_context),
            threads: Vec::with_capacity(n_threads),
            id_generator: JobIdGenerator::new(),
            state: Mutex::new(ShaderCompilationServiceState {
                shader_compilation_info: HashMap::default(),
            }),
        };

        for i in 0..n_threads {
            let ctx = Arc::clone(&thread_context);
            r.threads.push(std::thread::spawn(move || {
                log::info!("Starting shader compilation thread {i}");
                thread_work(&ctx, i);
            }));
        }

        r
    }

    // TODO: This helper function should not take self but instead the lock guards
    fn queue_work(&self, work: &[WorkItem]) {
        log::debug!(
            "Queued {} jobs to the shader compilation thread pool",
            work.len()
        );
        {
            let mut thread_state = self.thread_context.shared_state.lock().unwrap();
            thread_state.work_queue.extend_from_slice(work);
        }
        self.thread_context.has_work.notify_all();
    }

    pub fn register_permutation(&self, path: ShaderAbsPath, sci: Arc<ShaderCompilationInfo>) {
        // TODO: Check if permutation exists?
        {
            let mut state = self.state.lock().unwrap();
            state
                .shader_compilation_info
                .entry(path)
                .or_default()
                .push(sci);
        }
    }

    pub fn queue(&self, loc: &ShaderLocation) -> Result<(), FileNotFound> {
        log::debug!("Got shader compilation request for {}", loc);
        let path = self.thread_context.compiler.find(loc)?;
        // TOOD perf: There are some temporary allocations here that are used
        // to avoid locking both the mutexes at once.
        let permutations: Vec<Arc<ShaderCompilationInfo>> = {
            let service_state = self.state.lock().unwrap();
            service_state
                .shader_compilation_info
                .get(&path)
                .expect("TODO")
                .clone()
        };

        let new_jobs: Vec<WorkItem> = permutations
            .into_iter()
            .map(|sci| WorkItem {
                job_id: JobId::Async,
                path: path.clone(),
                compilation_info: Arc::clone(&sci),
            })
            .collect();
        self.queue_work(&new_jobs);
        Ok(())
    }

    pub fn flush(&self, results: &mut Vec<CompiledShader>) {
        let mut state = self.thread_context.shared_state.lock().unwrap();
        results.append(&mut state.async_jobs);
    }

    pub fn compile_shaders(
        &self,
        shaders: &[(ShaderLocation, Arc<ShaderCompilationInfo>)],
        results: &mut [Option<CompiledShader>],
    ) {
        let id_range = self.id_generator.take(shaders.len() as u64);
        let mut work_items: Vec<WorkItem> = Vec::with_capacity(shaders.len());
        for (id, (loc, info)) in id_range.clone().zip(shaders.iter()) {
            work_items.push(WorkItem {
                job_id: JobId::Blocking(id),
                path: self
                    .thread_context
                    .compiler
                    .find(loc)
                    .expect("TODO: Return error"),
                compilation_info: info.clone(),
            });
        }
        self.queue_work(&work_items);

        for (idx, id) in id_range.enumerate() {
            let mut state = self.thread_context.shared_state.lock().unwrap();

            let mut compiled_shader = state.sync_jobs.remove(&id);
            while compiled_shader.is_none() {
                state = self.thread_context.has_done.wait(state).unwrap();
                compiled_shader = state.sync_jobs.remove(&id);
            }

            results[idx] = compiled_shader;
        }
    }
}
