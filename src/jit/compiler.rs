//! Native compiler and background worker.
use std::collections::HashMap;
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use cranelift_codegen::ir::{AbiParam, Signature, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};

use super::{emit, jitdump};

const DEFAULT_THRESHOLD: u32 = 8;
const DEFAULT_HOT_THRESHOLD: u32 = 1000;

/// Maximum number of native functions (blocks and regions) one engine creates.
const MAX_BLOCKS: u32 = 1 << 20;

/// Configuration for [`crate::PpcCpu::enable_jit`].
#[derive(Debug, Clone)]
pub struct PpcJitConfig {
    /// Number of times a cached basic block must run in the interpreter
    /// before it is compiled. `0` compiles on first execution.
    pub threshold: u32,
    /// Run the Cranelift IR verifier on every block (slow; for debugging).
    pub verify: bool,
    /// Write separate `jit-<pid>-<timestamp>-{block,hot}.dump` files and
    /// sibling `.ppc` listings to the temporary directory so `samply`/`perf`
    /// can symbolicate generated code.
    pub jitdump: bool,
    /// Compile on a background thread. A hot block keeps running in the
    /// interpreter until its native code is ready, so only timing changes.
    pub background: bool,
    /// Observations of a standalone outgoing edge before its source is
    /// compiled with its dominant successor path as one hot region.
    /// `u32::MAX` disables regions.
    pub hot_threshold: u32,
}

impl Default for PpcJitConfig {
    /// Defaults, overridable with `PPC_JIT_THRESHOLD`, `PPC_JIT_VERIFY=1`
    /// `PPC_JIT_JITDUMP=1`, `PPC_JIT_SYNC=1` (compile on the CPU thread) and
    /// `PPC_JIT_HOT_THRESHOLD` (`off` disables regions).
    fn default() -> Self {
        let flag = |name: &str| std::env::var(name).as_deref() == Ok("1");
        Self {
            threshold: std::env::var("PPC_JIT_THRESHOLD")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_THRESHOLD),
            verify: flag("PPC_JIT_VERIFY"),
            jitdump: flag("PPC_JIT_JITDUMP"),
            background: !flag("PPC_JIT_SYNC"),
            hot_threshold: match std::env::var("PPC_JIT_HOT_THRESHOLD").as_deref() {
                Ok("off") => u32::MAX,
                Ok(value) => value.parse().unwrap_or(DEFAULT_HOT_THRESHOLD),
                Err(_) => DEFAULT_HOT_THRESHOLD,
            },
        }
    }
}

/// Counters describing JIT activity on one [`crate::PpcCpu`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PpcJitStats {
    /// Blocks compiled to native code.
    pub blocks_compiled: u64,
    /// Blocks the JIT declined (they keep running in the interpreter).
    pub blocks_rejected: u64,
    /// Native block executions.
    pub block_runs: u64,
    /// Guest instructions retired by native blocks.
    pub instructions: u64,
    /// Hot regions compiled to native code.
    pub regions_compiled: u64,
}

/// Native code for a block being compiled in the background: `0` while
/// pending, [`JIT_DECLINED`] if the compiler declined it, else the entry point.
pub(crate) type JitTicket = Arc<AtomicUsize>;
/// Entry point value meaning "declined; keep interpreting".
pub(crate) const JIT_DECLINED: usize = usize::MAX;

/// Outcome of [`JitShared::request`].
pub(crate) enum JitRequest {
    /// Compiled (or declined) synchronously.
    Ready(Option<usize>),
    /// Queued for the background compiler.
    Pending(JitTicket),
}

/// Read a ticket; `Some` once the background compiler has finished.
pub(crate) fn poll_ticket(ticket: &JitTicket) -> Option<Option<usize>> {
    match ticket.load(Ordering::Acquire) {
        0 => None,
        JIT_DECLINED => Some(None),
        code => {
            // The code was written by another core; make sure this one does
            // not run stale prefetched instructions. Once per block.
            #[cfg(target_arch = "aarch64")]
            // SAFETY: `isb` only flushes this core's pipeline.
            unsafe {
                std::arch::asm!("isb", options(nostack, preserves_flags));
            }
            Some(Some(code))
        }
    }
}

/// A block of a region to compile: start, words, and the indices of the
/// region blocks it may continue into.
pub(crate) type RegionSpec = Vec<(u32, Vec<u32>, Vec<usize>)>;

enum Work {
    Block(u32, Vec<u32>),
    Region(RegionSpec),
}

struct Job {
    work: Work,
    ticket: JitTicket,
}

/// Engine state shared by every [`crate::PpcCpu`] cloned from one another.
pub(crate) struct JitShared {
    pub(crate) config: PpcJitConfig,
    compiler: Arc<Mutex<Compiler>>,
    queue: Option<Arc<Queue>>,
}

/// Work for the background compiler. Blocks go first: a block waiting to
/// compile runs in the interpreter, a region only replaces native code.
#[derive(Default)]
struct Queue {
    state: Mutex<QueueState>,
    ready: Condvar,
    /// Jobs finished so far, so CPUs know when to look at their tickets.
    finished: AtomicUsize,
}

#[derive(Default)]
struct QueueState {
    blocks: VecDeque<Job>,
    regions: VecDeque<Job>,
    closed: bool,
}

impl Drop for JitShared {
    fn drop(&mut self) {
        if let Some(queue) = &self.queue {
            queue
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .closed = true;
            queue.ready.notify_all();
        }
    }
}

// SAFETY: the compiler (and the executable memory it owns) is only reached
// through the mutex, and generated code does not depend on the creating
// thread. Executable pages are never made writable again, so the worker can
// emit new code while other blocks run.
unsafe impl Send for JitShared {}
unsafe impl Sync for JitShared {}
struct SendCompiler(Arc<Mutex<Compiler>>);
// SAFETY: as above.
unsafe impl Send for SendCompiler {}

impl fmt::Debug for JitShared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JitShared")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

fn lock(compiler: &Mutex<Compiler>) -> std::sync::MutexGuard<'_, Compiler> {
    compiler.lock().unwrap_or_else(|poison| poison.into_inner())
}

impl JitShared {
    pub(crate) fn new(config: PpcJitConfig) -> Result<Self, String> {
        let compiler = Arc::new(Mutex::new(Compiler::new(&config)?));
        let queue = config.background.then(|| Self::spawn(&compiler)).flatten();
        Ok(Self {
            config,
            compiler,
            queue,
        })
    }

    fn spawn(compiler: &Arc<Mutex<Compiler>>) -> Option<Arc<Queue>> {
        let queue = Arc::new(Queue::default());
        let worker = SendCompiler(Arc::clone(compiler));
        let jobs = Arc::clone(&queue);
        std::thread::Builder::new()
            .name("ppc-jit".into())
            .spawn(move || {
                let worker = worker;
                loop {
                    let job = {
                        let mut state = jobs.state.lock().unwrap_or_else(|p| p.into_inner());
                        loop {
                            if let Some(job) = state.blocks.pop_front() {
                                break job;
                            }
                            if let Some(job) = state.regions.pop_front() {
                                break job;
                            }
                            if state.closed {
                                return;
                            }
                            state = jobs.ready.wait(state).unwrap_or_else(|p| p.into_inner());
                        }
                    };
                    // A CPU that dropped its ticket no longer wants the code.
                    if Arc::strong_count(&job.ticket) == 1 {
                        continue;
                    }
                    let mut compiler = lock(&worker.0);
                    let code = match &job.work {
                        Work::Block(pc, words) => compiler.compile(*pc, words),
                        Work::Region(region) => compiler.compile_region(region),
                    };
                    drop(compiler);
                    job.ticket
                        .store(code.unwrap_or(JIT_DECLINED), Ordering::Release);
                    jobs.finished.fetch_add(1, Ordering::Release);
                }
            })
            .ok()?;
        Some(queue)
    }

    /// Number of background jobs finished so far (changes whenever a ticket
    /// may have become ready).
    pub(crate) fn finished(&self) -> usize {
        self.queue
            .as_ref()
            .map_or(0, |queue| queue.finished.load(Ordering::Acquire))
    }

    pub(crate) fn request(&self, pc: u32, words: &[u32]) -> JitRequest {
        self.submit(Work::Block(pc, words.to_vec()))
    }

    pub(crate) fn request_region(&self, region: RegionSpec) -> JitRequest {
        self.submit(Work::Region(region))
    }

    fn submit(&self, work: Work) -> JitRequest {
        if let Some(queue) = &self.queue {
            let ticket = JitTicket::default();
            let job = Job {
                work,
                ticket: Arc::clone(&ticket),
            };
            let mut state = queue.state.lock().unwrap_or_else(|p| p.into_inner());
            match job.work {
                Work::Block(..) => state.blocks.push_back(job),
                Work::Region(_) => state.regions.push_back(job),
            }
            drop(state);
            queue.ready.notify_one();
            return JitRequest::Pending(ticket);
        }
        let mut compiler = lock(&self.compiler);
        JitRequest::Ready(match &work {
            Work::Block(pc, words) => compiler.compile(*pc, words),
            Work::Region(region) => compiler.compile_region(region),
        })
    }
}

struct Compiler {
    module: JITModule,
    count: u32,
    blocks: HashMap<(u32, Vec<u32>), Option<usize>>,
    regions: HashMap<RegionSpec, Option<usize>>,
    block_dump: Option<jitdump::Writer>,
    region_dump: Option<jitdump::Writer>,
    signature: Signature,
}

impl Compiler {
    fn new(config: &PpcJitConfig) -> Result<Self, String> {
        let builder = JITBuilder::with_flags(
            &[
                (
                    "enable_verifier",
                    if config.verify { "true" } else { "false" },
                ),
                ("opt_level", "speed"),
            ],
            default_libcall_names(),
        )
        .map_err(|error| error.to_string())?;
        let module = JITModule::new(builder);
        let target = module.target_config();
        let ptr = target.pointer_type();
        let mut signature = Signature::new(target.default_call_conv);
        signature
            .params
            .extend([AbiParam::new(ptr), AbiParam::new(ptr)]);
        signature.returns.push(AbiParam::new(types::I32));
        let block_dump = config
            .jitdump
            .then(|| jitdump::Writer::new("block"))
            .flatten();
        let region_dump = config
            .jitdump
            .then(|| jitdump::Writer::new("hot"))
            .flatten();
        Ok(Self {
            module,
            count: 0,
            blocks: HashMap::new(),
            regions: HashMap::new(),
            block_dump,
            region_dump,
            signature,
        })
    }

    fn compile(&mut self, pc: u32, words: &[u32]) -> Option<usize> {
        let key = (pc, words.to_vec());
        if let Some(known) = self.blocks.get(&key) {
            return *known;
        }
        let code = if self.count >= MAX_BLOCKS {
            None
        } else {
            let instrs = listing(pc, words);
            self.compile_uncached(
                false,
                &jitdump::block_name(pc),
                &instrs,
                |b, call_conv, ptr| {
                    emit::build(b, call_conv, ptr, pc, words);
                },
            )
        };
        self.blocks.insert(key, code);
        code
    }

    fn compile_region(&mut self, region: &RegionSpec) -> Option<usize> {
        if let Some(known) = self.regions.get(region) {
            return *known;
        }
        let code = if self.count >= MAX_BLOCKS || region.is_empty() {
            None
        } else {
            let instrs: Vec<(u32, u32)> = region
                .iter()
                .flat_map(|(pc, words, _)| listing(*pc, words))
                .collect();
            let blocks: Vec<emit::RegionBlock<'_>> = region
                .iter()
                .map(|(pc, words, successors)| emit::RegionBlock {
                    pc: *pc,
                    words,
                    successors: successors.clone(),
                })
                .collect();
            let name = jitdump::region_name(region[0].0, region.len());
            self.compile_uncached(true, &name, &instrs, |b, call_conv, ptr| {
                emit::build_region(b, call_conv, ptr, &blocks);
            })
        };
        self.regions.insert(region.clone(), code);
        code
    }

    fn compile_uncached(
        &mut self,
        hot: bool,
        name: &str,
        instrs: &[(u32, u32)],
        emit: impl FnOnce(&mut FunctionBuilder<'_>, cranelift_codegen::isa::CallConv, types::Type),
    ) -> Option<usize> {
        let ptr = self.module.target_config().pointer_type();
        let call_conv = self.module.target_config().default_call_conv;
        let mut ctx = self.module.make_context();
        ctx.func.signature = self.signature.clone();
        let mut builder_ctx = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);
            emit(&mut b, call_conv, ptr);
            b.seal_all_blocks();
            b.finalize();
        }
        let id = self
            .module
            .declare_function(name, Linkage::Local, &self.signature)
            .ok()?;
        self.module.define_function(id, &mut ctx).ok()?;
        let compiled = ctx.compiled_code()?;
        let size = compiled.code_buffer().len();
        let dump = if hot {
            &mut self.region_dump
        } else {
            &mut self.block_dump
        };
        let locations = dump.is_some().then(|| {
            let mut locations = std::collections::BTreeMap::new();
            locations.insert(0u32, 0u32);
            for loc in compiled.buffer.get_srclocs_sorted() {
                let index = if loc.loc.is_default() {
                    0
                } else {
                    loc.loc.bits()
                };
                locations.insert(loc.start, index);
                locations.insert(loc.end, 0);
            }
            locations.remove(&(size as u32));
            locations.into_iter().collect::<Vec<_>>()
        });
        self.module.finalize_definitions().ok()?;
        self.count += 1;
        let code = self.module.get_finalized_function(id);
        if let (Some(dump), Some(locations)) = (dump.as_mut(), locations) {
            // SAFETY: `code` points at `size` bytes of finalized machine code.
            let bytes = unsafe { std::slice::from_raw_parts(code, size) };
            dump.record(name, instrs, &locations, bytes);
        }
        Some(code as usize)
    }
}

/// `(pc, word)` for each instruction of a block, for the jitdump listing.
fn listing(pc: u32, words: &[u32]) -> Vec<(u32, u32)> {
    words
        .iter()
        .enumerate()
        .map(|(index, &word)| (pc.wrapping_add(index as u32 * 4), word))
        .collect()
}
