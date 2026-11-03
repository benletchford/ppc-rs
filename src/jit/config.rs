//! JIT configuration and counters, available on every target.
const DEFAULT_THRESHOLD: u32 = 8;
const DEFAULT_HOT_THRESHOLD: u32 = 1000;

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
