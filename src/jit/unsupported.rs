//! Keep feature unification harmless on WebAssembly.
use crate::{PpcCpu, PpcJitConfig, PpcJitStats};

impl PpcCpu {
    /// Native code generation is unavailable on WebAssembly.
    pub fn enable_jit(&mut self, _config: PpcJitConfig) -> Result<(), String> {
        Err("Cranelift JIT is unavailable on WebAssembly targets".into())
    }

    /// No-op on WebAssembly, where the JIT cannot be enabled.
    pub fn disable_jit(&mut self) {}

    /// No native code executes on WebAssembly.
    pub fn jit_stats(&self) -> PpcJitStats {
        PpcJitStats::default()
    }
}
