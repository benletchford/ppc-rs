//! Native execution sessions.
use crate::{PpcCpu, PpcMemory, PpcStepResult};
/// Mutable state a native block shares with its helper functions.
#[repr(C)]
pub(crate) struct JitCtx {
    pub(crate) cpu: *mut PpcCpu,
    pub(crate) mem: *mut dyn PpcMemory,
    pub(crate) stop: Option<PpcStepResult>,
    pub(crate) panic: Option<Box<dyn std::any::Any + Send>>,
    pub(crate) stop_pc: u32,
    /// Instructions a region may retire in this call (a block's own length
    /// is checked by the caller before entering it).
    pub(crate) budget: u64,
    /// Code epoch the running region was entered under.
    pub(crate) epoch: u64,
}

/// Adapter that lets a `?Sized` memory bus be used as `dyn PpcMemory`.
struct MemAdapter<'a, M: PpcMemory + ?Sized>(&'a mut M);

impl<M: PpcMemory + ?Sized> PpcMemory for MemAdapter<'_, M> {
    fn read_u8(&mut self, addr: u32) -> Option<u8> {
        self.0.read_u8(addr)
    }
    fn write_u8(&mut self, addr: u32, value: u8) -> Option<()> {
        self.0.write_u8(addr, value)
    }
    fn read_u16_be(&mut self, addr: u32) -> Option<u16> {
        self.0.read_u16_be(addr)
    }
    fn write_u16_be(&mut self, addr: u32, value: u16) -> Option<()> {
        self.0.write_u16_be(addr, value)
    }
    fn read_u32_be(&mut self, addr: u32) -> Option<u32> {
        self.0.read_u32_be(addr)
    }
    fn write_u32_be(&mut self, addr: u32, value: u32) -> Option<()> {
        self.0.write_u32_be(addr, value)
    }
    fn read_u64_be(&mut self, addr: u32) -> Option<u64> {
        self.0.read_u64_be(addr)
    }
    fn write_u64_be(&mut self, addr: u32, value: u64) -> Option<()> {
        self.0.write_u64_be(addr, value)
    }
    fn read_instruction_u32_be(&mut self, addr: u32) -> Option<u32> {
        self.0.read_instruction_u32_be(addr)
    }
    fn instruction_cache_token(&mut self, addr: u32) -> Option<u64> {
        self.0.instruction_cache_token(addr)
    }
    fn global_instruction_cache_token(&self) -> Option<u64> {
        self.0.global_instruction_cache_token()
    }
    fn code_epoch(&self) -> Option<u64> {
        self.0.code_epoch()
    }
}

/// A run of one or more native blocks sharing one memory adapter and
/// context, so chained blocks do not pay the setup cost per block.
pub(crate) struct Session<'a> {
    cpu: *mut PpcCpu,
    ctx: JitCtx,
    _marker: std::marker::PhantomData<&'a mut PpcCpu>,
}

/// Set in a native block's return value when a helper stopped the run (the
/// result is then in `JitCtx::stop`); the other bits count completed
/// instructions.
pub(crate) const STOPPED: u32 = 1 << 31;

/// A native block stopped the run: `(pc, completed, result)`.
pub(crate) type JitStop = (u32, u64, PpcStepResult);

impl Session<'_> {
    pub(crate) fn cpu(&mut self) -> &mut PpcCpu {
        // SAFETY: `cpu` came from a live `&mut PpcCpu` for this session, and no
        // native code is running while the caller holds this reference.
        unsafe { &mut *self.cpu }
    }

    /// Run the native block `code` at the current `pc`. Returns the number of
    /// guest instructions retired, or the stop that ended the run.
    ///
    /// # Safety
    /// `code` must have been returned by [`super::JitShared::request`] on an engine
    /// that is still alive, for a block that starts at `cpu.pc`.
    /// A region continues past its first block while `budget` allows, as
    /// long as the code epoch stays `epoch`.
    pub(crate) unsafe fn run(
        &mut self,
        code: usize,
        budget: u64,
        epoch: u64,
    ) -> Result<u64, JitStop> {
        // Completed counts share a `u32` with `STOPPED`.
        self.ctx.budget = budget.min(u64::from(STOPPED >> 1));
        self.ctx.epoch = epoch;
        // SAFETY: guaranteed by the caller.
        let function: extern "C" fn(*mut PpcCpu, *mut JitCtx) -> u32 =
            unsafe { std::mem::transmute(code) };
        let returned = function(self.cpu, &mut self.ctx);
        // Helpers return normally through generated code before we resume the
        // original panic. Check even normal exits: the epoch helper uses one.
        if let Some(payload) = self.ctx.panic.take() {
            std::panic::resume_unwind(payload);
        }
        if returned & STOPPED == 0 {
            return Ok(u64::from(returned));
        }
        let completed = u64::from(returned & !STOPPED);
        match self.ctx.stop.take() {
            Some(result) => Err((self.ctx.stop_pc, completed, result)),
            None => unreachable!("native block reported a stop without a result"),
        }
    }
}

/// Run `body` with a native-block session over `cpu` and `mem`.
pub(crate) fn with_session<M: PpcMemory + ?Sized, R>(
    cpu: &mut PpcCpu,
    mem: &mut M,
    body: impl FnOnce(&mut Session<'_>) -> R,
) -> R {
    let mut adapter = MemAdapter(mem);
    let dyn_mem: *mut (dyn PpcMemory + '_) = &mut adapter;
    // SAFETY: the pointer is only dereferenced while `adapter` is alive, which
    // is for the whole of this function.
    let mem: *mut dyn PpcMemory = unsafe { std::mem::transmute(dyn_mem) };
    let cpu_ptr: *mut PpcCpu = cpu;
    let mut session = Session {
        cpu: cpu_ptr,
        ctx: JitCtx {
            cpu: cpu_ptr,
            mem,
            stop: None,
            panic: None,
            stop_pc: 0,
            budget: 0,
            epoch: 0,
        },
        _marker: std::marker::PhantomData,
    };
    body(&mut session)
}
