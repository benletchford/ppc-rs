//! Cranelift baseline JIT (cargo feature `jit`).
//!
//! Compiles cached basic blocks and hot regions. Integer and branch operations
//! lower to CLIF; memory and floating-point operations use runtime helpers.
//! Other instructions fall back to the interpreter.
#![allow(unsafe_code)]

mod compiler;
pub(crate) mod diag;
pub(crate) mod runtime;
mod session;
pub(crate) use compiler::{
    JIT_DECLINED, JitRequest, JitShared, JitTicket, RegionSpec, poll_ticket,
};
pub use compiler::{PpcJitConfig, PpcJitStats};
pub(crate) use session::{JitCtx, STOPPED, with_session};
mod emit;
mod helpers;
mod jitdump;
