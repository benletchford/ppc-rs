//! Cranelift baseline JIT (cargo feature `jit`).
//!
//! Compiles cached basic blocks and hot regions. Integer and branch operations
//! lower to CLIF; memory and floating-point operations use runtime helpers.
//! Other instructions fall back to the interpreter.
#![allow(unsafe_code)]

#[cfg(not(target_family = "wasm"))]
mod compiler;
#[cfg(not(target_family = "wasm"))]
pub(crate) mod diag;
#[cfg(not(target_family = "wasm"))]
pub(crate) mod runtime;
#[cfg(not(target_family = "wasm"))]
mod session;
#[cfg(not(target_family = "wasm"))]
pub(crate) use compiler::{
    JIT_DECLINED, JitRequest, JitShared, JitTicket, RegionSpec, poll_ticket,
};
mod config;
pub use config::{PpcJitConfig, PpcJitStats};

#[cfg(target_family = "wasm")]
mod unsupported;
#[cfg(not(target_family = "wasm"))]
pub(crate) use session::{JitCtx, STOPPED, with_session};
#[cfg(not(target_family = "wasm"))]
mod emit;
#[cfg(not(target_family = "wasm"))]
mod helpers;
#[cfg(not(target_family = "wasm"))]
mod jitdump;
