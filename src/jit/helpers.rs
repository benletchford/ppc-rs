//! `extern "C"` helpers called from generated code.
//!
//! Every helper returns a `u64`: the low 32 bits carry a loaded value (when
//! there is one) and bit 32 ([`FAULT`]) is set when the instruction stopped the
//! run. A stopping helper records the result in [`JitCtx`] and leaves
//! `cpu.pc` at the faulting instruction, exactly like the interpreter.

use super::JitCtx;
use crate::{
    ApDouble, ApRound, FPSCR_VXIDI, FPSCR_VXSOFT, FPSCR_VXZDZ, Float, FpBinaryOperation,
    PpcAlignmentPolicy, PpcCpu, PpcException, PpcMemoryAccess, PpcStepResult,
};

pub(super) const FAULT_BIT: u32 = 32;
const FAULT: u64 = 1 << FAULT_BIT;

// Compact argument shared by floating-point helpers and the emitter.
const FP_REGISTER_MASK: u32 = 31;
const FP_A_SHIFT: u32 = 5;
const FP_B_SHIFT: u32 = 10;
const FP_OPERATION_SHIFT: u32 = 15;
const FP_OPERATION_MASK: u32 = 3;
const FP_RECORD_BIT: u32 = 1 << 17;

pub(super) fn fp_fields(dest: u8, a: u8, b: u8, operation: u32, record: bool) -> u32 {
    u32::from(dest)
        | (u32::from(a) << FP_A_SHIFT)
        | (u32::from(b) << FP_B_SHIFT)
        | (operation << FP_OPERATION_SHIFT)
        | if record { FP_RECORD_BIT } else { 0 }
}

/// Execute a decoded double-precision operation or ordered comparison.
pub(super) extern "C" fn double_op(ctx: *mut JitCtx, fields: u32, pc: u32) -> u64 {
    let ctx = unsafe { &mut *ctx };
    let cpu = unsafe { &mut *ctx.cpu };
    if !cpu.msr_fp_available() {
        return stop(
            ctx,
            pc,
            PpcStepResult::Exception(PpcException::FloatingPointUnavailable),
        );
    }
    let dest = (fields & FP_REGISTER_MASK) as u8;
    let a = cpu.fpr[((fields >> FP_A_SHIFT) & FP_REGISTER_MASK) as usize];
    let b = cpu.fpr[((fields >> FP_B_SHIFT) & FP_REGISTER_MASK) as usize];
    let rc = fields & FP_RECORD_BIT != 0;
    match (fields >> FP_OPERATION_SHIFT) & FP_OPERATION_MASK {
        0 => {
            if let Some(value) = cpu
                .fast_convert_to_i32_toward_zero(b)
                .or_else(|| cpu.fp_convert_to_i32(b, ApRound::TowardZero))
            {
                cpu.fpr[dest as usize] = u64::from(value);
            }
            cpu.finish_fp_record(rc);
        }
        1 => {
            let mut result = ApDouble::from_bits(u128::from(a))
                .div_r(ApDouble::from_bits(u128::from(b)), cpu.fp_rounding_mode());
            let truncated = ApDouble::from_bits(u128::from(a))
                .div_r(ApDouble::from_bits(u128::from(b)), ApRound::TowardZero)
                .value;
            let default = if PpcCpu::fp_is_zero(a) && PpcCpu::fp_is_zero(b) {
                FPSCR_VXZDZ
            } else if PpcCpu::fp_is_infinite(a) && PpcCpu::fp_is_infinite(b) {
                FPSCR_VXIDI
            } else {
                FPSCR_VXSOFT
            };
            let invalid = PpcCpu::fp_invalid_flags(a, Some(b), default);
            let adjusted =
                cpu.fp_adjusted_binary_result(a, b, FpBinaryOperation::Divide, false, &mut result);
            cpu.finish_ap_double_result_adjusted(dest, result, truncated, adjusted, invalid, rc);
        }
        _ => {
            // The block runner uses its unobserved comparison path.
            let (a, b) = (f64::from_bits(a), f64::from_bits(b));
            let nibble = if a.is_nan() || b.is_nan() {
                1
            } else if a < b {
                8
            } else if a > b {
                4
            } else {
                2
            };
            cpu.set_cr_field(dest, nibble);
            cpu.set_fpscr_compare_result(nibble);
        }
    }
    0
}

/// Execute a decoded single-precision operation using the shared FPSCR logic.
pub(super) extern "C" fn single_binary(ctx: *mut JitCtx, fields: u32, pc: u32) -> u64 {
    let ctx = unsafe { &mut *ctx };
    let cpu = unsafe { &mut *ctx.cpu };
    if !cpu.msr_fp_available() {
        return stop(
            ctx,
            pc,
            PpcStepResult::Exception(PpcException::FloatingPointUnavailable),
        );
    }
    let frt = (fields & FP_REGISTER_MASK) as u8;
    let a = cpu.fpr[((fields >> FP_A_SHIFT) & FP_REGISTER_MASK) as usize];
    let b = cpu.fpr[((fields >> FP_B_SHIFT) & FP_REGISTER_MASK) as usize];
    let operation = match (fields >> FP_OPERATION_SHIFT) & FP_OPERATION_MASK {
        0 => FpBinaryOperation::Add,
        1 => FpBinaryOperation::Subtract,
        2 => FpBinaryOperation::Multiply,
        _ => FpBinaryOperation::Divide,
    };
    let rc = fields & FP_RECORD_BIT != 0;
    if matches!(operation, FpBinaryOperation::Divide) {
        cpu.execute_single_binary(frt, a, b, operation, rc);
    } else {
        // Match the unobserved execution path used by the block runner.
        let (a, b) = (f64::from_bits(a), f64::from_bits(b));
        let result = match operation {
            FpBinaryOperation::Add => a + b,
            FpBinaryOperation::Subtract => a - b,
            FpBinaryOperation::Multiply => a * b,
            _ => unreachable!(),
        } as f32;
        cpu.finish_fp_result(frt, f64::from(result).to_bits(), rc);
    }
    0
}

#[cold]
fn stop(ctx: &mut JitCtx, pc: u32, result: PpcStepResult) -> u64 {
    // SAFETY: `ctx.cpu` is valid for the duration of the native call.
    unsafe { (*ctx.cpu).pc = pc };
    ctx.stop = Some(result);
    ctx.stop_pc = pc;
    FAULT
}

fn misaligned(cpu: &PpcCpu, ea: u32, size: u8) -> bool {
    cpu.alignment_policy == PpcAlignmentPolicy::Trap && (ea & (u32::from(size) - 1)) != 0
}

#[cold]
fn alignment(ctx: &mut JitCtx, pc: u32, ea: u32, size: u8, access: PpcMemoryAccess) -> u64 {
    stop(
        ctx,
        pc,
        PpcStepResult::Exception(PpcException::Alignment {
            addr: ea,
            size,
            access,
        }),
    )
}

/// Interpret one instruction with `cpu.pc` set to its address.
pub(super) extern "C" fn step(ctx: *mut JitCtx, word: u32, pc: u32) -> u64 {
    // SAFETY: generated code passes the context it was invoked with.
    let ctx = unsafe { &mut *ctx };
    let (cpu, mem) = unsafe { (&mut *ctx.cpu, &mut *ctx.mem) };
    cpu.pc = pc;
    let result = cpu
        .step_fast_unobserved(mem, word)
        .unwrap_or_else(|| cpu.step(mem, word));
    match result {
        PpcStepResult::Stepped => 0,
        other => stop(ctx, pc, other),
    }
}

/// Whether the code epoch moved since the region was entered: generated
/// region code calls this after a block that may have written memory.
pub(super) extern "C" fn code_changed(ctx: *mut JitCtx) -> u64 {
    // SAFETY: see `step`.
    let ctx = unsafe { &*ctx };
    let mem = unsafe { &*ctx.mem };
    u64::from(mem.code_epoch().unwrap_or(0) != ctx.epoch)
}

macro_rules! load_helper {
    ($name:ident, $size:expr, |$mem:ident, $ea:ident| $read:expr) => {
        pub(super) extern "C" fn $name(ctx: *mut JitCtx, $ea: u32, pc: u32) -> u64 {
            // SAFETY: see `step`.
            let ctx = unsafe { &mut *ctx };
            let (cpu, $mem) = unsafe { (&*ctx.cpu, &mut *ctx.mem) };
            if $size > 1 && misaligned(cpu, $ea, $size) {
                return alignment(ctx, pc, $ea, $size, PpcMemoryAccess::Load);
            }
            match $read {
                Some(value) => u64::from(value),
                None => stop(
                    ctx,
                    pc,
                    PpcStepResult::MemoryFault {
                        addr: $ea,
                        was_write: false,
                    },
                ),
            }
        }
    };
}

load_helper!(lbz, 1, |mem, ea| mem.read_u8(ea).map(u32::from));
load_helper!(lhz, 2, |mem, ea| mem.read_u16_be(ea).map(u32::from));
load_helper!(lha, 2, |mem, ea| mem
    .read_u16_be(ea)
    .map(|value| value as i16 as i32 as u32));
load_helper!(lwz, 4, |mem, ea| mem.read_u32_be(ea));

macro_rules! store_helper {
    ($name:ident, $size:expr, |$mem:ident, $ea:ident, $value:ident| $write:expr) => {
        pub(super) extern "C" fn $name(ctx: *mut JitCtx, $ea: u32, $value: u32, pc: u32) -> u64 {
            // SAFETY: see `step`.
            let ctx = unsafe { &mut *ctx };
            let (cpu, $mem) = unsafe { (&*ctx.cpu, &mut *ctx.mem) };
            if $size > 1 && misaligned(cpu, $ea, $size) {
                return alignment(ctx, pc, $ea, $size, PpcMemoryAccess::Store);
            }
            match $write {
                Some(()) => 0,
                None => stop(
                    ctx,
                    pc,
                    PpcStepResult::MemoryFault {
                        addr: $ea,
                        was_write: true,
                    },
                ),
            }
        }
    };
}

store_helper!(stb, 1, |mem, ea, value| mem.write_u8(ea, value as u8));
store_helper!(sth, 2, |mem, ea, value| mem.write_u16_be(ea, value as u16));
store_helper!(stw, 4, |mem, ea, value| mem.write_u32_be(ea, value));

macro_rules! fp_helper {
    ($name:ident, $size:expr, $access:ident, |$cpu:ident, $mem:ident, $ea:ident, $fr:ident| $body:expr) => {
        /// Floating-point load/store; the FPR is read or written in place.
        pub(super) extern "C" fn $name(ctx: *mut JitCtx, $ea: u32, $fr: u32, pc: u32) -> u64 {
            // SAFETY: see `step`.
            let ctx = unsafe { &mut *ctx };
            let ($cpu, $mem) = unsafe { (&mut *ctx.cpu, &mut *ctx.mem) };
            if misaligned($cpu, $ea, $size) {
                return alignment(ctx, pc, $ea, $size, PpcMemoryAccess::$access);
            }
            let $fr = $fr as usize;
            match $body {
                Some(()) => 0,
                None => stop(
                    ctx,
                    pc,
                    PpcStepResult::MemoryFault {
                        addr: $ea,
                        was_write: matches!(PpcMemoryAccess::$access, PpcMemoryAccess::Store),
                    },
                ),
            }
        }
    };
}

fp_helper!(lfs, 4, Load, |cpu, mem, ea, fr| mem
    .read_u32_be(ea)
    .map(|bits| cpu.fpr[fr] = (f32::from_bits(bits) as f64).to_bits()));
fp_helper!(lfd, 8, Load, |cpu, mem, ea, fr| mem
    .read_u64_be(ea)
    .map(|bits| cpu.fpr[fr] = bits));
fp_helper!(stfs, 4, Store, |cpu, mem, ea, fr| mem
    .write_u32_be(ea, cpu.fp_single_bits(cpu.fpr[fr])));
fp_helper!(stfd, 8, Store, |cpu, mem, ea, fr| mem
    .write_u64_be(ea, cpu.fpr[fr]));
