//! CLIF generation for basic blocks and regions.
//!
//! Registers live in the `PpcCpu` struct; the block receives `(cpu, ctx)` and
//! reads/writes them with plain loads and stores (write-through, with a small
//! per-block value cache that is dropped whenever a helper might have changed
//! architectural state).
//!
//! Bookkeeping the interpreter does per instruction is batched:
//!
//! * `time_base` is advanced once per run of inline instructions
//!   (`pending_cycles`), flushed before any helper that reads it or at block exit.
//! * `cpu.pc` is written only at block exit (or by helpers), tracked by
//!   `pc_in_sync`.

use std::mem::offset_of;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, InstBuilder, MemFlags, SigRef, Signature, SourceLoc, Type, Value,
    types,
};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::FunctionBuilder;

use super::helpers;
use crate::{PpcCpu, PpcInstr, decode};

// PowerPC branch-option bits (BO).
const BO_IGNORE_CTR: u8 = 0x04;
const BO_CTR_ZERO: u8 = 0x02;
const BO_IGNORE_CONDITION: u8 = 0x10;
const BO_CONDITION_TRUE: u8 = 0x08;

const CA_MASK: u32 = 1 << 29;

/// Address of a helper, for `Emitter::call`.
macro_rules! helper {
    ($name:ident) => {
        helpers::$name as *const () as usize
    };
}

#[derive(Clone, Copy)]
enum Load {
    Byte,
    Half,
    HalfAlgebraic,
    Word,
}

impl Load {
    fn helper(self) -> usize {
        match self {
            Load::Byte => helper!(lbz),
            Load::Half => helper!(lhz),
            Load::HalfAlgebraic => helper!(lha),
            Load::Word => helper!(lwz),
        }
    }
}

#[derive(Clone, Copy)]
enum Special {
    Cr,
    Lr,
    Ctr,
    Xer,
}

impl Special {
    fn offset(self) -> i32 {
        (match self {
            Special::Cr => offset_of!(PpcCpu, cr),
            Special::Lr => offset_of!(PpcCpu, lr),
            Special::Ctr => offset_of!(PpcCpu, ctr),
            Special::Xer => offset_of!(PpcCpu, xer),
        }) as i32
    }

    fn slot(self) -> usize {
        self as usize
    }
}

struct Emitter<'a, 'f> {
    b: &'a mut FunctionBuilder<'f>,
    cpu: Value,
    ctx: Value,
    ptr: Type,
    sig3: SigRef,
    sig4: SigRef,
    sig1: SigRef,
    gpr: [Option<Value>; 32],
    special: [Option<Value>; 4],
    pending_cycles: u32,
    pc_in_sync: bool,
    /// `cpu.pc` as last written by a branch in this block, if any.
    next_pc: Option<Value>,
    /// Instructions completed by earlier blocks of the region.
    done: Value,
    may_change_code: bool,
    fault: Block,
    normal_exit: Block,
    /// `(exit block, pending cycles, instruction index, prior completed count)`.
    fault_exits: Vec<(Block, u32, u32, Value)>,
}

/// One block of a region: its guest code and the region blocks it may be
/// followed by (indices into the region).
pub(super) struct RegionBlock<'a> {
    pub(super) pc: u32,
    pub(super) words: &'a [u32],
    pub(super) successors: Vec<usize>,
}

/// Build the function for one block.
pub(super) fn build(
    b: &mut FunctionBuilder<'_>,
    call_conv: CallConv,
    ptr: Type,
    start_pc: u32,
    words: &[u32],
) {
    let block = RegionBlock {
        pc: start_pc,
        words,
        successors: Vec::new(),
    };
    build_region(b, call_conv, ptr, std::slice::from_ref(&block));
}

/// Build one function for a region of blocks. Control enters at the first
/// block; at the end of each block it continues directly into a successor
/// whose start matches the new `pc` (while the cycle budget in
/// [`JitCtx::budget`](super::JitCtx) allows and, after a block that may have
/// changed code mappings, the code epoch is unchanged) and otherwise returns the
/// instructions completed so far. Faults return the same count plus
/// [`STOPPED`](super::STOPPED), exactly like a single block.
pub(super) fn build_region(
    b: &mut FunctionBuilder<'_>,
    call_conv: CallConv,
    ptr: Type,
    blocks: &[RegionBlock<'_>],
) {
    let entry = b.create_block();
    b.switch_to_block(entry);
    b.append_block_params_for_function_params(entry);
    let [cpu, ctx] = b.block_params(entry).try_into().unwrap();
    let fault = b.create_block();
    b.append_block_param(fault, types::I32);
    b.set_cold_block(fault);
    let normal_exit = b.create_block();
    b.append_block_param(normal_exit, types::I32);

    let mut sig = |params: usize| {
        let mut signature = Signature::new(call_conv);
        signature.params.push(AbiParam::new(ptr));
        for _ in 1..params {
            signature.params.push(AbiParam::new(types::I32));
        }
        signature.returns.push(AbiParam::new(types::I64));
        b.import_signature(signature)
    };
    let (sig1, sig3, sig4) = (sig(1), sig(3), sig(4));

    let has_edges = blocks.iter().any(|block| !block.successors.is_empty());
    let budget = has_edges.then(|| {
        b.ins().load(
            types::I64,
            MemFlags::trusted(),
            ctx,
            std::mem::offset_of!(super::JitCtx, budget) as i32,
        )
    });
    // The first block is emitted inline in the entry block unless another
    // block of the region jumps back to it.
    let targeted: Vec<bool> = (0..blocks.len())
        .map(|index| blocks.iter().any(|block| block.successors.contains(&index)))
        .collect();
    let heads: Vec<Option<Block>> = targeted
        .iter()
        .enumerate()
        .map(|(index, &targeted)| {
            (index > 0 || targeted).then(|| {
                let head = b.create_block();
                b.append_block_param(head, types::I32);
                head
            })
        })
        .collect();
    let zero = b.ins().iconst(types::I32, 0);
    if let Some(head) = heads[0] {
        b.ins().jump(head, &[BlockArg::Value(zero)]);
    }

    let mut e = Emitter {
        b,
        cpu,
        ctx,
        ptr,
        sig3,
        sig4,
        sig1,
        gpr: [None; 32],
        special: [None; 4],
        pending_cycles: 0,
        pc_in_sync: true,
        next_pc: None,
        done: zero,
        may_change_code: false,
        fault,
        normal_exit,
        fault_exits: Vec::new(),
    };
    let mut line = 1;
    for (index, block) in blocks.iter().enumerate() {
        if let Some(head) = heads[index] {
            e.b.switch_to_block(head);
            e.done = e.b.block_params(head)[0];
        }
        e.invalidate();
        e.pending_cycles = 0;
        e.pc_in_sync = true;
        e.next_pc = None;
        e.may_change_code = false;
        for (offset, &word) in block.words.iter().enumerate() {
            let pc = block.pc.wrapping_add(offset as u32 * 4);
            e.b.set_srcloc(SourceLoc::new(line));
            line += 1;
            if !e.inline(offset as u32, pc, word) {
                e.step(offset as u32, pc, word);
            }
        }
        e.b.set_srcloc(SourceLoc::default());
        let successors: Vec<(u32, u32, Block)> = block
            .successors
            .iter()
            .map(|&next| {
                let target = &blocks[next];
                (target.pc, target.words.len() as u32, heads[next].unwrap())
            })
            .collect();
        e.end_block(block.pc, block.words.len() as u32, &successors, budget);
    }
    e.finish();
}

impl Emitter<'_, '_> {
    // ---- plumbing -------------------------------------------------------

    fn flags() -> MemFlags {
        MemFlags::trusted()
    }

    fn c32(&mut self, value: u32) -> Value {
        self.b.ins().iconst(types::I32, i64::from(value as i32))
    }

    fn gpr_offset(register: u8) -> i32 {
        (offset_of!(PpcCpu, gpr) + 4 * usize::from(register)) as i32
    }

    fn get(&mut self, register: u8) -> Value {
        if let Some(value) = self.gpr[usize::from(register)] {
            return value;
        }
        let value = self.b.ins().load(
            types::I32,
            Self::flags(),
            self.cpu,
            Self::gpr_offset(register),
        );
        self.gpr[usize::from(register)] = Some(value);
        value
    }

    fn set(&mut self, register: u8, value: Value) {
        self.b
            .ins()
            .store(Self::flags(), value, self.cpu, Self::gpr_offset(register));
        self.gpr[usize::from(register)] = Some(value);
    }

    fn sp(&mut self, special: Special) -> Value {
        if let Some(value) = self.special[special.slot()] {
            return value;
        }
        let value = self
            .b
            .ins()
            .load(types::I32, Self::flags(), self.cpu, special.offset());
        self.special[special.slot()] = Some(value);
        value
    }

    fn set_sp(&mut self, special: Special, value: Value) {
        self.b
            .ins()
            .store(Self::flags(), value, self.cpu, special.offset());
        self.special[special.slot()] = Some(value);
    }

    fn set_pc(&mut self, value: Value) {
        self.b.ins().store(
            Self::flags(),
            value,
            self.cpu,
            offset_of!(PpcCpu, pc) as i32,
        );
    }

    fn invalidate(&mut self) {
        self.gpr = [None; 32];
        self.special = [None; 4];
    }

    fn flush_time_base(&mut self) {
        if self.pending_cycles == 0 {
            return;
        }
        let offset = offset_of!(PpcCpu, time_base) as i32;
        let current = self
            .b
            .ins()
            .load(types::I64, Self::flags(), self.cpu, offset);
        let next = self
            .b
            .ins()
            .iadd_imm(current, i64::from(self.pending_cycles));
        self.b.ins().store(Self::flags(), next, self.cpu, offset);
        self.pending_cycles = 0;
    }

    fn call(&mut self, helper: usize, four: bool, args: &[Value]) -> Value {
        let callee = self.b.ins().iconst(self.ptr, helper as i64);
        let signature = if four { self.sig4 } else { self.sig3 };
        let mut all = Vec::with_capacity(args.len() + 1);
        all.push(self.ctx);
        all.extend_from_slice(args);
        let call = self.b.ins().call_indirect(signature, callee, &all);
        self.b.inst_results(call)[0]
    }

    /// Branch to a cold fault exit, flushing pending time-base increments there.
    fn check_fault(&mut self, result: Value, index: u32) {
        let tramp = self.b.create_block();
        self.b.set_cold_block(tramp);
        let next = self.b.create_block();
        let faulted = self.b.ins().ushr_imm(result, i64::from(helpers::FAULT_BIT));
        self.b.ins().brif(faulted, tramp, &[], next, &[]);
        self.fault_exits
            .push((tramp, self.pending_cycles, index, self.done));
        self.b.switch_to_block(next);
    }

    /// End the current block: continue into the successor (`(pc, len,
    /// head)`) matching the new `pc` if the budget allows, else return the
    /// instructions completed so far.
    fn end_block(
        &mut self,
        start_pc: u32,
        len: u32,
        successors: &[(u32, u32, Block)],
        budget: Option<Value>,
    ) {
        self.flush_time_base();
        let next_pc = if self.pc_in_sync {
            self.next_pc
        } else {
            let next = self.c32(start_pc.wrapping_add(len * 4));
            self.set_pc(next);
            Some(next)
        };
        let total = self.b.ins().iadd_imm(self.done, i64::from(len));
        let Some(budget) = budget.filter(|_| !successors.is_empty()) else {
            self.b
                .ins()
                .jump(self.normal_exit, &[BlockArg::Value(total)]);
            return;
        };
        let next_pc = next_pc.unwrap_or_else(|| {
            self.b.ins().load(
                types::I32,
                Self::flags(),
                self.cpu,
                offset_of!(PpcCpu, pc) as i32,
            )
        });
        let exit = self.b.create_block();
        if self.may_change_code {
            // A callback may have changed code later blocks were built from.
            let same = self.b.create_block();
            let callee = self
                .b
                .ins()
                .iconst(self.ptr, helpers::code_changed as *const () as i64);
            let call = self.b.ins().call_indirect(self.sig1, callee, &[self.ctx]);
            let changed = self.b.inst_results(call)[0];
            self.b.ins().brif(changed, exit, &[], same, &[]);

            self.b.switch_to_block(same);
        }
        let total64 = self.b.ins().uextend(types::I64, total);
        for &(pc, len, head) in successors {
            let after = self.b.ins().iadd_imm(total64, i64::from(len));
            let fits = self
                .b
                .ins()
                .icmp(IntCC::UnsignedLessThanOrEqual, after, budget);
            let matches = self
                .b
                .ins()
                .icmp_imm(IntCC::Equal, next_pc, i64::from(pc as i32));
            let go = self.b.ins().band(fits, matches);
            let otherwise = self.b.create_block();
            self.b
                .ins()
                .brif(go, head, &[BlockArg::Value(total)], otherwise, &[]);
            self.b.switch_to_block(otherwise);
        }
        self.b.ins().jump(exit, &[]);
        self.b.switch_to_block(exit);
        self.b
            .ins()
            .jump(self.normal_exit, &[BlockArg::Value(total)]);
    }

    /// Emit the normal return and cold fault exits shared by every block.
    fn finish(mut self) {
        self.b.switch_to_block(self.normal_exit);
        let completed = self.b.block_params(self.normal_exit)[0];
        self.b.ins().return_(&[completed]);
        for (tramp, pending_cycles, index, done) in std::mem::take(&mut self.fault_exits) {
            self.b.switch_to_block(tramp);
            self.pending_cycles = pending_cycles;
            self.flush_time_base();
            let completed = self.b.ins().iadd_imm(done, i64::from(index));
            self.b.ins().jump(self.fault, &[BlockArg::Value(completed)]);
        }
        self.b.switch_to_block(self.fault);
        let completed = self.b.block_params(self.fault)[0];
        let stopped = self
            .b
            .ins()
            .bor_imm(completed, i64::from(super::STOPPED as i32));
        self.b.ins().return_(&[stopped]);
    }

    // ---- catch-all ------------------------------------------------------

    fn step(&mut self, index: u32, pc: u32, word: u32) {
        self.flush_time_base();
        let word_value = self.c32(word);
        let pc_value = self.c32(pc);
        let result = self.call(
            helpers::step as *const () as usize,
            false,
            &[word_value, pc_value],
        );
        self.check_fault(result, index);
        self.invalidate();
        self.pc_in_sync = true;
        self.next_pc = None;
        self.may_change_code = true;
    }

    // ---- flags ----------------------------------------------------------

    /// `nibble = LT/GT/EQ || XER.SO`.
    fn compare_nibble(&mut self, lt: Value, gt: Value) -> Value {
        let (eight, four, two) = (self.c32(8), self.c32(4), self.c32(2));
        let not_lt = self.b.ins().select(gt, four, two);
        let ordering = self.b.ins().select(lt, eight, not_lt);
        let xer = self.sp(Special::Xer);
        let so = self.b.ins().ushr_imm(xer, 31);
        self.b.ins().bor(ordering, so)
    }

    fn set_cr_field(&mut self, field: u8, nibble: Value) {
        let shift = 28 - 4 * u32::from(field);
        let cr = self.sp(Special::Cr);
        let kept = self
            .b
            .ins()
            .band_imm(cr, i64::from((!(0xfu32 << shift)) as i32));
        let placed = if shift == 0 {
            nibble
        } else {
            self.b.ins().ishl_imm(nibble, i64::from(shift))
        };
        let merged = self.b.ins().bor(kept, placed);
        self.set_sp(Special::Cr, merged);
    }

    fn cr0_from(&mut self, result: Value) {
        let lt = self.b.ins().icmp_imm(IntCC::SignedLessThan, result, 0);
        let gt = self.b.ins().icmp_imm(IntCC::SignedGreaterThan, result, 0);
        let nibble = self.compare_nibble(lt, gt);
        self.set_cr_field(0, nibble);
    }

    fn set_ca(&mut self, carry: Value) {
        let xer = self.sp(Special::Xer);
        let cleared = self.b.ins().band_imm(xer, i64::from((!CA_MASK) as i32));
        let bit = self.b.ins().uextend(types::I32, carry);
        let placed = self.b.ins().ishl_imm(bit, 29);
        let merged = self.b.ins().bor(cleared, placed);
        self.set_sp(Special::Xer, merged);
    }

    fn record(&mut self, rc: bool, result: Value) {
        if rc {
            self.cr0_from(result);
        }
    }

    // ---- branches -------------------------------------------------------

    /// Evaluate BO/BI; `None` means "always taken". Mirrors
    /// `evaluate_branch_condition` including the CTR side effect.
    fn branch_taken(&mut self, bo: u8, bi: u8) -> Option<Value> {
        let mut taken: Option<Value> = None;
        if bo & BO_IGNORE_CTR == 0 {
            let ctr = self.sp(Special::Ctr);
            let decremented = self.b.ins().iadd_imm(ctr, -1);
            self.set_sp(Special::Ctr, decremented);
            let cc = if bo & BO_CTR_ZERO != 0 {
                IntCC::Equal
            } else {
                IntCC::NotEqual
            };
            taken = Some(self.b.ins().icmp_imm(cc, decremented, 0));
        }
        if bo & BO_IGNORE_CONDITION == 0 {
            let cr = self.sp(Special::Cr);
            let shifted = self.b.ins().ushr_imm(cr, i64::from(31 - u32::from(bi)));
            let bit = self.b.ins().band_imm(shifted, 1);
            let cc = if bo & BO_CONDITION_TRUE != 0 {
                IntCC::NotEqual
            } else {
                IntCC::Equal
            };
            let cond = self.b.ins().icmp_imm(cc, bit, 0);
            taken = Some(match taken {
                Some(ctr_ok) => self.b.ins().band(ctr_ok, cond),
                None => cond,
            });
        }
        taken
    }

    /// Finish a branch: write LR (if `lk`) and `cpu.pc`.
    fn finish_branch(&mut self, pc: u32, taken: Option<Value>, target: Value, lk: bool) {
        let fallthrough = self.c32(pc.wrapping_add(4));
        let next = match taken {
            Some(taken) => self.b.ins().select(taken, target, fallthrough),
            None => target,
        };
        if lk {
            self.set_sp(Special::Lr, fallthrough);
        }
        self.set_pc(next);
        self.pc_in_sync = true;
        self.next_pc = Some(next);
    }

    // ---- memory ---------------------------------------------------------

    fn load(&mut self, kind: Load, index: u32, pc: u32, rt: u8, ea: Value) {
        let pc_value = self.c32(pc);
        let result = self.call(kind.helper(), false, &[ea, pc_value]);
        self.may_change_code = true;
        self.check_fault(result, index);
        let value = self.b.ins().ireduce(types::I32, result);
        self.set(rt, value);
    }

    fn store(&mut self, helper: usize, index: u32, pc: u32, ea: Value, value: Value) {
        let pc_value = self.c32(pc);
        let result = self.call(helper, true, &[ea, value, pc_value]);
        self.check_fault(result, index);
        self.may_change_code = true;
    }

    fn fp_access(&mut self, helper: usize, index: u32, pc: u32, fr: u8, ea: Value) {
        let fr_value = self.c32(u32::from(fr));
        let pc_value = self.c32(pc);
        let result = self.call(helper, true, &[ea, fr_value, pc_value]);
        self.check_fault(result, index);
        self.may_change_code = true;
    }

    #[allow(clippy::too_many_arguments)]
    fn double_op(&mut self, index: u32, pc: u32, dest: u8, fra: u8, frb: u8, op: u32, rc: bool) {
        let fields = helpers::fp_fields(dest, fra, frb, op, rc);
        let fields = self.c32(fields);
        let pc = self.c32(pc);
        let result = self.call(helper!(double_op), false, &[fields, pc]);
        self.check_fault(result, index);
        self.invalidate();
    }

    fn fp_helper(instr: PpcInstr) -> usize {
        match instr {
            PpcInstr::Lfs { .. }
            | PpcInstr::Lfsx { .. }
            | PpcInstr::Lfsu { .. }
            | PpcInstr::Lfsux { .. } => helper!(lfs),
            PpcInstr::Lfd { .. }
            | PpcInstr::Lfdx { .. }
            | PpcInstr::Lfdu { .. }
            | PpcInstr::Lfdux { .. } => helper!(lfd),
            PpcInstr::Stfs { .. }
            | PpcInstr::Stfsx { .. }
            | PpcInstr::Stfsu { .. }
            | PpcInstr::Stfsux { .. } => helper!(stfs),
            _ => helper!(stfd),
        }
    }

    fn d_ea(&mut self, ra: u8, d: i16) -> Value {
        if ra == 0 {
            return self.c32(i32::from(d) as u32);
        }
        let base = self.get(ra);
        self.b.ins().iadd_imm(base, i64::from(d))
    }

    fn x_ea(&mut self, ra: u8, rb: u8) -> Value {
        let index = self.get(rb);
        if ra == 0 {
            return index;
        }
        let base = self.get(ra);
        self.b.ins().iadd(base, index)
    }

    // ---- inline lowering ------------------------------------------------

    /// Lower `word` inline. Returns `false` (having emitted nothing) when the
    /// instruction must go through the catch-all helper.
    fn inline(&mut self, index: u32, pc: u32, word: u32) -> bool {
        let Ok(instr) = decode(word) else {
            return false;
        };
        // Instructions whose interpreter implementation can fault, trap or
        // report an invalid form are filtered out before any code is emitted.
        let supported = match instr {
            PpcInstr::Cmp { l, .. }
            | PpcInstr::Cmpl { l, .. }
            | PpcInstr::Cmpi { l, .. }
            | PpcInstr::Cmpli { l, .. } => !l,
            PpcInstr::Add { oe, .. }
            | PpcInstr::Addc { oe, .. }
            | PpcInstr::Adde { oe, .. }
            | PpcInstr::Subf { oe, .. }
            | PpcInstr::Neg { oe, .. }
            | PpcInstr::Mullw { oe, .. } => !oe,
            PpcInstr::Divw { oe, .. } => !oe,
            PpcInstr::Mfspr { spr, .. } | PpcInstr::Mtspr { spr, .. } => {
                matches!(spr, 1 | 8 | 9)
            }
            PpcInstr::Bcctr { bo, .. } => bo & BO_IGNORE_CTR != 0,
            PpcInstr::Lmw { rt, ra, .. } => ra < rt,
            PpcInstr::Lbzu { rt, ra, .. }
            | PpcInstr::Lhzu { rt, ra, .. }
            | PpcInstr::Lhau { rt, ra, .. }
            | PpcInstr::Lwzu { rt, ra, .. }
            | PpcInstr::Lbzux { rt, ra, .. }
            | PpcInstr::Lhzux { rt, ra, .. }
            | PpcInstr::Lhaux { rt, ra, .. }
            | PpcInstr::Lwzux { rt, ra, .. } => ra != 0 && ra != rt,
            PpcInstr::Stwu { ra, .. }
            | PpcInstr::Stbu { ra, .. }
            | PpcInstr::Sthu { ra, .. }
            | PpcInstr::Stwux { ra, .. }
            | PpcInstr::Stbux { ra, .. }
            | PpcInstr::Sthux { ra, .. }
            | PpcInstr::Lfsu { ra, .. }
            | PpcInstr::Lfdu { ra, .. }
            | PpcInstr::Stfsu { ra, .. }
            | PpcInstr::Stfdu { ra, .. }
            | PpcInstr::Lfsux { ra, .. }
            | PpcInstr::Lfdux { ra, .. }
            | PpcInstr::Stfsux { ra, .. }
            | PpcInstr::Stfdux { ra, .. } => ra != 0,
            PpcInstr::Lhax { .. }
            | PpcInstr::Fadds { .. }
            | PpcInstr::Fsubs { .. }
            | PpcInstr::Fmuls { .. }
            | PpcInstr::Fdivs { .. }
            | PpcInstr::Fctiwz { .. }
            | PpcInstr::Fdiv { .. }
            | PpcInstr::Fcmpo { .. }
            | PpcInstr::Lfs { .. }
            | PpcInstr::Lfd { .. }
            | PpcInstr::Stfs { .. }
            | PpcInstr::Stfd { .. }
            | PpcInstr::Lfsx { .. }
            | PpcInstr::Lfdx { .. }
            | PpcInstr::Stfsx { .. }
            | PpcInstr::Stfdx { .. } => true,
            PpcInstr::Addi { .. }
            | PpcInstr::Addis { .. }
            | PpcInstr::Ori { .. }
            | PpcInstr::Oris { .. }
            | PpcInstr::Xori { .. }
            | PpcInstr::Xoris { .. }
            | PpcInstr::AndiDot { .. }
            | PpcInstr::AndisDot { .. }
            | PpcInstr::Mulli { .. }
            | PpcInstr::Addic { .. }
            | PpcInstr::AddicDot { .. }
            | PpcInstr::Subfic { .. }
            | PpcInstr::And { .. }
            | PpcInstr::Or { .. }
            | PpcInstr::Xor { .. }
            | PpcInstr::Nor { .. }
            | PpcInstr::Nand { .. }
            | PpcInstr::Andc { .. }
            | PpcInstr::Orc { .. }
            | PpcInstr::Eqv { .. }
            | PpcInstr::Slw { .. }
            | PpcInstr::Srw { .. }
            | PpcInstr::Srawi { .. }
            | PpcInstr::Sraw { .. }
            | PpcInstr::Rlwinm { .. }
            | PpcInstr::Rlwimi { .. }
            | PpcInstr::Rlwnm { .. }
            | PpcInstr::Extsb { .. }
            | PpcInstr::Extsh { .. }
            | PpcInstr::Cntlzw { .. }
            | PpcInstr::B { .. }
            | PpcInstr::Bc { .. }
            | PpcInstr::Bclr { .. }
            | PpcInstr::Lwz { .. }
            | PpcInstr::Lbz { .. }
            | PpcInstr::Lhz { .. }
            | PpcInstr::Lha { .. }
            | PpcInstr::Stw { .. }
            | PpcInstr::Stmw { .. }
            | PpcInstr::Stb { .. }
            | PpcInstr::Sth { .. }
            | PpcInstr::Lwzx { .. }
            | PpcInstr::Lbzx { .. }
            | PpcInstr::Lhzx { .. }
            | PpcInstr::Stwx { .. }
            | PpcInstr::Stbx { .. }
            | PpcInstr::Sthx { .. } => true,
            _ => false,
        };
        if !supported {
            return false;
        }
        self.lower(index, pc, instr);
        true
    }

    fn lower(&mut self, index: u32, pc: u32, instr: PpcInstr) {
        let is_branch = matches!(
            instr,
            PpcInstr::B { .. }
                | PpcInstr::Bc { .. }
                | PpcInstr::Bclr { .. }
                | PpcInstr::Bcctr { .. }
        );
        match instr {
            PpcInstr::Fctiwz { frt: dest, frb, rc } => {
                self.double_op(index, pc, dest, 0, frb, 0, rc);
            }
            PpcInstr::Fdiv {
                frt: dest,
                fra,
                frb,
                rc,
            } => {
                self.double_op(index, pc, dest, fra, frb, 1, rc);
            }
            PpcInstr::Fcmpo { bf, fra, frb } => {
                self.double_op(index, pc, bf, fra, frb, 2, false);
            }
            PpcInstr::Fadds { frt, fra, frb, rc }
            | PpcInstr::Fsubs { frt, fra, frb, rc }
            | PpcInstr::Fdivs { frt, fra, frb, rc }
            | PpcInstr::Fmuls {
                frt,
                fra,
                frc: frb,
                rc,
            } => {
                let op = match instr {
                    PpcInstr::Fadds { .. } => 0,
                    PpcInstr::Fsubs { .. } => 1,
                    PpcInstr::Fmuls { .. } => 2,
                    _ => 3,
                };
                let fields = helpers::fp_fields(frt, fra, frb, op, rc);
                let fields = self.c32(fields);
                let pc_value = self.c32(pc);
                let result = self.call(helper!(single_binary), false, &[fields, pc_value]);
                self.check_fault(result, index);
                self.invalidate();
            }
            PpcInstr::Addi { rt, ra, si } => {
                let value = if ra == 0 {
                    self.c32(i32::from(si) as u32)
                } else {
                    let base = self.get(ra);
                    self.b.ins().iadd_imm(base, i64::from(si))
                };
                self.set(rt, value);
            }
            PpcInstr::Addis { rt, ra, si } => {
                let imm = (i32::from(si) << 16) as u32;
                let value = if ra == 0 {
                    self.c32(imm)
                } else {
                    let base = self.get(ra);
                    self.b.ins().iadd_imm(base, i64::from(imm as i32))
                };
                self.set(rt, value);
            }
            PpcInstr::Ori { ra, rs, ui } => {
                let source = self.get(rs);
                let value = self.b.ins().bor_imm(source, i64::from(ui));
                self.set(ra, value);
            }
            PpcInstr::Oris { ra, rs, ui } => {
                let source = self.get(rs);
                let value = self
                    .b
                    .ins()
                    .bor_imm(source, i64::from((u32::from(ui) << 16) as i32));
                self.set(ra, value);
            }
            PpcInstr::Xori { ra, rs, ui } => {
                let source = self.get(rs);
                let value = self.b.ins().bxor_imm(source, i64::from(ui));
                self.set(ra, value);
            }
            PpcInstr::Xoris { ra, rs, ui } => {
                let source = self.get(rs);
                let value = self
                    .b
                    .ins()
                    .bxor_imm(source, i64::from((u32::from(ui) << 16) as i32));
                self.set(ra, value);
            }
            PpcInstr::AndiDot { ra, rs, ui } => {
                let source = self.get(rs);
                let value = self.b.ins().band_imm(source, i64::from(ui));
                self.set(ra, value);
                self.cr0_from(value);
            }
            PpcInstr::AndisDot { ra, rs, ui } => {
                let source = self.get(rs);
                let value = self
                    .b
                    .ins()
                    .band_imm(source, i64::from((u32::from(ui) << 16) as i32));
                self.set(ra, value);
                self.cr0_from(value);
            }
            PpcInstr::Mulli { rt, ra, si } => {
                let source = self.get(ra);
                let value = self.b.ins().imul_imm(source, i64::from(si));
                self.set(rt, value);
            }
            PpcInstr::Addic { rt, ra, si } | PpcInstr::AddicDot { rt, ra, si } => {
                let source = self.get(ra);
                let value = self.b.ins().iadd_imm(source, i64::from(si));
                let carry = self.b.ins().icmp(IntCC::UnsignedLessThan, value, source);
                self.set(rt, value);
                self.set_ca(carry);
                if matches!(instr, PpcInstr::AddicDot { .. }) {
                    self.cr0_from(value);
                }
            }
            PpcInstr::Subfic { rt, ra, si } => {
                let source = self.get(ra);
                let imm = self.c32(i32::from(si) as u32);
                let value = self.b.ins().isub(imm, source);
                let carry = self
                    .b
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, imm, source);
                self.set(rt, value);
                self.set_ca(carry);
            }
            PpcInstr::Add { rt, ra, rb, rc, .. } => {
                let (a, b) = (self.get(ra), self.get(rb));
                let value = self.b.ins().iadd(a, b);
                self.set(rt, value);
                self.record(rc, value);
            }
            PpcInstr::Addc { rt, ra, rb, rc, .. } | PpcInstr::Adde { rt, ra, rb, rc, .. } => {
                let (a, b) = (self.get(ra), self.get(rb));
                let sum = self.b.ins().iadd(a, b);
                let carry1 = self.b.ins().icmp(IntCC::UnsignedLessThan, sum, a);
                let (value, carry) = if matches!(instr, PpcInstr::Adde { .. }) {
                    let xer = self.sp(Special::Xer);
                    let ca = self.b.ins().ushr_imm(xer, 29);
                    let ca = self.b.ins().band_imm(ca, 1);
                    let value = self.b.ins().iadd(sum, ca);
                    let carry2 = self.b.ins().icmp(IntCC::UnsignedLessThan, value, sum);
                    (value, self.b.ins().bor(carry1, carry2))
                } else {
                    (sum, carry1)
                };
                self.set(rt, value);
                self.set_ca(carry);
                self.record(rc, value);
            }
            PpcInstr::Subf { rt, ra, rb, rc, .. } => {
                let (a, b) = (self.get(ra), self.get(rb));
                let value = self.b.ins().isub(b, a);
                self.set(rt, value);
                self.record(rc, value);
            }
            PpcInstr::Neg { rt, ra, rc, .. } => {
                let a = self.get(ra);
                let value = self.b.ins().ineg(a);
                self.set(rt, value);
                self.record(rc, value);
            }
            PpcInstr::Mullw { rt, ra, rb, rc, .. } => {
                let (a, b) = (self.get(ra), self.get(rb));
                let value = self.b.ins().imul(a, b);
                self.set(rt, value);
                self.record(rc, value);
            }
            PpcInstr::Divw { rt, ra, rb, rc, .. } => {
                let (a, b) = (self.get(ra), self.get(rb));
                let zero_divisor = self.b.ins().icmp_imm(IntCC::Equal, b, 0);
                let min = self.b.ins().icmp_imm(IntCC::Equal, a, i64::from(i32::MIN));
                let minus_one = self.b.ins().icmp_imm(IntCC::Equal, b, -1);
                let overflow = self.b.ins().band(min, minus_one);
                let invalid = self.b.ins().bor(zero_divisor, overflow);
                let one = self.c32(1);
                let safe_b = self.b.ins().select(invalid, one, b);
                let quotient = self.b.ins().sdiv(a, safe_b);
                let zero = self.c32(0);
                let value = self.b.ins().select(invalid, zero, quotient);
                self.set(rt, value);
                self.record(rc, value);
            }
            PpcInstr::And { ra, rs, rb, rc } => {
                self.logical(ra, rs, rb, rc, |b, l, r| b.ins().band(l, r))
            }
            PpcInstr::Or { ra, rs, rb, rc } => {
                self.logical(ra, rs, rb, rc, |b, l, r| b.ins().bor(l, r))
            }
            PpcInstr::Xor { ra, rs, rb, rc } => {
                self.logical(ra, rs, rb, rc, |b, l, r| b.ins().bxor(l, r))
            }
            PpcInstr::Nor { ra, rs, rb, rc } => self.logical(ra, rs, rb, rc, |b, l, r| {
                let or = b.ins().bor(l, r);
                b.ins().bnot(or)
            }),
            PpcInstr::Nand { ra, rs, rb, rc } => self.logical(ra, rs, rb, rc, |b, l, r| {
                let and = b.ins().band(l, r);
                b.ins().bnot(and)
            }),
            PpcInstr::Andc { ra, rs, rb, rc } => {
                self.logical(ra, rs, rb, rc, |b, l, r| b.ins().band_not(l, r))
            }
            PpcInstr::Orc { ra, rs, rb, rc } => {
                self.logical(ra, rs, rb, rc, |b, l, r| b.ins().bor_not(l, r))
            }
            PpcInstr::Eqv { ra, rs, rb, rc } => self.logical(ra, rs, rb, rc, |b, l, r| {
                let xor = b.ins().bxor(l, r);
                b.ins().bnot(xor)
            }),
            PpcInstr::Slw { ra, rs, rb, rc } | PpcInstr::Srw { ra, rs, rb, rc } => {
                let (source, amount) = (self.get(rs), self.get(rb));
                let n = self.b.ins().band_imm(amount, 0x3f);
                let shifted = if matches!(instr, PpcInstr::Slw { .. }) {
                    self.b.ins().ishl(source, n)
                } else {
                    self.b.ins().ushr(source, n)
                };
                let too_far = self
                    .b
                    .ins()
                    .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, n, 32);
                let zero = self.c32(0);
                let value = self.b.ins().select(too_far, zero, shifted);
                self.set(ra, value);
                self.record(rc, value);
            }
            PpcInstr::Srawi { ra, rs, sh, rc } => {
                let source = self.get(rs);
                let carry = if sh == 0 {
                    self.b.ins().iconst(types::I8, 0)
                } else {
                    let lost = self.b.ins().band_imm(source, (1i64 << sh) - 1);
                    let negative = self.b.ins().icmp_imm(IntCC::SignedLessThan, source, 0);
                    let any = self.b.ins().icmp_imm(IntCC::NotEqual, lost, 0);
                    self.b.ins().band(negative, any)
                };
                let value = self.b.ins().sshr_imm(source, i64::from(sh));
                self.set(ra, value);
                self.set_ca(carry);
                self.record(rc, value);
            }
            PpcInstr::Sraw { ra, rs, rb, rc } => {
                let (source, amount) = (self.get(rs), self.get(rb));
                let n = self.b.ins().band_imm(amount, 63);
                let too_far = self
                    .b
                    .ins()
                    .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, n, 32);
                let shifted = self.b.ins().sshr(source, n);
                let sign = self.b.ins().sshr_imm(source, 31);
                let value = self.b.ins().select(too_far, sign, shifted);
                let restored = self.b.ins().ishl(shifted, n);
                let lost = self.b.ins().icmp(IntCC::NotEqual, source, restored);
                let negative = self.b.ins().icmp_imm(IntCC::SignedLessThan, source, 0);
                let lost = self.b.ins().bor(lost, too_far);
                let carry = self.b.ins().band(negative, lost);
                self.set(ra, value);
                self.set_ca(carry);
                self.record(rc, value);
            }
            PpcInstr::Rlwinm {
                ra,
                rs,
                sh,
                mb,
                me,
                rc,
            } => {
                let source = self.get(rs);
                let rotated = self.b.ins().rotl_imm(source, i64::from(sh));
                let mask = PpcCpu::mask32(mb, me);
                let value = self.b.ins().band_imm(rotated, i64::from(mask as i32));
                self.set(ra, value);
                self.record(rc, value);
            }
            PpcInstr::Rlwimi {
                ra,
                rs,
                sh,
                mb,
                me,
                rc,
            } => {
                let source = self.get(rs);
                let rotated = self.b.ins().rotl_imm(source, i64::from(sh));
                let mask = i64::from(PpcCpu::mask32(mb, me) as i32);
                let inserted = self.b.ins().band_imm(rotated, mask);
                let old = self.get(ra);
                let kept = self.b.ins().band_imm(old, !mask);
                let value = self.b.ins().bor(inserted, kept);
                self.set(ra, value);
                self.record(rc, value);
            }
            PpcInstr::Rlwnm {
                ra,
                rs,
                rb,
                mb,
                me,
                rc,
            } => {
                let (source, amount) = (self.get(rs), self.get(rb));
                let n = self.b.ins().band_imm(amount, 0x1f);
                let rotated = self.b.ins().rotl(source, n);
                let mask = PpcCpu::mask32(mb, me);
                let value = self.b.ins().band_imm(rotated, i64::from(mask as i32));
                self.set(ra, value);
                self.record(rc, value);
            }
            PpcInstr::Extsb { ra, rs, rc } | PpcInstr::Extsh { ra, rs, rc } => {
                let source = self.get(rs);
                let narrow = if matches!(instr, PpcInstr::Extsb { .. }) {
                    types::I8
                } else {
                    types::I16
                };
                let low = self.b.ins().ireduce(narrow, source);
                let value = self.b.ins().sextend(types::I32, low);
                self.set(ra, value);
                self.record(rc, value);
            }
            PpcInstr::Cntlzw { ra, rs, rc } => {
                let source = self.get(rs);
                let value = self.b.ins().clz(source);
                self.set(ra, value);
                self.record(rc, value);
            }
            PpcInstr::Cmp { bf, ra, rb, .. } | PpcInstr::Cmpl { bf, ra, rb, .. } => {
                let (signed_lt, signed_gt) = if matches!(instr, PpcInstr::Cmp { .. }) {
                    (IntCC::SignedLessThan, IntCC::SignedGreaterThan)
                } else {
                    (IntCC::UnsignedLessThan, IntCC::UnsignedGreaterThan)
                };
                let (a, b) = (self.get(ra), self.get(rb));
                let lt = self.b.ins().icmp(signed_lt, a, b);
                let gt = self.b.ins().icmp(signed_gt, a, b);
                let nibble = self.compare_nibble(lt, gt);
                self.set_cr_field(bf, nibble);
            }
            PpcInstr::Cmpi { bf, ra, si, .. } => {
                let a = self.get(ra);
                let lt = self
                    .b
                    .ins()
                    .icmp_imm(IntCC::SignedLessThan, a, i64::from(si));
                let gt = self
                    .b
                    .ins()
                    .icmp_imm(IntCC::SignedGreaterThan, a, i64::from(si));
                let nibble = self.compare_nibble(lt, gt);
                self.set_cr_field(bf, nibble);
            }
            PpcInstr::Cmpli { bf, ra, ui, .. } => {
                let a = self.get(ra);
                let lt = self
                    .b
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThan, a, i64::from(ui));
                let gt = self
                    .b
                    .ins()
                    .icmp_imm(IntCC::UnsignedGreaterThan, a, i64::from(ui));
                let nibble = self.compare_nibble(lt, gt);
                self.set_cr_field(bf, nibble);
            }
            PpcInstr::Mfspr { rt, spr } => {
                let value = self.sp(Self::spr(spr));
                self.set(rt, value);
            }
            PpcInstr::Mtspr { spr, rs } => {
                let value = self.get(rs);
                self.set_sp(Self::spr(spr), value);
            }
            PpcInstr::B {
                displacement,
                aa,
                lk,
            } => {
                let target = if aa {
                    displacement as u32
                } else {
                    pc.wrapping_add(displacement as u32)
                };
                let target = self.c32(target);
                self.finish_branch(pc, None, target, lk);
            }
            PpcInstr::Bc {
                bo,
                bi,
                displacement,
                aa,
                lk,
            } => {
                let taken = self.branch_taken(bo, bi);
                let target = if aa {
                    displacement as u32
                } else {
                    pc.wrapping_add(displacement as u32)
                };
                let target = self.c32(target);
                self.finish_branch(pc, taken, target, lk);
            }
            PpcInstr::Bclr { bo, bi, lk } => {
                let taken = self.branch_taken(bo, bi);
                let lr = self.sp(Special::Lr);
                let target = self.b.ins().band_imm(lr, !3);
                self.finish_branch(pc, taken, target, lk);
            }
            PpcInstr::Bcctr { bo, bi, lk } => {
                let taken = self.branch_taken(bo, bi);
                let ctr = self.sp(Special::Ctr);
                let target = self.b.ins().band_imm(ctr, !3);
                self.finish_branch(pc, taken, target, lk);
            }
            PpcInstr::Lwz { rt, ra, d } => {
                let ea = self.d_ea(ra, d);
                self.load(Load::Word, index, pc, rt, ea);
            }
            PpcInstr::Lmw { rt, ra, d } => {
                let base = self.d_ea(ra, d);
                for reg in rt..=31 {
                    let ea = self.b.ins().iadd_imm(base, i64::from(reg - rt) * 4);
                    self.load(Load::Word, index, pc, reg, ea);
                }
            }
            PpcInstr::Stmw { rs, ra, d } => {
                let base = self.d_ea(ra, d);
                for reg in rs..=31 {
                    let ea = self.b.ins().iadd_imm(base, i64::from(reg - rs) * 4);
                    let value = self.get(reg);
                    self.store(helper!(stw), index, pc, ea, value);
                }
            }
            PpcInstr::Lbz { rt, ra, d } => {
                let ea = self.d_ea(ra, d);
                self.load(Load::Byte, index, pc, rt, ea);
            }
            PpcInstr::Lhz { rt, ra, d } => {
                let ea = self.d_ea(ra, d);
                self.load(Load::Half, index, pc, rt, ea);
            }
            PpcInstr::Lha { rt, ra, d } => {
                let ea = self.d_ea(ra, d);
                self.load(Load::HalfAlgebraic, index, pc, rt, ea);
            }
            PpcInstr::Lwzx { rt, ra, rb } => {
                let ea = self.x_ea(ra, rb);
                self.load(Load::Word, index, pc, rt, ea);
            }
            PpcInstr::Lbzx { rt, ra, rb } => {
                let ea = self.x_ea(ra, rb);
                self.load(Load::Byte, index, pc, rt, ea);
            }
            PpcInstr::Lhzx { rt, ra, rb } => {
                let ea = self.x_ea(ra, rb);
                self.load(Load::Half, index, pc, rt, ea);
            }
            PpcInstr::Stw { rs, ra, d } => {
                let (ea, value) = (self.d_ea(ra, d), self.get(rs));
                self.store(helper!(stw), index, pc, ea, value);
            }
            PpcInstr::Stb { rs, ra, d } => {
                let (ea, value) = (self.d_ea(ra, d), self.get(rs));
                self.store(helper!(stb), index, pc, ea, value);
            }
            PpcInstr::Sth { rs, ra, d } => {
                let (ea, value) = (self.d_ea(ra, d), self.get(rs));
                self.store(helper!(sth), index, pc, ea, value);
            }
            PpcInstr::Stwx { rs, ra, rb } => {
                let (ea, value) = (self.x_ea(ra, rb), self.get(rs));
                self.store(helper!(stw), index, pc, ea, value);
            }
            PpcInstr::Stbx { rs, ra, rb } => {
                let (ea, value) = (self.x_ea(ra, rb), self.get(rs));
                self.store(helper!(stb), index, pc, ea, value);
            }
            PpcInstr::Sthx { rs, ra, rb } => {
                let (ea, value) = (self.x_ea(ra, rb), self.get(rs));
                self.store(helper!(sth), index, pc, ea, value);
            }
            PpcInstr::Stwu { rs, ra, d } => {
                let ea = self.d_ea(ra, d);
                let value = self.get(rs);
                self.store(helper!(stw), index, pc, ea, value);
                // RA is updated only once the store has succeeded.
                self.set(ra, ea);
            }
            PpcInstr::Lhax { rt, ra, rb } => {
                let ea = self.x_ea(ra, rb);
                self.load(Load::HalfAlgebraic, index, pc, rt, ea);
            }
            // Update forms: RA receives the EA once the access succeeded.
            PpcInstr::Lbzu { rt, ra, d }
            | PpcInstr::Lhzu { rt, ra, d }
            | PpcInstr::Lhau { rt, ra, d }
            | PpcInstr::Lwzu { rt, ra, d } => {
                let ea = self.d_ea(ra, d);
                let function = match instr {
                    PpcInstr::Lbzu { .. } => Load::Byte,
                    PpcInstr::Lhzu { .. } => Load::Half,
                    PpcInstr::Lhau { .. } => Load::HalfAlgebraic,
                    _ => Load::Word,
                };
                self.load(function, index, pc, rt, ea);
                self.set(ra, ea);
            }
            PpcInstr::Lbzux { rt, ra, rb }
            | PpcInstr::Lhzux { rt, ra, rb }
            | PpcInstr::Lhaux { rt, ra, rb }
            | PpcInstr::Lwzux { rt, ra, rb } => {
                let ea = self.x_ea(ra, rb);
                let function = match instr {
                    PpcInstr::Lbzux { .. } => Load::Byte,
                    PpcInstr::Lhzux { .. } => Load::Half,
                    PpcInstr::Lhaux { .. } => Load::HalfAlgebraic,
                    _ => Load::Word,
                };
                self.load(function, index, pc, rt, ea);
                self.set(ra, ea);
            }
            PpcInstr::Stbu { rs, ra, d } | PpcInstr::Sthu { rs, ra, d } => {
                let ea = self.d_ea(ra, d);
                let value = self.get(rs);
                let function = if matches!(instr, PpcInstr::Stbu { .. }) {
                    helper!(stb)
                } else {
                    helper!(sth)
                };
                self.store(function, index, pc, ea, value);
                self.set(ra, ea);
            }
            PpcInstr::Stbux { rs, ra, rb }
            | PpcInstr::Sthux { rs, ra, rb }
            | PpcInstr::Stwux { rs, ra, rb } => {
                let ea = self.x_ea(ra, rb);
                let value = self.get(rs);
                let function = match instr {
                    PpcInstr::Stbux { .. } => helper!(stb),
                    PpcInstr::Sthux { .. } => helper!(sth),
                    _ => helper!(stw),
                };
                self.store(function, index, pc, ea, value);
                self.set(ra, ea);
            }
            // Floating point: the helper moves the FPR itself.
            PpcInstr::Lfs { frt: fr, ra, d }
            | PpcInstr::Lfd { frt: fr, ra, d }
            | PpcInstr::Stfs { frs: fr, ra, d }
            | PpcInstr::Stfd { frs: fr, ra, d } => {
                let ea = self.d_ea(ra, d);
                self.fp_access(Self::fp_helper(instr), index, pc, fr, ea);
            }
            PpcInstr::Lfsx { frt: fr, ra, rb }
            | PpcInstr::Lfdx { frt: fr, ra, rb }
            | PpcInstr::Stfsx { frs: fr, ra, rb }
            | PpcInstr::Stfdx { frs: fr, ra, rb } => {
                let ea = self.x_ea(ra, rb);
                self.fp_access(Self::fp_helper(instr), index, pc, fr, ea);
            }
            PpcInstr::Lfsu { frt: fr, ra, d }
            | PpcInstr::Lfdu { frt: fr, ra, d }
            | PpcInstr::Stfsu { frs: fr, ra, d }
            | PpcInstr::Stfdu { frs: fr, ra, d } => {
                let ea = self.d_ea(ra, d);
                self.fp_access(Self::fp_helper(instr), index, pc, fr, ea);
                self.set(ra, ea);
            }
            PpcInstr::Lfsux { frt: fr, ra, rb }
            | PpcInstr::Lfdux { frt: fr, ra, rb }
            | PpcInstr::Stfsux { frs: fr, ra, rb }
            | PpcInstr::Stfdux { frs: fr, ra, rb } => {
                let ea = self.x_ea(ra, rb);
                self.fp_access(Self::fp_helper(instr), index, pc, fr, ea);
                self.set(ra, ea);
            }
            _ => unreachable!("filtered by `inline`"),
        }
        self.pending_cycles += 1;
        if is_branch {
            // `finish_branch` already wrote `cpu.pc`.
        } else {
            self.pc_in_sync = false;
        }
    }

    fn spr(spr: u16) -> Special {
        match spr {
            1 => Special::Xer,
            8 => Special::Lr,
            9 => Special::Ctr,
            _ => unreachable!("filtered by `inline`"),
        }
    }

    fn logical(
        &mut self,
        ra: u8,
        rs: u8,
        rb: u8,
        rc: bool,
        op: impl FnOnce(&mut FunctionBuilder<'_>, Value, Value) -> Value,
    ) {
        let (lhs, rhs) = (self.get(rs), self.get(rb));
        let value = op(self.b, lhs, rhs);
        self.set(ra, value);
        self.record(rc, value);
    }
}
