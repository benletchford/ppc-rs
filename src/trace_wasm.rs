//! A small, conservative PowerPC trace compiler for browser WebAssembly.
//! Unsupported instructions and paths are declined during planning.

use crate::{PpcCpu, PpcInstr, decode};
use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, MemArg, MemoryType, Module, TypeSection, ValType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceStep {
    pub pc: u32,
    pub word: u32,
    pub next_pc: u32,
}

/// Fixed little-endian scratch state shared with a generated WebAssembly trace.
/// Its byte offsets are independent of Rust's field layout. The host must apply
/// it to the CPU only when the trace returns a nonzero cycle count.
#[repr(align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceState {
    bytes: [u8; 152],
}

impl TraceState {
    pub fn from_cpu(cpu: &PpcCpu) -> Self {
        let mut state = Self { bytes: [0; 152] };
        for (register, value) in cpu.gpr.iter().enumerate() {
            state.write_u32(register * 4, *value);
        }
        state.write_u32(128, cpu.pc);
        state.write_u32(132, cpu.ctr);
        state.bytes[136..144].copy_from_slice(&cpu.time_base().to_le_bytes());
        state.write_u32(144, cpu.cr);
        state.write_u32(148, cpu.xer);
        state
    }

    pub fn apply_to_cpu(self, cpu: &mut PpcCpu) {
        for (register, value) in cpu.gpr.iter_mut().enumerate() {
            *value = self.read_u32(register * 4);
        }
        cpu.pc = self.read_u32(128);
        cpu.ctr = self.read_u32(132);
        cpu.set_time_base(u64::from_le_bytes(self.bytes[136..144].try_into().unwrap()));
        cpu.cr = self.read_u32(144);
        cpu.xer = self.read_u32(148);
    }

    pub fn as_bytes(&self) -> &[u8; 152] {
        &self.bytes
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8; 152] {
        &mut self.bytes
    }

    fn read_u32(&self, offset: usize) -> u32 {
        u32::from_le_bytes(self.bytes[offset..offset + 4].try_into().unwrap())
    }

    fn write_u32(&mut self, offset: usize, value: u32) {
        self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
}

const _: () = assert!(std::mem::size_of::<TraceState>() == 152);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TraceOp {
    Addi {
        rt: u8,
        ra: u8,
        si: i16,
    },
    Add {
        rt: u8,
        ra: u8,
        rb: u8,
    },
    Rlwinm {
        ra: u8,
        rs: u8,
        sh: u8,
        mb: u8,
        me: u8,
    },
    Cmpi {
        bf: u8,
        ra: u8,
        si: i16,
    },
    Cmpli {
        bf: u8,
        ra: u8,
        ui: u16,
    },
    Cmp {
        bf: u8,
        ra: u8,
        rb: u8,
    },
    Lwz {
        rt: u8,
        ra: u8,
        d: i16,
    },
    Bc {
        bo: u8,
        bi: u8,
        target: u32,
        fallthrough: u32,
        taken: bool,
    },
    B {
        target: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TracePlan {
    entry: u32,
    source: Vec<TraceStep>,
    steps: Vec<(u32, TraceOp)>,
}

impl TracePlan {
    pub fn entry(&self) -> u32 {
        self.entry
    }

    /// PCs whose instruction mappings must still match the recorded code.
    /// The caller must also reject a trace crossing the active halt/import range.
    pub fn instruction_pcs(&self) -> impl Iterator<Item = u32> + '_ {
        self.steps.iter().map(|(pc, _)| *pc)
    }

    /// Original fetched words and observed successors for code validation.
    pub fn source(&self) -> &[TraceStep] {
        &self.source
    }
}

/// Plan one observed, closed path. Only supported instructions are accepted.
/// `next_pc` records the observed successor, including the final backedge.
pub fn plan(entry: u32, fetched: &[TraceStep]) -> Result<TracePlan, String> {
    if fetched.len() > 256 {
        return Err("trace exceeds 256 instructions".into());
    }
    if fetched.is_empty() || fetched[0].pc != entry || entry & 3 != 0 {
        return Err("trace does not start at its entry".into());
    }
    let mut steps = Vec::with_capacity(fetched.len());
    for (index, step) in fetched.iter().enumerate() {
        if step.pc & 3 != 0 {
            return Err(format!("unaligned instruction at ${:08x}", step.pc));
        }
        if index > 0 && fetched[index - 1].next_pc != step.pc {
            return Err(format!("disconnected successor at ${:08x}", step.pc));
        }
        let fallthrough = step.pc.wrapping_add(4);
        let instruction = decode(step.word).map_err(|error| format!("decode: {error:?}"))?;
        let op = match instruction {
            PpcInstr::Addi { rt, ra, si } => TraceOp::Addi { rt, ra, si },
            PpcInstr::Add {
                rt,
                ra,
                rb,
                oe: false,
                rc: false,
            } => TraceOp::Add { rt, ra, rb },
            PpcInstr::Rlwinm {
                ra,
                rs,
                sh,
                mb,
                me,
                rc: false,
            } => TraceOp::Rlwinm { ra, rs, sh, mb, me },
            PpcInstr::Cmpi {
                bf,
                l: false,
                ra,
                si,
            } => TraceOp::Cmpi { bf, ra, si },
            PpcInstr::Cmpli {
                bf,
                l: false,
                ra,
                ui,
            } => TraceOp::Cmpli { bf, ra, ui },
            PpcInstr::Cmp {
                bf,
                l: false,
                ra,
                rb,
            } => TraceOp::Cmp { bf, ra, rb },
            PpcInstr::Lwz { rt, ra, d } => TraceOp::Lwz { rt, ra, d },
            PpcInstr::Bc {
                bo: bo @ (4 | 12),
                bi,
                displacement,
                aa: false,
                lk: false,
            } => {
                let target = step.pc.wrapping_add(displacement as u32);
                let taken = step.next_pc == target;
                if !taken && step.next_pc != fallthrough {
                    return Err(format!("invalid conditional successor at ${:08x}", step.pc));
                }
                TraceOp::Bc {
                    bo,
                    bi,
                    target,
                    fallthrough,
                    taken,
                }
            }
            PpcInstr::B {
                displacement,
                aa: false,
                lk: false,
            } => {
                let target = step.pc.wrapping_add(displacement as u32);
                if step.next_pc != target {
                    return Err(format!("invalid branch successor at ${:08x}", step.pc));
                }
                TraceOp::B { target }
            }
            _ => {
                return Err(format!(
                    "unsupported instruction at ${:08x}: {instruction:?}",
                    step.pc
                ));
            }
        };
        if !matches!(op, TraceOp::Bc { .. } | TraceOp::B { .. }) && step.next_pc != fallthrough {
            return Err(format!("invalid fallthrough at ${:08x}", step.pc));
        }
        steps.push((step.pc, op));
    }
    if fetched.last().unwrap().next_pc != entry {
        return Err("trace path does not close at its entry".into());
    }
    Ok(TracePlan {
        entry,
        source: fetched.to_vec(),
        steps,
    })
}

fn memarg(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

fn get_gpr(i: &mut wasm_encoder::InstructionSink<'_>, register: u8) {
    i.local_get(0).i32_load(memarg(u64::from(register) * 4, 2));
}

fn set_gpr_prefix(i: &mut wasm_encoder::InstructionSink<'_>) {
    i.local_get(0);
}

fn emit_guest_load(i: &mut wasm_encoder::InstructionSink<'_>, rt: u8, ra: u8, d: i16, pc: u32) {
    // Args: state=0, budget=1, guest-base=2, bytes=3, byte-len=4.
    // Locals: completed=5, current-PC=6, address=7, raw-word=8.
    if ra == 0 {
        i.i32_const(0);
    } else {
        get_gpr(i, ra);
    }
    i.i32_const(i32::from(d)).i32_add().local_set(7);
    // Decline the load before charging it. The interpreter owns alignment,
    // mapping, and fault behavior once the trace returns at this PC.
    i.local_get(7).i32_const(3).i32_and().i32_eqz();
    i.local_get(7).local_get(2).i32_ge_u().i32_and();
    i.local_get(4).i32_const(4).i32_ge_u().i32_and();
    i.local_get(7).local_get(2).i32_sub();
    i.local_get(4).i32_const(4).i32_sub().i32_le_u().i32_and();
    i.i32_eqz()
        .if_(BlockType::Empty)
        .i32_const(pc as i32)
        .local_set(6)
        .br(2)
        .end();
    i.local_get(3).local_get(7).local_get(2).i32_sub().i32_add();
    i.i32_load(memarg(0, 2)).local_set(8);
    set_gpr_prefix(i);
    i.local_get(8)
        .i32_const(0xff)
        .i32_and()
        .i32_const(24)
        .i32_shl();
    i.local_get(8)
        .i32_const(0xff00)
        .i32_and()
        .i32_const(8)
        .i32_shl()
        .i32_or();
    i.local_get(8)
        .i32_const(0x00ff_0000)
        .i32_and()
        .i32_const(8)
        .i32_shr_u()
        .i32_or();
    i.local_get(8).i32_const(24).i32_shr_u().i32_or();
    i.i32_store(memarg(u64::from(rt) * 4, 2));
}

fn mask32(mb: u8, me: u8) -> u32 {
    let left = u32::MAX >> mb;
    let right = u32::MAX << (31 - me);
    if mb <= me { left & right } else { left | right }
}

fn emit_compare(
    i: &mut wasm_encoder::InstructionSink<'_>,
    bf: u8,
    ra: u8,
    rhs: impl Fn(&mut wasm_encoder::InstructionSink<'_>),
    signed: bool,
) {
    let shift = 28 - u32::from(bf) * 4;
    let mask = 0xfu32 << shift;
    i.local_get(0).local_get(0).i32_load(memarg(144, 2));
    i.i32_const(!mask as i32).i32_and();
    get_gpr(i, ra);
    rhs(i);
    if signed {
        i.i32_lt_s();
    } else {
        i.i32_lt_u();
    }
    i.if_(BlockType::Result(ValType::I32))
        .i32_const((8u32 << shift) as i32);
    i.else_();
    get_gpr(i, ra);
    rhs(i);
    i.i32_eq().if_(BlockType::Result(ValType::I32));
    i.i32_const((2u32 << shift) as i32);
    i.else_().i32_const((4u32 << shift) as i32).end();
    i.end().i32_or();
    i.local_get(0).i32_load(memarg(148, 2));
    i.i32_const(31)
        .i32_shr_u()
        .i32_const(shift as i32)
        .i32_shl()
        .i32_or();
    i.i32_store(memarg(144, 2));
}

/// Compile a plan into a module importing `env.memory` and exporting
/// `run(state_ptr, budget, guest_data_base, data_ptr, data_len) -> cycles`.
///
/// The host must provide valid in-memory `TraceState` and data spans, verify
/// the instruction mappings recorded by the plan, exclude active halt/import
/// PCs, and prevent reentry while the generated module executes. A zero result
/// leaves the state untouched. A nonzero result contains complete instructions;
/// the host may commit it only after checking it does not exceed `budget`.
/// Guest reads outside the provided address span return at the current PC for
/// interpreter fallback. The module never writes guest memory.
pub fn compile_wasm(path: &TracePlan) -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function([ValType::I32; 5], [ValType::I32]);
    module.section(&types);
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    module.section(&imports);
    let mut funcs = FunctionSection::new();
    funcs.function(0);
    module.section(&funcs);
    let mut exports = ExportSection::new();
    exports.export("run", ExportKind::Func, 0);
    module.section(&exports);
    let mut code = CodeSection::new();
    let mut function = Function::new([(4, ValType::I32)]);
    let mut i = function.instructions();
    i.local_get(0)
        .i32_load(memarg(128, 2))
        .i32_const(path.entry as i32)
        .i32_ne();
    i.if_(BlockType::Empty).i32_const(0).return_().end();
    i.i32_const(path.entry as i32).local_set(6);
    i.block(BlockType::Empty).loop_(BlockType::Empty);
    i.local_get(1)
        .i32_const(path.steps.len() as i32)
        .i32_lt_u()
        .br_if(1);
    i.local_get(5)
        .local_get(1)
        .i32_const(path.steps.len() as i32)
        .i32_sub()
        .i32_gt_u()
        .br_if(1);
    for &(pc, op) in &path.steps {
        let next = match op {
            TraceOp::Addi { rt, ra, si } => {
                set_gpr_prefix(&mut i);
                if ra == 0 {
                    i.i32_const(0);
                } else {
                    get_gpr(&mut i, ra);
                }
                i.i32_const(i32::from(si))
                    .i32_add()
                    .i32_store(memarg(u64::from(rt) * 4, 2));
                pc.wrapping_add(4)
            }
            TraceOp::Add { rt, ra, rb } => {
                set_gpr_prefix(&mut i);
                get_gpr(&mut i, ra);
                get_gpr(&mut i, rb);
                i.i32_add().i32_store(memarg(u64::from(rt) * 4, 2));
                pc.wrapping_add(4)
            }
            TraceOp::Rlwinm { ra, rs, sh, mb, me } => {
                set_gpr_prefix(&mut i);
                get_gpr(&mut i, rs);
                i.i32_const(i32::from(sh))
                    .i32_rotl()
                    .i32_const(mask32(mb, me) as i32)
                    .i32_and();
                i.i32_store(memarg(u64::from(ra) * 4, 2));
                pc.wrapping_add(4)
            }
            TraceOp::Cmpi { bf, ra, si } => {
                emit_compare(
                    &mut i,
                    bf,
                    ra,
                    |i| {
                        i.i32_const(i32::from(si));
                    },
                    true,
                );
                pc.wrapping_add(4)
            }
            TraceOp::Cmpli { bf, ra, ui } => {
                emit_compare(
                    &mut i,
                    bf,
                    ra,
                    |i| {
                        i.i32_const(i32::from(ui));
                    },
                    false,
                );
                pc.wrapping_add(4)
            }
            TraceOp::Cmp { bf, ra, rb } => {
                emit_compare(&mut i, bf, ra, |i| get_gpr(i, rb), true);
                pc.wrapping_add(4)
            }
            TraceOp::Lwz { rt, ra, d } => {
                emit_guest_load(&mut i, rt, ra, d, pc);
                pc.wrapping_add(4)
            }
            TraceOp::Bc {
                bo,
                bi,
                target,
                fallthrough,
                taken,
            } => {
                i.local_get(0).i32_load(memarg(144, 2));
                i.i32_const((31 - bi) as i32)
                    .i32_shr_u()
                    .i32_const(1)
                    .i32_and();
                if bo == 4 {
                    i.i32_eqz();
                }
                let unexpected = if taken { fallthrough } else { target };
                if taken {
                    i.i32_eqz();
                }
                i.if_(BlockType::Empty);
                i.i32_const(unexpected as i32).local_set(6);
                i.local_get(5).i32_const(1).i32_add().local_set(5);
                i.br(2);
                i.end();
                if taken { target } else { fallthrough }
            }
            TraceOp::B { target } => target,
        };
        i.local_get(5).i32_const(1).i32_add().local_set(5);
        i.i32_const(next as i32).local_set(6);
    }
    i.br(0).end().end();
    i.local_get(0).local_get(6).i32_store(memarg(128, 2));
    i.local_get(0).local_get(0).i64_load(memarg(136, 3));
    i.local_get(5)
        .i64_extend_i32_u()
        .i64_add()
        .i64_store(memarg(136, 3));
    i.local_get(5).end();
    code.function(&function);
    module.section(&code);
    module.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arithmetic_loop() -> [TraceStep; 2] {
        [
            TraceStep {
                pc: 0x1000,
                word: 0x3863_0001,
                next_pc: 0x1004,
            },
            TraceStep {
                pc: 0x1004,
                word: 0x4bff_fffc,
                next_pc: 0x1000,
            },
        ]
    }

    #[test]
    fn plans_and_emits_valid_closed_loop() {
        let plan = plan(0x1000, &arithmetic_loop()).unwrap();
        assert_eq!(plan.entry(), 0x1000);
        assert_eq!(plan.source(), &arithmetic_loop());
        assert_eq!(plan.instruction_pcs().collect::<Vec<_>>(), [0x1000, 0x1004]);
        let bytes = compile_wasm(&plan);
        wasmparser::Validator::new().validate_all(&bytes).unwrap();
    }

    #[test]
    fn declines_disconnected_and_unsupported_paths() {
        let mut steps = arithmetic_loop();
        steps[0].next_pc = 0x1234;
        assert!(plan(0x1000, &steps).is_err());
        steps = arithmetic_loop();
        steps[0].word = 0x9063_0000;
        assert!(plan(0x1000, &steps).is_err());
        steps = arithmetic_loop();
        steps[1].next_pc = 0x1008;
        assert!(plan(0x1000, &steps).is_err());
        steps = arithmetic_loop();
        steps[0].pc = 0x1002;
        assert!(plan(0x1002, &steps).is_err());
    }

    #[test]
    fn architectural_state_round_trips() {
        let mut cpu = PpcCpu::new();
        cpu.pc = 0x1000;
        cpu.gpr[3] = 27;
        cpu.ctr = 4;
        cpu.cr = 0x1234_5678;
        cpu.xer = 0x8000_0000;
        cpu.set_time_base(99);
        let state = TraceState::from_cpu(&cpu);
        assert_eq!(&state.as_bytes()[12..16], &27u32.to_le_bytes());
        assert_eq!(&state.as_bytes()[128..132], &0x1000u32.to_le_bytes());
        assert_eq!(&state.as_bytes()[136..144], &99u64.to_le_bytes());
        assert_eq!(&state.as_bytes()[148..152], &0x8000_0000u32.to_le_bytes());
        let mut restored = PpcCpu::new();
        state.apply_to_cpu(&mut restored);
        assert_eq!(TraceState::from_cpu(&restored), state);
    }
}
