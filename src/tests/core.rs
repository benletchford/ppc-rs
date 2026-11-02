use super::*;

/// Operands biased towards the fast path's edges: normal singles with
/// few significant bits (exact results and ties), nearby and distant
/// exponents, the overflow/underflow binades, and arbitrary doubles.
fn fast_path_operand(state: &mut u64) -> u64 {
    let mut next = || {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };
    let r = next();
    let sign = ((r >> 63) as u32) << 31;
    let exponent = match r % 8 {
        0 => 1 + (next() % 4) as u32,
        1 => 250 + (next() % 4) as u32,
        2 => (next() % 256) as u32,
        _ => 117 + (next() % 20) as u32,
    };
    let fraction = match (r >> 8) % 4 {
        0 => (next() as u32 & 0x7f_ffff) & !((1 << (next() % 23)) - 1),
        1 => 0,
        _ => next() as u32 & 0x7f_ffff,
    };
    match (r >> 16) % 16 {
        0 => next(),
        1 => ((next() % 4096) as f64).to_bits() ^ (next() & 0xffff),
        2 => f64::from(f32::from_bits(sign | fraction)).to_bits(),
        _ => f64::from(f32::from_bits(sign | (exponent.min(254) << 23) | fraction)).to_bits(),
    }
}

#[test]
fn fast_single_arithmetic_matches_the_software_reference() {
    let iterations: u64 = std::env::var("PPC_FP_FAST_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut fast_hits = 0u64;
    let (mut cpu, mut reference) = (PpcCpu::new(), PpcCpu::new());
    for i in 0..iterations {
        let a = fast_path_operand(&mut state);
        let b = fast_path_operand(&mut state);
        let operation = [
            FpBinaryOperation::Add,
            FpBinaryOperation::Subtract,
            FpBinaryOperation::Multiply,
            FpBinaryOperation::Divide,
        ][(i % 4) as usize];
        // Rounding mode, enable bits and sticky exception bits.
        for target in [&mut cpu, &mut reference] {
            target.fpscr = (state as u32) & 0x9ff8_00ff;
            target.fpr[3] = state;
            target.cr = 0;
            target.pc = 0;
        }
        let rc = state & (1 << 40) != 0;
        let fast = cpu.try_fast_single_binary(3, a, b, operation, rc);
        reference.single_binary_reference(3, a, b, operation, rc);
        if fast {
            fast_hits += 1;
            assert_eq!(
                (cpu.fpr[3], cpu.fpscr, cpu.cr, cpu.pc),
                (
                    reference.fpr[3],
                    reference.fpscr,
                    reference.cr,
                    reference.pc
                ),
                "op {} a {a:#018x} b {b:#018x} fpscr {:#010x}",
                i % 4,
                reference.fpscr
            );
        }
        // fctiwz on the same operand.
        for target in [&mut cpu, &mut reference] {
            target.fpscr = (state as u32 >> 3) & 0x9ff8_00ff;
        }
        let bits = if i % 3 == 0 {
            (f64::from_bits(a) * 2f64.powi((i % 40) as i32)).to_bits()
        } else {
            a
        };
        if let Some(value) = cpu.fast_convert_to_i32_toward_zero(bits) {
            let expected = reference.fp_convert_to_i32(bits, ApRound::TowardZero);
            assert_eq!(
                (Some(value), cpu.fpscr),
                (expected, reference.fpscr),
                "fctiwz {bits:#018x}"
            );
        }
    }
    assert!(
        fast_hits > iterations / 4,
        "fast path rarely taken: {fast_hits}"
    );
}

struct CountingImmutableMemory {
    base: u32,
    bytes: Vec<u8>,
    instruction_reads: usize,
    token: u64,
}

impl CountingImmutableMemory {
    fn new(base: u32, words: &[u32]) -> Self {
        Self::with_token(base, words, 1)
    }

    fn with_token(base: u32, words: &[u32], token: u64) -> Self {
        Self {
            base,
            bytes: words.iter().flat_map(|word| word.to_be_bytes()).collect(),
            instruction_reads: 0,
            token,
        }
    }
}

impl PpcMemory for CountingImmutableMemory {
    fn read_u8(&mut self, addr: u32) -> Option<u8> {
        let offset = usize::try_from(addr.checked_sub(self.base)?).ok()?;
        self.bytes.get(offset).copied()
    }

    fn write_u8(&mut self, _addr: u32, _value: u8) -> Option<()> {
        None
    }

    fn read_instruction_u32_be(&mut self, addr: u32) -> Option<u32> {
        self.instruction_reads += 1;
        self.read_u32_be(addr)
    }

    fn instruction_cache_token(&mut self, addr: u32) -> Option<u64> {
        let offset = usize::try_from(addr.checked_sub(self.base)?).ok()?;
        (self.bytes.len().saturating_sub(offset) >= 4).then_some(self.token)
    }
}

struct TokenizedWritableMemory {
    base: u32,
    bytes: Vec<u8>,
    token: u64,
}

impl TokenizedWritableMemory {
    fn new(base: u32, words: &[u32]) -> Self {
        Self {
            base,
            bytes: words.iter().flat_map(|word| word.to_be_bytes()).collect(),
            token: 1,
        }
    }

    fn offset(&self, addr: u32) -> Option<usize> {
        usize::try_from(addr.checked_sub(self.base)?).ok()
    }
}

impl PpcMemory for TokenizedWritableMemory {
    fn read_u8(&mut self, addr: u32) -> Option<u8> {
        self.bytes.get(self.offset(addr)?).copied()
    }

    fn write_u8(&mut self, addr: u32, value: u8) -> Option<()> {
        let offset = self.offset(addr)?;
        *self.bytes.get_mut(offset)? = value;
        self.token = self.token.checked_add(1).expect("test token exhausted");
        Some(())
    }

    fn instruction_cache_token(&mut self, addr: u32) -> Option<u64> {
        let offset = self.offset(addr)?;
        (self.bytes.len().saturating_sub(offset) >= 4).then_some(self.token)
    }
}

#[test]
fn cpu_keeps_large_host_caches_off_the_stack() {
    let size = std::mem::size_of::<PpcCpu>();
    assert!(size <= 1024, "PpcCpu grew to {size} stack bytes");
}

#[test]
fn fresh_execution_context_matches_cpu_architectural_defaults() {
    let context = PpcExecutionContext::fresh();
    let cpu = PpcCpu::new();

    assert_eq!(
        context.architectural(),
        cpu.capture_execution_context().architectural()
    );
    assert!(context.import_call_stack.is_empty());
}

#[test]
fn execution_context_round_trips_registers_and_retains_engine_state() {
    let expected_architectural = PpcArchitecturalState {
        gpr: std::array::from_fn(|index| 0x1000_0000 | index as u32),
        fpr: std::array::from_fn(|index| 0x2000_0000_0000_0000 | index as u64),
        cr: 0x3456_789a,
        lr: 0x4567_89ab,
        ctr: 0x5678_9abc,
        xer: 0x6789_abcd,
        fpscr: 0x789a_bcde,
        msr: 0x89ab_cdef,
        pc: 0x9abc_def0,
    };
    let mut source = PpcCpu::new();
    source.gpr = expected_architectural.gpr;
    source.fpr = expected_architectural.fpr;
    source.cr = expected_architectural.cr;
    source.lr = expected_architectural.lr;
    source.ctr = expected_architectural.ctr;
    source.xer = expected_architectural.xer;
    source.fpscr = expected_architectural.fpscr;
    source.msr = expected_architectural.msr;
    source.pc = expected_architectural.pc;
    let captured = source.capture_execution_context();

    assert_eq!(captured.architectural(), &expected_architectural);

    source.gpr.fill(0);
    source.fpr.fill(0);
    source.cr = 0;
    source.lr = 0;
    source.ctr = 0;
    source.xer = 0;
    source.fpscr = 0;
    source.msr = 0;
    source.pc = 0;

    let mut destination = PpcCpu::new();
    destination.alignment_policy = PpcAlignmentPolicy::EmulateData;
    assert_eq!(
        destination.step_instruction(0x6000_0000),
        PpcStepResult::Stepped
    );
    destination.set_time_base(0x1122_3344_5566_7788);
    let cached_decodes = destination.decode_cache_entry_count();
    destination.install_execution_context(captured);

    assert_eq!(
        destination.capture_execution_context().architectural(),
        &expected_architectural
    );
    assert_eq!(
        destination.alignment_policy,
        PpcAlignmentPolicy::EmulateData
    );
    assert_eq!(destination.time_base(), 0x1122_3344_5566_7788);
    assert_eq!(destination.decode_cache_entry_count(), cached_decodes);
}

#[test]
fn execution_context_resumes_nested_native_import_returns() {
    const TRAP_BASE: u32 = 0x1000;
    const OUTER_RETURN: u32 = 0x2000;
    const INNER_RETURN: u32 = 0x2100;
    const HALT_PC: u32 = 0x3000;

    let mut cpu = PpcCpu::new();
    let mut memory = PpcSectionMem::new();
    cpu.pc = TRAP_BASE;
    let first = cpu.run_with_imports(&mut memory, 1, HALT_PC, TRAP_BASE, 2, |index, _, _| {
        assert_eq!(index, 0);
        PpcImportAction::CallNative {
            entry: TRAP_BASE + 4,
            rtoc: 0xaaaa_0001,
            return_pc: OUTER_RETURN,
            final_pc: HALT_PC,
            restore_rtoc: 0x1111_0001,
            return_gpr3: PpcNativeReturnGpr3::Set(0x3333_0003),
        }
    });
    assert_eq!(first, PpcRunResult::CycleLimit { cycles: 1 });
    let context = cpu.capture_execution_context();
    assert_eq!(context.import_call_stack.len(), 1);

    let mut resumed = PpcCpu::new();
    resumed.install_execution_context(context);
    let second = resumed.run_with_imports(&mut memory, 10, HALT_PC, TRAP_BASE, 2, |index, _, _| {
        assert_eq!(index, 1);
        PpcImportAction::CallNative {
            entry: INNER_RETURN,
            rtoc: 0xbbbb_0002,
            return_pc: INNER_RETURN,
            final_pc: OUTER_RETURN,
            restore_rtoc: 0xaaaa_0001,
            return_gpr3: PpcNativeReturnGpr3::Set(0x2222_0002),
        }
    });

    assert_eq!(
        second,
        PpcRunResult::Halted {
            pc: HALT_PC,
            cycles: 3
        }
    );
    assert_eq!(resumed.pc, HALT_PC);
    assert_eq!(resumed.lr, HALT_PC);
    assert_eq!(resumed.gpr[2], 0x1111_0001);
    assert_eq!(resumed.gpr[3], 0x3333_0003);
    assert!(resumed.import_call_stack.is_empty());
}

#[test]
fn execution_context_install_replaces_pending_import_frames_a_b_a() {
    fn pending_context(
        trap_pc: u32,
        return_pc: u32,
        final_pc: u32,
        callback_rtoc: u32,
        restored_rtoc: u32,
        result: u32,
    ) -> PpcExecutionContext {
        let mut cpu = PpcCpu::new();
        let mut memory = PpcSectionMem::new();
        cpu.pc = trap_pc;
        assert_eq!(
            cpu.run_with_imports(&mut memory, 1, 0, trap_pc, 1, |index, _, _| {
                assert_eq!(index, 0);
                PpcImportAction::CallNative {
                    entry: return_pc,
                    rtoc: callback_rtoc,
                    return_pc,
                    final_pc,
                    restore_rtoc: restored_rtoc,
                    return_gpr3: PpcNativeReturnGpr3::Set(result),
                }
            },),
            PpcRunResult::CycleLimit { cycles: 1 }
        );
        cpu.capture_execution_context()
    }

    const A_RETURN: u32 = 0x2000;
    const A_FINAL: u32 = 0x3000;
    const A_RTOC: u32 = 0xaaaa_0001;
    const A_RESULT: u32 = 0xaaaa_0003;
    const B_RETURN: u32 = 0x2100;
    const B_FINAL: u32 = 0x3100;
    const B_RTOC: u32 = 0xbbbb_0002;
    const B_RESULT: u32 = 0xbbbb_0003;

    let context_a = pending_context(0x1000, A_RETURN, A_FINAL, 0xaaaa_1000, A_RTOC, A_RESULT);
    let context_b = pending_context(0x1100, B_RETURN, B_FINAL, 0xbbbb_1000, B_RTOC, B_RESULT);
    let mut engine = PpcCpu::new();
    let mut memory = PpcSectionMem::new();

    engine.install_execution_context(context_a.clone());
    engine.install_execution_context(context_b);
    assert_eq!(
        engine.run_with_imports(&mut memory, 2, B_FINAL, 0, 0, |_, _, _| unreachable!()),
        PpcRunResult::Halted {
            pc: B_FINAL,
            cycles: 1
        }
    );
    assert_eq!(engine.pc, B_FINAL);
    assert_eq!(engine.lr, B_FINAL);
    assert_eq!(engine.gpr[2], B_RTOC);
    assert_eq!(engine.gpr[3], B_RESULT);
    assert!(engine.import_call_stack.is_empty());

    engine.install_execution_context(context_a);
    assert_eq!(
        engine.run_with_imports(&mut memory, 2, A_FINAL, 0, 0, |_, _, _| unreachable!()),
        PpcRunResult::Halted {
            pc: A_FINAL,
            cycles: 1
        }
    );
    assert_eq!(engine.pc, A_FINAL);
    assert_eq!(engine.lr, A_FINAL);
    assert_eq!(engine.gpr[2], A_RTOC);
    assert_eq!(engine.gpr[3], A_RESULT);
    assert!(engine.import_call_stack.is_empty());
}

#[test]
fn execution_context_runs_across_distinct_section_memories() {
    const BASE: u32 = 0x1000;
    const BRANCH_TO_BASE: u32 = 0x4bff_fffc;

    let mut first_memory = PpcSectionMem::new();
    first_memory.add_readonly_region(
        BASE,
        [0x3863_0001, BRANCH_TO_BASE]
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect(),
    );
    let mut equal_token_memory = first_memory.clone();
    let mut different_token_memory = PpcSectionMem::new();
    different_token_memory.add_readonly_region(
        BASE,
        [0x3863_0005, BRANCH_TO_BASE]
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect(),
    );

    let mut cpu = PpcCpu::new();
    cpu.pc = BASE;
    assert_eq!(
        cpu.run_with_imports(&mut first_memory, 2, 0, 0, 0, |_, _, _| unreachable!()),
        PpcRunResult::CycleLimit { cycles: 2 }
    );
    assert_eq!(cpu.gpr[3], 1);

    let mut context = cpu.capture_execution_context();
    context.architectural_mut().gpr[3] = 10;
    cpu.install_execution_context(context);
    assert_eq!(
        cpu.run_with_imports(
            &mut equal_token_memory,
            2,
            0,
            0,
            0,
            |_, _, _| unreachable!()
        ),
        PpcRunResult::CycleLimit { cycles: 2 }
    );
    assert_eq!(cpu.gpr[3], 11);

    let mut context = cpu.capture_execution_context();
    context.architectural_mut().gpr[3] = 20;
    cpu.install_execution_context(context);
    assert_eq!(
        cpu.run_with_imports(&mut different_token_memory, 2, 0, 0, 0, |_, _, _| {
            unreachable!()
        }),
        PpcRunResult::CycleLimit { cycles: 2 }
    );
    assert_eq!(cpu.gpr[3], 25);
}

#[test]
fn execution_context_retains_token_validated_basic_block_cache() {
    const BASE: u32 = 0x1000;
    const BRANCH_TO_BASE: u32 = 0x4bff_fffc;
    let words = [0x3863_0001, BRANCH_TO_BASE];
    let mut first_memory = CountingImmutableMemory::with_token(BASE, &words, 41);
    let mut equal_token_memory = CountingImmutableMemory::with_token(BASE, &words, 41);
    let mut different_token_memory =
        CountingImmutableMemory::with_token(BASE, &[0x3863_0005, BRANCH_TO_BASE], 42);
    let mut cpu = PpcCpu::new();
    cpu.pc = BASE;

    assert_eq!(
        cpu.run_with_imports(&mut first_memory, 2, 0, 0, 0, |_, _, _| unreachable!()),
        PpcRunResult::CycleLimit { cycles: 2 }
    );
    assert_eq!(first_memory.instruction_reads, 2);

    let context = cpu.capture_execution_context();
    cpu.install_execution_context(context);
    assert_eq!(
        cpu.run_with_imports(
            &mut equal_token_memory,
            2,
            0,
            0,
            0,
            |_, _, _| unreachable!()
        ),
        PpcRunResult::CycleLimit { cycles: 2 }
    );
    assert_eq!(equal_token_memory.instruction_reads, 0);
    assert_eq!(cpu.gpr[3], 2);

    let context = cpu.capture_execution_context();
    cpu.install_execution_context(context);
    assert_eq!(
        cpu.run_with_imports(&mut different_token_memory, 2, 0, 0, 0, |_, _, _| {
            unreachable!()
        }),
        PpcRunResult::CycleLimit { cycles: 2 }
    );
    assert_eq!(different_token_memory.instruction_reads, 2);
    assert_eq!(cpu.gpr[3], 7);
}

#[test]
fn capture_preserves_reservation_but_install_invalidates_it() {
    const ADDRESS: u32 = 0x1000;
    const ORIGINAL: u32 = 0x1122_3344;
    const REPLACEMENT: u32 = 0xaabb_ccdd;
    const LWARX_R3_R4_R5: u32 = (31 << 26) | (3 << 21) | (4 << 16) | (5 << 11) | (20 << 1);
    const STWCX_R6_R4_R5: u32 = (31 << 26) | (6 << 21) | (4 << 16) | (5 << 11) | (150 << 1) | 1;

    let mut source_memory = PpcSectionMem::new();
    source_memory.add_region(ADDRESS, ORIGINAL.to_be_bytes().to_vec());
    let mut source = PpcCpu::new();
    source.gpr[4] = ADDRESS;
    source.gpr[5] = 0;
    source.gpr[6] = REPLACEMENT;
    assert_eq!(
        source.step(&mut source_memory, LWARX_R3_R4_R5),
        PpcStepResult::Stepped
    );

    let context = source.capture_execution_context();
    assert_eq!(source.reservation_address(), Some(ADDRESS));
    assert_eq!(
        source.step(&mut source_memory, STWCX_R6_R4_R5),
        PpcStepResult::Stepped
    );
    assert_eq!(source.cr_field(0), 0b0010);
    assert_eq!(source_memory.read_u32_be(ADDRESS), Some(REPLACEMENT));

    let mut installed_memory = PpcSectionMem::new();
    installed_memory.add_region(ADDRESS, ORIGINAL.to_be_bytes().to_vec());
    let mut installed = PpcCpu::new();
    installed.gpr[4] = ADDRESS;
    installed.gpr[5] = 0;
    assert_eq!(
        installed.step(&mut installed_memory, LWARX_R3_R4_R5),
        PpcStepResult::Stepped
    );
    assert_eq!(installed.reservation_address(), Some(ADDRESS));

    installed.install_execution_context(context);
    assert_eq!(installed.reservation_address(), None);
    assert_eq!(
        installed.step(&mut installed_memory, STWCX_R6_R4_R5),
        PpcStepResult::Stepped
    );
    assert_eq!(installed.cr_field(0), 0);
    assert_eq!(installed_memory.read_u32_be(ADDRESS), Some(ORIGINAL));
}

#[test]
fn immutable_basic_block_cache_avoids_repeated_instruction_fetches() {
    let base = 0x1000;
    let mut memory = CountingImmutableMemory::new(
        base,
        &[
            0x3863_0001,
            0x3863_0001,
            0x3863_0001,
            0x3863_0001,
            0x4bff_fff0,
        ],
    );
    let mut cpu = PpcCpu::new();
    cpu.pc = base;

    let result = cpu.run_with_imports(&mut memory, 100, 0, 0, 0, |_, _, _| {
        unreachable!("test program has no imports")
    });

    assert_eq!(result, PpcRunResult::CycleLimit { cycles: 100 });
    assert_eq!(cpu.gpr[3], 80);
    assert_eq!(memory.instruction_reads, 5);
}

#[test]
fn remapping_readonly_code_invalidates_a_cached_block() {
    const BASE: u32 = 0x1000;
    const BRANCH_TO_BASE: u32 = 0x4bff_fffc;
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(
        BASE,
        [0x3863_0001u32, BRANCH_TO_BASE]
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect(),
    );
    let mut cpu = PpcCpu::new();
    cpu.pc = BASE;
    assert_eq!(
        cpu.run(&mut memory, 20, 0),
        PpcRunResult::CycleLimit { cycles: 20 }
    );
    assert_eq!(cpu.gpr[3], 10);

    memory.add_readonly_region(BASE, 0x3863_0005u32.to_be_bytes().to_vec());
    assert_eq!(
        cpu.run(&mut memory, 20, 0),
        PpcRunResult::CycleLimit { cycles: 20 }
    );
    assert_eq!(cpu.gpr[3], 60);
}

fn assert_store_redecodes_the_following_instruction(base: u32) {
    const STORE_R4_AT_R5_PLUS_4: u32 = 0x9085_0004;
    const ADD_ONE: u32 = 0x3863_0001;
    const ADD_TWO: u32 = 0x3863_0002;
    const BLR: u32 = 0x4e80_0020;

    let mut memory = TokenizedWritableMemory::new(base, &[STORE_R4_AT_R5_PLUS_4, ADD_ONE, BLR]);
    let mut cpu = PpcCpu::new();
    cpu.pc = base;
    cpu.lr = 0;
    cpu.gpr[4] = ADD_TWO;
    cpu.gpr[5] = base;

    assert_eq!(
        cpu.run(&mut memory, 8, 0),
        PpcRunResult::Halted { pc: 0, cycles: 3 }
    );
    assert_eq!(
        cpu.gpr[3], 2,
        "the instruction after the store came from a stale cached block"
    );
}

#[test]
fn store_rewriting_the_next_instruction_revalidates_the_cached_block() {
    assert_store_redecodes_the_following_instruction(0x1000);
}

#[test]
fn store_rewriting_the_next_page_revalidates_the_cached_block() {
    assert_store_redecodes_the_following_instruction(0x1ffc);
}

#[test]
fn cached_block_stops_before_interior_halt_pc() {
    let base = 0x1000;
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(
        base,
        [0x3863_0001u32, 0x3863_0001, 0x3863_0001, 0x4bff_fff4]
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect(),
    );
    let mut cpu = PpcCpu::new();
    cpu.pc = base;

    assert_eq!(
        cpu.run(&mut memory, 10, base + 8),
        PpcRunResult::Halted {
            pc: base + 8,
            cycles: 2
        }
    );
    assert_eq!(cpu.gpr[3], 2);
}

#[test]
fn cached_word_copy_preserves_load_store_boundaries() {
    const CODE: u32 = 0x1000;
    const DATA: u32 = 0x2000;
    // Loading r4 also changes the base used by the following store.
    let instructions = [0x8085_0000u32, 0x9084_0004];
    for (source, destination, max_cycles, halt_pc) in [
        (DATA, DATA + 8, 2, 0),
        (DATA, DATA + 8, 1, 0),
        (DATA, DATA + 8, 2, CODE + 4),
        (DATA + 1, DATA + 8, 2, 0),
        (DATA, DATA + 1, 2, 0),
        (DATA + 16, DATA + 8, 2, 0),
        (DATA, DATA + 16, 2, 0),
    ] {
        let mut memory = PpcSectionMem::new();
        memory.add_readonly_region(
            CODE,
            instructions
                .into_iter()
                .flat_map(u32::to_be_bytes)
                .collect(),
        );
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(&(destination - 4).to_be_bytes());
        memory.add_region(DATA, data);

        let mut cached = PpcCpu::new();
        cached.pc = CODE;
        cached.gpr[5] = source;
        let mut expected = PpcCpu::new();
        expected.pc = CODE;
        expected.gpr[5] = source;
        let mut expected_memory = memory.clone();

        let result = cached.run_with_imports(
            &mut memory,
            max_cycles,
            halt_pc,
            0,
            0,
            |_, _, _| unreachable!(),
        );
        let reference = expected.run_with_imports_and_fetch_observer(
            &mut expected_memory,
            max_cycles,
            halt_pc,
            0,
            0,
            &mut Vec::new(),
            |_, _, _| unreachable!(),
        );
        assert_eq!(
            result, reference,
            "source={source:#x}, destination={destination:#x}"
        );
        assert_eq!(cached.pc, expected.pc);
        assert_eq!(cached.gpr, expected.gpr);
        assert_eq!(cached.time_base(), expected.time_base());
        assert_eq!(
            memory.read_u32_be(DATA + 8),
            expected_memory.read_u32_be(DATA + 8)
        );
    }
}

#[test]
fn cached_block_stops_before_interior_import_trap() {
    let base = 0x1000;
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(
        base,
        [0x3863_0001u32, 0x3863_0001, 0x3863_0001, 0x4bff_fff4]
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect(),
    );
    let mut cpu = PpcCpu::new();
    cpu.pc = base;
    cpu.lr = 0;
    let mut calls = 0;

    assert_eq!(
        cpu.run_with_imports(&mut memory, 10, 0, base + 8, 1, |index, _, _| {
            assert_eq!(index, 0);
            calls += 1;
            PpcImportAction::Return(42)
        }),
        PpcRunResult::Halted { pc: 0, cycles: 3 }
    );
    assert_eq!(calls, 1);
    assert_eq!(cpu.gpr[3], 42);
}

#[test]
fn cached_block_stops_before_interior_native_callback_return() {
    let base = 0x1000;
    let trap_base = 0x2000;
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(
        base,
        [0x3863_0001u32, 0x3863_0001, 0x3863_0001, 0x4bff_fff4]
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect(),
    );
    let mut cpu = PpcCpu::new();
    cpu.pc = trap_base;

    assert_eq!(
        cpu.run_with_imports(&mut memory, 10, 0, trap_base, 1, |index, _, _| {
            assert_eq!(index, 0);
            PpcImportAction::CallNative {
                entry: base,
                rtoc: 0,
                return_pc: base + 8,
                final_pc: 0,
                restore_rtoc: 0,
                return_gpr3: PpcNativeReturnGpr3::Preserve,
            }
        }),
        PpcRunResult::Halted { pc: 0, cycles: 4 }
    );
    assert_eq!(cpu.gpr[3], 2);
}

#[test]
fn immutable_basic_block_cache_executes_predecoded_fallback_without_decode_cache() {
    // `stbux` is intentionally outside the raw fast dispatcher.  Its
    // decoded form must still execute from an immutable code block while
    // routing the guest store through the normal memory bus.
    let code_base = 0x1000;
    let data_base = 0x2000;
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(code_base, 0x7F2B_61EEu32.to_be_bytes().to_vec());
    memory.add_region(data_base, vec![0; 0x20]);

    let mut cpu = PpcCpu::new();
    cpu.pc = code_base;
    cpu.gpr[25] = 0xAB;
    cpu.gpr[11] = data_base;
    cpu.gpr[12] = 4;

    assert_eq!(
        cpu.run_with_imports(&mut memory, 1, 0, 0, 0, |_, _, _| unreachable!()),
        PpcRunResult::CycleLimit { cycles: 1 }
    );
    assert_eq!(cpu.gpr[11], data_base + 4);
    assert_eq!(memory.read_u8(data_base + 4), Some(0xAB));
    assert_eq!(cpu.decode_cache_entry_count(), 0);
}

#[test]
fn fast_instruction_classifier_matches_dispatcher() {
    let mut memory = PpcSectionMem::new();
    let mut cpu = PpcCpu::new();

    // Exercise every primary/XO combination with the remaining fields
    // zeroed. This keeps the cached classification conservative: a word
    // is only omitted from predecoding when the raw dispatcher accepts it.
    for primary in 0..64u32 {
        for xo in 0..1024u32 {
            let word = (primary << 26) | (xo << 1);
            let dispatched = cpu.step_fast_unobserved(&mut memory, word).is_some();
            assert_eq!(
                PpcCpu::fast_instruction(word),
                dispatched,
                "primary={primary}, xo={xo}, word={word:#010x}"
            );
        }
    }
}

#[test]
fn decode_cache_reuses_repeated_successful_instruction_words() {
    let mut cpu = PpcCpu::new();

    assert_eq!(cpu.step_instruction(0x6000_0000), PpcStepResult::Stepped);
    assert_eq!(cpu.pc, 4);
    assert_eq!(cpu.decode_cache_entry_count(), 1);
    assert_eq!(cpu.step_instruction(0x6000_0000), PpcStepResult::Stepped);
    assert_eq!(cpu.pc, 8);
    assert_eq!(cpu.decode_cache_entry_count(), 1);
}

#[test]
fn decode_cache_reuses_repeated_decode_errors() {
    let mut cpu = PpcCpu::new();
    let unsupported_vector_instruction = 0x1000_0000;

    assert_eq!(
        cpu.step_instruction(unsupported_vector_instruction),
        PpcStepResult::Unimplemented(PpcDecodeError::UnsupportedPrimaryOpcode(4))
    );
    assert_eq!(cpu.decode_cache_entry_count(), 1);
    assert_eq!(
        cpu.step_instruction(unsupported_vector_instruction),
        PpcStepResult::Unimplemented(PpcDecodeError::UnsupportedPrimaryOpcode(4))
    );
    assert_eq!(cpu.decode_cache_entry_count(), 1);
}

#[test]
fn stbux_decodes_and_updates_the_base_register() {
    let word = 0x7F2B_61EE;
    assert_eq!(
        crate::decode::decode(word),
        Ok(PpcInstr::Stbux {
            rs: 25,
            ra: 11,
            rb: 12,
        })
    );

    let mut cpu = PpcCpu::new();
    let mut mem = PpcSectionMem::new();
    mem.add_region(0x1000, vec![0; 0x100]);
    cpu.gpr[25] = 0x1234_56AB;
    cpu.gpr[11] = 0x1000;
    cpu.gpr[12] = 0x20;

    assert_eq!(cpu.step(&mut mem, word), PpcStepResult::Stepped);
    assert_eq!(mem.read_u8(0x1020), Some(0xAB));
    assert_eq!(cpu.gpr[11], 0x1020);
    assert_eq!(cpu.pc, 4);
}

#[test]
fn stbux_rejects_ra_zero() {
    let word = 0x7F20_61EE;
    let mut cpu = PpcCpu::new();
    let mut mem = PpcSectionMem::new();

    assert_eq!(
        cpu.step(&mut mem, word),
        PpcStepResult::Exception(PpcException::IllegalInstruction {
            word,
            reason: PpcIllegalInstructionReason::InvalidForm,
        })
    );
    assert_eq!(cpu.pc, 0);
}

#[test]
fn bcctr_null_target_branches_to_zero() {
    let mut cpu = PpcCpu::new();
    cpu.ctr = 0;

    assert_eq!(cpu.step_instruction(0x4E80_0420), PpcStepResult::Stepped);
    assert_eq!(cpu.pc, 0);
    assert_eq!(cpu.lr, 0);
    assert_eq!(cpu.gpr[3], 0);
}
