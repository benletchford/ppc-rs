use ppc::{PpcCpu, PpcImportAction, PpcRunResult, PpcSectionMem};

const CODE: u32 = 0x1000;

fn loop_memory() -> PpcSectionMem {
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(
        CODE,
        [0x3863_0001u32, 0x4bff_fffcu32]
            .into_iter()
            .flat_map(u32::to_be_bytes)
            .collect(),
    );
    memory
}

fn loop_cpu() -> PpcCpu {
    let mut cpu = PpcCpu::new();
    cpu.pc = CODE;
    cpu
}

#[test]
fn declined_trace_preserves_interpreter_result() {
    let mut baseline = loop_cpu();
    let mut baseline_memory = loop_memory();
    let expected = baseline.run_with_imports_and_cycle_handler(
        &mut baseline_memory,
        5,
        0,
        0,
        0,
        |_, _, _, _| unreachable!(),
    );

    let mut traced = loop_cpu();
    let mut traced_memory = loop_memory();
    let mut attempts = 0;
    let actual = traced.run_with_imports_and_cycle_handler_and_trace(
        &mut traced_memory,
        5,
        0,
        0,
        0,
        |_, _, _, _| unreachable!(),
        |_, _, _, _, _, _| {
            attempts += 1;
            None
        },
    );

    assert!(attempts > 0);
    assert_eq!(actual, expected);
    assert_eq!(traced.pc, baseline.pc);
    assert_eq!(traced.gpr, baseline.gpr);
    assert_eq!(traced.cr, baseline.cr);
    assert_eq!(traced.time_base(), baseline.time_base());
}

#[test]
fn accepted_trace_charges_cycles_and_resumes_at_resulting_pc() {
    let mut baseline = loop_cpu();
    let expected = baseline.run_with_imports_and_cycle_handler(
        &mut loop_memory(),
        5,
        0,
        0,
        0,
        |_, _, _, _| unreachable!(),
    );

    let mut traced = loop_cpu();
    let mut accepted = false;
    let actual = traced.run_with_imports_and_cycle_handler_and_trace(
        &mut loop_memory(),
        5,
        0,
        0,
        0,
        |_, _, _, _| unreachable!(),
        |cpu, _, remaining, halt_pc, trap_base, import_count| {
            assert_eq!((halt_pc, trap_base, import_count), (0, 0, 0));
            if accepted || remaining < 4 || cpu.pc != CODE {
                return None;
            }
            accepted = true;
            cpu.gpr[3] = cpu.gpr[3].wrapping_add(2);
            cpu.set_time_base(cpu.time_base() + 4);
            Some(4)
        },
    );

    assert!(accepted);
    assert_eq!(actual, expected);
    assert_eq!(actual, PpcRunResult::CycleLimit { cycles: 5 });
    assert_eq!(traced.pc, baseline.pc);
    assert_eq!(traced.gpr, baseline.gpr);
    assert_eq!(traced.time_base(), baseline.time_base());
}

#[test]
fn trace_is_not_offered_before_halt_import_or_alignment_handling() {
    let mut halted = loop_cpu();
    let result = halted.run_with_imports_and_cycle_handler_and_trace(
        &mut loop_memory(),
        5,
        CODE,
        0,
        0,
        |_, _, _, _| unreachable!(),
        |_, _, _, _, _, _| panic!("trace offered at halt PC"),
    );
    assert_eq!(
        result,
        PpcRunResult::Halted {
            pc: CODE,
            cycles: 0
        }
    );

    let mut imported = loop_cpu();
    let result = imported.run_with_imports_and_cycle_handler_and_trace(
        &mut loop_memory(),
        5,
        0,
        CODE,
        1,
        |_, _, _, _| PpcImportAction::Halt,
        |_, _, _, _, _, _| panic!("trace offered at import PC"),
    );
    assert_eq!(
        result,
        PpcRunResult::Halted {
            pc: CODE,
            cycles: 0
        }
    );

    let mut unaligned = loop_cpu();
    unaligned.pc = CODE + 2;
    let result = unaligned.run_with_imports_and_cycle_handler_and_trace(
        &mut loop_memory(),
        5,
        0,
        0,
        0,
        |_, _, _, _| unreachable!(),
        |_, _, _, _, _, _| panic!("trace offered at unaligned PC"),
    );
    assert!(matches!(result, PpcRunResult::Exception { pc, cycles: 0, .. } if pc == CODE + 2));
}

#[test]
fn trace_reaches_import_handler_with_accumulated_cycles() {
    const IMPORT: u32 = 0x2000;
    let mut cpu = loop_cpu();
    let mut elapsed_at_import = None;
    let mut trace_calls = 0;
    let result = cpu.run_with_imports_and_cycle_handler_and_trace(
        &mut loop_memory(),
        5,
        0,
        IMPORT,
        1,
        |elapsed, index, _, _| {
            elapsed_at_import = Some((elapsed, index));
            PpcImportAction::Halt
        },
        |cpu, _, remaining, _, trap_base, _| {
            trace_calls += 1;
            assert_eq!(remaining, 5);
            assert_eq!(cpu.pc, CODE);
            cpu.pc = trap_base;
            cpu.set_time_base(cpu.time_base() + 2);
            Some(2)
        },
    );

    assert_eq!(trace_calls, 1);
    assert_eq!(elapsed_at_import, Some((2, 0)));
    assert_eq!(
        result,
        PpcRunResult::Halted {
            pc: IMPORT,
            cycles: 2
        }
    );
}
