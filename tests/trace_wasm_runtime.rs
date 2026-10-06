#![cfg(feature = "trace-wasm")]

use ppc::trace_wasm::{TraceStep, compile_wasm, plan};
use ppc::{PpcCpu, PpcRunResult, PpcSectionMem};
use std::path::PathBuf;
use std::process::Command;

fn run_case(name: &str, steps: &[TraceStep], value: u32, budget: u32) {
    let mut memory = PpcSectionMem::new();
    let base = steps[0].pc;
    memory.add_readonly_region(
        base,
        steps
            .iter()
            .flat_map(|step| step.word.to_be_bytes())
            .collect(),
    );
    memory.add_region(0x2000, value.to_be_bytes().to_vec());
    let mut cpu = PpcCpu::new();
    cpu.pc = base;
    cpu.gpr[3] = 7;
    cpu.gpr[4] = 0x2000;
    cpu.ctr = 19;
    cpu.cr = 0x1234_5678;
    cpu.xer = 0x8000_0000;
    cpu.set_time_base(42);
    let result = cpu.run(
        &mut memory,
        u64::from(budget),
        base + (steps.len() as u32) * 4,
    );
    assert!(matches!(
        result,
        PpcRunResult::CycleLimit { .. } | PpcRunResult::Halted { .. }
    ));
    let cycles = match result {
        PpcRunResult::CycleLimit { cycles } | PpcRunResult::Halted { cycles, .. } => cycles,
        _ => unreachable!(),
    };
    let module = compile_wasm(&plan(base, steps).unwrap());
    let path = std::env::temp_dir().join(format!(
        "ppc-trace-{}-{}-{}.wasm",
        std::process::id(),
        name,
        budget
    ));
    std::fs::write(&path, module).unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/trace_wasm_runtime.mjs");
    let output = Command::new("node")
        .arg(script)
        .arg(&path)
        .arg(name)
        .arg(value.to_string())
        .arg(budget.to_string())
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let fields = String::from_utf8(output.stdout).unwrap();
    let actual: Vec<u64> = fields
        .trim()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(
        actual,
        [
            cycles,
            u64::from(cpu.pc),
            u64::from(cpu.gpr[3]),
            u64::from(cpu.ctr),
            cpu.time_base(),
            u64::from(cpu.cr),
            u64::from(cpu.xer)
        ]
    );
}

#[test]
fn generated_wasm_matches_interpreter_for_load_branch_and_arithmetic() {
    Command::new("node")
        .arg("--version")
        .output()
        .expect("Node.js is required to execute generated WebAssembly in this feature test");
    let read_loop = [
        TraceStep {
            pc: 0x1000,
            word: 0x8064_0000,
            next_pc: 0x1004,
        },
        TraceStep {
            pc: 0x1004,
            word: 0x2c03_0000,
            next_pc: 0x1008,
        },
        TraceStep {
            pc: 0x1008,
            word: 0x4182_fff8,
            next_pc: 0x1000,
        },
    ];
    run_case("read", &read_loop, 0, 6);
    run_case("read", &read_loop, 1, 100);
    let arithmetic = [
        TraceStep {
            pc: 0x3000,
            word: 0x3863_0001,
            next_pc: 0x3004,
        },
        TraceStep {
            pc: 0x3004,
            word: 0x4bff_fffc,
            next_pc: 0x3000,
        },
    ];
    run_case("arithmetic", &arithmetic, 0, 4);
}
