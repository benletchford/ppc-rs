use ppc::{PpcCpu, PpcRunResult, PpcSectionMem};
use std::hint::black_box;
use std::time::{Duration, Instant};

const CODE_BASE: u32 = 0x1000;
const DATA_BASE: u32 = 0x2000;
const DEFAULT_CYCLES: u64 = 20_000_000;

fn run_case(words: &[u32], with_data: bool, cycles: u64) -> Duration {
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(
        CODE_BASE,
        words.iter().flat_map(|word| word.to_be_bytes()).collect(),
    );
    if with_data {
        memory.add_region(DATA_BASE, vec![0; 4]);
    }
    let mut cpu = PpcCpu::new();
    cpu.pc = CODE_BASE;
    cpu.gpr[5] = DATA_BASE;

    let start = Instant::now();
    let result = cpu.run_with_imports(&mut memory, cycles, 0, 0, 0, |_, _, _| unreachable!());
    let elapsed = start.elapsed();
    assert_eq!(result, PpcRunResult::CycleLimit { cycles });
    black_box(cpu.gpr[3]);
    black_box(memory);
    elapsed
}

fn main() {
    let cycles = std::env::var("PPC_BENCH_CYCLES")
        .ok()
        .map(|value| value.parse().expect("PPC_BENCH_CYCLES must be an integer"))
        .unwrap_or(DEFAULT_CYCLES);
    assert!(cycles > 0);

    // Four straight-line integer instructions followed by a back branch.
    let arithmetic = [
        0x3863_0001, // addi r3,r3,1
        0x3884_0001, // addi r4,r4,1
        0x38a5_0001, // addi r5,r5,1
        0x38c6_0001, // addi r6,r6,1
        0x4bff_fff0, // b CODE_BASE
    ];
    // The branch is taken after the first comparison and exercises CR lookup.
    let conditional = [
        0x3863_0001, // addi r3,r3,1
        0x2c03_0000, // cmpwi r3,0
        0x4082_fff8, // bne to CODE_BASE
    ];
    // A store ends a cached block so the next instruction view is rechecked.
    let memory = [
        0x8085_0000, // lwz r4,0(r5)
        0x3884_0001, // addi r4,r4,1
        0x9085_0000, // stw r4,0(r5)
        0x4bff_fff4, // b CODE_BASE
    ];

    println!("case\tcycles\tmedian_ms\tMcycles/s");
    for (name, words, with_data) in [
        ("arithmetic", arithmetic.as_slice(), false),
        ("conditional", conditional.as_slice(), false),
        ("load_store", memory.as_slice(), true),
    ] {
        let mut runs = [Duration::ZERO; 3];
        for elapsed in &mut runs {
            *elapsed = run_case(words, with_data, cycles);
        }
        runs.sort_unstable();
        let seconds = runs[1].as_secs_f64();
        println!(
            "{name}\t{cycles}\t{:.1}\t{:.2}",
            seconds * 1000.0,
            cycles as f64 / seconds / 1_000_000.0
        );
    }
}
