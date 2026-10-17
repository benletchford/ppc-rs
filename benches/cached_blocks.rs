use ppc::{PpcCpu, PpcRunResult, PpcSectionMem};
use std::hint::black_box;
use std::time::{Duration, Instant};

const CODE_BASE: u32 = 0x1000;
const DATA_BASE: u32 = 0x2000;
const DEFAULT_CYCLES: u64 = 20_000_000;

fn run_case(words: &[u32], with_data: bool, floating_value: f64, cycles: u64) -> Duration {
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(
        CODE_BASE,
        words.iter().flat_map(|word| word.to_be_bytes()).collect(),
    );
    if with_data {
        memory.add_region(DATA_BASE, vec![0; 64]);
    }
    let mut cpu = PpcCpu::new();
    cpu.pc = CODE_BASE;
    cpu.gpr[5] = DATA_BASE;
    cpu.gpr[6] = DATA_BASE + 32;
    cpu.fpr[1] = floating_value.to_bits();

    let start = Instant::now();
    let result = cpu.run_with_imports(&mut memory, cycles, 0, 0, 0, |_, _, _| unreachable!());
    let elapsed = start.elapsed();
    assert_eq!(result, PpcRunResult::CycleLimit { cycles });
    black_box(&cpu);
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
    let mut copy_words = Vec::with_capacity(17);
    for offset in (0..32).step_by(4) {
        copy_words.push(0x8085_0000 | offset); // lwz r4,offset(r5)
        copy_words.push(0x9086_0000 | offset); // stw r4,offset(r6)
    }
    copy_words.push(0x4bff_ffc0); // b CODE_BASE

    let single_stores = [
        0xd025_0000, // stfs f1,0(r5)
        0xd025_0004, // stfs f1,4(r5)
        0xd025_0008, // stfs f1,8(r5)
        0xd025_000c, // stfs f1,12(r5)
        0x4bff_fff0, // b CODE_BASE
    ];

    // Repeated independent results keep f1 unchanged, so inexact cases do
    // not become exact after their first iteration.
    let single_rounding = [
        0xfc40_0818, // frsp f2,f1
        0xfc60_0818, // frsp f3,f1
        0xfc80_0818, // frsp f4,f1
        0xfca0_0818, // frsp f5,f1
        0x4bff_fff0, // b CODE_BASE
    ];
    let reciprocal_roots = [
        0xfc40_0834, // frsqrte f2,f1
        0xfc60_0834, // frsqrte f3,f1
        0xfc80_0834, // frsqrte f4,f1
        0xfca0_0834, // frsqrte f5,f1
        0x4bff_fff0, // b CODE_BASE
    ];

    println!("case\tcycles\tmedian_ms\tMcycles/s");
    for (name, words, with_data, floating_value) in [
        ("arithmetic", arithmetic.as_slice(), false, 0.0),
        ("conditional", conditional.as_slice(), false, 0.0),
        ("load_store", memory.as_slice(), true, 0.0),
        ("copy_words", copy_words.as_slice(), true, 0.0),
        ("single_store_exact", single_stores.as_slice(), true, 2.5),
        ("single_store_rounding", single_stores.as_slice(), true, 0.1),
        ("single_round_exact", single_rounding.as_slice(), false, 2.5),
        (
            "single_round_inexact",
            single_rounding.as_slice(),
            false,
            0.1,
        ),
        (
            "single_round_subnormal",
            single_rounding.as_slice(),
            false,
            f64::from_bits(893u64 << 52),
        ),
        (
            "reciprocal_root_exact",
            reciprocal_roots.as_slice(),
            false,
            4.0,
        ),
        (
            "reciprocal_root_inexact",
            reciprocal_roots.as_slice(),
            false,
            2.5,
        ),
        (
            "reciprocal_root_invalid",
            reciprocal_roots.as_slice(),
            false,
            -1.0,
        ),
    ] {
        let mut runs = [Duration::ZERO; 3];
        for elapsed in &mut runs {
            *elapsed = run_case(words, with_data, floating_value, cycles);
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
