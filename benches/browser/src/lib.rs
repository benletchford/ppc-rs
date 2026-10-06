use ppc::{PpcCpu, PpcMemory, PpcRunResult, PpcSectionMem};

const CODE_BASE: u32 = 0x1000;
const DATA_BASE: u32 = 0x2000;

#[unsafe(no_mangle)]
pub extern "C" fn run_bench(kind: u32, cycles: u32) -> u32 {
    let arithmetic = [
        0x3863_0001,
        0x3884_0001,
        0x38a5_0001,
        0x38c6_0001,
        0x4bff_fff0,
    ];
    let conditional = [0x3863_0001, 0x2c03_0000, 0x4082_fff8];
    let load_store = [0x8085_0000, 0x3884_0001, 0x9085_0000, 0x4bff_fff4];
    let mut copy_words = Vec::with_capacity(17);
    for offset in (0..32).step_by(4) {
        copy_words.push(0x8085_0000 | offset);
        copy_words.push(0x9086_0000 | offset);
    }
    copy_words.push(0x4bff_ffc0);
    let words: &[u32] = match kind {
        0 => &arithmetic,
        1 => &conditional,
        2 => &load_store,
        3 => &copy_words,
        _ => return 0,
    };
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(
        CODE_BASE,
        words.iter().flat_map(|word| word.to_be_bytes()).collect(),
    );
    if kind >= 2 {
        memory.add_region(DATA_BASE, vec![0; 64]);
    }
    let mut cpu = PpcCpu::new();
    cpu.pc = CODE_BASE;
    cpu.gpr[5] = DATA_BASE;
    cpu.gpr[6] = DATA_BASE + 32;
    let result = cpu.run_with_imports(
        &mut memory,
        u64::from(cycles),
        0,
        0,
        0,
        |_, _, _| unreachable!(),
    );
    assert_eq!(
        result,
        PpcRunResult::CycleLimit {
            cycles: u64::from(cycles)
        }
    );
    (cpu.time_base() as u32)
        ^ cpu.gpr[3].rotate_left(7)
        ^ cpu.gpr[4]
        ^ memory.read_u32_be(DATA_BASE).unwrap_or(0)
}
