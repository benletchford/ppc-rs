use ppc::{PpcCpu, PpcImportAction, PpcMemory};

struct MirroredRam(Vec<u8>);
impl PpcMemory for MirroredRam {
    fn read_u8(&mut self, addr: u32) -> Option<u8> {
        self.0.get((addr & 0xffff) as usize).copied()
    }
    fn write_u8(&mut self, addr: u32, val: u8) -> Option<()> {
        *self.0.get_mut((addr & 0xffff) as usize)? = val;
        Some(())
    }
    // Writable RAM offers no immutable instruction tokens.
    fn code_epoch(&self) -> Option<u64> {
        Some(1)
    }
}
fn run(budget: u64) -> (u32, u32, usize) {
    let mut ram = MirroredRam(vec![0; 65536]);
    for (i, w) in [
        0x81820010, 0x90410014, 0x800c0000, 0x804c0004, 0x7c0903a6, 0x4e800420,
    ]
    .into_iter()
    .enumerate()
    {
        ram.write_u32_be(0x1000 + i as u32 * 4, w).unwrap();
    }
    // Store a valid blr through the mirror at the third instruction.
    let mut cpu = PpcCpu::new();
    cpu.gpr[2] = 0x4e800020;
    ram.write_u32_be(cpu.gpr[2] + 16, 0x4000).unwrap();
    ram.write_u32_be(0x4000, 0x5000).unwrap();
    ram.write_u32_be(0x4004, 0x7000).unwrap();
    cpu.gpr[1] = 0x11008 - 20;
    cpu.pc = 0x1000;
    cpu.lr = 0x6000;
    let mut calls = 0;
    for _ in 0..8 {
        cpu.run_with_imports(&mut ram, budget, 0x6000, 0x5000, 1, |_, _, _| {
            calls += 1;
            PpcImportAction::ReturnPreserve
        });
        if cpu.pc == 0x6000 {
            break;
        }
    }
    (cpu.pc, cpu.gpr[2], calls)
}
#[test]
fn mirrored_ram_observes_store_before_fetch() {
    assert_eq!(run(8), run(1));
}
