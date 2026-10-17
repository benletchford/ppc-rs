//! Single stores must preserve the software converter's bits in every mode,
//! including when an exact binary32 value can bypass rounding altogether.

use ppc::{PpcCpu, PpcMemory, PpcRunResult, PpcSectionMem, PpcStepResult};
use rustc_apfloat::ieee::{Double, Single};
use rustc_apfloat::{Float, FloatConvert, Round, StatusAnd};

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;
const STFS: u32 = (52 << 26) | (1 << 21) | (4 << 16);

fn reference(bits: u64, mode: u32) -> u32 {
    let round = match mode {
        0 => Round::NearestTiesToEven,
        1 => Round::TowardZero,
        2 => Round::TowardPositive,
        3 => Round::TowardNegative,
        _ => unreachable!(),
    };
    let result: StatusAnd<Single> =
        Double::from_bits(u128::from(bits)).convert_r(round, &mut false);
    result.value.to_bits() as u32
}

fn check(cpu: &mut PpcCpu, memory: &mut PpcSectionMem, bits: u64) {
    for mode in 0..4 {
        cpu.fpr[1] = bits;
        // Existing status bits and exception enables must not be modified.
        let fpscr = 0xa502_40f8 | mode;
        cpu.fpscr = fpscr;
        cpu.gpr[4] = DATA;
        cpu.pc = CODE;
        assert_eq!(cpu.step(memory, STFS), PpcStepResult::Stepped);
        assert_eq!(
            memory.read_u32_be(DATA),
            Some(reference(bits, mode)),
            "source={bits:016x} rounding={mode}"
        );
        assert_eq!(cpu.fpscr, fpscr);
        assert_eq!(cpu.fpr[1], bits);
    }
}

#[test]
fn single_store_matches_software_at_every_binary32_exponent_and_rounding_boundary() {
    let mut cpu = PpcCpu::new();
    let mut memory = PpcSectionMem::new();
    memory.add_region(DATA, vec![0; 4]);
    for exponent in 0..255 {
        for fraction in [0, 1, 0x3f_ffff, 0x40_0000, 0x7f_fffe, 0x7f_ffff] {
            for sign in [0, 0x8000_0000] {
                let single = f32::from_bits(sign | (exponent << 23) | fraction);
                let bits = f64::from(single).to_bits();
                check(&mut cpu, &mut memory, bits);
                // Neighboring doubles require rounding. Half a normal
                // single's ulp also exercises ties and directed modes.
                for delta in [1, (1 << 28) - 1, 1 << 28, (1 << 28) + 1] {
                    check(&mut cpu, &mut memory, bits.wrapping_add(delta));
                    check(&mut cpu, &mut memory, bits.wrapping_sub(delta));
                }
            }
        }
    }
}

#[test]
fn single_store_matches_software_for_random_doubles_and_exceptional_values() {
    let mut cpu = PpcCpu::new();
    let mut memory = PpcSectionMem::new();
    memory.add_region(DATA, vec![0; 4]);
    for bits in [
        0,
        1,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0x7ff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff7_ffff_ffff_ffff,
        0x7ff8_0000_0000_0000,
        0x7ff8_1234_5678_9abc,
        0x7fff_ffff_ffff_ffff,
    ] {
        check(&mut cpu, &mut memory, bits);
        check(&mut cpu, &mut memory, bits | (1 << 63));
    }
    let mut random = 0xd1b5_4a32_d192_ed03u64;
    for _ in 0..16_384 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        check(&mut cpu, &mut memory, random);
        let single = f32::from_bits(random as u32);
        if single.is_finite() {
            check(&mut cpu, &mut memory, f64::from(single).to_bits());
        }
    }
}

#[test]
fn single_store_forms_agree_in_stepped_and_cached_execution() {
    let forms: [(u32, bool); 4] = [
        ((52 << 26) | (1 << 21) | (4 << 16) | 4, false),
        ((53 << 26) | (1 << 21) | (4 << 16) | 4, true),
        (
            (31 << 26) | (1 << 21) | (4 << 16) | (6 << 11) | (663 << 1),
            false,
        ),
        (
            (31 << 26) | (1 << 21) | (4 << 16) | (6 << 11) | (695 << 1),
            true,
        ),
    ];
    for (word, updates) in forms {
        let mut memory = PpcSectionMem::new();
        memory.add_readonly_region(CODE, word.to_be_bytes().to_vec());
        memory.add_region(DATA, vec![0; 4]);
        let mut cpu = PpcCpu::new();
        for bits in [
            0,
            (-0.0f64).to_bits(),
            2.5f64.to_bits(),
            (-2.75f64).to_bits(),
            f64::from(f32::MIN_POSITIVE).to_bits(),
            f64::from(f32::MAX).to_bits(),
            f64::from(f32::from_bits(1)).to_bits(),
            0.1f64.to_bits(),
            f64::MAX.to_bits(),
            0x7ff0_0000_0000_0001,
        ] {
            for mode in 0..4 {
                for execution in ["step", "run", "cached"] {
                    cpu.pc = CODE;
                    cpu.gpr[4] = DATA - 4;
                    cpu.gpr[6] = 4;
                    cpu.fpr[1] = bits;
                    let fpscr = 0xa502_40f8 | mode;
                    cpu.fpscr = fpscr;
                    match execution {
                        "cached" => assert_eq!(
                            cpu.run_with_imports(&mut memory, 1, 0, 0, 0, |_, _, _| unreachable!()),
                            PpcRunResult::CycleLimit { cycles: 1 }
                        ),
                        "run" => assert_eq!(
                            cpu.run(&mut memory, 1, 0),
                            PpcRunResult::CycleLimit { cycles: 1 }
                        ),
                        "step" => assert_eq!(cpu.step(&mut memory, word), PpcStepResult::Stepped),
                        _ => unreachable!(),
                    }
                    assert_eq!(memory.read_u32_be(DATA), Some(reference(bits, mode)));
                    assert_eq!(cpu.gpr[4], if updates { DATA } else { DATA - 4 });
                    assert_eq!(cpu.pc, CODE + 4);
                    assert_eq!(cpu.fpr[1], bits);
                    assert_eq!(cpu.fpscr, fpscr);
                }
            }
        }
    }
}
