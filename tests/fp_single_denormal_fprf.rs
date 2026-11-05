//! FPRF classifies a single-precision result as a single (issue #87).
//!
//! MPCFPE32B Rev. 2 §3.3.4 (p. 3-19): status bits "are set to reflect the
//! single-precision result". Table 3-10 (p. 3-25) gives C = 1 for a denormal.
//! A single denormal is stored widened to a normal double, so classifying the
//! register value reports it as normalized.

use ppc::{PpcCpu, PpcRunResult, PpcSectionMem};

const CODE: u32 = 0x1000;
const SIGN: u64 = 1 << 63;
const POSITIVE_DENORMAL: u32 = 0x0001_4000;
const NEGATIVE_DENORMAL: u32 = 0x0001_8000;
const POSITIVE_NORMAL: u32 = 0x0000_4000;

const fn a_form(opcd: u32, xo: u32, a: u32, b: u32, c: u32) -> u32 {
    (opcd << 26) | (1 << 21) | (a << 16) | (b << 11) | (c << 6) | (xo << 1)
}

const FADDS: u32 = a_form(59, 21, 2, 3, 0);
const FSUBS: u32 = a_form(59, 20, 2, 3, 0);
const FMULS: u32 = a_form(59, 25, 2, 0, 4);
const FDIVS: u32 = a_form(59, 18, 2, 3, 0);
const FMADDS: u32 = a_form(59, 29, 2, 3, 4);
const FMSUBS: u32 = a_form(59, 28, 2, 3, 4);
const FNMADDS: u32 = a_form(59, 31, 2, 3, 4);
const FNMSUBS: u32 = a_form(59, 30, 2, 3, 4);
const FRES: u32 = a_form(59, 24, 0, 3, 0);
const FRSP: u32 = (63 << 26) | (1 << 21) | (3 << 11) | (12 << 1);
const FMUL: u32 = a_form(63, 25, 2, 0, 4);

fn pow2(exponent: i32) -> u64 {
    ((exponent + 1023) as u64) << 52
}

fn step(word: u32, fpscr: u32, a: u64, b: u64, c: u64) -> (u64, u32) {
    let mut cpu = PpcCpu::new();
    cpu.fpscr = fpscr;
    cpu.fpr[2] = a;
    cpu.fpr[3] = b;
    cpu.fpr[4] = c;
    cpu.step_instruction(word);
    (cpu.fpr[1], cpu.fpscr)
}

#[test]
fn fmuls_denormal_is_classified_as_denormal() {
    assert_eq!(
        step(FMULS, 0, pow2(-70), 0, pow2(-70)),
        (pow2(-140), POSITIVE_DENORMAL)
    );
    assert_eq!(
        step(FMULS, 0, pow2(-70) | SIGN, 0, pow2(-70)),
        (pow2(-140) | SIGN, NEGATIVE_DENORMAL)
    );
}

#[test]
fn rounded_denormal_is_classified_as_denormal() {
    assert_eq!(
        step(FMULS, 1, 0x3fef_ffff_e000_0000, 0, 0x3810_0000_0000_0000),
        (0x380f_ffff_c000_0000, 0x8a03_4001)
    );
}

#[test]
fn other_single_instructions_classify_denormals() {
    for (word, a, b, c, value) in [
        (FADDS, pow2(-130), pow2(-140), 0, 0x37d0_0400_0000_0000),
        (FSUBS, pow2(-126), pow2(-127), 0, pow2(-127)),
        (FDIVS, pow2(-100), pow2(40), 0, pow2(-140)),
        (FMADDS, pow2(-70), 0, pow2(-70), pow2(-140)),
        (FMSUBS, pow2(-70), 0, pow2(-70), pow2(-140)),
        (FRSP, 0, pow2(-140), 0, pow2(-140)),
        (FRES, 0, pow2(127), 0, pow2(-127)),
    ] {
        assert_eq!(
            step(word, 0, a, b, c),
            (value, POSITIVE_DENORMAL),
            "{word:#010x}"
        );
    }
    for word in [FNMADDS, FNMSUBS] {
        assert_eq!(
            step(word, 0, pow2(-70), 0, pow2(-70)),
            (pow2(-140) | SIGN, NEGATIVE_DENORMAL),
            "{word:#010x}"
        );
    }
}

#[test]
fn run_loops_classify_denormals() {
    // `run_with_imports` takes the unobserved native path for `fmuls`.
    for negative in [false, true] {
        let mut memory = PpcSectionMem::new();
        memory.add_readonly_region(CODE, FMULS.to_be_bytes().to_vec());
        let a = pow2(-70) | if negative { SIGN } else { 0 };
        let expected = if negative {
            NEGATIVE_DENORMAL
        } else {
            POSITIVE_DENORMAL
        };
        for imports in [false, true] {
            let mut cpu = PpcCpu::new();
            cpu.pc = CODE;
            cpu.fpr[2] = a;
            cpu.fpr[4] = pow2(-70);
            let result = if imports {
                cpu.run_with_imports(&mut memory, 1, 0, 0, 0, |_, _, _| unreachable!())
            } else {
                cpu.run(&mut memory, 1, 0)
            };
            assert_eq!(result, PpcRunResult::CycleLimit { cycles: 1 });
            assert_eq!(cpu.fpscr, expected);
        }
    }
}

#[test]
fn smallest_normal_single_stays_normal() {
    assert_eq!(
        step(FMULS, 0, pow2(-63), 0, pow2(-63)),
        (pow2(-126), POSITIVE_NORMAL)
    );
    assert_eq!(
        step(FRSP, 0, 0, pow2(-126), 0),
        (pow2(-126), POSITIVE_NORMAL)
    );
}

#[test]
fn single_zero_results_are_classified_as_zero() {
    assert_eq!(step(FMULS, 0, 0, 0, pow2(0)), (0, 0x0000_2000));
    assert_eq!(step(FMULS, 0, SIGN, 0, pow2(0)), (SIGN, 0x0001_2000));
}

#[test]
fn double_instructions_classify_the_double() {
    // 2^-140 is a normal double for `fmul`.
    assert_eq!(
        step(FMUL, 0, pow2(-70), 0, pow2(-70)),
        (pow2(-140), POSITIVE_NORMAL)
    );
    // 2^-1060 is a double denormal.
    assert_eq!(
        step(FMUL, 0, pow2(-530), 0, pow2(-530)),
        (0x0000_0000_0000_4000, POSITIVE_DENORMAL)
    );
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
#[test]
fn jit_classifies_denormals() {
    let mut code: Vec<u8> = FMULS.to_be_bytes().to_vec();
    code.extend_from_slice(&0x4e80_0020u32.to_be_bytes()); // blr
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(CODE, code);
    let mut cpu = PpcCpu::new();
    cpu.pc = CODE;
    cpu.lr = 0;
    cpu.fpr[2] = pow2(-70);
    cpu.fpr[4] = pow2(-70);
    cpu.enable_jit(ppc::PpcJitConfig {
        threshold: 0,
        verify: true,
        jitdump: false,
        background: false,
        hot_threshold: 0,
    })
    .unwrap();
    cpu.run_with_imports(&mut memory, 100, 0, 0, 0, |_, _, _| unreachable!());
    assert!(cpu.jit_stats().block_runs > 0, "ran natively");
    assert_eq!(cpu.fpr[1], pow2(-140));
    assert_eq!(cpu.fpscr, POSITIVE_DENORMAL);
}
