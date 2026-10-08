//! Overflow when the rounding mode delivers the largest finite number
//! (issue #83).
//!
//! MPCFPE32B Rev. 2 §3.3.6.2.1 (p. 3-33) sets FPSCR[OX] whenever the result,
//! rounded as though the exponent range were unbounded, exceeds the format's
//! largest finite number, "regardless of the FPSCR[OE] value" and in every
//! rounding mode. Table 3-15 picks the delivered value for OE = 0 (the largest
//! finite number when rounding toward zero); Table 3-14 subtracts 1536
//! (double) or 192 (single and `frsp`) from the exponent when OE = 1 and sets
//! FEX.
//!
//! The expected values are the reproductions from issue #83. FR is undefined
//! for a disabled overflow (Table 3-14); the crate keeps its usual rule (FR is
//! set when rounding incremented the fraction), which gives 0 for the largest
//! finite number and 1 for infinity.

use ppc::PpcCpu;

const FPSCR_OE: u32 = 0x0000_0040;
const RN_TOWARD_ZERO: u32 = 0b01;
const RN_TOWARD_POSITIVE: u32 = 0b10;
const RN_TOWARD_NEGATIVE: u32 = 0b11;

const FX: u32 = 0x8000_0000;
const FEX: u32 = 0x4000_0000;
const OX: u32 = 0x1000_0000;

/// A-form floating-point instruction word, operands in f2 (A), f3 (B) and f4
/// (C), result in f1. Fields an instruction doesn't use are zero.
fn a_form(opcd: u32, xo: u32) -> u32 {
    let (a, b, c) = match xo {
        25 => (2, 0, 4),           // fmul
        18 | 20 | 21 => (2, 3, 0), // fdiv, fsub, fadd
        22 | 24 | 26 => (0, 3, 0), // fsqrt, fres, frsqrte
        _ => (2, 3, 4),            // multiply-adds
    };
    (opcd << 26) | (1 << 21) | (a << 16) | (b << 11) | (c << 6) | (xo << 1)
}

/// X-form `frsp f1,f3`.
fn frsp() -> u32 {
    (63 << 26) | (1 << 21) | (3 << 11) | (12 << 1)
}

fn run(word: u32, fpscr: u32, a: u64, b: u64, c: u64) -> (u64, u32) {
    let mut cpu = PpcCpu::new();
    cpu.fpscr = fpscr;
    cpu.fpr[2] = a;
    cpu.fpr[3] = b;
    cpu.fpr[4] = c;
    cpu.step_instruction(word);
    (cpu.fpr[1], cpu.fpscr)
}

const FADD: (u32, u32) = (63, 21);
const FSUB: (u32, u32) = (63, 20);
const FMUL: (u32, u32) = (63, 25);
const FDIV: (u32, u32) = (63, 18);
const FMADD: (u32, u32) = (63, 29);
const FNMADD: (u32, u32) = (63, 31);
const FADDS: (u32, u32) = (59, 21);
const FSUBS: (u32, u32) = (59, 20);
const FMULS: (u32, u32) = (59, 25);
const FDIVS: (u32, u32) = (59, 18);
const FMADDS: (u32, u32) = (59, 29);
const FNMADDS: (u32, u32) = (59, 31);
const FRES: (u32, u32) = (59, 24);

fn op((opcd, xo): (u32, u32)) -> u32 {
    a_form(opcd, xo)
}

fn pow2(exponent: i32) -> u64 {
    assert!((-1022..=1023).contains(&exponent));
    ((exponent + 1023) as u64) << 52
}

const DOUBLE_MAX: u64 = 0x7fef_ffff_ffff_ffff;
const NEG_DOUBLE_MAX: u64 = 0xffef_ffff_ffff_ffff;
const SINGLE_MAX: u64 = 0x47ef_ffff_e000_0000;
const NEG_SINGLE_MAX: u64 = 0xc7ef_ffff_e000_0000;

#[test]
fn fmul_overflow_toward_zero_sets_ox() {
    // Issue #83: 2^600 × 2^600 toward zero.
    assert_eq!(
        run(op(FMUL), RN_TOWARD_ZERO, pow2(600), 0, pow2(600)),
        (DOUBLE_MAX, 0x9202_4001)
    );
}

#[test]
fn fmul_overflow_to_nearest_is_unchanged() {
    assert_eq!(
        run(op(FMUL), 0, pow2(600), 0, pow2(600)),
        (0x7ff0_0000_0000_0000, 0x9206_5000)
    );
}

#[test]
fn fmul_enabled_overflow_toward_zero_delivers_adjusted_result() {
    // 2^1200 × 2^-1536 = 2^-336, with OX and FEX set, as in round to nearest.
    let (value, fpscr) = run(op(FMUL), FPSCR_OE | RN_TOWARD_ZERO, pow2(600), 0, pow2(600));
    assert_eq!(value, 0x2af0_0000_0000_0000);
    assert_eq!(fpscr, 0xd202_4041);
    let (nearest_value, nearest_fpscr) = run(op(FMUL), FPSCR_OE, pow2(600), 0, pow2(600));
    assert_eq!(nearest_value, 0x2af0_0000_0000_0000);
    assert_eq!(nearest_fpscr & (FX | FEX | OX), FX | FEX | OX);
}

#[test]
fn fmuls_overflow_toward_zero_sets_ox() {
    // Issue #83: 2^100 × 2^100 in single precision.
    assert_eq!(
        run(op(FMULS), RN_TOWARD_ZERO, pow2(100), 0, pow2(100)),
        (SINGLE_MAX, 0x9202_4001)
    );
    // OE = 1: 2^200 × 2^-192 = 2^8.
    let (value, fpscr) = run(
        op(FMULS),
        FPSCR_OE | RN_TOWARD_ZERO,
        pow2(100),
        0,
        pow2(100),
    );
    assert_eq!(value, 0x4070_0000_0000_0000);
    assert_eq!(fpscr & (FX | FEX | OX), FX | FEX | OX);
}

#[test]
fn every_issue_reproduction_toward_zero_and_negative_sets_ox() {
    let f32_max = SINGLE_MAX;
    let neg_f32_max = NEG_SINGLE_MAX;
    // (instruction, A, B, C, delivered value)
    let cases: [(u32, u64, u64, u64, u64); 12] = [
        (op(FADD), DOUBLE_MAX, DOUBLE_MAX, 0, DOUBLE_MAX),
        (op(FSUB), DOUBLE_MAX, NEG_DOUBLE_MAX, 0, DOUBLE_MAX),
        (op(FMUL), pow2(600), 0, pow2(600), DOUBLE_MAX),
        (op(FDIV), pow2(600), pow2(-600), 0, DOUBLE_MAX),
        (op(FMADD), pow2(600), 0, pow2(600), DOUBLE_MAX),
        (op(FADDS), f32_max, f32_max, 0, SINGLE_MAX),
        (op(FSUBS), f32_max, neg_f32_max, 0, SINGLE_MAX),
        (op(FMULS), pow2(100), 0, pow2(100), SINGLE_MAX),
        (op(FDIVS), pow2(100), pow2(-100), 0, SINGLE_MAX),
        (op(FMADDS), pow2(100), 0, pow2(100), SINGLE_MAX),
        (frsp(), 0, pow2(200), 0, SINGLE_MAX),
        // 1 / 2^-140, a single-precision subnormal operand.
        (op(FRES), 0, 0x3730_0000_0000_0000, 0, SINGLE_MAX),
    ];
    for (word, a, b, c, expected) in cases {
        for mode in [RN_TOWARD_ZERO, RN_TOWARD_NEGATIVE] {
            let (value, fpscr) = run(word, mode, a, b, c);
            assert_eq!(value, expected, "{word:#010x} mode {mode}");
            assert_eq!(
                fpscr & (FX | FEX | OX),
                FX | OX,
                "{word:#010x} mode {mode}: {fpscr:#010x}"
            );
        }
    }
}

#[test]
fn negative_overflow_toward_positive_and_zero_sets_ox() {
    for mode in [RN_TOWARD_ZERO, RN_TOWARD_POSITIVE] {
        let (value, fpscr) = run(op(FMUL), mode, pow2(600), 0, pow2(600) | 1 << 63);
        assert_eq!(value, NEG_DOUBLE_MAX);
        assert_eq!(fpscr & (FX | OX), FX | OX, "mode {mode}");
        let (value, fpscr) = run(op(FMULS), mode, pow2(100), 0, pow2(100) | 1 << 63);
        assert_eq!(value, NEG_SINGLE_MAX);
        assert_eq!(fpscr & (FX | OX), FX | OX, "mode {mode}");
    }
    // Issue #83: fnmadds rounds 2^100 × 2^100 + 0 and then negates it.
    assert_eq!(
        run(op(FNMADDS), RN_TOWARD_ZERO, pow2(100), 0, pow2(100)),
        (NEG_SINGLE_MAX, 0x9202_8001)
    );
    // fnmadd toward −∞ rounds the positive 2^1200 down to the largest double
    // and then negates it.
    let (value, fpscr) = run(op(FNMADD), RN_TOWARD_NEGATIVE, pow2(600), 0, pow2(600));
    assert_eq!(value, NEG_DOUBLE_MAX);
    assert_eq!(fpscr & (FX | OX), FX | OX);
}

#[test]
fn results_between_max_and_the_next_binade_do_not_overflow_toward_zero() {
    // DOUBLE_MAX + 2^970 = 2^1024 - 2^970 rounds toward zero to DOUBLE_MAX
    // with an unbounded exponent too, so it is inexact but not an overflow.
    let (value, fpscr) = run(op(FADD), RN_TOWARD_ZERO, DOUBLE_MAX, pow2(970), 0);
    assert_eq!(value, DOUBLE_MAX);
    assert_eq!(fpscr & OX, 0);
    assert_ne!(fpscr & 0x0200_0000, 0); // XX

    // 2^1023 × 2 − 2^-1074 is just below 2^1024. Rounded to nearest in quad
    // precision it would be 2^1024, so the test must truncate.
    let (value, fpscr) = run(
        op(FMADD),
        RN_TOWARD_ZERO | FPSCR_OE,
        pow2(1023),
        1 << 63 | 1,
        pow2(1),
    );
    assert_eq!(value, DOUBLE_MAX);
    assert_eq!(fpscr & (FEX | OX), 0);

    // Single precision: f32::MAX + 2^103 = 2^128 − 2^103.
    let (value, fpscr) = run(op(FADDS), RN_TOWARD_ZERO, SINGLE_MAX, pow2(103), 0);
    assert_eq!(value, SINGLE_MAX);
    assert_eq!(fpscr & OX, 0);
    // frsp of the largest double below 2^128.
    let (value, fpscr) = run(frsp(), RN_TOWARD_ZERO, 0, pow2(128) - 1, 0);
    assert_eq!(value, SINGLE_MAX);
    assert_eq!(fpscr & OX, 0);
}

#[test]
fn ox_is_sticky_and_fx_reports_only_a_new_exception() {
    let before = OX | RN_TOWARD_ZERO;
    let (_, fpscr) = run(op(FMUL), before, pow2(600), 0, pow2(600));
    // OX was already set; FX is still set because XX is new.
    assert_eq!(fpscr & OX, OX);
    assert_eq!(fpscr & FX, FX);
    let before = OX | 0x0200_0000 | RN_TOWARD_ZERO;
    // OX and XX were both already set: nothing is new, so FX stays clear.
    let (_, fpscr) = run(op(FMUL), before, pow2(600), 0, pow2(600));
    assert_eq!(fpscr & (FX | OX), OX);
}

#[test]
fn enabled_overflow_adjusted_value_is_rounded_once_in_directed_modes() {
    // fmsub toward −∞: a × c (about 2^1189) is a double, and subtracting
    // the much smaller positive b puts the exact result just below it.
    // Rounding to nearest in quad first would land on that double; rounding
    // toward −∞ once gives the double below it.
    const FMSUB: (u32, u32) = (63, 28);
    let (value, fpscr) = run(
        op(FMSUB),
        FPSCR_OE | RN_TOWARD_NEGATIVE,
        0x7920_0000_0000_0000,
        0x4b02_052e_5e82_12f7,
        0x6f22_c4b4_5000_0000,
    );
    assert_eq!(value, 0x4852_c4b4_4fff_ffff);
    assert_eq!(fpscr & (FEX | OX), FEX | OX);
}

#[test]
fn enabled_overflow_of_negated_multiply_add_rounds_before_negating() {
    // fnmadd toward +∞ rounds the positive a × c + b up, then negates it,
    // so the adjusted result is the larger magnitude.
    let (value, fpscr) = run(
        op(FNMADD),
        FPSCR_OE | RN_TOWARD_POSITIVE,
        0x639f_ffff_ffff_fffa,
        DOUBLE_MAX,
        0x5c40_0000_0000_0002,
    );
    assert_eq!(value, 0x9fff_ffff_ffff_ffff);
    assert_eq!(fpscr & (FEX | OX), FEX | OX);
}

#[test]
fn frsp_enabled_overflow_toward_zero_delivers_adjusted_result() {
    // 2^200 × 2^-192 = 2^8.
    let (value, fpscr) = run(frsp(), FPSCR_OE | RN_TOWARD_ZERO, 0, pow2(200), 0);
    assert_eq!(value, 0x4070_0000_0000_0000);
    assert_eq!(fpscr & (FX | FEX | OX), FX | FEX | OX);
}
