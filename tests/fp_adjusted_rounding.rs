//! XX, FR and FI describe the rounding of the exponent-adjusted result
//! (issue #86).
//!
//! MPCFPE32B Rev. 2 Table 3-14 (p. 3-33) and Table 3-16 (pp. 3-34-3-35): with
//! FPSCR[OE] = 1 or FPSCR[UE] = 1, XX is "set if rounded result differs from
//! intermediate result" and FR and FI "reflect rounding". §3.3.6.2.3 (p. 3-35):
//! after an enabled overflow or underflow, an inexact condition occurs only if
//! the significand of the rounded result differs from the intermediate's.

use ppc::PpcCpu;

const OE: u32 = 0x40;
const UE: u32 = 0x20;
const XE: u32 = 0x08;
const TOWARD_POSITIVE: u32 = 2;

/// FX, FEX, OX, FPRF = +normal.
const OVERFLOW_EXACT: u32 = 0xd000_4040;
/// FX, FEX, UX, FPRF = +normal.
const UNDERFLOW_EXACT: u32 = 0xc800_4020;

const fn a_form(opcd: u32, xo: u32, a: u32, b: u32, c: u32) -> u32 {
    (opcd << 26) | (1 << 21) | (a << 16) | (b << 11) | (c << 6) | (xo << 1)
}

const FMUL: u32 = a_form(63, 25, 2, 0, 4);
const FMULS: u32 = a_form(59, 25, 2, 0, 4);
const FDIVS: u32 = a_form(59, 18, 2, 3, 0);
const FMADDS: u32 = a_form(59, 29, 2, 3, 4);
const FNMADDS: u32 = a_form(59, 31, 2, 3, 4);
const FRES: u32 = a_form(59, 24, 0, 3, 0);
const FRSP: u32 = (63 << 26) | (1 << 21) | (3 << 11) | (12 << 1);

fn run(word: u32, fpscr: u32, a: u64, b: u64, c: u64) -> (u64, u32) {
    let mut cpu = PpcCpu::new();
    cpu.fpscr = fpscr;
    cpu.fpr[2] = a;
    cpu.fpr[3] = b;
    cpu.fpr[4] = c;
    cpu.step_instruction(word);
    (cpu.fpr[1], cpu.fpscr)
}

fn pow2(exponent: i32) -> u64 {
    ((exponent + 1023) as u64) << 52
}

#[test]
fn exact_scaled_overflow_is_not_inexact() {
    assert_eq!(
        run(FMULS, OE, pow2(100), 0, pow2(100)),
        (pow2(8), OVERFLOW_EXACT)
    );
    assert_eq!(
        run(FDIVS, OE, pow2(0), pow2(-140), 0),
        (0x3cb0_0000_0000_0000, OVERFLOW_EXACT)
    );
    assert_eq!(
        run(FMADDS, OE, pow2(100), 0, pow2(100)),
        (pow2(8), OVERFLOW_EXACT)
    );
    assert_eq!(run(FRSP, OE, 0, pow2(200), 0), (pow2(8), OVERFLOW_EXACT));
    assert_eq!(
        run(FMUL, OE, pow2(600), 0, pow2(600)),
        (0x2af0_0000_0000_0000, OVERFLOW_EXACT)
    );
    // `fres` is an estimate, but 1 / 2^-140 has an exact reciprocal here.
    assert_eq!(
        run(FRES, OE, 0, pow2(-140), 0),
        (0x3cb0_0000_0000_0000, OVERFLOW_EXACT)
    );
}

#[test]
fn exact_scaled_underflow_is_not_inexact() {
    // (1 - 2^-24) x 2^-126 is a tiny exact product, also exact once scaled.
    assert_eq!(
        run(FMULS, UE, 0x3fef_ffff_e000_0000, 0, 0x3810_0000_0000_0000),
        (0x440f_ffff_e000_0000, UNDERFLOW_EXACT)
    );
    assert_eq!(
        run(FMULS, UE, pow2(-75), 0, pow2(-75)),
        (pow2(42), UNDERFLOW_EXACT)
    );
    assert_eq!(
        run(FMUL, UE, pow2(-600), 0, pow2(-600)),
        (pow2(336), UNDERFLOW_EXACT)
    );
    assert_eq!(run(FRSP, UE, 0, pow2(-150), 0), (pow2(42), UNDERFLOW_EXACT));
}

#[test]
fn negated_result_reports_the_same_rounding() {
    let (value, fpscr) = run(FNMADDS, OE, pow2(100), 0, pow2(100));
    assert_eq!(value, pow2(8) | (1 << 63));
    assert_eq!(fpscr, 0xd000_8040);
}

#[test]
fn exact_scaled_result_does_not_raise_an_enabled_inexact() {
    let (_, fpscr) = run(FMULS, OE | XE, pow2(100), 0, pow2(100));
    assert_eq!(fpscr, OVERFLOW_EXACT | XE);
}

#[test]
fn sticky_xx_survives_while_fr_and_fi_clear() {
    // XX is sticky, and FR and FI describe only this instruction.
    let (_, fpscr) = run(FMULS, OE | 0x0206_0000, pow2(100), 0, pow2(100));
    assert_eq!(fpscr, 0xd200_4040);
}

#[test]
fn scaled_result_rounded_down_sets_xx_and_fi_only() {
    // 2^8 x (1 + 2^-22 + 2^-46) rounds down.
    assert_eq!(
        run(FMULS, OE, 0x4630_0000_2000_0000, 0, 0x4630_0000_2000_0000),
        (0x4070_0000_4000_0000, 0xd202_4040)
    );
    // 2^-336 x (1 + 2^-29 + 2^-60) rounds down, in double precision.
    assert_eq!(
        run(FMUL, OE, 0x6570_0000_0040_0000, 0, 0x6570_0000_0040_0000),
        (0x2af0_0000_0080_0000, 0xd202_4040)
    );
    assert_eq!(
        run(FMUL, UE, 0x1a70_0000_0040_0000, 0, 0x1a70_0000_0040_0000),
        (0x54f0_0000_0080_0000, 0xca02_4020)
    );
}

#[test]
fn scaled_result_rounded_up_sets_fr_too() {
    // 2^8 x (1 + 2^-11 + 2^-23 + 2^-24 + 2^-35) is above a tie, so rounds up.
    assert_eq!(
        run(FMULS, OE, 0x4630_0100_0000_0000, 0, 0x4630_0100_2000_0000),
        (0x4070_0200_4000_0000, 0xd206_4040)
    );
    // The double-precision product above, toward +infinity.
    assert_eq!(
        run(
            FMUL,
            OE | TOWARD_POSITIVE,
            0x6570_0000_0040_0000,
            0,
            0x6570_0000_0040_0000
        ),
        (0x2af0_0000_0080_0001, 0xd206_4042)
    );
    assert_eq!(
        run(
            FMUL,
            UE | TOWARD_POSITIVE,
            0x1a70_0000_0040_0000,
            0,
            0x1a70_0000_0040_0000
        ),
        (0x54f0_0000_0080_0001, 0xca06_4022)
    );
}

#[test]
fn disabled_overflow_and_underflow_keep_their_status() {
    assert_eq!(
        run(FMUL, 0, pow2(600), 0, pow2(600)),
        (0x7ff0_0000_0000_0000, 0x9206_5000)
    );
    // 2^-126 - 2^-150 toward zero stays subnormal, with XX and FI set.
    let (value, fpscr) = run(FMULS, 1, 0x3fef_ffff_e000_0000, 0, 0x3810_0000_0000_0000);
    assert_eq!(value, 0x380f_ffff_c000_0000);
    assert_eq!(fpscr & 0x0a02_0000, 0x0a02_0000);
}

#[test]
fn in_range_results_with_the_enables_set_keep_their_status() {
    // 1 / 3 is in range: inexact as usual, with no adjustment.
    let (value, fpscr) = run(
        FDIVS,
        OE | UE,
        0x3ff0_0000_0000_0000,
        0x4008_0000_0000_0000,
        0,
    );
    assert_eq!(value, 0x3fd5_5555_6000_0000);
    assert_eq!(fpscr, 0x8206_4060);
}
