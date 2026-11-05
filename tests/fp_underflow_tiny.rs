//! Disabled underflow detects tininess before rounding (issue #84).
//!
//! MPCFPE32B Rev. 2 §3.3.6.2.2 (p. 3-34): with FPSCR[UE] = 0, underflow occurs
//! when the intermediate result, computed as though with infinite precision and
//! unbounded exponent range, is below the smallest normal number and the rounded
//! result is inexact. So UX is set even when rounding carries the result up to
//! exactly the smallest normal number.

use ppc::PpcCpu;

const RN_TOWARD_ZERO: u32 = 1;
const RN_TOWARD_POSITIVE: u32 = 2;
const RN_TOWARD_NEGATIVE: u32 = 3;
const FPSCR_UE: u32 = 0x0000_0020;
const UX: u32 = 0x0800_0000;

/// FX, UX, XX, FR, FI, FPRF = +normal.
const UX_ROUNDED_UP: u32 = 0x8a06_4000;
const SINGLE_MIN_NORMAL: u64 = 0x3810_0000_0000_0000;
const DOUBLE_MIN_NORMAL: u64 = 0x0010_0000_0000_0000;
/// 1 - 2^-24, the largest single below 1.
const BELOW_ONE_SINGLE: u64 = 0x3fef_ffff_e000_0000;
/// 1 - 2^-53, the largest double below 1.
const BELOW_ONE_DOUBLE: u64 = 0x3fef_ffff_ffff_ffff;
const SIGN: u64 = 1 << 63;

fn a_form(opcd: u32, xo: u32) -> u32 {
    let (a, b, c) = match xo {
        25 => (2, 0, 4),           // fmul
        18 | 20 | 21 => (2, 3, 0), // fdiv, fsub, fadd
        22 | 24 => (0, 3, 0),      // fsqrt, fres
        _ => (2, 3, 4),            // multiply-adds
    };
    (opcd << 26) | (1 << 21) | (a << 16) | (b << 11) | (c << 6) | (xo << 1)
}

const FADD: u32 = (63 << 26) | (1 << 21) | (2 << 16) | (3 << 11) | (21 << 1);
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

fn fmuls() -> u32 {
    a_form(59, 25)
}
fn fmul() -> u32 {
    a_form(63, 25)
}

#[test]
fn fmuls_rounding_up_to_smallest_normal_sets_ux() {
    assert_eq!(
        run(fmuls(), 0, BELOW_ONE_SINGLE, 0, SINGLE_MIN_NORMAL),
        (SINGLE_MIN_NORMAL, UX_ROUNDED_UP)
    );
    assert_eq!(
        run(
            fmuls(),
            RN_TOWARD_POSITIVE,
            BELOW_ONE_SINGLE,
            0,
            SINGLE_MIN_NORMAL
        ),
        (SINGLE_MIN_NORMAL, UX_ROUNDED_UP | RN_TOWARD_POSITIVE)
    );
}

#[test]
fn negative_fmuls_rounding_away_from_zero_sets_ux() {
    let (value, fpscr) = run(
        fmuls(),
        RN_TOWARD_NEGATIVE,
        BELOW_ONE_SINGLE | SIGN,
        0,
        SINGLE_MIN_NORMAL,
    );
    assert_eq!(value, SINGLE_MIN_NORMAL | SIGN);
    assert_eq!(fpscr, 0x8a06_8003);
}

#[test]
fn double_instructions_rounding_up_to_smallest_normal_set_ux() {
    assert_eq!(
        run(fmul(), 0, BELOW_ONE_DOUBLE, 0, DOUBLE_MIN_NORMAL),
        (DOUBLE_MIN_NORMAL, UX_ROUNDED_UP)
    );
    assert_eq!(
        run(
            a_form(63, 18),
            0,
            BELOW_ONE_DOUBLE,
            0x7fd0_0000_0000_0000,
            0
        ),
        (DOUBLE_MIN_NORMAL, UX_ROUNDED_UP)
    );
    assert_eq!(
        run(a_form(63, 29), 0, BELOW_ONE_DOUBLE, 0, DOUBLE_MIN_NORMAL),
        (DOUBLE_MIN_NORMAL, UX_ROUNDED_UP)
    );
}

#[test]
fn single_instructions_rounding_up_to_smallest_normal_set_ux() {
    assert_eq!(
        run(
            a_form(59, 18),
            0,
            BELOW_ONE_SINGLE,
            0x47d0_0000_0000_0000,
            0
        ),
        (SINGLE_MIN_NORMAL, UX_ROUNDED_UP)
    );
    assert_eq!(
        run(a_form(59, 29), 0, BELOW_ONE_SINGLE, 0, SINGLE_MIN_NORMAL),
        (SINGLE_MIN_NORMAL, UX_ROUNDED_UP)
    );
    // 2^-126 x (1 - 2^-30).
    assert_eq!(
        run(FRSP, 0, 0, 0x380f_ffff_ff80_0000, 0),
        (SINGLE_MIN_NORMAL, UX_ROUNDED_UP)
    );
}

#[test]
fn fres_rounding_up_to_smallest_normal_sets_ux() {
    // 2^126 x (1 + 2^-23), toward +infinity.
    assert_eq!(
        run(
            a_form(59, 24),
            RN_TOWARD_POSITIVE,
            0,
            0x47d0_0000_2000_0000,
            0
        ),
        (SINGLE_MIN_NORMAL, 0x8a06_4002)
    );
}

#[test]
fn tiny_result_that_quad_rounds_to_the_boundary_is_still_tiny() {
    // 2^-1022 - 2^-2148 needs more bits than quad has, so the quad result
    // rounded to nearest is 2^-1022 itself.
    let smallest_denormal = 1u64;
    assert_eq!(
        run(
            a_form(63, 29),
            0,
            smallest_denormal,
            DOUBLE_MIN_NORMAL,
            smallest_denormal | SIGN,
        ),
        (DOUBLE_MIN_NORMAL, UX_ROUNDED_UP)
    );
}

#[test]
fn staying_subnormal_toward_zero_is_unchanged() {
    // The issue's control: the rounded result stays subnormal.
    assert_eq!(
        run(
            fmul(),
            RN_TOWARD_ZERO,
            BELOW_ONE_DOUBLE,
            0,
            DOUBLE_MIN_NORMAL
        ),
        (0x000f_ffff_ffff_ffff, 0x8a03_4001)
    );
}

#[test]
fn exact_tiny_results_do_not_set_ux() {
    // 2^-70 x 2^-70 = 2^-140 is an exact single denormal.
    let (value, fpscr) = run(fmuls(), 0, 0x3b90_0000_0000_0000, 0, 0x3b90_0000_0000_0000);
    assert_eq!(value, 0x3730_0000_0000_0000);
    assert_eq!(fpscr & UX, 0);
    // 2^-530 x 2^-530 = 2^-1060 is an exact double denormal.
    let (_, fpscr) = run(fmul(), 0, 0x1ed0_0000_0000_0000, 0, 0x1ed0_0000_0000_0000);
    assert_eq!(fpscr & UX, 0);
}

#[test]
fn inexact_results_at_or_above_the_boundary_do_not_set_ux() {
    // (1 + 2^-23)^2 x 2^-126 is inexact, and not tiny.
    let (_, fpscr) = run(fmuls(), 0, 0x3ff0_0000_2000_0000, 0, 0x3810_0000_2000_0000);
    assert_ne!(fpscr & 0x0200_0000, 0, "XX");
    assert_eq!(fpscr & UX, 0);
    // (1 + 2^-52)^2 x 2^-1022 as a double.
    let (_, fpscr) = run(fmul(), 0, 0x3ff0_0000_0000_0001, 0, 0x0010_0000_0000_0001);
    assert_ne!(fpscr & 0x0200_0000, 0, "XX");
    assert_eq!(fpscr & UX, 0);
}

#[test]
fn sums_of_doubles_near_the_boundary_are_exact() {
    // 2^-1022 - 2^-1074 is an exact denormal.
    let (value, fpscr) = run(FADD, 0, DOUBLE_MIN_NORMAL, 1 | SIGN, 0);
    assert_eq!(value, 0x000f_ffff_ffff_ffff);
    assert_eq!(fpscr & UX, 0);
}

#[test]
fn fsqrt_of_the_smallest_denormal_is_not_tiny() {
    let (_, fpscr) = run(a_form(63, 22), 0, 0, 1, 0);
    assert_eq!(fpscr & UX, 0);
}

#[test]
fn enabled_underflow_still_adjusts() {
    let (value, fpscr) = run(fmuls(), FPSCR_UE, BELOW_ONE_SINGLE, 0, SINGLE_MIN_NORMAL);
    assert_eq!(value, 0x440f_ffff_e000_0000);
    assert_ne!(fpscr & UX, 0);
}
