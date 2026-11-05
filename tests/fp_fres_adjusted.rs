//! `fres` applies the enabled overflow and underflow exponent adjustment
//! (issue #85).
//!
//! MPCFPE32B Rev. 2 Table 3-14 (p. 3-33) and Table 3-16 (pp. 3-34-3-35): with
//! FPSCR[OE] = 1 or FPSCR[UE] = 1, a single-precision arithmetic instruction
//! subtracts or adds 192 to the exponent and implicitly sets FEX. FR and FI are
//! undefined for `fres` (p. 8-80), so they are not asserted here.

use ppc::PpcCpu;

const FPSCR_OE: u32 = 0x0000_0040;
const FPSCR_UE: u32 = 0x0000_0020;
const FX: u32 = 0x8000_0000;
const FEX: u32 = 0x4000_0000;
const OX: u32 = 0x1000_0000;
const UX: u32 = 0x0800_0000;
/// FPRF = C, FL, FG, FE, FU.
const FPRF: u32 = 0x0001_f000;
const FPRF_POSITIVE_NORMAL: u32 = 0x0000_4000;
const FPRF_POSITIVE_INFINITY: u32 = 0x0000_5000;
/// `fres f1,f3`.
const FRES: u32 = (59 << 26) | (1 << 21) | (3 << 11) | (24 << 1);
/// `fdivs f1,f2,f3`.
const FDIVS: u32 = (59 << 26) | (1 << 21) | (2 << 16) | (3 << 11) | (18 << 1);
const ONE: u64 = 0x3ff0_0000_0000_0000;

fn run(word: u32, fpscr: u32, b: u64) -> (u64, u32) {
    let mut cpu = PpcCpu::new();
    cpu.fpscr = fpscr;
    cpu.fpr[2] = ONE;
    cpu.fpr[3] = b;
    cpu.step_instruction(word);
    (cpu.fpr[1], cpu.fpscr)
}

fn status(fpscr: u32) -> u32 {
    fpscr & (FX | FEX | OX | UX | FPRF)
}

#[test]
fn enabled_overflow_is_adjusted_like_fdivs() {
    let (value, fpscr) = run(FRES, FPSCR_OE, 0x3730_0000_0000_0000);
    assert_eq!(value, 0x3cb0_0000_0000_0000);
    assert_eq!(status(fpscr), FX | FEX | OX | FPRF_POSITIVE_NORMAL);
    let (fdivs_value, fdivs_fpscr) = run(FDIVS, FPSCR_OE, 0x3730_0000_0000_0000);
    assert_eq!((value, status(fpscr)), (fdivs_value, status(fdivs_fpscr)));
}

#[test]
fn enabled_underflow_is_adjusted_like_fdivs() {
    let (value, fpscr) = run(FRES, FPSCR_UE, 0x47e0_0000_0000_0000);
    assert_eq!(value, 0x4400_0000_0000_0000);
    assert_eq!(status(fpscr), FX | FEX | UX | FPRF_POSITIVE_NORMAL);
    let (fdivs_value, fdivs_fpscr) = run(FDIVS, FPSCR_UE, 0x47e0_0000_0000_0000);
    assert_eq!((value, status(fpscr)), (fdivs_value, status(fdivs_fpscr)));
}

#[test]
fn negative_operand_keeps_its_sign() {
    let (value, _) = run(FRES, FPSCR_OE, 0xb730_0000_0000_0000);
    assert_eq!(value, 0xbcb0_0000_0000_0000);
    let (value, _) = run(FRES, FPSCR_UE, 0xc7e0_0000_0000_0000);
    assert_eq!(value, 0xc400_0000_0000_0000);
}

#[test]
fn disabled_overflow_still_delivers_infinity() {
    let (value, fpscr) = run(FRES, 0, 0x3730_0000_0000_0000);
    assert_eq!(value, 0x7ff0_0000_0000_0000);
    assert_eq!(status(fpscr), FX | OX | FPRF_POSITIVE_INFINITY);
}

#[test]
fn disabled_underflow_still_delivers_the_unadjusted_value() {
    let (value, fpscr) = run(FRES, 0, 0x47e0_0000_0000_0000);
    assert_eq!(value, 0x3800_0000_0000_0000);
    assert_eq!(fpscr & (FX | FEX | OX | UX), 0);
}

#[test]
fn in_range_reciprocal_ignores_the_enables() {
    // 1 / 2 is exact and in range, so OE = UE = 1 changes nothing.
    let (value, fpscr) = run(FRES, FPSCR_OE | FPSCR_UE, 0x4000_0000_0000_0000);
    assert_eq!(value, 0x3fe0_0000_0000_0000);
    assert_eq!(status(fpscr), FPRF_POSITIVE_NORMAL);
}

#[test]
fn zero_divide_and_invalid_operands_are_not_adjusted() {
    // 1 / +0 raises ZX and delivers +infinity with ZE clear.
    let (value, fpscr) = run(FRES, FPSCR_OE | FPSCR_UE, 0);
    assert_eq!(value, 0x7ff0_0000_0000_0000);
    assert_eq!(fpscr & (OX | UX), 0);
    // 1 / +infinity is an exact zero, which is not tiny.
    let (value, fpscr) = run(FRES, FPSCR_UE, 0x7ff0_0000_0000_0000);
    assert_eq!(value, 0);
    assert_eq!(fpscr & (OX | UX), 0);
}
