//! Correctly rounded square roots for `fsqrt`, `fsqrts` and `frsqrte`.
//!
//! The architecture rounds the infinitely precise root once, to the target
//! precision, in the FPSCR rounding mode; an exactly representable root is
//! exact in every mode, with FR and FI clear.
//!
//! For a positive, nonzero, finite `x`:
//!
//! 1. `s = sqrt(x)` natively.  IEEE 754 requires the native square root to be
//!    correctly rounded to nearest, so `s` is the round-to-nearest root.
//! 2. The residual `x - s * s` is computed exactly as a 128-bit integer once
//!    both sides are scaled to the last bit of `s * s` (see
//!    [`nearest_root()`]).  Its sign is the sign of `t - s`, where `t` is the
//!    true root, because `t - s = (x - s * s) / (t + s)` and `t + s > 0`.
//!    A zero residual means `t = s` exactly.
//! 3. When `t` is not a double it lies strictly between `s` and one of its
//!    neighbours, and the residual's sign says which.  Rounding toward zero
//!    or toward -infinity (the same for a positive root) picks the lower of
//!    the two, rounding toward +infinity the upper; stepping the bit pattern
//!    by one reaches the neighbour, including across a binade boundary, where
//!    the ulp below a power of two is half the ulp above it.
//!
//! No margin and no Newton step are needed: the residual is exact, so every
//! decision above is exact.  The root of a positive finite double lies in
//! `[2^-537, 2^512)`, so it is a normal double and stepping never reaches
//! zero or infinity.
//!
//! `fsqrts` rounds the same root to single precision.  Rounding to nearest
//! double first and then to single would round twice; instead,
//! [`sqrt_round_to_odd()`] returns the root *rounded to odd*: `t` itself when
//! it is a double, otherwise whichever of its two neighbouring doubles has an
//! odd significand.  Every single-precision value, every single-precision
//! rounding midpoint, the overflow threshold and the tininess threshold
//! `2^-126` has at most 25 significant bits, so as a double (all of them are
//! normal doubles) it has an even significand.  Hence no such boundary lies
//! strictly between `t` and its odd neighbour, neither `t` nor the odd
//! neighbour is a boundary, and both lie on the same side of every boundary.
//! Rounding the round-to-odd double to single precision, in any mode and
//! including single-precision subnormals and overflow, therefore gives the
//! same value and flags as rounding `t` once.  This is the standard
//! round-to-odd argument, which needs the intermediate format to have at
//! least two more significand bits than the target (53 >= 24 + 2).
//!
//! Zeros, negatives, infinities and NaNs take [`special()`], which returns
//! the results the crate has always returned for them.

use std::cmp::Ordering;

use rustc_apfloat::{Float, Status as ApStatus, StatusAnd, ieee::Double as ApDouble};

pub(crate) const HIDDEN_BIT: u64 = 1 << 52;
pub(crate) const FRACTION_MASK: u64 = HIDDEN_BIT - 1;
pub(crate) const POSITIVE_INFINITY: u64 = 0x7ff0_0000_0000_0000;
const SIGN_BIT: u64 = 1 << 63;
const QUIET_BIT: u64 = 1 << 51;
const DEFAULT_NAN: u64 = 0x7ff8_0000_0000_0000;

/// A square root with its status and its value rounded toward zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Root {
    /// The result, as double-precision bits.
    pub(crate) value: u64,
    /// The root rounded toward zero, for the FPSCR FR (fraction rounded)
    /// bit: the finish paths set FR when the magnitude of the rounded result
    /// exceeds this value's.
    pub(crate) truncated: u64,
    /// `OK`, `INEXACT`, or `INVALID_OP` for a signaling NaN or a negative.
    pub(crate) status: ApStatus,
}

impl Root {
    fn exact(value: u64, status: ApStatus) -> Self {
        Self {
            value,
            truncated: value,
            status,
        }
    }

    /// The result in the form the FPSCR finish paths take.
    pub(crate) fn result(self) -> StatusAnd<ApDouble> {
        StatusAnd {
            status: self.status,
            value: ApDouble::from_bits(u128::from(self.value)),
        }
    }

    pub(crate) fn truncated(self) -> ApDouble {
        ApDouble::from_bits(u128::from(self.truncated))
    }
}

/// Splits a positive finite double into `(significand, exponent)` with
/// `value = significand * 2^exponent`.
#[inline]
pub(crate) fn decompose(bits: u64) -> (u64, i32) {
    let biased = (bits >> 52) as i32;
    if biased == 0 {
        (bits & FRACTION_MASK, -1074)
    } else {
        ((bits & FRACTION_MASK) | HIDDEN_BIT, biased - 1075)
    }
}

/// Whether `bits` is a positive, nonzero, finite double.
#[inline]
pub(crate) fn is_positive_finite(bits: u64) -> bool {
    bits.wrapping_sub(1) < POSITIVE_INFINITY - 1
}

/// For a positive, nonzero, finite `bits`, returns the round-to-nearest root
/// `s` and where the true root lies relative to it: `Less` below `s`,
/// `Greater` above, `Equal` when `s` is exact.
///
/// With `x = mx * 2^ex` and `s = ms * 2^es`, both sides of `x <=> s * s` are
/// scaled by `2^(-2 * es)`: `mx << (ex - 2 * es)` against `ms^2`.  `s` is a
/// normal double, so `ms^2` lies in `[2^104, 2^106)`, and `s * s` is within
/// a factor `1 ± 2^-51` of `x`, so the shifted `mx` is below `2^107` and the
/// shift is between 51 (for `mx < 2^53`) and 107 (for `mx >= 1`).  Both
/// sides fit in a `u128`, and the comparison is exact.
#[inline]
pub(crate) fn nearest_root(bits: u64) -> (u64, Ordering) {
    debug_assert!(is_positive_finite(bits));
    let (mx, ex) = decompose(bits);
    let s = f64::from_bits(bits).sqrt().to_bits();
    let (ms, es) = decompose(s);
    let shift = ex - 2 * es;
    debug_assert!((51..=107).contains(&shift), "shift {shift}");
    let square = u128::from(ms) * u128::from(ms);
    let scaled_x = u128::from(mx) << shift;
    (s, scaled_x.cmp(&square))
}

/// Square roots of zeros, negatives, infinities and NaNs; `None` for
/// positive, nonzero, finite inputs.
///
/// `sqrt(±0) = ±0` and `sqrt(+inf) = +inf`, exactly.  A NaN gives the default
/// NaN, invalid only when signaling; a negative (including `-inf`) gives the
/// default NaN and is invalid.  These are the results of
/// `ieee_apsqrt::sqrt_accurate`, which the interpreter used before.
#[inline]
pub(crate) fn special(bits: u64) -> Option<Root> {
    if is_positive_finite(bits) {
        return None;
    }
    let magnitude = bits & !SIGN_BIT;
    Some(if magnitude == 0 || bits == POSITIVE_INFINITY {
        Root::exact(bits, ApStatus::OK)
    } else if magnitude > POSITIVE_INFINITY && bits & QUIET_BIT != 0 {
        Root::exact(DEFAULT_NAN, ApStatus::OK)
    } else {
        // Signaling NaNs and negatives.
        Root::exact(DEFAULT_NAN, ApStatus::INVALID_OP)
    })
}

/// `fsqrt`: the square root of `bits`, correctly rounded to double precision
/// under FPSCR rounding control `rn` (the low two FPSCR bits).
#[inline]
pub(crate) fn sqrt(bits: u64, rn: u32) -> Root {
    if let Some(root) = special(bits) {
        return root;
    }
    let (s, side) = nearest_root(bits);
    let (value, truncated) = match side {
        Ordering::Equal => return Root::exact(s, ApStatus::OK),
        // t is in (s - ulp, s): toward zero and toward -infinity step down.
        Ordering::Less => (if rn & 1 == 1 { s - 1 } else { s }, s - 1),
        // t is in (s, s + ulp): toward +infinity steps up.
        Ordering::Greater => (if rn & 3 == 2 { s + 1 } else { s }, s),
    };
    Root {
        value,
        truncated,
        status: ApStatus::INEXACT,
    }
}

/// The square root of `bits` rounded to odd at double precision: exact when
/// the root is a double, otherwise its neighbouring double with an odd
/// significand.  Rounding this value to single precision in any mode equals
/// rounding the true root once; see the module documentation.
#[inline]
pub(crate) fn sqrt_round_to_odd(bits: u64) -> Root {
    if let Some(root) = special(bits) {
        return root;
    }
    let (s, side) = nearest_root(bits);
    let truncated = match side {
        Ordering::Equal => return Root::exact(s, ApStatus::OK),
        Ordering::Less => s - 1,
        Ordering::Greater => s,
    };
    Root {
        value: truncated | 1,
        truncated,
        status: ApStatus::INEXACT,
    }
}

/// Deterministic input generators shared by the square-root tests here and in
/// [`crate::frsqrte`].
#[cfg(test)]
pub(crate) mod test_support {
    use super::{FRACTION_MASK, HIDDEN_BIT};

    /// SplitMix64: deterministic, dependency-free test input stream.
    pub(crate) struct Rng(pub(crate) u64);

    impl Rng {
        pub(crate) fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    /// Square roots of `c` modulo `2^n` for odd `c ≡ 1 (mod 8)`, by Hensel
    /// lifting.  Returns all four roots.
    fn sqrt_mod_pow2(c: u128, n: u32) -> [u128; 4] {
        let modulus = 1u128 << n;
        let mask = modulus - 1;
        let mut y: u128 = 1;
        for bits in 3..n {
            // y^2 ≡ c (mod 2^bits); lift to 2^(bits+1).
            let next = 1u128 << (bits + 1);
            if (y.wrapping_mul(y).wrapping_sub(c)) & (next - 1) != 0 {
                y += 1u128 << (bits - 1);
            }
        }
        let y = y & mask;
        let half = 1u128 << (n - 1);
        [
            y,
            modulus.wrapping_sub(y) & mask,
            (y + half) & mask,
            (modulus.wrapping_sub(y) + half) & mask,
        ]
    }

    fn compose(significand: u128, exponent: i32) -> Option<u64> {
        if !(HIDDEN_BIT as u128..(HIDDEN_BIT as u128) << 1).contains(&significand) {
            return None;
        }
        let biased = exponent + 1075;
        if !(1..=2046).contains(&biased) {
            return None;
        }
        Some(((biased as u64) << 52) | (significand as u64 & FRACTION_MASK))
    }

    /// Inputs whose square root lies a chosen residual `d` away from a double
    /// (or from a double-precision rounding midpoint), on both sides and
    /// across many binades: the cases where an imprecise square root would
    /// round differently.  `|d|` spans 1 to ~2^32 residual units.
    pub(crate) fn hard_cases(rng: &mut Rng, count: usize) -> Vec<u64> {
        let mut cases = Vec::new();
        while cases.len() < count {
            let magnitude = 1u128 << (rng.next() % 32);
            let c = magnitude + u128::from(rng.next()) % magnitude;
            let midpoint = rng.next() & 1 == 1;
            let n = 52 + (rng.next() % 2) as u32;
            let modulus_bits = if midpoint { n + 2 } else { n };
            let modulus = 1u128 << modulus_bits;
            // root^2 + d ≡ 0 (mod 2^modulus_bits) needs -d ≡ 1 (mod 8).
            let (d_positive, d) = if rng.next() & 1 == 1 {
                (true, (c & !7) | 7)
            } else {
                (false, (c & !7) | 1)
            };
            let target = if d_positive { modulus - d } else { d };
            for root in sqrt_mod_pow2(target, modulus_bits) {
                let base = if midpoint {
                    if root & 1 == 0 {
                        continue;
                    }
                    (root - 1) / 2
                } else {
                    root
                };
                for ms in [base, base + (1 << 52), base.wrapping_sub(1 << 52)] {
                    if !(HIDDEN_BIT as u128..(HIDDEN_BIT as u128) << 1).contains(&ms) {
                        continue;
                    }
                    let square = if midpoint {
                        (2 * ms + 1) * (2 * ms + 1)
                    } else {
                        ms * ms
                    };
                    let scaled = if d_positive { square + d } else { square - d };
                    if scaled % modulus != 0 {
                        continue;
                    }
                    let exponent = ((rng.next() % 1800) as i32 - 900) & !1;
                    if let Some(bits) = compose(scaled / modulus, exponent + n as i32) {
                        cases.push(bits);
                    }
                }
            }
        }
        cases
    }

    /// The square of a random double with at most 26 significant bits, so the
    /// square, and therefore its root, is exact: normal squares from the
    /// root's exponent range ±500, and subnormal squares `n^2 * 2^-1074`.
    pub(crate) fn exact_square(rng: &mut Rng) -> u64 {
        loop {
            let raw = rng.next();
            if raw.is_multiple_of(8) {
                // Subnormal: the bit pattern n^2 is n^2 * 2^-1074, with root
                // n * 2^-537.
                let n = (raw >> 40) % (1 << 26);
                if n != 0 {
                    return n * n;
                }
                continue;
            }
            let root = ((raw >> 38) | (1 << 25)) as f64 * 2f64.powi((raw % 1000) as i32 - 500);
            let square = root * root;
            if square != 0.0 && square.is_finite() {
                return square.to_bits();
            }
        }
    }

    /// Powers of four: the normal ones (2^-1022 ..= 2^1022) and the
    /// subnormal ones (2^-1074 ..= 2^-1024), with both neighbours of each.
    pub(crate) fn powers_of_four() -> Vec<u64> {
        let mut cases = Vec::new();
        for exponent in (-1074i32..=1022).step_by(2) {
            let power = if exponent < -1022 {
                1u64 << (exponent + 1074)
            } else {
                ((exponent + 1023) as u64) << 52
            };
            cases.extend([power - 1, power, power + 1]);
        }
        cases
    }

    /// Multiplies by `2^exponent` with a single final rounding.
    pub(crate) fn scale(mut value: f64, mut exponent: i32) -> f64 {
        while exponent > 600 {
            value *= 2f64.powi(600);
            exponent -= 600;
        }
        while exponent < -600 {
            value *= 2f64.powi(-600);
            exponent += 600;
        }
        value * 2f64.powi(exponent)
    }

    /// Inputs whose root lies near a single-precision rounding midpoint (or,
    /// with `d = 0`, exactly on one), across the whole exponent range of the
    /// root, including single-precision subnormal and overflowing roots.
    /// With `|d|` small the root's round-to-nearest double is the midpoint
    /// itself, which is where rounding to double and then to single
    /// rounds twice.
    pub(crate) fn single_midpoint_case(rng: &mut Rng) -> u64 {
        loop {
            let raw = rng.next();
            // A 25-bit odd significand: a single-precision midpoint.
            let midpoint = ((raw >> 39) | (1 << 24)) | 1;
            // Scale its square to 51 or 52 bits so that small offsets stay
            // within the 53-bit significand.
            let square = midpoint * midpoint; // 49 or 50 bits
            let square = if square < 1 << 49 {
                square << 4
            } else {
                square << 2
            };
            let offset = match rng.next() % 4 {
                0 => 0,
                1 => 1 + rng.next() % 4,
                2 => (1 + rng.next() % 4).wrapping_neg(),
                _ => (rng.next() % 2048).wrapping_sub(1024),
            };
            let n = square.wrapping_add(offset);
            if n == 0 || n >= 1 << 53 {
                continue;
            }
            // Root exponents from about 2^-560 to 2^520.
            let exponent = 2 * ((rng.next() % 1080) as i32 - 560) - 26;
            let x = scale(n as f64, exponent);
            if x != 0.0 && x.is_finite() {
                return x.to_bits();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::{FPSCR_VXSQRT, PpcCpu};
    use rustc_apfloat::Round as ApRound;
    use std::collections::BTreeMap;

    const MODES: [u32; 4] = [0, 1, 2, 3];

    // ---- Exact oracle -------------------------------------------------

    /// The true root of a positive finite `bits` as `(root + f) * 2^exponent`
    /// with `root` in `[2^54, 2^55)` and `f` in `[0, 1)`; the flag is
    /// `f != 0`.  Computed with an integer square root, independently of
    /// the native square root.
    fn exact_root(bits: u64) -> (u128, bool, i32) {
        let (mx, ex) = decompose(bits);
        // mx << shift in [2^108, 2^110) with ex - shift even.
        let mut shift = 108 - mx.ilog2() as i32;
        if (ex - shift) % 2 != 0 {
            shift += 1;
        }
        let n = u128::from(mx) << shift;
        let root = n.isqrt();
        assert!((1u128 << 54..1u128 << 55).contains(&root));
        (root, root * root != n, (ex - shift) / 2)
    }

    /// Rounds the true root of `bits` to `precision` significant bits, with
    /// quanta no finer than `2^min_exponent`, in FPSCR mode `rn`.  Returns
    /// the rounded value and the value rounded toward zero (both exact as
    /// `f64`), and whether rounding was inexact.
    fn oracle(bits: u64, rn: u32, precision: i32, min_exponent: i32) -> (f64, f64, bool) {
        let (root, sticky, exponent) = exact_root(bits);
        let quantum = (exponent + 54 - (precision - 1)).max(min_exponent);
        // Past 56 dropped bits everything is below half a quantum; cap the
        // shift there so it stays in range.
        let drop = ((quantum - exponent) as u32).min(56);
        assert!(drop >= 2);
        let kept = root >> drop;
        let rest = root & ((1 << drop) - 1);
        let half = 1u128 << (drop - 1);
        let inexact = rest != 0 || sticky;
        let up = match rn {
            0 => rest > half || (rest == half && (sticky || kept & 1 == 1)),
            2 => inexact,
            _ => false,
        };
        let value = |m: u128| scale(m as f64, quantum);
        (value(kept + u128::from(up)), value(kept), inexact)
    }

    fn oracle_double(bits: u64, rn: u32) -> Root {
        let (value, truncated, inexact) = oracle(bits, rn, 53, -1074);
        Root {
            value: value.to_bits(),
            truncated: truncated.to_bits(),
            status: if inexact {
                ApStatus::INEXACT
            } else {
                ApStatus::OK
            },
        }
    }

    /// What an `fsqrt` or `fsqrts` instruction should leave behind: the FPR
    /// value and the FPSCR exception and rounding bits it can change.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Expected {
        frt: u64,
        xx: bool,
        fi: bool,
        fr: bool,
        ox: bool,
        ux: bool,
    }

    impl Expected {
        fn observed(outcome: Outcome) -> Self {
            let bit = |mask: u32| outcome.fpscr & mask != 0;
            Self {
                frt: outcome.frt,
                xx: bit(FPSCR_XX_MASK),
                fi: bit(FPSCR_FI_MASK),
                fr: bit(FPSCR_FR_MASK),
                ox: bit(FPSCR_OX_MASK),
                ux: bit(FPSCR_UX_MASK),
            }
        }
    }

    fn oracle_instruction(bits: u64, rn: u32, single: bool) -> Expected {
        if !single {
            let root = oracle_double(bits, rn);
            let inexact = root.status == ApStatus::INEXACT;
            return Expected {
                frt: root.value,
                xx: inexact,
                fi: inexact,
                fr: root.value != root.truncated,
                ox: false,
                ux: false,
            };
        }
        let (value, truncated, inexact) = oracle(bits, rn, 24, -149);
        let max = f64::from(f32::MAX);
        let beyond = value > max;
        // Overflow is inexact even when the root is exact.
        let inexact = inexact || beyond;
        // Overflow to the largest finite value (toward zero or -infinity).
        let value = if beyond && rn & 1 == 1 {
            max
        } else {
            f64::from(value as f32)
        };
        Expected {
            frt: value.to_bits(),
            xx: inexact,
            fi: inexact,
            fr: value > truncated.min(max),
            // The finish path follows rustc_apfloat (and LLVM's APFloat),
            // which reports OVERFLOW only when the result becomes infinite.
            ox: value.is_infinite(),
            // Tiny after rounding, and inexact.
            ux: inexact && value < f64::from(f32::MIN_POSITIVE),
        }
    }

    // ---- Instruction-level harness -------------------------------------

    const FPSCR_FR_MASK: u32 = 1 << (31 - 13);
    const FPSCR_FI_MASK: u32 = 1 << (31 - 14);
    const FPSCR_XX_MASK: u32 = 1 << (31 - 6);
    const FPSCR_OX_MASK: u32 = 1 << (31 - 3);
    const FPSCR_UX_MASK: u32 = 1 << (31 - 4);

    /// `fsqrt[s][.] f1, f2`.
    fn instruction(single: bool, rc: bool) -> u32 {
        let opcd = if single { 59 } else { 63 };
        (opcd << 26) | (1 << 21) | (2 << 11) | (22 << 1) | u32::from(rc)
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Outcome {
        frt: u64,
        fpscr: u32,
        cr: u32,
    }

    fn prepare(cpu: &mut PpcCpu, bits: u64, fpscr: u32) {
        cpu.fpr[1] = 0x5555_5555_5555_5555;
        cpu.fpr[2] = bits;
        cpu.fpscr = fpscr;
        cpu.cr = 0;
    }

    fn outcome(cpu: &PpcCpu) -> Outcome {
        Outcome {
            frt: cpu.fpr[1],
            fpscr: cpu.fpscr,
            cr: cpu.cr,
        }
    }

    fn run(cpu: &mut PpcCpu, bits: u64, fpscr: u32, single: bool, rc: bool) -> Outcome {
        prepare(cpu, bits, fpscr);
        cpu.step_instruction(instruction(single, rc));
        outcome(cpu)
    }

    /// The `Fsqrt`/`Fsqrts` arms as they were before this module: the root
    /// from `ieee_apsqrt::sqrt_accurate` in the FPSCR mode, its truncation
    /// from the same function toward zero.
    fn run_legacy(cpu: &mut PpcCpu, bits: u64, fpscr: u32, single: bool, rc: bool) -> Outcome {
        prepare(cpu, bits, fpscr);
        let (sqrt, _) = ieee_apsqrt::sqrt_accurate(bits, cpu.fp_rounding_mode());
        let (truncated, _) = ieee_apsqrt::sqrt_accurate(bits, ApRound::TowardZero);
        let result = StatusAnd {
            status: sqrt.status,
            value: ApDouble::from_bits(u128::from(sqrt.value)),
        };
        let truncated = ApDouble::from_bits(u128::from(truncated.value));
        let invalid = PpcCpu::fp_invalid_flags(bits, None, FPSCR_VXSQRT);
        if single {
            cpu.finish_ap_single_result_rounded(1, result, truncated, invalid, rc);
        } else {
            cpu.finish_ap_double_result_rounded(1, result, truncated, invalid, rc);
        }
        outcome(cpu)
    }

    /// Checks `fsqrt` and `fsqrts` on `bits` in mode `rn` against the oracle,
    /// at the function level and through the instruction.
    fn check(cpu: &mut PpcCpu, bits: u64, rn: u32) {
        if !is_positive_finite(bits) {
            let legacy = ieee_apsqrt::sqrt_accurate(bits, crate::frsqrte::rounding_mode(rn)).0;
            for root in [sqrt(bits, rn), sqrt_round_to_odd(bits)] {
                assert_eq!(
                    (root.value, root.status),
                    (legacy.value, legacy.status),
                    "special {bits:#018x} rn={rn}"
                );
            }
            for single in [false, true] {
                assert_eq!(
                    run(cpu, bits, rn, single, true),
                    run_legacy(cpu, bits, rn, single, true),
                    "special {bits:#018x} rn={rn} single={single}"
                );
            }
            return;
        }

        let expected = oracle_double(bits, rn);
        assert_eq!(sqrt(bits, rn), expected, "fsqrt({bits:#018x}) rn={rn}");
        for single in [false, true] {
            let got = run(cpu, bits, rn, single, false);
            assert_eq!(
                Expected::observed(got),
                oracle_instruction(bits, rn, single),
                "{}({bits:#018x}) rn={rn}: {got:x?}",
                if single { "fsqrts" } else { "fsqrt" }
            );
        }
    }

    fn check_all_modes(cpu: &mut PpcCpu, bits: u64) {
        for rn in MODES {
            check(cpu, bits, rn);
        }
    }

    fn targeted_cases() -> Vec<u64> {
        let mut cases = vec![
            0,
            1,
            2,
            3,
            HIDDEN_BIT - 1,
            HIDDEN_BIT,
            HIDDEN_BIT + 1,
            0x7fef_ffff_ffff_ffff,
            POSITIVE_INFINITY,
            POSITIVE_INFINITY + 1,
            0x7ff4_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0x7fff_ffff_ffff_ffff,
            9f64.to_bits(),
            2.25f64.to_bits(),
            2f64.to_bits(),
        ];
        let negatives: Vec<u64> = cases.iter().map(|bits| bits | SIGN_BIT).collect();
        cases.extend(negatives);
        // Every binade, at the power of two and both neighbours.
        for exponent in -1074i32..=1023 {
            let power = if exponent < -1022 {
                1u64 << (exponent + 1074)
            } else {
                ((exponent + 1023) as u64) << 52
            };
            cases.extend([power - 1, power, power + 1]);
        }
        // Exact squares and neighbours.
        for root in (1u64..300).chain([4095, 4097, 1 << 26, (1 << 26) - 1, 94_906_265]) {
            let square = (root as f64) * (root as f64);
            for exponent in [0, -2, -600, 600, -1074 + 52, -1074] {
                let bits = scale(square, exponent).to_bits();
                if bits != 0 {
                    cases.extend([bits - 1, bits, bits + 1]);
                }
            }
        }
        // Roots at the edges of single precision: 2^-149, 2^-126, FLT_MAX.
        for root in [
            f64::from(f32::from_bits(1)),
            f64::from(f32::MIN_POSITIVE),
            f64::from(f32::MAX),
            2f64.powi(128),
        ] {
            let bits = (root * root).to_bits();
            cases.extend([bits - 1, bits, bits + 1]);
        }
        cases
    }

    #[test]
    fn issue_80_exact_squares_are_exact_toward_positive_infinity() {
        let mut cpu = PpcCpu::new();
        for (x, root) in [(9.0f64, 3.0f64), (2.25, 1.5)] {
            for single in [false, true] {
                let got = run(&mut cpu, x.to_bits(), 2, single, false);
                assert_eq!(got.frt, root.to_bits(), "sqrt({x}) single={single}");
                assert_eq!(
                    got.fpscr & (FPSCR_XX_MASK | FPSCR_FI_MASK | FPSCR_FR_MASK),
                    0
                );
            }
        }
        // frsqrte f1, f2 on 9.0: 1/3 rounded toward +infinity.
        prepare(&mut cpu, 9f64.to_bits(), 2);
        cpu.step_instruction((63 << 26) | (1 << 21) | (2 << 11) | (26 << 1));
        assert_eq!(cpu.fpr[1], 0x3fd5_5555_5555_5556);
    }

    #[test]
    fn targeted_inputs_match_oracle_in_every_mode() {
        let mut cpu = PpcCpu::new();
        let cases = targeted_cases();
        for &bits in &cases {
            check_all_modes(&mut cpu, bits);
        }
        for &bits in &powers_of_four() {
            check_all_modes(&mut cpu, bits);
        }
    }

    fn random_inputs(cpu: &mut PpcCpu, seed: u64, count: usize) {
        let mut rng = Rng(seed);
        for i in 0..count {
            let raw = rng.next();
            // Half the draws are forced positive.
            let bits = if i % 2 == 0 { raw & !SIGN_BIT } else { raw };
            check(cpu, bits, (i as u32 / 2) % 4);
        }
    }

    fn exact_squares(cpu: &mut PpcCpu, seed: u64, count: usize) {
        let mut rng = Rng(seed);
        for _ in 0..count {
            check_all_modes(cpu, exact_square(&mut rng));
        }
    }

    fn near_boundary(cpu: &mut PpcCpu, seed: u64, count: usize) {
        let mut rng = Rng(seed);
        for &bits in &hard_cases(&mut rng, count) {
            check_all_modes(cpu, bits);
        }
    }

    fn single_midpoints(cpu: &mut PpcCpu, seed: u64, count: usize) {
        let mut rng = Rng(seed);
        for _ in 0..count {
            check_all_modes(cpu, single_midpoint_case(&mut rng));
        }
    }

    #[test]
    fn random_inputs_match_oracle() {
        random_inputs(&mut PpcCpu::new(), 0x5eed_5001, 20_000);
    }

    #[test]
    fn exact_squares_match_oracle_in_every_mode() {
        exact_squares(&mut PpcCpu::new(), 0x5eed_5002, 3_000);
    }

    #[test]
    fn near_boundary_roots_match_oracle_in_every_mode() {
        near_boundary(&mut PpcCpu::new(), 0x5eed_5003, 2_000);
    }

    #[test]
    fn single_midpoint_roots_match_oracle_in_every_mode() {
        single_midpoints(&mut PpcCpu::new(), 0x5eed_5004, 3_000);
    }

    /// The large oracle run: `cargo test --release -- --ignored`.
    #[test]
    #[ignore = "slow; run with --release -- --ignored"]
    fn exhaustive_oracle() {
        let mut cpu = PpcCpu::new();
        random_inputs(&mut cpu, 0x5eed_6001, 8_000_000);
        exact_squares(&mut cpu, 0x5eed_6002, 500_000);
        near_boundary(&mut cpu, 0x5eed_6003, 400_000);
        single_midpoints(&mut cpu, 0x5eed_6004, 1_000_000);
    }

    // ---- Old-versus-new census --------------------------------------

    /// Counts of inputs where the new arms differ from the old ones.
    #[derive(Default)]
    struct Census {
        checked: u64,
        /// Exact double root toward +infinity (issue #80), per instruction.
        exact_toward_positive: [u64; 2],
        /// Other `fsqrts` cases where the old result disagreed with the
        /// oracle: the value (by rounding mode), or only flags (by bit).
        single_value: [u64; 4],
        single_flags: BTreeMap<&'static str, u64>,
        /// Differences the categories above do not explain.
        other: Vec<String>,
        examples: Vec<String>,
    }

    impl Census {
        fn record(&mut self, cpu: &mut PpcCpu, bits: u64, fpscr: u32) {
            let rn = fpscr & 3;
            for single in [false, true] {
                self.checked += 1;
                let new = run(cpu, bits, fpscr, single, true);
                let old = run_legacy(cpu, bits, fpscr, single, true);
                if new == old {
                    continue;
                }
                let name = if single { "fsqrts" } else { "fsqrt" };
                let line =
                    format!("{name}({bits:#018x}) fpscr={fpscr:#x}: new {new:x?}, old {old:x?}");
                if !is_positive_finite(bits) {
                    self.other.push(line);
                    continue;
                }
                let exact = nearest_root(bits).1 == Ordering::Equal;
                let expected = oracle_instruction(bits, rn, single);
                let old_flags = Expected::observed(old);
                if exact && rn == 2 {
                    self.exact_toward_positive[usize::from(single)] += 1;
                } else if single && old_flags.frt != expected.frt {
                    self.single_value[rn as usize] += 1;
                    if self.examples.len() < 6 {
                        self.examples.push(line);
                    }
                } else if single && old_flags != expected {
                    for (flag, old_bit, bit) in [
                        ("XX", old_flags.xx, expected.xx),
                        ("FI", old_flags.fi, expected.fi),
                        ("FR", old_flags.fr, expected.fr),
                        ("OX", old_flags.ox, expected.ox),
                        ("UX", old_flags.ux, expected.ux),
                    ] {
                        if old_bit != bit {
                            *self.single_flags.entry(flag).or_default() += 1;
                        }
                    }
                    if self.examples.len() < 12 {
                        self.examples.push(line);
                    }
                } else if self.other.len() < 20 {
                    self.other.push(line);
                }
            }
        }

        /// Sweeps `count` generated inputs, plus the targeted ones if asked.
        fn sweep(&mut self, seed: u64, count: usize, targeted: bool) {
            let mut cpu = PpcCpu::new();
            let mut rng = Rng(seed);
            let mut inputs = Vec::new();
            if targeted {
                inputs.extend(targeted_cases());
                inputs.extend(powers_of_four());
            }
            for i in 0..count {
                inputs.push(match i % 5 {
                    0 => rng.next(),
                    1 => rng.next() & !SIGN_BIT,
                    2 => exact_square(&mut rng),
                    3 => single_midpoint_case(&mut rng),
                    _ => hard_cases(&mut rng, 1)[0],
                });
            }
            for bits in inputs {
                for rn in MODES {
                    // Vary the exception enables (VE, OE, UE, ZE, XE) too.
                    let enables = (rng.next() as u32 & 0x1f) << 3;
                    self.record(&mut cpu, bits, rn | enables);
                }
            }
        }
    }

    /// The new arms differ from the old ones only where the old ones
    /// disagreed with the oracle: exact roots toward +infinity (issue #80)
    /// and `fsqrts` double rounding.
    #[test]
    fn differs_from_legacy_only_where_legacy_was_wrong() {
        let mut census = Census::default();
        census.sweep(0x5eed_7001, 1_000, false);
        assert!(census.other.is_empty(), "{:#?}", census.other);
        assert!(census.exact_toward_positive[0] > 0 && census.exact_toward_positive[1] > 0);
        assert!(census.single_value[0] > 0);
    }

    /// The census at volume: `--release -- --ignored --nocapture`.
    #[test]
    #[ignore = "slow; run with --release -- --ignored --nocapture"]
    fn legacy_census() {
        let mut census = Census::default();
        census.sweep(0x5eed_8001, 2_000_000, true);
        println!(
            "checked {} instruction/mode pairs: exact roots toward +inf fsqrt {} fsqrts {}; \
             other fsqrts values wrong (rn 0..3) {:?}; fsqrts flags only {:?}; other {}",
            census.checked,
            census.exact_toward_positive[0],
            census.exact_toward_positive[1],
            census.single_value,
            census.single_flags,
            census.other.len()
        );
        for example in &census.examples {
            println!("  {example}");
        }
        assert!(census.other.is_empty(), "{:#?}", census.other);
    }

    /// Throughput of the old and new `fsqrt` computation (the old arm ran
    /// `sqrt_accurate` twice: once in the mode, once toward zero):
    /// `--release -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement; run with --release -- --ignored --nocapture"]
    fn throughput() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = Rng(0x5eed_9001);
        // Positive normal doubles from 2^-20 to 2^20.
        let inputs: Vec<u64> = (0..4096)
            .map(|_| (rng.next() & FRACTION_MASK) | ((1003 + rng.next() % 40) << 52))
            .collect();
        for rn in MODES {
            let mode = crate::frsqrte::rounding_mode(rn);
            let rounds = 20;
            let start = Instant::now();
            for _ in 0..rounds {
                for &bits in &inputs {
                    let _ = black_box(ieee_apsqrt::sqrt_accurate(black_box(bits), mode));
                    let _ = black_box(ieee_apsqrt::sqrt_accurate(
                        black_box(bits),
                        ApRound::TowardZero,
                    ));
                }
            }
            let old = start.elapsed().as_secs_f64() * 1e9 / (rounds * inputs.len()) as f64;
            let rounds = 2_000;
            let start = Instant::now();
            for _ in 0..rounds {
                for &bits in &inputs {
                    let _ = black_box(sqrt(black_box(bits), rn));
                }
            }
            let new = start.elapsed().as_secs_f64() * 1e9 / (rounds * inputs.len()) as f64;
            let start = Instant::now();
            for _ in 0..rounds {
                for &bits in &inputs {
                    let _ = black_box(sqrt_round_to_odd(black_box(bits)));
                }
            }
            let odd = start.elapsed().as_secs_f64() * 1e9 / (rounds * inputs.len()) as f64;
            println!(
                "rn={rn}: old {old:.0} ns/op, new {new:.2} ns/op ({:.0}x), round-to-odd {odd:.2} ns/op",
                old / new
            );
        }
    }
}
