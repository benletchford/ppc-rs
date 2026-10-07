//! `frsqrte` evaluation with a native fast path.
//!
//! The interpreter defines `frsqrte` as `round(1 / round(sqrt(x)))`, both
//! roundings in the FPSCR rounding mode, computed by [`reference()`]: a
//! widened-precision Newton square root (`ieee_apsqrt::sqrt_accurate`)
//! followed by an arbitrary-precision division.  That is exact enough, but
//! slow.
//!
//! [`fast()`] reproduces the same value with native `f64` arithmetic and proves
//! it with exact integer residuals before accepting it:
//!
//! 1. `s = sqrt(x)` in round-to-nearest.  The residual
//!    `r = x - s * s` (an exact 128-bit integer once both sides are scaled to
//!    the last bit of `s * s`) locates the true root `t` relative to `s`: with
//!    `ms` the 53-bit significand of `s`, `(t - s) / ulp(s) = r / (2 * ms + θ)`
//!    for some `|θ| < 1/2`.  The candidate is accepted only when
//!    `MARGIN <= |r| <= ms - MARGIN`, which proves that `s` is the
//!    round-to-nearest root and that `t` is about `2^-30` ulp or more away
//!    from `s` and from both rounding midpoints.  The reference square root
//!    is a Newton iteration in a format with 56 extra significand bits that
//!    stops within a few of its own ulps (`2^-56` of a double ulp each) of
//!    `t`; that bound is observed rather than proven.  With a margin of about
//!    `2^26` of those ulps it narrows to the same double and reports
//!    `INEXACT`; the differential tests below check this, including
//!    on generated roots just either side of the margin.  Directed modes step
//!    `s` one ulp toward the mode using the sign of `r`.
//! 2. `q = 1 / s` in round-to-nearest.  The residual `1 - q * s` is again an
//!    exact integer; it verifies that `q` is the round-to-nearest quotient
//!    and, by its sign, picks the directed-mode neighbour.  The reference
//!    division is correctly rounded, so the values agree.
//!
//! An exact square (`r == 0`) is accepted with an exact root: near an exact
//! root, the reference's Newton step `(e + x/e) / 2` returns the root itself
//! when rounding to nearest, toward zero or toward -infinity, and the check
//! `e * e == x` then reports it exact.  Toward +infinity the step stalls one
//! ulp above the root and the reference reports an inexact, one-ulp-high
//! root, so exact squares in that mode take the reference.
//!
//! Anything else the proof does not cover returns `None` and takes
//! [`reference()`]: zeros, negatives, infinities, NaNs, inputs within the
//! margin, and inexact roots whose significand is a power of two (the
//! residual bound assumes equal ulps on both sides of `s`, and below a power
//! of two the midpoint sits at a quarter ulp).  Subnormal inputs with fewer
//! than about 31 significant bits also fall back, because aligning `x` with
//! `s * s` would need a shift of more than 74 bits.
//!
//! For accepted inputs the status is `INEXACT` or `OK` and nothing else: `x`
//! is positive and finite, so `1 / sqrt(x)` lies in `[2^-512, 2^537]` and can
//! neither overflow nor underflow.

use rustc_apfloat::{
    Float, Round as ApRound, Status as ApStatus, StatusAnd, ieee::Double as ApDouble,
};

const HIDDEN_BIT: u64 = 1 << 52;
const FRACTION_MASK: u64 = HIDDEN_BIT - 1;
const POSITIVE_INFINITY: u64 = 0x7ff0_0000_0000_0000;

/// Distance, in units of the square-root residual, that a root must keep from
/// every rounding boundary for [`fast()`] to accept it (about `2^-30` ulp).
const MARGIN: u128 = 1 << 24;

/// Evaluates `frsqrte` for `bits` under FPSCR rounding control `rn` (the low
/// two FPSCR bits), returning the same value and status as [`reference()`].
#[inline]
pub(crate) fn evaluate(bits: u64, rn: u32) -> StatusAnd<ApDouble> {
    match fast(bits, rn) {
        Some((value, status)) => StatusAnd {
            status,
            value: ApDouble::from_bits(u128::from(value)),
        },
        None => reference(bits, rounding_mode(rn)),
    }
}

/// The slow, arbitrary-precision definition of `frsqrte`.
pub(crate) fn reference(bits: u64, round: ApRound) -> StatusAnd<ApDouble> {
    let (sqrt, _) = ieee_apsqrt::sqrt_accurate(bits, round);
    let one = ApDouble::from_bits(u128::from(1.0f64.to_bits()));
    let mut result = one.div_r(ApDouble::from_bits(u128::from(sqrt.value)), round);
    result.status |= sqrt.status;
    result
}

fn rounding_mode(rn: u32) -> ApRound {
    match rn & 0x3 {
        0 => ApRound::NearestTiesToEven,
        1 => ApRound::TowardZero,
        2 => ApRound::TowardPositive,
        _ => ApRound::TowardNegative,
    }
}

/// Splits a positive finite double into `(significand, exponent)` with
/// `value = significand * 2^exponent`.
#[inline]
fn decompose(bits: u64) -> (u64, i32) {
    let biased = (bits >> 52) as i32;
    if biased == 0 {
        (bits & FRACTION_MASK, -1074)
    } else {
        ((bits & FRACTION_MASK) | HIDDEN_BIT, biased - 1075)
    }
}

/// Returns the `frsqrte` result bits and status for `bits` when the exact
/// residual checks prove they match [`reference()`].
#[inline]
pub(crate) fn fast(bits: u64, rn: u32) -> Option<(u64, ApStatus)> {
    // Positive, nonzero, finite inputs only.
    if bits.wrapping_sub(1) >= POSITIVE_INFINITY - 1 {
        return None;
    }
    let toward_positive = rn & 0x3 == 2;
    let toward_negative = rn & 0x3 == 1 || rn & 0x3 == 3;

    // Step 1: s = round(sqrt(x)).
    let (mx, ex) = decompose(bits);
    let s_nearest = f64::from_bits(bits).sqrt().to_bits();
    let (ms, es) = decompose(s_nearest);
    // x = mx * 2^ex and s*s = ms^2 * 2^(2*es); align both to 2^(2*es).
    let shift = ex - 2 * es;
    if !(0..=74).contains(&shift) {
        return None;
    }
    let square = u128::from(ms) * u128::from(ms);
    let scaled_x = u128::from(mx) << shift;
    let (s, mut status) = if scaled_x == square {
        // Exact root.  The reference's Newton iteration lands on it exactly
        // in every mode except toward +infinity, where it stalls one ulp
        // high and reports INEXACT; leave that mode to the reference.
        if toward_positive {
            return None;
        }
        (s_nearest, ApStatus::OK)
    } else {
        // A power-of-two root has a narrower ulp below it; leave it to the
        // reference rather than special-case the asymmetric bounds.
        if ms == HIDDEN_BIT {
            return None;
        }
        let below = scaled_x < square; // true root is below s
        let r = scaled_x.abs_diff(square);
        if r < MARGIN || r > u128::from(ms) - MARGIN {
            return None;
        }
        let s = if below && toward_negative {
            s_nearest - 1
        } else if !below && toward_positive {
            s_nearest + 1
        } else {
            s_nearest
        };
        (s, ApStatus::INEXACT)
    };

    // Step 2: q = round(1 / s).
    let q_nearest = (1.0 / f64::from_bits(s)).to_bits();
    let (mq, eq) = decompose(q_nearest);
    let (ms, es) = decompose(s);
    // q*s = mq*ms * 2^(eq+es); compare against 1 = 2^k * 2^(eq+es).
    let k = -(eq + es);
    if !(0..=126).contains(&k) {
        return None;
    }
    let one = 1u128 << k;
    let product = u128::from(mq) * u128::from(ms);
    let above = product > one; // q is above the true quotient
    let rho = product.abs_diff(one);
    // Prove q is the round-to-nearest quotient: |1/s - q| < ulp(q)/2, i.e.
    // 2|rho| < ms, or 4|rho| < ms below a power-of-two q.
    let limit = if above && mq == HIDDEN_BIT {
        u128::from(ms) / 4
    } else {
        u128::from(ms) / 2
    };
    if rho >= limit {
        return None;
    }
    if rho == 0 {
        return Some((q_nearest, status));
    }
    status |= ApStatus::INEXACT;
    let q = if above && toward_negative {
        q_nearest - 1
    } else if !above && toward_positive {
        q_nearest + 1
    } else {
        q_nearest
    };
    Some((q, status))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64: deterministic, dependency-free test input stream.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    const MODES: [u32; 4] = [0, 1, 2, 3];

    /// Asserts that the dispatching evaluation matches the reference bit for
    /// bit, value and status; returns whether the fast path accepted it.
    fn check(bits: u64, rn: u32) -> bool {
        let expected = reference(bits, rounding_mode(rn));
        let accepted = fast(bits, rn).is_some();
        let actual = evaluate(bits, rn);
        assert!(
            actual.value.to_bits() == expected.value.to_bits() && actual.status == expected.status,
            "frsqrte({bits:#018x}) rn={rn}: fast {:#018x} {:?}, reference {:#018x} {:?}",
            actual.value.to_bits() as u64,
            actual.status,
            expected.value.to_bits() as u64,
            expected.status,
        );
        accepted
    }

    fn check_all_modes(bits: u64) {
        for rn in MODES {
            check(bits, rn);
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
    /// (or from a rounding midpoint), on both sides and across many binades:
    /// the cases where an imprecise square root would round differently.
    /// `|d|` spans 1 to ~2^32 residual units, straddling `MARGIN`.
    fn hard_cases(rng: &mut Rng, count: usize) -> Vec<u64> {
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
        ];
        let negatives: Vec<u64> = cases.iter().map(|bits| bits | (1 << 63)).collect();
        cases.extend(negatives);
        // Every binade at the subnormal and overflow ends, a stride between.
        let exponents = (-1074i32..=-1018)
            .chain((-1017..1000).step_by(5))
            .chain(1000..=1023);
        for exponent in exponents {
            // Built from bits: `powi` flushes 2^-1024 and below to zero.
            let power = if exponent < -1022 {
                1u64 << (exponent + 1074)
            } else {
                ((exponent + 1023) as u64) << 52
            };
            cases.extend([power.wrapping_sub(1), power, power + 1]);
        }
        // Exact squares and neighbours.
        for root in (1u64..300).chain([4095, 4097, 1 << 26, (1 << 26) + 1, 94_906_265]) {
            let square = (root as f64) * (root as f64);
            for scale in [1.0, 0.25, 2f64.powi(-600), 2f64.powi(600)] {
                let bits = (square * scale).to_bits();
                cases.extend([bits - 1, bits, bits + 1]);
            }
        }
        for subnormal in [1u64, 2, 3, 4, 5, 9, 1 << 20, (1 << 51) + 3, FRACTION_MASK] {
            cases.push(subnormal);
        }
        cases
    }

    #[test]
    fn targeted_inputs_match_reference_in_every_mode() {
        let cases = targeted_cases();
        for &bits in &cases {
            check_all_modes(bits);
        }
        // Every subnormal binade contributes its power of two and neighbours.
        let subnormals = cases
            .iter()
            .filter(|&&bits| bits != 0 && bits < HIDDEN_BIT)
            .count();
        assert!(subnormals >= 150, "only {subnormals} subnormal inputs");
    }

    #[test]
    fn near_boundary_roots_match_reference_in_every_mode() {
        let mut rng = Rng(0x5eed_0001);
        let cases = hard_cases(&mut rng, 1_000);
        let mut accepted = 0;
        for &bits in &cases {
            for rn in MODES {
                accepted += usize::from(check(bits, rn));
            }
        }
        // The generator must exercise both sides of the margin.
        assert!(accepted > 0 && accepted < cases.len() * MODES.len());
    }

    fn random_inputs(seed: u64, count: usize) {
        let mut rng = Rng(seed);
        for i in 0..count {
            let raw = rng.next();
            // Half the draws are forced positive so most reach the fast path.
            let bits = if i % 2 == 0 { raw & !(1 << 63) } else { raw };
            check(bits, (i as u32 / 2) % 4);
        }
    }

    /// The squared length a guest feeds `frsqrte` when normalising a vector:
    /// a vector of length 0.01..1e4 in double precision, or (for `unit`) an
    /// already-normalised single-precision vector whose squared length is
    /// 1.0 or within a few single-precision ulps of it.
    fn vector_square(rng: &mut Rng, unit_length: bool) -> u64 {
        let mut unit = || (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        let (a, b, c) = (unit() - 0.5, unit() - 0.5, unit() - 0.5);
        let length = 10f64.powf(-2.0 + 6.0 * unit());
        let norm = (a * a + b * b + c * c).sqrt().max(f64::MIN_POSITIVE);
        if unit_length {
            let (x, y, z) = ((a / norm) as f32, (b / norm) as f32, (c / norm) as f32);
            f64::from(x.mul_add(x, y.mul_add(y, z * z))).to_bits()
        } else {
            let (x, y, z) = (a / norm * length, b / norm * length, c / norm * length);
            (x * x + y * y + z * z).to_bits()
        }
    }

    fn realistic_inputs(seed: u64, count: usize) {
        let mut rng = Rng(seed);
        for i in 0..count {
            check(vector_square(&mut rng, i % 8 < 4), (i % 4) as u32);
        }
    }

    /// Squares of random doubles with at most 26 significant bits, so the
    /// square, and therefore the root, is exact.
    fn exact_squares(seed: u64, count: usize) {
        let mut rng = Rng(seed);
        let mut tested = 0;
        while tested < count {
            let raw = rng.next();
            let root = ((raw >> 38) | (1 << 25)) as f64 * 2f64.powi((raw % 1000) as i32 - 500);
            let square = root * root;
            if square == 0.0 || !square.is_finite() {
                continue;
            }
            check_all_modes(square.to_bits());
            tested += 1;
        }
    }

    #[test]
    fn exact_squares_match_reference_in_every_mode() {
        exact_squares(0x5eed_0004, 1_500);
    }

    #[test]
    fn random_inputs_match_reference_in_every_mode() {
        random_inputs(0x5eed_0002, 6_000);
    }

    #[test]
    fn realistic_vector_lengths_match_reference_in_every_mode() {
        realistic_inputs(0x5eed_0003, 3_000);
    }

    /// The large differential run: `cargo test --release -- --ignored`.
    #[test]
    #[ignore = "slow; run with --release -- --ignored"]
    fn exhaustive_differential() {
        random_inputs(0x5eed_1000, 8_000_000);
        realistic_inputs(0x5eed_1001, 2_000_000);
        exact_squares(0x5eed_1003, 200_000);
        let mut rng = Rng(0x5eed_1002);
        for &bits in &hard_cases(&mut rng, 400_000) {
            check_all_modes(bits);
        }
    }

    /// Reports how often the fast path declines: run with `--nocapture`.
    #[test]
    #[ignore = "measurement; run with --release -- --ignored --nocapture"]
    fn fallback_rates() {
        let mut rng = Rng(0x5eed_2000);
        let total = 10_000_000u64;
        for rn in MODES {
            let mut random_declined = 0u64;
            let mut lengths_declined = 0u64;
            let mut units_declined = 0u64;
            for _ in 0..total {
                let bits = rng.next() & !(1 << 63);
                if bits < POSITIVE_INFINITY && fast(bits, rn).is_none() {
                    random_declined += 1;
                }
                if fast(vector_square(&mut rng, false), rn).is_none() {
                    lengths_declined += 1;
                }
                if fast(vector_square(&mut rng, true), rn).is_none() {
                    units_declined += 1;
                }
            }
            println!(
                "rn={rn}: declined random positive finite {random_declined}/{total}, \
                 lengths 0.01..1e4 {lengths_declined}/{total}, \
                 unit single-precision vectors {units_declined}/{total}"
            );
        }
    }

    /// Throughput of the reference and the dispatching evaluation on
    /// realistic inputs: run with `--release -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement; run with --release -- --ignored --nocapture"]
    fn throughput() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = Rng(0x5eed_3000);
        let unit_length = std::env::var_os("FRSQRTE_BENCH_UNIT").is_some();
        let inputs: Vec<u64> = (0..4096)
            .map(|_| vector_square(&mut rng, unit_length))
            .collect();
        let rounds = 50;
        let start = Instant::now();
        for _ in 0..rounds {
            for &bits in &inputs {
                let _ = black_box(reference(black_box(bits), ApRound::NearestTiesToEven));
            }
        }
        let slow = start.elapsed().as_secs_f64();
        let rounds_fast = rounds * 200;
        let start = Instant::now();
        for _ in 0..rounds_fast {
            for &bits in &inputs {
                let _ = black_box(evaluate(black_box(bits), 0));
            }
        }
        let quick = start.elapsed().as_secs_f64();
        let per_slow = slow * 1e9 / (rounds * inputs.len()) as f64;
        let per_fast = quick * 1e9 / (rounds_fast * inputs.len()) as f64;
        println!(
            "reference {per_slow:.1} ns/op, evaluate {per_fast:.2} ns/op, {:.0}x",
            per_slow / per_fast
        );
    }
}
