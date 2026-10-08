//! `frsqrte` evaluation with a native fast path.
//!
//! The interpreter defines `frsqrte` as `round(1 / sqrt(x))`, where the square
//! root is correctly rounded ([`crate::sqrt::sqrt()`], the `fsqrt` result) and
//! the division is rounded again, both in the FPSCR rounding mode.
//! [`reference()`] computes the division with arbitrary-precision arithmetic.
//!
//! [`fast()`] reproduces the same value with native `f64` division and proves
//! it with an exact integer residual before accepting it: `q = 1 / s` in
//! round-to-nearest, and the residual `1 - q * s`, an exact integer once both
//! sides are scaled to the last bit of `q * s`, verifies that `q` is the
//! round-to-nearest quotient (including the narrower ulp below a
//! power-of-two `q`) and, by its sign, picks the directed-mode neighbour.  A
//! zero residual means the division is exact.  Anything the proof does not
//! cover returns `None` and takes [`reference()`]: zeros, negatives,
//! infinities and NaNs.
//!
//! For accepted inputs the status is `INEXACT` or `OK` and nothing else: `x`
//! is positive and finite, so `1 / sqrt(x)` lies in `[2^-512, 2^537]` and can
//! neither overflow nor underflow.

use rustc_apfloat::{
    Float, Round as ApRound, Status as ApStatus, StatusAnd, ieee::Double as ApDouble,
};

use crate::sqrt::{self, decompose, is_positive_finite};

/// Evaluates `frsqrte` for `bits` under FPSCR rounding control `rn` (the low
/// two FPSCR bits), returning the same value and status as [`reference()`].
#[inline]
pub(crate) fn evaluate(bits: u64, rn: u32) -> StatusAnd<ApDouble> {
    match fast(bits, rn) {
        Some((value, status)) => StatusAnd {
            status,
            value: ApDouble::from_bits(u128::from(value)),
        },
        None => reference(bits, rn),
    }
}

/// The slow, arbitrary-precision definition of `frsqrte`.
pub(crate) fn reference(bits: u64, rn: u32) -> StatusAnd<ApDouble> {
    let root = sqrt::sqrt(bits, rn);
    let one = ApDouble::from_bits(u128::from(1.0f64.to_bits()));
    let mut result = one.div_r(
        ApDouble::from_bits(u128::from(root.value)),
        rounding_mode(rn),
    );
    result.status |= root.status;
    result
}

pub(crate) fn rounding_mode(rn: u32) -> ApRound {
    match rn & 0x3 {
        0 => ApRound::NearestTiesToEven,
        1 => ApRound::TowardZero,
        2 => ApRound::TowardPositive,
        _ => ApRound::TowardNegative,
    }
}

/// Returns the `frsqrte` result bits and status for `bits` when the exact
/// residual check proves they match [`reference()`].
#[inline]
pub(crate) fn fast(bits: u64, rn: u32) -> Option<(u64, ApStatus)> {
    // Positive, nonzero, finite inputs only.
    if !is_positive_finite(bits) {
        return None;
    }
    let toward_positive = rn & 0x3 == 2;
    let toward_negative = rn & 0x3 == 1 || rn & 0x3 == 3;

    // Step 1: s = sqrt(x), correctly rounded in the mode.
    let root = sqrt::sqrt(bits, rn);
    let (s, mut status) = (root.value, root.status);

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
    // 2|rho| < ms, or 4|rho| < ms below a power-of-two q.  (Compared
    // exactly: `rho < ms / 2` would wrongly decline `rho = (ms - 1) / 2`.)
    let scale = if above && mq == sqrt::HIDDEN_BIT {
        4
    } else {
        2
    };
    if rho * scale >= u128::from(ms) {
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
    use crate::sqrt::test_support::{Rng, exact_square, hard_cases, powers_of_four};
    use crate::sqrt::{FRACTION_MASK, HIDDEN_BIT, POSITIVE_INFINITY};

    /// `frsqrte` as it was defined before the square root was correctly
    /// rounded: the root from `ieee_apsqrt::sqrt_accurate`, which is one ulp
    /// high and inexact for exact squares toward +infinity (issue #80).
    fn legacy_reference(bits: u64, round: ApRound) -> StatusAnd<ApDouble> {
        let (sqrt, _) = ieee_apsqrt::sqrt_accurate(bits, round);
        let one = ApDouble::from_bits(u128::from(1.0f64.to_bits()));
        let mut result = one.div_r(ApDouble::from_bits(u128::from(sqrt.value)), round);
        result.status |= sqrt.status;
        result
    }

    const MODES: [u32; 4] = [0, 1, 2, 3];

    /// Asserts that the dispatching evaluation matches the reference bit for
    /// bit, value and status; returns whether the fast path accepted it.
    fn check(bits: u64, rn: u32) -> bool {
        let expected = reference(bits, rn);
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
        for &bits in cases.iter().chain(&powers_of_four()) {
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
        // The square root is exact, so only the reciprocal's proof could
        // decline, and native division is correctly rounded.
        assert_eq!(accepted, cases.len() * MODES.len());
    }

    /// Quotients exactly at the floor of the round-to-nearest bound.  For
    /// `s = 0x1ffffff8000001 * 2^-52` (`ms` divides `2^106 + 1`), `q =
    /// round(1 / s)` has residual `rho = (ms - 1) / 2`: inside the bound
    /// `2 * rho < ms`, but not below the floor of `ms / 2`.  The inputs are
    /// squares near `s * s` (scaled by even powers of two) whose rounded root
    /// is `s` in the listed modes; the fast path must accept them.
    #[test]
    fn quotients_at_the_half_ulp_floor_take_the_fast_path() {
        let ms = 0x1f_ffff_f800_0001u64;
        let s = f64::from_bits((1023 << 52) | (ms & FRACTION_MASK));
        let (mq, eq) = decompose((1.0 / s).to_bits());
        let product = u128::from(mq) * u128::from(ms);
        let rho = product.abs_diff(1 << -(eq - 52));
        assert_eq!(2 * rho, u128::from(ms) - 1);
        let cases: [(u64, &[u32]); 3] = [
            (0x400f_ffff_f000_0003, &[0, 2]),
            (0x400f_ffff_f000_0004, &[0, 1, 3]),
            (0x400f_ffff_f000_0005, &[1, 3]),
        ];
        for (base, modes) in cases {
            for binades in [-800i64, -200, -2, 0, 2, 200, 800] {
                let bits = base.wrapping_add_signed(binades << 52);
                for &rn in modes {
                    assert!(check(bits, rn), "frsqrte({bits:#018x}) rn={rn} declined");
                }
            }
        }
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

    /// Exact squares, normal and subnormal, so the root is exact.
    fn exact_squares(seed: u64, count: usize) {
        let mut rng = Rng(seed);
        for _ in 0..count {
            check_all_modes(exact_square(&mut rng));
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

    /// Compares [`reference()`] with [`legacy_reference()`] on `count` inputs
    /// of each kind in every mode.  They must agree except on exact squares
    /// toward +infinity, where the legacy square root was one ulp high
    /// (issue #80); returns how many of those differed and how many inputs
    /// were checked.
    fn legacy_census(seed: u64, count: usize) -> (usize, usize) {
        let mut rng = Rng(seed);
        let mut inputs = targeted_cases();
        inputs.extend(powers_of_four());
        inputs.extend(hard_cases(&mut rng, count));
        for i in 0..count {
            inputs.push(rng.next());
            inputs.push(exact_square(&mut rng));
            inputs.push(vector_square(&mut rng, i % 2 == 0));
        }
        let mut differed = 0;
        for &bits in &inputs {
            for rn in MODES {
                let new = reference(bits, rn);
                let old = legacy_reference(bits, rounding_mode(rn));
                if new.value.to_bits() == old.value.to_bits() && new.status == old.status {
                    continue;
                }
                let exact = is_positive_finite(bits)
                    && sqrt::nearest_root(bits).1 == std::cmp::Ordering::Equal;
                assert!(
                    exact && rn == 2,
                    "frsqrte({bits:#018x}) rn={rn}: {:#018x} {:?}, legacy {:#018x} {:?}",
                    new.value.to_bits() as u64,
                    new.status,
                    old.value.to_bits() as u64,
                    old.status,
                );
                differed += 1;
            }
        }
        (differed, inputs.len() * MODES.len())
    }

    #[test]
    fn differs_from_legacy_only_on_exact_squares_toward_positive_infinity() {
        let (differed, _) = legacy_census(0x5eed_0005, 1_000);
        assert!(differed > 0);
    }

    /// The large differential run: `cargo test --release -- --ignored`.
    #[test]
    #[ignore = "slow; run with --release -- --ignored --nocapture"]
    fn exhaustive_differential() {
        random_inputs(0x5eed_1000, 8_000_000);
        realistic_inputs(0x5eed_1001, 2_000_000);
        exact_squares(0x5eed_1003, 200_000);
        let mut rng = Rng(0x5eed_1002);
        for &bits in &hard_cases(&mut rng, 400_000) {
            check_all_modes(bits);
        }
        let (differed, checked) = legacy_census(0x5eed_1004, 1_000_000);
        println!("legacy census: {differed} of {checked} input/mode pairs differ");
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
                let _ = black_box(legacy_reference(
                    black_box(bits),
                    ApRound::NearestTiesToEven,
                ));
            }
        }
        let legacy = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for _ in 0..rounds {
            for &bits in &inputs {
                let _ = black_box(reference(black_box(bits), 0));
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
        let per_legacy = legacy * 1e9 / (rounds * inputs.len()) as f64;
        let per_slow = slow * 1e9 / (rounds * inputs.len()) as f64;
        let per_fast = quick * 1e9 / (rounds_fast * inputs.len()) as f64;
        println!(
            "legacy reference {per_legacy:.1} ns/op, reference {per_slow:.1} ns/op, \
             evaluate {per_fast:.2} ns/op, {:.0}x over legacy",
            per_legacy / per_fast
        );
    }
}
