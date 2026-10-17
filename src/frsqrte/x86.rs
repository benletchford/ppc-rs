//! Exact round-to-nearest evaluation using baseline x86-64 SSE2.
//!
//! SQRTSD followed by DIVSD implements the interpreter's two rounded
//! operations. This is not the approximate reciprocal-square-root instruction.
//! Intel SDM vol. 1 sections 10.2.3 and E.4 specify MXCSR and IEEE-754 behavior.

use rustc_apfloat::Status;

#[inline]
fn control_is_supported(control: u32) -> bool {
    // All exceptions masked, nearest-even, gradual underflow, no DAZ.
    control & 0xffc0 == 0x1f80
}

#[inline]
fn control() -> u32 {
    let mut value = 0u32;
    // SAFETY: SSE2 is baseline on x86-64; value is a valid writable u32.
    unsafe {
        core::arch::asm!(
            "stmxcsr [{ptr}]",
            ptr = in(reg) &mut value,
            options(nostack, preserves_flags),
        );
    }
    value
}

#[inline]
pub(super) fn nearest(bits: u64) -> Option<(u64, Status)> {
    if !crate::sqrt::is_positive_finite(bits) || !control_is_supported(control()) {
        return None;
    }
    let root = f64::from_bits(bits);
    let mut reciprocal = 1.0f64;
    // SAFETY: SSE2 is baseline on x86-64. MXCSR was checked without changing
    // it. All accepted inputs produce finite, normal roots and reciprocals.
    // Inline assembly keeps the operations and their rounding boundary exact;
    // neither instruction is an estimate or a compiler algebraic rewrite.
    // Do not use preserves_flags: these instructions can set MXCSR exception
    // flags even though they leave EFLAGS and MXCSR control bits unchanged.
    unsafe {
        core::arch::asm!(
            "sqrtsd {root}, {root}",
            "divsd {reciprocal}, {root}",
            root = inout(xmm_reg) root => _,
            reciprocal = inout(xmm_reg) reciprocal,
            options(nostack, nomem),
        );
    }
    // Both operations are exact iff x is a power of four: an exact reciprocal
    // of a binary rational is representable only when its significand is a
    // power of two. Any rounded root from a different input is already inexact.
    let exact = if bits < (1u64 << 52) {
        // Subnormal bit k represents 2^(k-1074), whose exponent is even iff k is.
        bits.is_power_of_two() && bits.trailing_zeros().is_multiple_of(2)
    } else {
        // Normal unbiased exponent is even iff the biased exponent is odd.
        bits & 0x001f_ffff_ffff_ffff == 0x0010_0000_0000_0000
    };
    Some((
        reciprocal.to_bits(),
        if exact { Status::OK } else { Status::INEXACT },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_status(flags: u32) {
        // Preserve every control bit: changing those across Rust code is UB,
        // even if the source contains no intervening floating-point operations.
        let value = (control() & !0x3f) | (flags & 0x3f);
        // SAFETY: only exception status flags change, which Rust permits.
        unsafe {
            core::arch::asm!("ldmxcsr [{ptr}]", ptr = in(reg) &value,
                options(nostack, readonly));
        }
    }

    struct Restore(u32);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_status(self.0);
        }
    }

    #[test]
    fn arithmetic_can_set_status_flags_but_preserves_control_bits() {
        let saved = control();
        assert!(control_is_supported(saved));
        let _restore = Restore(saved);
        set_status(0);
        let exact = nearest(4.0f64.to_bits());
        let exact_control = control();
        let inexact = nearest(2.0f64.to_bits());
        let inexact_control = control();
        set_status(saved);
        assert_eq!(exact, Some((0.5f64.to_bits(), Status::OK)));
        assert_eq!(exact_control, saved & !0x3f);
        assert_eq!(inexact.unwrap().1, Status::INEXACT);
        assert_eq!(inexact_control & 0xffc0, 0x1f80);
        assert_ne!(inexact_control & 0x20, 0); // precision/inexact exception
    }

    #[test]
    fn control_mask_rejects_incompatible_settings_and_accepts_status_flags() {
        for flags in 0..64 {
            assert!(control_is_supported(0x1f80 | flags));
        }
        for bit in 6..16 {
            assert!(!control_is_supported(0x1f80 ^ (1 << bit)));
        }
    }

    #[test]
    fn supported_path_matches_reference_for_every_exponent_and_subnormal_power() {
        let saved = control();
        assert!(control_is_supported(saved));
        let _restore = Restore(saved);
        let mut cases = vec![0, u64::MAX, 0x7ff0_0000_0000_0000, 0x8000_0000_0000_0000];
        for exponent in 1..2047u64 {
            for fraction in [0, 1, 0x000f_ffff_ffff_ffff] {
                cases.push((exponent << 52) | fraction);
            }
        }
        for bit in 0..52 {
            cases.extend([(1u64 << bit) - 1, 1 << bit, (1 << bit) + 1]);
        }
        for bits in cases {
            let actual = nearest(bits);
            assert_eq!(actual.is_some(), crate::sqrt::is_positive_finite(bits));
            if let Some((value, status)) = actual {
                use rustc_apfloat::Float;
                let expected = super::super::reference(bits, 0);
                assert_eq!(
                    (value, status),
                    (expected.value.to_bits() as u64, expected.status),
                    "bits={bits:016x}"
                );
            }
        }
    }
}
