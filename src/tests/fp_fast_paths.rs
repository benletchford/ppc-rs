use super::*;

/// Operands biased towards the fast path's edges: normal singles with
/// few significant bits (exact results and ties), nearby and distant
/// exponents, the overflow/underflow binades, and arbitrary doubles.
fn fast_path_operand(state: &mut u64) -> u64 {
    let mut next = || {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };
    let r = next();
    let sign = ((r >> 63) as u32) << 31;
    let exponent = match r % 8 {
        0 => 1 + (next() % 4) as u32,
        1 => 250 + (next() % 4) as u32,
        2 => (next() % 256) as u32,
        _ => 117 + (next() % 20) as u32,
    };
    let fraction = match (r >> 8) % 4 {
        0 => (next() as u32 & 0x7f_ffff) & !((1 << (next() % 23)) - 1),
        1 => 0,
        _ => next() as u32 & 0x7f_ffff,
    };
    match (r >> 16) % 16 {
        0 => next(),
        1 => ((next() % 4096) as f64).to_bits() ^ (next() & 0xffff),
        2 => f64::from(f32::from_bits(sign | fraction)).to_bits(),
        _ => f64::from(f32::from_bits(sign | (exponent.min(254) << 23) | fraction)).to_bits(),
    }
}

#[test]
fn fast_single_arithmetic_matches_the_software_reference() {
    let iterations: u64 = std::env::var("PPC_FP_FAST_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut fast_hits = 0u64;
    let (mut cpu, mut reference) = (PpcCpu::new(), PpcCpu::new());
    for i in 0..iterations {
        let a = fast_path_operand(&mut state);
        let b = fast_path_operand(&mut state);
        let operation = [
            FpBinaryOperation::Add,
            FpBinaryOperation::Subtract,
            FpBinaryOperation::Multiply,
            FpBinaryOperation::Divide,
        ][(i % 4) as usize];
        // Rounding mode, enable bits and sticky exception bits.
        for target in [&mut cpu, &mut reference] {
            target.fpscr = (state as u32) & 0x9ff8_00ff;
            target.fpr[3] = state;
            target.cr = 0;
            target.pc = 0;
        }
        let rc = state & (1 << 40) != 0;
        let fast = cpu.try_fast_single_binary(3, a, b, operation, rc);
        reference.single_binary_reference(3, a, b, operation, rc);
        if fast {
            fast_hits += 1;
            assert_eq!(
                (cpu.fpr[3], cpu.fpscr, cpu.cr, cpu.pc),
                (
                    reference.fpr[3],
                    reference.fpscr,
                    reference.cr,
                    reference.pc
                ),
                "op {} a {a:#018x} b {b:#018x} fpscr {:#010x}",
                i % 4,
                reference.fpscr
            );
        }
        // fctiwz on the same operand.
        for target in [&mut cpu, &mut reference] {
            target.fpscr = (state as u32 >> 3) & 0x9ff8_00ff;
        }
        let bits = if i % 3 == 0 {
            (f64::from_bits(a) * 2f64.powi((i % 40) as i32)).to_bits()
        } else {
            a
        };
        if let Some(value) = cpu.fast_convert_to_i32_toward_zero(bits) {
            let expected = reference.fp_convert_to_i32(bits, ApRound::TowardZero);
            assert_eq!(
                (Some(value), cpu.fpscr),
                (expected, reference.fpscr),
                "fctiwz {bits:#018x}"
            );
        }
    }
    assert!(
        fast_hits > iterations / 4,
        "fast path rarely taken: {fast_hits}"
    );
}
