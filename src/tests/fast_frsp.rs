use super::*;

#[test]
fn unary_fp_reserved_fra_matches_decoded_exceptions_in_all_paths() {
    for xo in [12u32, 26] {
        for fra in 0..32 {
            for rc in [0, 1] {
                let word = (63 << 26) | (3 << 21) | (fra << 16) | (2 << 11) | (xo << 1) | rc;
                assert_eq!(PpcCpu::fast_instruction(word), fra == 0);
                for fp_available in [false, true] {
                    if fra == 0 && fp_available {
                        continue; // Valid arithmetic is covered below.
                    }
                    let exception = if fra == 0 {
                        PpcException::FloatingPointUnavailable
                    } else {
                        PpcException::IllegalInstruction {
                            word,
                            reason: PpcIllegalInstructionReason::InvalidForm,
                        }
                    };
                    for cached in [false, true] {
                        let mut memory = PpcSectionMem::new();
                        let code = word.to_be_bytes().to_vec();
                        if cached {
                            memory.add_readonly_region(0x1000, code);
                        } else {
                            memory.add_region(0x1000, code);
                        }
                        let mut cpu = PpcCpu::new();
                        cpu.pc = 0x1000;
                        cpu.fpr.fill(2.0f64.to_bits());
                        cpu.fpscr = 0x1234_5678;
                        cpu.cr = 0x8765_4321;
                        cpu.set_time_base(123);
                        cpu.set_msr_fp_available(fp_available);
                        assert_eq!(
                            cpu.step(&mut memory, word),
                            PpcStepResult::Exception(exception)
                        );
                        // Repeat to cover a cache hit as well as its fill.
                        for _ in 0..2 {
                            assert_eq!(
                                cpu.run_with_imports(
                                    &mut memory,
                                    1,
                                    0,
                                    0,
                                    0,
                                    |_, _, _| unreachable!()
                                ),
                                PpcRunResult::Exception {
                                    pc: 0x1000,
                                    exception,
                                    cycles: 0
                                },
                                "xo={xo} fra={fra} rc={rc} fp={fp_available} cached={cached}"
                            );
                            assert_eq!(cpu.pc, 0x1000);
                            assert_eq!(cpu.time_base(), 123);
                            assert_eq!(cpu.fpr, [2.0f64.to_bits(); 32]);
                            assert_eq!(cpu.fpscr, 0x1234_5678);
                            assert_eq!(cpu.cr, 0x8765_4321);
                        }
                    }
                }
            }
        }
    }
}

fn reference(cpu: &mut PpcCpu, bits: u64, rc: bool) {
    let source = ApDouble::from_bits(u128::from(bits));
    let narrowed: StatusAnd<ApSingle> = source.convert_r(cpu.fp_rounding_mode(), &mut false);
    let mut status = narrowed.status
        | if PpcCpu::is_signaling_nan(bits) {
            ApStatus::INVALID_OP
        } else {
            ApStatus::OK
        };
    let largest = (narrowed.value.to_bits() as u32 & 0x7fff_ffff) == 0x7f7f_ffff;
    let adjusted = cpu.fp_adjusted_single_conversion(bits, &mut status, largest);
    cpu.finish_ap_single_result_adjusted(
        3,
        StatusAnd {
            status,
            value: source,
        },
        source,
        adjusted,
        1u32 << FPSCR_VXSNAN,
        rc,
    );
    cpu.set_time_base(cpu.time_base().wrapping_add(1));
}
#[test]
fn normal_single_rounding_matches_software_in_all_modes_and_execution_paths() {
    let mut inputs = vec![
        0,
        0x8000_0000_0000_0000,
        1,
        u64::MAX,
        0x7ff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff8_0000_1234_5678,
    ];
    for exponent in 896u64..=1151 {
        for fraction in [
            0,
            1,
            0x0fff_ffff,
            0x1000_0000,
            0x1000_0001,
            0x1fff_ffff,
            0x2000_0000,
            0x3000_0000,
            0x000f_ffff_ffff_ffff,
        ] {
            for sign in [0, 0x8000_0000_0000_0000] {
                inputs.push(sign | exponent << 52 | fraction);
            }
        }
    }
    let mut seed = 0x5a8c_1234_0066_17a3u64;
    for _ in 0..8192 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        inputs.push(seed);
    }
    let mut memory = PpcSectionMem::new();
    for rc in [0u32, 1] {
        memory.add_readonly_region(
            0x1000 + 4 * rc,
            ((63u32 << 26) | (3 << 21) | (2 << 11) | (12 << 1) | rc)
                .to_be_bytes()
                .to_vec(),
        );
    }
    let mut expected = PpcCpu::new();
    let mut stepped = PpcCpu::new();
    let mut cached = PpcCpu::new();
    let mut accepted = 0;
    for bits in inputs {
        for mode in 0..4 {
            if PpcCpu::fast_single_round(bits, mode).is_some() {
                accepted += 1;
            }
            for prior in [0u32, 0xa502_40f8] {
                for rc in [false, true] {
                    for cpu in [&mut expected, &mut stepped, &mut cached] {
                        cpu.pc = 0x1000 + 4 * u32::from(rc);
                        cpu.fpscr = prior | mode;
                        cpu.cr = 0x1234_5678;
                        cpu.fpr.fill(0x4020_0000_0000_0000);
                        cpu.fpr[2] = bits;
                        cpu.set_time_base(u64::MAX);
                    }
                    reference(&mut expected, bits, rc);
                    let word = (63u32 << 26) | (3 << 21) | (2 << 11) | (12 << 1) | u32::from(rc);
                    assert_eq!(stepped.step(&mut memory, word), PpcStepResult::Stepped);
                    assert_eq!(
                        cached.run_with_imports(&mut memory, 1, 0, 0, 0, |_, _, _| unreachable!()),
                        PpcRunResult::CycleLimit { cycles: 1 }
                    );
                    for actual in [&stepped, &cached] {
                        assert_eq!(
                            (
                                &actual.fpr,
                                actual.fpscr,
                                actual.cr,
                                actual.pc,
                                actual.time_base()
                            ),
                            (
                                &expected.fpr,
                                expected.fpscr,
                                expected.cr,
                                expected.pc,
                                expected.time_base()
                            ),
                            "bits={bits:016x} mode={mode} rc={rc}"
                        );
                    }
                }
            }
        }
    }
    assert!(
        accepted > 16000,
        "insufficient fast-path coverage: {accepted}"
    );
}
