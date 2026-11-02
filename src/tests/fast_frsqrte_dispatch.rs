use super::*;

#[test]
fn direct_frsqrte_result_matches_software_in_both_execution_paths() {
    let mut inputs = vec![
        0,
        1,
        2,
        3,
        0x8000_0000_0000_0000,
        u64::MAX,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0x7ff8_0000_1234_5678,
    ];
    for exponent in 0u64..2047 {
        for fraction in [0, 1, 0x000f_ffff_ffff_ffff] {
            inputs.push((exponent << 52) | fraction);
        }
    }
    let mut seed = 0x890c_1337_5184_eeeeu64;
    for _ in 0..4096 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        inputs.push(seed);
    }
    let mut memory = PpcSectionMem::new();
    for rc in [0u32, 1] {
        memory.add_readonly_region(
            0x1000 + 4 * rc,
            ((63u32 << 26) | (3 << 21) | (3 << 11) | (26 << 1) | rc)
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
            accepted += usize::from(frsqrte::fast(bits, mode).is_some());
            for prior in [0u32, 0xa502_40f8] {
                for rc in [false, true] {
                    for cpu in [&mut expected, &mut stepped, &mut cached] {
                        cpu.pc = 0x1000 + 4 * u32::from(rc);
                        cpu.fpscr = prior | mode;
                        cpu.cr = 0x1234_5678;
                        cpu.fpr.fill(0x4020_0000_0000_0000);
                        cpu.fpr[3] = bits;
                        cpu.set_time_base(u64::MAX);
                    }
                    let result = frsqrte::reference(bits, mode);
                    let invalid = PpcCpu::fp_invalid_flags(bits, None, FPSCR_VXSQRT);
                    expected.finish_ap_double_result(3, result, invalid, rc);
                    expected.set_time_base(0);
                    let word = (63u32 << 26) | (3 << 21) | (3 << 11) | (26 << 1) | u32::from(rc);
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
                            "bits={bits:016x} mode={mode} rc={rc} prior={prior:08x}"
                        );
                    }
                }
            }
        }
    }
    assert!(
        accepted > 24000,
        "insufficient direct path coverage: {accepted}"
    );
}
