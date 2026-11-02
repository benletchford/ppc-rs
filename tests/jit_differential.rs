//! Differential test: random programs must behave identically in the
//! interpreter and the Cranelift JIT.
#![cfg(all(feature = "jit", not(target_family = "wasm")))]

use ppc::{PpcAlignmentPolicy, PpcCpu, PpcJitConfig, PpcRunResult, PpcSectionMem};

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;
const DATA_LEN: usize = 8192;

static NATIVE_RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static REGIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }
    fn reg(&mut self) -> u32 {
        self.below(32)
    }
    /// A register likely to be a valid data pointer.
    fn base(&mut self) -> u32 {
        if self.below(8) == 0 {
            self.reg()
        } else {
            5 + self.below(2)
        }
    }
    fn imm16(&mut self) -> u32 {
        match self.below(4) {
            0 => 0,
            1 => self.below(64),
            2 => 0xffff - self.below(4),
            _ => self.next() as u32 & 0xffff,
        }
    }
}

fn d_form(op: u32, a: u32, b: u32, imm: u32) -> u32 {
    (op << 26) | (a << 21) | (b << 16) | (imm & 0xffff)
}

fn x_form(op: u32, a: u32, b: u32, c: u32, xo: u32, rc: u32) -> u32 {
    (op << 26) | (a << 21) | (b << 16) | (c << 11) | (xo << 1) | rc
}

fn random_word(rng: &mut Rng) -> u32 {
    let (a, b, c) = (rng.reg(), rng.reg(), rng.reg());
    let rc = rng.below(2);
    match rng.below(91) {
        0 => d_form(14, a, b, rng.imm16()),
        1 => d_form(15, a, b, rng.imm16()),
        2 => d_form(24, a, b, rng.imm16()),
        3 => d_form(25, a, b, rng.imm16()),
        4 => d_form(26, a, b, rng.imm16()),
        5 => d_form(27, a, b, rng.imm16()),
        6 => d_form(28, a, b, rng.imm16()),
        7 => d_form(29, a, b, rng.imm16()),
        8 => d_form(7, a, b, rng.imm16()),
        9 => d_form(12, a, b, rng.imm16()),
        10 => d_form(13, a, b, rng.imm16()),
        11 => d_form(8, a, b, rng.imm16()),
        12 => d_form(11, rng.below(8) << 2, b, rng.imm16()),
        13 => d_form(10, rng.below(8) << 2, b, rng.imm16()),
        // Loads and stores (D-form).
        14 => d_form(32, a, rng.base(), rng.below(64)),
        15 => d_form(34, a, rng.base(), rng.below(64)),
        16 => d_form(40, a, rng.base(), rng.below(64)),
        17 => d_form(42, a, rng.base(), rng.below(64)),
        18 => d_form(36, a, rng.base(), rng.below(64)),
        19 => d_form(38, a, rng.base(), rng.below(64)),
        20 => d_form(44, a, rng.base(), rng.below(64)),
        21 => d_form(37, a, rng.base(), rng.below(64)),
        // Loads and stores (indexed).
        22 => x_form(31, a, rng.base(), 7, 23, 0),
        23 => x_form(31, a, rng.base(), 7, 87, 0),
        24 => x_form(31, a, rng.base(), 7, 279, 0),
        25 => x_form(31, a, rng.base(), 7, 151, 0),
        26 => x_form(31, a, rng.base(), 7, 215, 0),
        27 => x_form(31, a, rng.base(), 7, 407, 0),
        // Logical / arithmetic X-forms.
        28 => x_form(31, a, b, c, 28, rc),
        29 => x_form(31, a, b, c, 444, rc),
        30 => x_form(31, a, b, c, 316, rc),
        31 => x_form(31, a, b, c, 124, rc),
        32 => x_form(31, a, b, c, 476, rc),
        33 => x_form(31, a, b, c, 60, rc),
        34 => x_form(31, a, b, c, 412, rc),
        35 => x_form(31, a, b, c, 284, rc),
        36 => x_form(31, a, b, c, 24, rc),
        37 => x_form(31, a, b, c, 536, rc),
        38 => x_form(31, a, b, c, 824, rc),
        39 => x_form(31, a, b, 0, 954, rc),
        40 => x_form(31, a, b, 0, 922, rc),
        41 => x_form(31, a, b, 0, 26, rc),
        42 => x_form(31, a, b, c, 266, rc),
        43 => x_form(31, a, b, c, 40, rc),
        44 => x_form(31, a, b, 0, 104, rc),
        45 => x_form(31, a, b, c, 235, rc),
        46 => x_form(31, rng.below(8) << 2, b, c, 0, 0),
        47 => x_form(31, rng.below(8) << 2, b, c, 32, 0),
        // Rotates.
        48 => {
            (21 << 26)
                | (a << 21)
                | (b << 16)
                | (rng.below(32) << 11)
                | (rng.below(32) << 6)
                | (rng.below(32) << 1)
                | rc
        }
        49 => {
            (20 << 26)
                | (a << 21)
                | (b << 16)
                | (rng.below(32) << 11)
                | (rng.below(32) << 6)
                | (rng.below(32) << 1)
                | rc
        }
        50 => {
            (23 << 26)
                | (a << 21)
                | (b << 16)
                | (c << 11)
                | (rng.below(32) << 6)
                | (rng.below(32) << 1)
                | rc
        }
        // SPR moves: XER(1), LR(8), CTR(9), plus one unsupported.
        51 => x_form(31, a, [1, 8, 9, 8][rng.below(4) as usize], 0, 339, 0),
        52 => x_form(31, a, [1, 8, 9, 9][rng.below(4) as usize], 0, 467, 0),
        // Instructions that always go through the catch-all helper.
        53 => x_form(31, a, b, c, 10, rc),
        54 => x_form(31, a, b, c, 75, rc),
        55 => x_form(31, a, b, c, 491, rc),
        56 => x_form(31, a, 0, 0, 19, 0),
        57 => x_form(31, a, b, c, 792, rc),
        88 => x_form(31, a, b, c, 138, rc),
        89 => d_form(46, a, rng.base(), rng.below(64)),
        90 => d_form(47, a, rng.base(), rng.below(64)),
        // Forward conditional branches (bc), possibly with CTR and link.
        58 => {
            (16 << 26)
                | (rng.below(32) << 21)
                | (rng.below(32) << 16)
                | ((1 + rng.below(3)) << 2)
                | rng.below(2)
        }
        // Update-form, algebraic and floating-point loads/stores.
        59 => d_form(
            [35, 41, 43, 33, 39, 45][rng.below(6) as usize],
            a,
            rng.base(),
            rng.below(64),
        ),
        60..=67 => d_form(48 + rng.below(8), a, rng.base(), rng.below(64)),
        68..=80 => x_form(
            31,
            a,
            rng.base(),
            7,
            [
                119, 311, 375, 55, 247, 439, 183, 343, 535, 567, 599, 631, 663, 695, 727, 759,
            ][rng.below(16) as usize],
            0,
        ),
        // Backward `bdnz`/`bdz` loops, so chained blocks and their
        // successor links run many times.
        81..=83 => {
            let back = (rng.below(8) + 1) * 4;
            (16 << 26) | ([16, 18][rng.below(2) as usize] << 21) | ((back.wrapping_neg()) & 0xfffc)
        }
        // bclr/bcctr/b
        _ => match rng.below(3) {
            0 => {
                (19 << 26)
                    | (rng.below(32) << 21)
                    | (rng.below(32) << 16)
                    | (16 << 1)
                    | rng.below(2)
            }
            1 => {
                (19 << 26)
                    | (rng.below(32) << 21)
                    | (rng.below(32) << 16)
                    | (528 << 1)
                    | rng.below(2)
            }
            _ => (18 << 26) | ((1 + rng.below(3)) << 2) | rng.below(2),
        },
    }
}

#[derive(Debug, PartialEq)]
struct Outcome {
    result: PpcRunResult,
    gpr: [u32; 32],
    cr: u32,
    lr: u32,
    ctr: u32,
    xer: u32,
    fpr: [u64; 32],
    fpscr: u32,
    pc: u32,
    time_base: u64,
    data: Vec<u8>,
}

fn run(
    words: &[u32],
    seed: u64,
    policy: PpcAlignmentPolicy,
    budget: u64,
    jit: bool,
    ctr: Option<u32>,
) -> Outcome {
    run_fp(words, seed, policy, budget, jit, ctr, 0, true)
}

#[allow(clippy::too_many_arguments)]
fn run_fp(
    words: &[u32],
    seed: u64,
    policy: PpcAlignmentPolicy,
    budget: u64,
    jit: bool,
    ctr: Option<u32>,
    fpscr: u32,
    fp_available: bool,
) -> Outcome {
    let mut rng = Rng(seed | 1);
    let mut code: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
    for _ in 0..4 {
        code.extend_from_slice(&0x6000_0000u32.to_be_bytes());
    }
    code.extend_from_slice(&0x4e80_0020u32.to_be_bytes());
    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(CODE, code);
    memory.add_region(DATA, (0..DATA_LEN).map(|i| (i * 7) as u8).collect());
    let mut cpu = PpcCpu::new();
    cpu.fpscr = fpscr;
    cpu.set_msr_fp_available(fp_available);
    cpu.alignment_policy = policy;
    cpu.pc = CODE;
    for r in 0..32 {
        cpu.gpr[r] = match rng.below(4) {
            0 => 0,
            1 => rng.next() as u32,
            2 => rng.below(300),
            _ => 0x8000_0000 | rng.below(4),
        };
    }
    for r in 0..32 {
        cpu.fpr[r] = match rng.below(6) {
            0 => 0,
            1 => rng.next(),
            // Exactly representable as a single.
            2 => f64::from(f32::from_bits(rng.next() as u32)).to_bits(),
            3 => (rng.below(1000) as f64 / 7.0).to_bits(),
            4 => 0x7ff0_0000_0000_0001 | (rng.next() & 0x8007_ffff_ffff_ffff),
            _ => rng.next() & 0x800f_ffff_ffff_ffff,
        };
    }
    cpu.gpr[7] = rng.below(96);
    cpu.gpr[5] = DATA + 16 * rng.below(4);
    cpu.gpr[6] = DATA + 64 + rng.below(8);
    cpu.cr = rng.next() as u32;
    cpu.xer = (rng.next() as u32) & 0xe000_0000;
    cpu.ctr = rng.below(5);
    if let Some(ctr) = ctr {
        cpu.ctr = ctr;
    }
    cpu.lr = 0;
    if jit {
        cpu.enable_jit(PpcJitConfig {
            threshold: 0,
            verify: true,
            jitdump: false,
            background: false,
            hot_threshold: 0,
        })
        .unwrap();
    }
    let result = cpu.run_with_imports(&mut memory, budget, 0, 0, 0, |_, _, _| unreachable!());
    let mut data = vec![0; DATA_LEN];
    memory.read_bytes_into(DATA, &mut data).unwrap();
    if jit {
        NATIVE_RUNS.fetch_add(
            cpu.jit_stats().block_runs,
            std::sync::atomic::Ordering::Relaxed,
        );
        REGIONS.fetch_add(
            cpu.jit_stats().regions_compiled,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    Outcome {
        result,
        gpr: cpu.gpr,
        cr: cpu.cr,
        lr: cpu.lr,
        ctr: cpu.ctr,
        xer: cpu.xer,
        fpr: cpu.fpr,
        fpscr: cpu.fpscr,
        pc: cpu.pc,
        time_base: cpu.time_base(),
        data,
    }
}

#[test]
fn double_precision_operations_match() {
    for seed in 1..=150 {
        for word in [
            x_form(63, 3, 0, 4, 15, 0),
            x_form(63, 3, 0, 4, 15, 1),
            x_form(63, 3, 2, 4, 18, 0),
            x_form(63, 3, 2, 4, 18, 1),
            x_form(63, 2 << 2, 2, 4, 32, 0),
        ] {
            let expected = run(&[word], seed, PpcAlignmentPolicy::Trap, 100, false, None);
            let actual = run(&[word], seed, PpcAlignmentPolicy::Trap, 100, true, None);
            assert_eq!(expected.fpr, actual.fpr, "seed={seed} word={word:#x}");
            assert_eq!(expected.fpscr, actual.fpscr, "seed={seed} word={word:#x}");
            assert_eq!(expected.cr, actual.cr, "seed={seed} word={word:#x}");
            assert_eq!(expected, actual, "seed={seed} word={word:#x}");
        }
    }
}

#[test]
fn double_precision_rounding_and_unavailable_match() {
    for word in [
        x_form(63, 3, 0, 4, 15, 1),
        x_form(63, 3, 2, 4, 18, 1),
        x_form(63, 2 << 2, 2, 4, 32, 0),
    ] {
        for fpscr in [0, 1, 2, 3, 0x80, 0x4000_00f3] {
            for fp_available in [true, false] {
                for seed in 1..=12 {
                    let expected = run_fp(
                        &[word],
                        seed,
                        PpcAlignmentPolicy::Trap,
                        100,
                        false,
                        None,
                        fpscr,
                        fp_available,
                    );
                    let actual = run_fp(
                        &[word],
                        seed,
                        PpcAlignmentPolicy::Trap,
                        100,
                        true,
                        None,
                        fpscr,
                        fp_available,
                    );
                    assert_eq!(
                        expected.fpr, actual.fpr,
                        "seed={seed} word={word:#x} fpscr={fpscr:#x} available={fp_available}"
                    );
                    assert_eq!(
                        expected.fpscr, actual.fpscr,
                        "seed={seed} word={word:#x} fpscr={fpscr:#x} available={fp_available}"
                    );
                    assert_eq!(
                        expected, actual,
                        "seed={seed} word={word:#x} fpscr={fpscr:#x} available={fp_available}"
                    );
                }
            }
        }
    }
}

#[test]
fn single_precision_arithmetic_matches() {
    for seed in 1..=100 {
        for &xo in &[18u32, 20, 21, 25] {
            for rc in 0..=1 {
                let operand = if xo == 25 { 4 << 6 } else { 4 << 11 };
                let word = (59 << 26) | (3 << 21) | (2 << 16) | operand | (xo << 1) | rc;
                let expected = run(&[word], seed, PpcAlignmentPolicy::Trap, 100, false, None);
                let actual = run(&[word], seed, PpcAlignmentPolicy::Trap, 100, true, None);
                assert_eq!(expected.fpr, actual.fpr, "seed={seed} xo={xo} rc={rc}");
                assert_eq!(expected.fpscr, actual.fpscr, "seed={seed} xo={xo} rc={rc}");
                assert_eq!(expected, actual, "seed={seed} xo={xo} rc={rc}");
            }
        }
    }
}

#[test]
fn multiple_word_transfers_match_on_partial_faults() {
    for &op in &[46, 47] {
        for &offset in &[0u32, 1, 4, 8, 12, 16] {
            for &policy in &[PpcAlignmentPolicy::Trap, PpcAlignmentPolicy::EmulateData] {
                let address = DATA + DATA_LEN as u32 - offset;
                let words = [
                    d_form(15, 5, 0, address >> 16),
                    d_form(24, 5, 5, address),
                    d_form(op, 28, 5, 0),
                ];
                let expected = run(&words, 7, policy, 100, false, None);
                let actual = run(&words, 7, policy, 100, true, None);
                assert_eq!(
                    expected, actual,
                    "op={op} offset={offset} policy={policy:?}"
                );
            }
        }
    }
}

#[test]
fn division_and_arithmetic_shift_edges_match() {
    for &a in &[0u32, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0x8000_0001] {
        for &b in &[0u32, 1, 31, 32, 33, 63, 64, 0xffff_ffff] {
            for &xo in &[491, 792] {
                let words = [
                    d_form(15, 3, 0, a >> 16),
                    d_form(24, 3, 3, a),
                    d_form(15, 4, 0, b >> 16),
                    d_form(24, 4, 4, b),
                    x_form(31, 3, 4, 4, xo, 1),
                ];
                let expected = run(&words, 1, PpcAlignmentPolicy::Trap, 100, false, None);
                let actual = run(&words, 1, PpcAlignmentPolicy::Trap, 100, true, None);
                assert_eq!(expected, actual, "a={a:#x} b={b:#x} xo={xo}");
            }
        }
    }
}

#[test]
fn random_programs_match_the_interpreter() {
    let iterations: u64 = std::env::var("PPC_JIT_FUZZ_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4000);
    for seed in 1..=iterations {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let len = 1 + rng.below(14) as usize;
        let words: Vec<u32> = (0..len).map(|_| random_word(&mut rng)).collect();
        let policy = if rng.below(2) == 0 {
            PpcAlignmentPolicy::Trap
        } else {
            PpcAlignmentPolicy::EmulateData
        };
        let budget = if rng.below(4) == 0 {
            1 + u64::from(rng.below(6))
        } else {
            1000
        };
        let expected = run(&words, seed, policy, budget, false, None);
        let actual = run(&words, seed, policy, budget, true, None);
        assert_eq!(
            expected, actual,
            "seed {seed}, policy {policy:?}, budget {budget}, words {:08x?}",
            words
        );
    }
    let runs = NATIVE_RUNS.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("native block runs: {runs}");
    assert!(runs > iterations, "JIT barely ran: {runs}");
    let regions = REGIONS.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("regions compiled: {regions}");
    assert!(regions > 0, "no regions formed");
}

/// Background compilation only changes when native code takes over: the
/// loop keeps running in the interpreter until the worker finishes, then
/// switches without changing the result.
#[test]
fn background_compilation_installs_code_without_changing_results() {
    // r3 = 0; loop: r3 += 1; cmpwi r3, 30000; blt loop; nop
    let words = [
        0x3860_0000,
        0x3863_0001,
        0x2c03_7530,
        0x4180_fff8,
        0x6000_0000,
    ];
    let halt = CODE + 16;
    let run = |background: Option<bool>| {
        let code: Vec<u8> = words.iter().flat_map(|w: &u32| w.to_be_bytes()).collect();
        let mut memory = PpcSectionMem::new();
        memory.add_readonly_region(CODE, code);
        let mut cpu = PpcCpu::new();
        cpu.pc = CODE;
        if let Some(background) = background {
            cpu.enable_jit(PpcJitConfig {
                threshold: 2,
                verify: true,
                jitdump: false,
                background,
                hot_threshold: 2,
            })
            .unwrap();
        }
        let mut slices = 0;
        while cpu.pc != halt {
            cpu.run_with_imports(&mut memory, 50, halt, 0, 0, |_, _, _| unreachable!());
            // Give the worker a chance (block, then region) before the loop finishes.
            if background == Some(true) && cpu.jit_stats().regions_compiled == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            slices += 1;
            assert!(slices < 1_000_000);
        }
        (cpu.gpr[3], cpu.cr, cpu.time_base(), cpu.jit_stats())
    };
    let interpreted = run(None);
    for background in [false, true] {
        let jitted = run(Some(background));
        assert_eq!(jitted.0, interpreted.0);
        assert_eq!(jitted.1, interpreted.1);
        assert_eq!(jitted.2, interpreted.2);
        assert!(jitted.3.blocks_compiled > 0, "background={background}");
        assert!(jitted.3.block_runs > 0, "background={background}");
        assert!(jitted.3.regions_compiled > 0, "background={background}");
    }
}

/// Loops whose bodies span several blocks run long enough to form regions,
/// enter them, follow their internal edges and leave them mid-way (faults,
/// budget, exits).
#[test]
fn random_loops_match_the_interpreter() {
    let iterations: u64 = std::env::var("PPC_JIT_FUZZ_ITERATIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let regions_before = REGIONS.load(std::sync::atomic::Ordering::Relaxed);
    for seed in 1..=iterations {
        let mut rng = Rng(seed.wrapping_mul(0xd1b5_4a32_d192_ed03) | 1);
        let len = 2 + rng.below(12) as usize;
        let mut words: Vec<u32> = (0..len).map(|_| random_word(&mut rng)).collect();
        // bdnz back to the start of the body.
        let back = (len as u32 * 4).wrapping_neg() & 0xfffc;
        words.push((16 << 26) | (16 << 21) | back);
        let policy = if rng.below(2) == 0 {
            PpcAlignmentPolicy::Trap
        } else {
            PpcAlignmentPolicy::EmulateData
        };
        let budget = [1000, 3000, 1 + u64::from(rng.below(200))][rng.below(3) as usize];
        let ctr = Some(20 + rng.below(300));
        let expected = run(&words, seed, policy, budget, false, ctr);
        let actual = run(&words, seed, policy, budget, true, ctr);
        assert_eq!(
            expected, actual,
            "loop seed {seed}, policy {policy:?}, budget {budget}, ctr {ctr:?}, words {:08x?}",
            words
        );
    }
    let regions = REGIONS.load(std::sync::atomic::Ordering::Relaxed) - regions_before;
    eprintln!("loop regions compiled: {regions}");
    assert!(regions > iterations / 4, "regions barely formed: {regions}");
}

/// Two copies of a routine 256 KiB apart share an interpreter cache slot and
/// keep evicting each other's entry there; their compiled code and regions
/// live in the JIT table and survive that: no block compiles twice (running
/// twice as long compiles nothing more) and each entry forms one region.
#[test]
fn routines_sharing_an_interpreter_cache_slot_keep_their_regions() {
    const TWIN: u32 = 0x4_0000;
    let routine = [
        0x3880_0000, // li r4, 0
        0x3884_0001, // loop: addi r4, r4, 1
        0x7080_0001, // andi. r0, r4, 1
        0x4182_0008, // beq skip
        0x3863_0001, // addi r3, r3, 1
        0x2c04_0032, // skip: cmpwi r4, 50
        0x4180_ffec, // blt loop
        0x4e80_0020, // blr
    ];
    let first = CODE + 0x1000;
    let second = first + TWIN;
    let bl = |from: u32, to: u32| (18 << 26) | (to.wrapping_sub(from) & 0x03ff_fffc) | 1;
    let driver = [
        bl(CODE, first),
        bl(CODE + 4, second),
        0x4200_fff8, // bdnz -8
        0x6000_0000, // halt
    ];
    let halt = CODE + 12;
    let mut code = vec![0u8; (second - CODE) as usize + routine.len() * 4];
    let mut place = |at: u32, words: &[u32]| {
        for (index, word) in words.iter().enumerate() {
            let offset = (at - CODE) as usize + index * 4;
            code[offset..offset + 4].copy_from_slice(&word.to_be_bytes());
        }
    };
    place(CODE, &driver);
    place(first, &routine);
    place(second, &routine);
    let run = |jit: bool, calls: u32| {
        let mut memory = PpcSectionMem::new();
        memory.add_readonly_region(CODE, code.clone());
        let mut cpu = PpcCpu::new();
        cpu.pc = CODE;
        cpu.ctr = calls;
        if jit {
            cpu.enable_jit(PpcJitConfig {
                threshold: 2,
                verify: false,
                jitdump: false,
                background: false,
                hot_threshold: 20,
            })
            .unwrap();
        }
        let mut slices = 0;
        while cpu.pc != halt {
            cpu.run_with_imports(&mut memory, 1000, halt, 0, 0, |_, _, _| unreachable!());
            slices += 1;
            assert!(slices < 1_000_000);
        }
        let stats = jit.then(|| cpu.jit_stats());
        ((cpu.gpr[3], cpu.cr, cpu.time_base()), stats)
    };
    let (interpreted, _) = run(false, 300);
    let (jitted, short) = run(true, 300);
    assert_eq!(jitted, interpreted);
    let (_, long) = run(true, 600);
    let (short, long) = (short.unwrap(), long.unwrap());
    eprintln!("{short:?}\n{long:?}");
    assert!(short.regions_compiled > 0, "no regions formed");
    assert_eq!(
        long.blocks_compiled, short.blocks_compiled,
        "blocks compiled again"
    );
    assert!(
        long.regions_compiled <= long.blocks_compiled,
        "regions formed again: {long:?}"
    );
}

#[test]
fn memory_remapping_during_a_native_block_matches_interpreter() {
    use ppc::PpcMemory;

    struct RemappingBus {
        code: PpcSectionMem,
        data: Box<[u8; 4096]>,
    }
    impl PpcMemory for RemappingBus {
        fn read_u8(&mut self, addr: u32) -> Option<u8> {
            if (DATA..DATA + 4096).contains(&addr) {
                Some(self.data[(addr - DATA) as usize])
            } else {
                self.code.read_u8(addr)
            }
        }
        #[allow(clippy::replace_box)] // Replacing the allocation simulates memory remapping.
        fn write_u8(&mut self, addr: u32, value: u8) -> Option<()> {
            if addr != DATA {
                return None;
            }
            self.data = Box::new([value; 4096]);
            Some(())
        }
        fn instruction_cache_token(&mut self, addr: u32) -> Option<u64> {
            self.code.instruction_cache_token(addr)
        }
        fn global_instruction_cache_token(&self) -> Option<u64> {
            self.code.global_instruction_cache_token()
        }
        fn code_epoch(&self) -> Option<u64> {
            self.code.code_epoch()
        }
    }
    let words = [
        d_form(34, 3, 5, 0),
        d_form(38, 4, 5, 0),
        d_form(34, 6, 5, 0),
        0x48000000,
    ];
    let code: Vec<_> = words.into_iter().flat_map(u32::to_be_bytes).collect();
    let run = |jit| {
        let mut code_mem = PpcSectionMem::new();
        code_mem.add_readonly_region(CODE, code.clone());
        let mut mem = RemappingBus {
            code: code_mem,
            data: Box::new([1; 4096]),
        };
        let mut cpu = PpcCpu::new();
        cpu.pc = CODE;
        cpu.gpr[4] = 7;
        cpu.gpr[5] = DATA;
        if jit {
            cpu.enable_jit(PpcJitConfig {
                threshold: 0,
                background: false,
                ..Default::default()
            })
            .unwrap();
        }
        cpu.run_with_imports(&mut mem, 3, 0, 0, 0, |_, _, _| unreachable!());
        if jit {
            assert!(cpu.jit_stats().block_runs > 0);
        }
        (cpu.gpr[3], cpu.gpr[6])
    };
    assert_eq!(run(false), (1, 7));
    assert_eq!(run(true), run(false));
}

#[test]
fn load_remapping_code_exits_a_hot_region() {
    use ppc::PpcMemory;

    struct BankSwitchingBus {
        memory: PpcSectionMem,
        switch_on_read: bool,
    }
    impl PpcMemory for BankSwitchingBus {
        fn read_u8(&mut self, addr: u32) -> Option<u8> {
            if (DATA..DATA + 8).contains(&addr) {
                if std::mem::take(&mut self.switch_on_read) {
                    self.memory
                        .add_readonly_region(CODE + 8, d_form(14, 3, 3, 7).to_be_bytes().to_vec());
                }
                Some(0)
            } else {
                self.memory.read_u8(addr)
            }
        }
        fn write_u8(&mut self, addr: u32, value: u8) -> Option<()> {
            self.memory.write_u8(addr, value)
        }
        fn instruction_cache_token(&mut self, addr: u32) -> Option<u64> {
            self.memory.instruction_cache_token(addr)
        }
        fn global_instruction_cache_token(&self) -> Option<u64> {
            self.memory.global_instruction_cache_token()
        }
        fn code_epoch(&self) -> Option<u64> {
            self.memory.code_epoch()
        }
    }

    // Two blocks: load and branch; increment and branch back.
    let run = |jit, load_opcode| {
        let words = [
            d_form(load_opcode, 4, 5, 0),
            0x4800_0004,
            d_form(14, 3, 3, 1),
            0x4bff_fff4,
        ];
        let mut memory = PpcSectionMem::new();
        memory.add_readonly_region(CODE, words.into_iter().flat_map(u32::to_be_bytes).collect());
        let mut bus = BankSwitchingBus {
            memory,
            switch_on_read: false,
        };
        let mut cpu = PpcCpu::new();
        cpu.pc = CODE;
        cpu.gpr[5] = DATA;
        if jit {
            cpu.enable_jit(PpcJitConfig {
                threshold: 0,
                hot_threshold: 2,
                background: false,
                verify: true,
                jitdump: false,
            })
            .unwrap();
        }
        cpu.run_with_imports(&mut bus, 400, 0, 0, 0, |_, _, _| unreachable!());
        assert_eq!(cpu.pc, CODE);
        if jit {
            assert!(cpu.jit_stats().regions_compiled > 0);
        }
        let before = cpu.gpr[3];
        bus.switch_on_read = true;
        cpu.run_with_imports(&mut bus, 4, 0, 0, 0, |_, _, _| unreachable!());
        (cpu.gpr[3] - before, cpu.pc, cpu.time_base())
    };
    for load_opcode in [34, 48, 50] {
        // lbz, lfs, lfd
        let interpreted = run(false, load_opcode);
        assert_eq!(interpreted.0, 7);
        assert_eq!(run(true, load_opcode), interpreted);
    }
}
