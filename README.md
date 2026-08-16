# ppc-rs

A safe, pure Rust 32-bit PowerPC user-mode CPU interpreter.

An architecture-only core for high-level emulation (HLE), executable analysis,
and other hosts that provide their own memory map, loader, and operating-system
services.

[![Rust CI](https://github.com/benletchford/ppc-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/benletchford/ppc-rs/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/ppc.svg)](https://crates.io/crates/ppc)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Features

- **32-bit PowerPC state**: 32 general-purpose registers, 32 floating-point registers, CR, LR, CTR, XER, FPSCR, MSR, PC, and load/store reservation state
- **Broad user ISA coverage**: integer arithmetic, logical and rotate operations, branches, compares, multiply/divide, byte-reversed and string memory operations, atomic reservation forms, cache hints, and common floating-point instructions
- **Structured exceptions**: alignment, memory, illegal-instruction, system-call, trap, and floating-point-unavailable exits leave the faulting PC available to the host
- **HLE-ready imports**: synthetic import ranges can return values, preserve GPR3, charge skipped guest cycles, enter native guest callbacks, halt, or raise host-import exceptions
- **Observable execution**: optional instruction-fetch histograms and guest-write observers support coverage audits, tracing, and compatibility work
- **Host-controlled alignment**: strict architectural traps by default, with optional unaligned data-access emulation for operating systems that provide fixups
- **Ready-made memory bus**: `PpcSectionMem` maps arbitrary read-only and writable guest regions, including deliberate overlays
- **No runtime dependencies**: the interpreter is self-contained and written in safe Rust

## Quick Start

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
ppc = "0.1"
```

### Basic Usage

```rust
use ppc::{PpcCpu, PpcRunResult, PpcSectionMem};

fn main() {
    // addi r3, r0, 42; blr
    let mut code = Vec::new();
    code.extend_from_slice(&0x3860_002au32.to_be_bytes());
    code.extend_from_slice(&0x4e80_0020u32.to_be_bytes());

    let mut memory = PpcSectionMem::new();
    memory.add_readonly_region(0x1000, code);

    let mut cpu = PpcCpu::new();
    cpu.pc = 0x1000;
    cpu.lr = 0;

    match cpu.run(&mut memory, 100, 0) {
        PpcRunResult::Halted { .. } => assert_eq!(cpu.gpr[3], 42),
        result => panic!("unexpected exit: {result:?}"),
    }
}
```

### Custom Memory

Implement `PpcMemory` when the guest address space needs devices, banked
memory, or host-managed mappings. Byte reads and writes are required; the
trait supplies big-endian 16-, 32-, and 64-bit accessors by default.

```rust
use ppc::PpcMemory;

struct Memory {
    bytes: Vec<u8>,
}

impl PpcMemory for Memory {
    fn read_u8(&mut self, address: u32) -> Option<u8> {
        self.bytes.get(address as usize).copied()
    }

    fn write_u8(&mut self, address: u32, value: u8) -> Option<()> {
        *self.bytes.get_mut(address as usize)? = value;
        Some(())
    }
}
```

### High-Level Emulation

`run_with_imports` treats a caller-selected address range as synthetic import
slots. The handler receives the slot index, complete CPU state, and memory bus:

```rust
use ppc::{PpcCpu, PpcImportAction, PpcSectionMem};

fn run_guest(cpu: &mut PpcCpu, memory: &mut PpcSectionMem) {
    let _ = cpu.run_with_imports(
        memory,
        10_000,
        0,
        0x01f0_0000,
        256,
        |index, cpu, _memory| match index {
            0 => PpcImportAction::Return(cpu.gpr[3].wrapping_add(cpu.gpr[4])),
            _ => PpcImportAction::Halt,
        },
    );
}
```

Native PowerPC callbacks are supported through `PpcImportAction::CallNative`,
including RTOC restoration and configurable GPR3 finalization when the callback
returns.

## Choosing an API

| API | Purpose | Host-visible exits |
| :--- | :--- | :--- |
| `decode()` | Decode one instruction word without executing it | `PpcInstr` or `PpcDecodeError` |
| `step_instruction()` | Execute one non-memory instruction | Step, exception, unsupported decode, or unexpected memory access |
| `step()` | Execute one supplied instruction against a memory bus | Step, exception, unsupported decode, or memory fault |
| `run()` | Execute an instruction-budgeted guest stream | Halt, budget exhaustion, exception, unsupported decode, or fetch/data fault |
| `run_with_imports()` | Run with synthetic HLE import slots | Normal run exits plus handler-controlled returns, callbacks, and halts |
| `run_with_fetch_observer()` | Run while observing successful instruction fetches | Same as `run()` |
| `run_with_imports_and_observers()` | Combine imports with fetch and guest-write observation | Same as `run_with_imports()` |

## Architectural Scope

The crate models the 32-bit user instruction set used by classic PowerPC
software. The current implementation includes the common fixed-point,
branching, memory, condition-register, special-register, and IEEE-754
floating-point surface exercised by real applications.

The library deliberately does not provide:

- supervisor-mode exception vectoring or privileged system behavior;
- MMU or address translation;
- PEF, CFM, Mach-O, or firmware loading;
- Macintosh Toolbox, Memory Manager, QuickDraw, or other platform services.

Those facilities belong in the host. The bundled `ppc-inspect` and
`ppc-disasm-window` binaries contain small, diagnostic-only PEF readers for
examining application fragments; they are not part of the CPU library API.

## Command-Line Tools

Inspect a PowerPC PEF fragment and summarize its sections, imports, entry point,
and decoded instruction coverage:

```sh
cargo run --bin ppc-inspect -- application.pef
```

Disassemble a window around a guest PC:

```sh
cargo run --bin ppc-disasm-window -- application.pef 0x10001234 32
```

## Validation and Testing

The test suite covers:

- decoder fields and reserved-form rejection;
- integer, branch, condition-register, special-register, memory, atomic, and floating-point semantics;
- alignment, fetch, data, and architected exception behavior;
- read-only, writable, overlapping, cached, and cross-region memory accesses;
- normal, observed, and import-intercepted run loops;
- callback entry/return behavior and ABI state restoration;
- PEF inspection CLI fixtures captured from real application fragments.

Run the same local checks used by CI:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo doc --all-features --locked --no-deps --document-private-items
cargo test --all-features --locked
cargo package --locked
```

## Architecture

```text
ppc-rs/
├── src/
│   ├── decode.rs             # Instruction decoder and typed instruction forms
│   ├── lib.rs                # CPU state, execution, exceptions, observers, and HLE runner
│   ├── memory.rs             # Memory trait and multi-region implementation
│   └── bin/
│       ├── ppc-inspect.rs
│       └── ppc-disasm-window.rs
├── tests/                    # ISA, runner, public API, and CLI integration tests
└── fixtures/pef/             # Compact PEF inspection expectations
```

### Key Types

| Type | Description |
| :--- | :--- |
| `PpcCpu` | Complete host-visible CPU state and execution APIs |
| `PpcInstr` / `PpcDecodeError` | Typed decoded instruction and decode failure |
| `PpcMemory` | Big-endian guest memory contract |
| `PpcSectionMem` | Ready-made multi-region memory implementation |
| `PpcStepResult` / `PpcRunResult` | Single-step and run-loop outcomes |
| `PpcException` | Structured architected and host-import exceptions |
| `PpcImportAction` | HLE import return, callback, exception, and halt control |
| `PpcFetchObserver` / `PpcMemoryWriteObserver` | Optional execution instrumentation |
| `PpcFetchHistogram` | Aggregate fetched-PC and decoder-coverage statistics |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

Maintainers must configure the GitHub Actions publishing secret described in
the [release setup instructions](CONTRIBUTING.md#required-github-setup) before
merging the first release pull request.

## References

- *PowerPC User Instruction Set Architecture, Book I, Version 2.01*
- Apple Computer, *PowerPC Numerics* (1994)
- Apple Computer, *PowerPC System Software* (1994)
