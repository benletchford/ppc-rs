//! Exercise panic transport through actual compiled blocks and regions.
use super::{JitRequest, JitShared, PpcJitConfig, with_session};
use crate::{PpcCpu, PpcMemory};
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind, panic_any};

const CODE: u32 = 0x1000;
const DATA: u32 = 0x2000;

#[derive(Debug, PartialEq)]
struct HostPanic(u32);

struct PanickingMemory {
    epoch_panic: bool,
    accesses: Cell<u32>,
}

impl PanickingMemory {
    fn access(&self, addr: u32) {
        assert!(addr >= DATA);
        self.accesses.set(self.accesses.get() + 1);
        if !self.epoch_panic {
            panic_any(HostPanic(addr));
        }
    }
}

impl PpcMemory for PanickingMemory {
    fn read_u8(&mut self, addr: u32) -> Option<u8> {
        self.access(addr);
        Some(0)
    }
    fn write_u8(&mut self, addr: u32, _value: u8) -> Option<()> {
        self.access(addr);
        Some(())
    }
    fn read_u32_be(&mut self, addr: u32) -> Option<u32> {
        self.access(addr);
        Some(0)
    }
    fn write_u32_be(&mut self, addr: u32, _value: u32) -> Option<()> {
        self.access(addr);
        Some(())
    }
    fn code_epoch(&self) -> Option<u64> {
        if self.epoch_panic && self.accesses.get() != 0 {
            panic_any(HostPanic(0));
        }
        Some(1)
    }
}

fn check_panic(word: u32, region: bool, epoch_panic: bool) {
    let engine = JitShared::new(PpcJitConfig {
        threshold: 0,
        verify: true,
        jitdump: false,
        background: false,
        hot_threshold: 1,
    })
    .unwrap();
    // The preceding instruction must complete; the following instructions must
    // not run after a memory panic. An epoch panic happens at the block boundary.
    let words = vec![0x38e7_0001, word, 0x3908_0001, 0x4800_0004];
    let request = if region {
        engine.request_region(vec![
            (CODE, words, vec![1]),
            (CODE + 16, vec![0x3929_0001], vec![]),
        ])
    } else {
        engine.request(CODE, &words)
    };
    let JitRequest::Ready(Some(code)) = request else {
        panic!("synchronous compilation failed");
    };
    let mut cpu = PpcCpu::new();
    cpu.pc = CODE;
    cpu.gpr[5] = DATA;
    let mut mem = PanickingMemory {
        epoch_panic,
        accesses: Cell::new(0),
    };
    let payload = catch_unwind(AssertUnwindSafe(|| {
        with_session(&mut cpu, &mut mem, |session| {
            // SAFETY: the engine remains alive and code starts at cpu.pc.
            unsafe { session.run(code, 100, 1) }
        })
    }))
    .expect_err("host panic must reach the Rust caller");
    assert_eq!(
        payload.downcast_ref::<HostPanic>(),
        Some(&HostPanic(if epoch_panic { 0 } else { DATA }))
    );
    assert_eq!(mem.accesses.get(), 1, "callback must not be retried");
    assert_eq!(cpu.gpr[7], 1);
    assert_eq!(cpu.gpr[8], u32::from(epoch_panic));
    assert_eq!(cpu.gpr[9], 0, "region successor must not execute");
    assert_eq!(cpu.pc, if epoch_panic { CODE + 16 } else { CODE + 4 });
}

#[test]
fn memory_panics_propagate_from_blocks_and_regions_without_retry() {
    for region in [false, true] {
        // Integer and FP load/store helpers, plus interpreter fallback for
        // multiple-register transfers. r5 is the base and r3 is the operand.
        for op in [32, 34, 40, 42, 36, 38, 44, 48, 50, 52, 54, 46, 47] {
            let register = if op == 46 || op == 47 { 28 } else { 3 };
            check_panic((op << 26) | (register << 21) | (5 << 16), region, false);
        }
    }
}

#[test]
fn epoch_panic_exits_region_before_its_successor() {
    check_panic((32 << 26) | (3 << 21) | (5 << 16), true, true);
}
