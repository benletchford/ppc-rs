//! Minimal `perf`/`samply` jitdump writer.
//!
//! Each compiled block is described to the profiler together with a mapping
//! from native code offsets to a generated `.ppc` listing, so guest
//! instructions show up as "source lines" in the profiler's assembly view.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::decode;

static NEXT_NAME: AtomicU64 = AtomicU64::new(0);

pub(super) fn block_name(pc: u32) -> String {
    let index = NEXT_NAME.fetch_add(1, Ordering::Relaxed);
    format!("ppc_{pc:08x}_{index}")
}

/// Name of a hot region entered at `pc` spanning `blocks` blocks.
pub(super) fn region_name(pc: u32, blocks: usize) -> String {
    let index = NEXT_NAME.fetch_add(1, Ordering::Relaxed);
    format!("IonIC: ppc_hot_{pc:08x}_{blocks}b_{index}")
}

#[cfg(target_os = "macos")]
mod sys {
    unsafe extern "C" {
        fn clock_gettime_nsec_np(clock_id: u32) -> u64;
        fn pthread_threadid_np(thread: usize, id: *mut u64) -> i32;
    }
    pub fn clock_ns() -> u64 {
        const CLOCK_UPTIME_RAW: u32 = 8;
        // SAFETY: plain libc call.
        unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
    }
    pub fn thread_id() -> u32 {
        let mut id = 0u64;
        // SAFETY: `id` is a valid out-pointer; thread 0 means "current".
        unsafe { pthread_threadid_np(0, &mut id) };
        id as u32
    }
}

#[cfg(target_os = "linux")]
mod sys {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock_id: i32, out: *mut Timespec) -> i32;
        fn gettid() -> i32;
    }
    pub fn clock_ns() -> u64 {
        const CLOCK_MONOTONIC: i32 = 1;
        let mut ts = Timespec { sec: 0, nsec: 0 };
        // SAFETY: `ts` is a valid out-pointer.
        unsafe { clock_gettime(CLOCK_MONOTONIC, &mut ts) };
        ts.sec as u64 * 1_000_000_000 + ts.nsec as u64
    }
    pub fn thread_id() -> u32 {
        // SAFETY: plain libc call.
        unsafe { gettid() as u32 }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod sys {
    pub fn clock_ns() -> u64 {
        0
    }
    pub fn thread_id() -> u32 {
        std::process::id()
    }
}

pub(super) struct Writer {
    pid: u32,
    file: File,
    source: File,
    source_path: String,
    next_line: u32,
    next_index: u64,
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

impl Writer {
    pub(super) fn new(tier: &str) -> Option<Self> {
        match Self::create(tier) {
            Ok(writer) => Some(writer),
            Err(error) => {
                eprintln!("ppc jit: jitdump disabled: {error}");
                None
            }
        }
    }

    fn create(tier: &str) -> io::Result<Self> {
        let arch = if cfg!(target_arch = "aarch64") {
            183
        } else if cfg!(target_arch = "x86_64") {
            62
        } else {
            return Err(io::Error::other("unsupported host architecture"));
        };
        let pid = std::process::id();
        let timestamp = sys::clock_ns();
        let path = std::env::temp_dir().join(format!("jit-{pid}-{timestamp}-{tier}.dump"));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let source_path = path.with_extension("ppc");
        let source = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&source_path)?;
        let mut header = Vec::with_capacity(40);
        for value in [0x4a69_5444, 1, 40, arch, 0, pid] {
            push_u32(&mut header, value);
        }
        push_u64(&mut header, timestamp);
        push_u64(&mut header, 0);
        file.write_all(&header)?;
        eprintln!("ppc jit: jitdump {}", path.display());
        Ok(Self {
            pid,
            file,
            source,
            source_path: source_path.to_string_lossy().into_owned(),
            next_line: 1,
            next_index: 0,
        })
    }

    pub(super) fn record(
        &mut self,
        name: &str,
        instrs: &[(u32, u32)],
        locations: &[(u32, u32)],
        code: &[u8],
    ) {
        if let Err(error) = self.try_record(name, instrs, locations, code) {
            eprintln!("ppc jit: jitdump write failed: {error}");
        }
    }

    fn try_record(
        &mut self,
        name: &str,
        instrs: &[(u32, u32)],
        locations: &[(u32, u32)],
        code: &[u8],
    ) -> io::Result<()> {
        let timestamp = sys::clock_ns();
        let address = code.as_ptr() as u64;
        // Line 0 of each function is its name; instruction `i` is line `i + 1`
        // (`instrs` lists `(pc, word)` in emission order).
        let mut lines = vec![self.next_line];
        let mut listing = format!("{name}\n");
        self.next_line += 1;
        for &(at, word) in instrs {
            lines.push(self.next_line);
            self.next_line += 1;
            match decode(word) {
                Ok(instr) => listing.push_str(&format!("{at:08x} {word:08x} {instr:?}\n")),
                Err(_) => listing.push_str(&format!("{at:08x} {word:08x} <undecoded>\n")),
            }
        }
        self.source.write_all(listing.as_bytes())?;

        let debug_size = 32 + locations.len() * (17 + self.source_path.len());
        let mut debug = Vec::with_capacity(debug_size);
        push_u32(&mut debug, 2);
        push_u32(&mut debug, debug_size as u32);
        push_u64(&mut debug, timestamp);
        push_u64(&mut debug, address);
        push_u64(&mut debug, locations.len() as u64);
        for &(offset, index) in locations {
            push_u64(&mut debug, address + u64::from(offset));
            push_u32(&mut debug, lines[index as usize]);
            push_u32(&mut debug, 0);
            debug.extend_from_slice(self.source_path.as_bytes());
            debug.push(0);
        }
        self.file.write_all(&debug)?;

        let total = 56 + name.len() + 1 + code.len();
        let mut record = Vec::with_capacity(total);
        push_u32(&mut record, 0);
        push_u32(&mut record, total as u32);
        push_u64(&mut record, timestamp);
        push_u32(&mut record, self.pid);
        push_u32(&mut record, sys::thread_id());
        push_u64(&mut record, address);
        push_u64(&mut record, address);
        push_u64(&mut record, code.len() as u64);
        push_u64(&mut record, self.next_index);
        record.extend_from_slice(name.as_bytes());
        record.push(0);
        record.extend_from_slice(code);
        self.file.write_all(&record)?;
        self.next_index += 1;
        Ok(())
    }
}
