//! Opt-in JIT diagnostics (`PPC_JIT_STATS=1`): region lifecycle and JIT
//! table replacement counters, printed to stderr every few seconds (counts
//! cover the interval since the previous summary). One flag check when off.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

const INTERVAL: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
pub(crate) enum Event {
    FormAttempt,
    FormRootStale,
    FormNoEdges,
    Formed,
    Installed,
    Declined,
    DroppedStale,
    Revalidated,
    /// A compiled block's JIT entry was taken by a block at another PC.
    EntryCollision,
    /// A compiled block's JIT entry was reset for the same PC (new code).
    EntrySamePc,
    /// A region was lost with its entry (either cause above).
    RegionLostWithEntry,
    /// A cache miss found the same words under a new token.
    TokenAdopted,
}

const EVENT_COUNT: usize = Event::TokenAdopted as usize + 1;
const MAX_REPORTED_ROOTS: usize = 12;

const NAMES: [&str; EVENT_COUNT] = [
    "form-attempt",
    "form-root-stale",
    "form-no-edges",
    "formed",
    "installed",
    "declined",
    "dropped-stale",
    "revalidated",
    "entry-collision",
    "entry-same-pc",
    "region-lost-with-entry",
    "token-adopted",
];

static COUNTS: [AtomicU64; EVENT_COUNT] = [const { AtomicU64::new(0) }; EVENT_COUNT];
static ROOTS: LazyLock<Mutex<HashMap<u32, u64>>> = LazyLock::new(Default::default);
static STALE_ROOTS: LazyLock<Mutex<HashMap<u32, u64>>> = LazyLock::new(Default::default);

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    let enabled = std::env::var("PPC_JIT_STATS").as_deref() == Ok("1");
    if enabled {
        eprintln!(
            "[jit-stats] enabled; summaries every {}s",
            INTERVAL.as_secs()
        );
        let _ = std::thread::Builder::new()
            .name("jit-stats".into())
            .spawn(|| {
                loop {
                    std::thread::sleep(INTERVAL);
                    print();
                }
            });
    }
    enabled
});

#[inline]
pub(crate) fn enabled() -> bool {
    *ENABLED
}

#[inline]
pub(crate) fn count(event: Event) {
    if enabled() {
        COUNTS[event as usize].fetch_add(1, Ordering::Relaxed);
    }
}

fn note(map: &Mutex<HashMap<u32, u64>>, pc: u32) {
    *map.lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .entry(pc)
        .or_default() += 1;
}

/// A region was formed (queued) at `pc`.
pub(crate) fn formed(pc: u32) {
    if enabled() {
        count(Event::Formed);
        note(&ROOTS, pc);
    }
}

/// A region could not form at `pc`: the block's words are gone.
pub(crate) fn root_stale(pc: u32) {
    if enabled() {
        count(Event::FormRootStale);
        note(&STALE_ROOTS, pc);
    }
}

fn print() {
    let counts: Vec<String> = NAMES
        .iter()
        .zip(&COUNTS)
        .map(|(name, count)| format!("{name}={}", count.swap(0, Ordering::Relaxed)))
        .collect();
    let mut out = format!("[jit-stats] {}\n", counts.join(" "));
    let top = |map: &Mutex<HashMap<u32, u64>>, min: u64| {
        let all = std::mem::take(&mut *map.lock().unwrap_or_else(|poison| poison.into_inner()));
        let mut top: Vec<(u32, u64)> = all.into_iter().filter(|&(_, n)| n >= min).collect();
        top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
        top.truncate(MAX_REPORTED_ROOTS);
        top
    };
    for (pc, n) in top(&ROOTS, 2) {
        out.push_str(&format!("[jit-stats]   formed again at {pc:08x}: {n}\n"));
    }
    for (pc, n) in top(&STALE_ROOTS, 1) {
        out.push_str(&format!("[jit-stats]   root stale at {pc:08x}: {n}\n"));
    }
    eprint!("{out}");
}
