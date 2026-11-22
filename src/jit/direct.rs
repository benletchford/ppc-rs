//! Direct guest-memory access for generated code.
//!
//! A small direct-mapped table maps guest pages to host pointers that loads
//! and stores may use directly, skipping the memory helpers. Entries come
//! only from [`PpcMemory::direct_read_page`] and [`PpcMemory::direct_write_page`]
//! and are valid while the matching epoch is unchanged. Each entry into
//! generated code from the run loop drops the entries whose epoch moved, and
//! the chaining loop checks the write entries again whenever the code epoch
//! has moved, the only change the memory may make while generated code runs
//! (see the trait documentation).

use crate::PpcMemory;

/// Guest page size covered by one entry: the 4 KiB pages of the
/// `PpcMemory::direct_*_page` methods.
pub(crate) const DIRECT_PAGE_SHIFT: u32 = 12;
/// Number of entries; a power of two.
pub(crate) const DIRECT_ENTRIES: usize = 256;

/// The table read by generated code. `tags[i]` is the page number plus one of
/// the page whose first byte is at `bases[i]`, or zero for an empty entry.
#[repr(C)]
#[derive(Clone)]
pub(crate) struct DirectTlb {
    pub(crate) tags: [u32; DIRECT_ENTRIES],
    pub(crate) bases: [usize; DIRECT_ENTRIES],
    /// Pages stores may write directly, laid out like `tags`/`bases`.
    pub(crate) write_tags: [u32; DIRECT_ENTRIES],
    pub(crate) write_bases: [usize; DIRECT_ENTRIES],
    /// Per slot, the page (plus one) last refused for direct reads or
    /// writes, so a page that is never offered costs one lookup per epoch.
    refused: [u32; DIRECT_ENTRIES],
    write_refused: [u32; DIRECT_ENTRIES],
    /// The epoch the read entries were filled under; `None` disables filling.
    epoch: Option<[u64; 2]>,
    /// The epoch the write entries were filled under; `None` disables filling.
    write_epoch: Option<[u64; 4]>,
}

impl Default for DirectTlb {
    fn default() -> Self {
        Self {
            tags: [0; DIRECT_ENTRIES],
            bases: [0; DIRECT_ENTRIES],
            write_tags: [0; DIRECT_ENTRIES],
            write_bases: [0; DIRECT_ENTRIES],
            refused: [0; DIRECT_ENTRIES],
            write_refused: [0; DIRECT_ENTRIES],
            epoch: None,
            write_epoch: None,
        }
    }
}

impl std::fmt::Debug for DirectTlb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectTlb")
            .field("epoch", &self.epoch)
            .field(
                "entries",
                &self.tags.iter().filter(|&&tag| tag != 0).count(),
            )
            .finish()
    }
}

impl DirectTlb {
    /// Keep the entries only if `mem` is still under the epoch they were
    /// filled under.
    pub(crate) fn validate<M: PpcMemory + ?Sized>(&mut self, mem: &M) {
        // Generated code byte-swaps guest values for a little-endian host.
        let epoch = mem
            .direct_read_epoch()
            .filter(|_| cfg!(target_endian = "little"));
        if epoch != self.epoch {
            self.tags = [0; DIRECT_ENTRIES];
            self.refused = [0; DIRECT_ENTRIES];
            self.epoch = epoch;
        }
        self.validate_writes(mem);
    }

    /// Keep the write entries only if `mem` is still under their epoch.
    pub(crate) fn validate_writes<M: PpcMemory + ?Sized>(&mut self, mem: &M) {
        let epoch = mem
            .direct_write_epoch()
            .filter(|_| cfg!(target_endian = "little"));
        if epoch != self.write_epoch {
            self.write_tags = [0; DIRECT_ENTRIES];
            self.write_refused = [0; DIRECT_ENTRIES];
            self.write_epoch = epoch;
        }
    }

    /// Record the page holding `addr` if `mem` offers it for direct writes.
    pub(crate) fn fill_write(&mut self, mem: &mut dyn PpcMemory, addr: u32) {
        if self.write_epoch.is_none() {
            return;
        }
        let page = addr >> DIRECT_PAGE_SHIFT;
        let slot = page as usize & (DIRECT_ENTRIES - 1);
        if self.write_tags[slot] == page + 1 || self.write_refused[slot] == page + 1 {
            return;
        }
        match mem.direct_write_page(page) {
            Some(base) => {
                self.write_tags[slot] = page + 1;
                self.write_bases[slot] = base as usize;
            }
            None => self.write_refused[slot] = page + 1,
        }
    }

    /// Record the page holding `addr` if `mem` offers it for direct reads.
    pub(crate) fn fill(&mut self, mem: &mut dyn PpcMemory, addr: u32) {
        if self.epoch.is_none() {
            return;
        }
        let page = addr >> DIRECT_PAGE_SHIFT;
        let slot = page as usize & (DIRECT_ENTRIES - 1);
        if self.tags[slot] == page + 1 || self.refused[slot] == page + 1 {
            return;
        }
        match mem.direct_read_page(page) {
            Some(base) => {
                self.tags[slot] = page + 1;
                self.bases[slot] = base as usize;
            }
            None => self.refused[slot] = page + 1,
        }
    }

    #[inline]
    fn read_page_base<M: PpcMemory + ?Sized>(&mut self, mem: &mut M, page: u32) -> Option<usize> {
        let slot = page as usize & (DIRECT_ENTRIES - 1);
        let tag = page + 1;
        if self.tags[slot] == tag {
            return Some(self.bases[slot]);
        }
        if self.epoch.is_none() || self.refused[slot] == tag {
            return None;
        }
        match mem.direct_read_page(page) {
            Some(base) => {
                let base = base as usize;
                self.tags[slot] = tag;
                self.bases[slot] = base;
                Some(base)
            }
            None => {
                self.refused[slot] = tag;
                None
            }
        }
    }

    #[inline]
    fn write_page_base<M: PpcMemory + ?Sized>(&mut self, mem: &mut M, page: u32) -> Option<usize> {
        let slot = page as usize & (DIRECT_ENTRIES - 1);
        let tag = page + 1;
        if self.write_tags[slot] == tag {
            return Some(self.write_bases[slot]);
        }
        if self.write_epoch.is_none() || self.write_refused[slot] == tag {
            return None;
        }
        match mem.direct_write_page(page) {
            Some(base) => {
                let base = base as usize;
                self.write_tags[slot] = tag;
                self.write_bases[slot] = base;
                Some(base)
            }
            None => {
                self.write_refused[slot] = tag;
                None
            }
        }
    }

    /// Read a big-endian 32-bit word via a validated direct-read page when the
    /// word fits inside one 4 KiB page, falling back to `mem.read_u32_be`.
    ///
    /// # Safety
    /// [`Self::validate`] must have been called for `mem` since the last host
    /// callback that could change `mem.direct_read_epoch()`.
    #[inline]
    pub(crate) unsafe fn read_u32_be<M: PpcMemory + ?Sized>(
        &mut self,
        mem: &mut M,
        addr: u32,
    ) -> Option<u32> {
        let offset = (addr as usize) & ((1usize << DIRECT_PAGE_SHIFT) - 1);
        if offset <= (1usize << DIRECT_PAGE_SHIFT) - 4
            && let Some(base) = self.read_page_base(mem, addr >> DIRECT_PAGE_SHIFT)
        {
            let ptr = (base + offset) as *const u32;
            // SAFETY: `validate` confirmed the epoch; `offset + 4 <= 4096` keeps
            // all four bytes inside the offered 4 KiB page.
            return Some(u32::from_be(unsafe { ptr.read_unaligned() }));
        }
        mem.read_u32_be(addr)
    }

    /// Read two consecutive big-endian 32-bit words (`addr` and `addr + 4`),
    /// sharing one direct-page lookup when all eight bytes fit in one page.
    ///
    /// # Safety
    /// As for [`Self::read_u32_be`].
    #[inline]
    pub(crate) unsafe fn read_u32_pair_be<M: PpcMemory + ?Sized>(
        &mut self,
        mem: &mut M,
        addr: u32,
    ) -> (Option<u32>, Option<u32>) {
        let offset = (addr as usize) & ((1usize << DIRECT_PAGE_SHIFT) - 1);
        if offset <= (1usize << DIRECT_PAGE_SHIFT) - 8
            && let Some(base) = self.read_page_base(mem, addr >> DIRECT_PAGE_SHIFT)
        {
            let ptr = (base + offset) as *const u64;
            // SAFETY: `validate` confirmed the epoch; `offset + 8 <= 4096` keeps
            // all eight bytes inside the offered 4 KiB page.
            let pair = u64::from_be(unsafe { ptr.read_unaligned() });
            return (Some((pair >> 32) as u32), Some(pair as u32));
        }
        // SAFETY: upheld by caller.
        unsafe {
            (
                self.read_u32_be(mem, addr),
                self.read_u32_be(mem, addr.wrapping_add(4)),
            )
        }
    }

    /// Write a big-endian 32-bit word via a validated direct-write page when
    /// the word fits inside one 4 KiB page, falling back to `mem.write_u32_be`.
    ///
    /// # Safety
    /// [`Self::validate`] must have been called for `mem` since the last host
    /// callback that could change `mem.direct_write_epoch()`.
    #[inline]
    pub(crate) unsafe fn write_u32_be<M: PpcMemory + ?Sized>(
        &mut self,
        mem: &mut M,
        addr: u32,
        value: u32,
    ) -> Option<()> {
        let offset = (addr as usize) & ((1usize << DIRECT_PAGE_SHIFT) - 1);
        if offset <= (1usize << DIRECT_PAGE_SHIFT) - 4
            && let Some(base) = self.write_page_base(mem, addr >> DIRECT_PAGE_SHIFT)
        {
            let ptr = (base + offset) as *mut u32;
            // SAFETY: `validate` confirmed the write epoch; `offset + 4 <= 4096`
            // keeps all four bytes inside the offered writable 4 KiB page.
            unsafe { ptr.write_unaligned(value.to_be()) };
            return Some(());
        }
        mem.write_u32_be(addr, value)
    }
}

/// The table as `PpcCpu` holds it. Generated code and helpers reach it
/// through a raw pointer that stays valid across sessions, so the CPU never
/// forms a `&mut` to it while one may be in use.
#[derive(Default)]
pub(crate) struct DirectTable(Box<std::cell::UnsafeCell<DirectTlb>>);

// SAFETY: the table is only written through `&mut PpcCpu` (a session, or a
// chaining decision inside one); shared references only read it (`Clone`,
// `Debug`), so sharing a `&PpcCpu` across threads cannot race.
unsafe impl Sync for DirectTable {}

impl DirectTable {
    pub(crate) fn get(&self) -> *mut DirectTlb {
        self.0.get()
    }
}

impl Clone for DirectTable {
    fn clone(&self) -> Self {
        // SAFETY: no generated code runs while the CPU is being cloned.
        Self(Box::new(std::cell::UnsafeCell::new(unsafe {
            (*self.get()).clone()
        })))
    }
}

impl std::fmt::Debug for DirectTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // SAFETY: as for `clone`.
        unsafe { (*self.get()).fmt(f) }
    }
}
