//! CPU-side JIT state and dispatch.
use crate::{PpcBasicBlockRunResult, PpcCpu, PpcJitConfig, PpcJitStats, PpcMemory, jit};

/// Run-loop boundaries a chain of native blocks must respect.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PpcJitLimits {
    pub(crate) max_cycles: u64,
    pub(crate) halt_pc: u32,
    pub(crate) trap_base: u32,
    pub(crate) import_count: u32,
    pub(crate) chain: bool,
}

/// JIT state for the block starting at `start_pc`, in a table of its own
/// indexed by PC (see [`PpcCpu::jit_table_index`]), so interpreter cache
/// replacements never discard compiled code, links or regions.
///
/// The entry describes the `len` instructions at `start_pc` as they were
/// under `memory_token`; it is current while memory still hands out that
/// token for `start_pc`.
#[derive(Debug, Clone, Default)]
pub(crate) struct PpcJitEntry {
    pub(crate) start_pc: u32,
    /// Instructions; `0` for an empty entry.
    pub(crate) len: u32,
    pub(crate) memory_token: u64,
    /// [`PpcMemory::code_epoch`] under which `memory_token` last matched
    /// (`0`: unknown).
    epoch: u64,
    interpreter_runs: u32,
    /// The block's own native code: `0` if not compiled, `usize::MAX` if
    /// declined.
    block_code: usize,
    /// Background compilation of the block in flight.
    ticket: Option<jit::JitTicket>,
    /// The block's words once compiled (as they were under `memory_token`),
    /// to form regions and check the block against memory without the
    /// interpreter cache.
    words: Option<Box<[u32]>>,
    /// Hot region entered at this block, if one was formed.
    region: Option<Box<PpcJitRegion>>,
    /// What chaining enters here: the region's code if it has a compiled
    /// one, else `block_code`; `0` when not chainable.
    code: usize,
    /// Guest code `code` may execute: `[lo, hi)`. Just the block, unless
    /// `code` is a region.
    lo: u32,
    hi: u32,
    /// Nonzero when `code` is a region: the only code epoch it is valid in.
    region_epoch: u64,
    /// Recent successors of the block, newest first.
    links: [PpcJitLink; 2],
    /// Observed standalone outgoing edges, independent of cached native links.
    edges: [(u32, u32); 2],
    /// The block was taken into a region, so it needs more standalone runs
    /// before it gets one of its own.
    covered: bool,
}

/// JIT table size. Indexed linearly by word address, so contiguous code
/// never collides with itself; the span (512 KiB of code) keeps routines
/// duplicated at power-of-two distances up to 256 KiB apart separate.
const PPC_JIT_TABLE_ENTRIES: usize = 1 << 17;

/// "The block at `pc` is `code` (`len` instructions)", proven by a token
/// check under code epoch `epoch` and therefore true while the epoch is
/// unchanged, whatever later happens to this slot or the target's.
#[derive(Debug, Clone, Copy, Default)]
struct PpcJitLink {
    pc: u32,
    len: u32,
    code: usize,
    epoch: u64,
    /// The target entry's `lo`/`hi`.
    lo: u32,
    hi: u32,
    /// [`PpcCpu::jit_link_gen`] when made; bumped to forget every link.
    generation: u32,
}

/// A hot region entered at a block, valid in one code epoch.
#[derive(Debug, Clone)]
struct PpcJitRegion {
    /// Entry point; `0` while compiling, `usize::MAX` if declined.
    code: usize,
    ticket: Option<jit::JitTicket>,
    epoch: u64,
    /// Bounds of `spans`.
    lo: u32,
    hi: u32,
    /// `[start, end)` of each block in the region.
    spans: Vec<(u32, u32)>,
    /// `(start, memory token, words)` of each block, to carry the region
    /// over to a new code epoch when none of its code changed.
    blocks: Vec<(u32, u64, Vec<u32>)>,
}

/// Region size limits.
const PPC_JIT_REGION_MAX_BLOCKS: usize = 32;
const PPC_JIT_REGION_MAX_INSTRUCTIONS: usize = 512;
/// A block already taken into a region needs this many times the hot
/// threshold in standalone runs to start a region of its own.
const PPC_JIT_COVERED_HOT_FACTOR: u32 = 8;

impl PpcCpu {
    /// Enable the Cranelift baseline JIT for cached basic blocks.
    ///
    /// Hot blocks are compiled to native code; behaviour is otherwise
    /// identical to the interpreter. Returns an error if the host cannot
    /// generate code.
    pub fn enable_jit(&mut self, config: PpcJitConfig) -> Result<(), String> {
        jit::diag::enabled();
        self.jit = Some(std::sync::Arc::new(jit::JitShared::new(config)?));
        self.jit_pending_regions.clear();
        self.jit_table = std::iter::repeat_with(PpcJitEntry::default)
            .take(PPC_JIT_TABLE_ENTRIES)
            .collect();
        Ok(())
    }

    /// Disable the JIT. Compiled code is released when the last engine
    /// handle is dropped.
    pub fn disable_jit(&mut self) {
        self.jit = None;
        self.jit_pending_regions.clear();
        self.jit_table = Box::default();
    }

    /// JIT activity counters for this CPU.
    pub fn jit_stats(&self) -> PpcJitStats {
        self.jit_stats
    }

    #[inline]
    pub(crate) fn jit_table_index(pc: u32) -> usize {
        ((pc >> 2) as usize) & (PPC_JIT_TABLE_ENTRIES - 1)
    }

    /// Run a hot cached block natively, or return `None` to interpret it.
    /// Chaining rechecks run-loop boundaries between blocks. Regions check
    /// halt/import/return PCs over their spans on entry, and budget and code
    /// validity between blocks.
    pub(crate) fn run_jit_block<M: PpcMemory + ?Sized>(
        &mut self,
        mem: &mut M,
        cache_index: usize,
        block_len: usize,
        limits: PpcJitLimits,
    ) -> Option<PpcBasicBlockRunResult> {
        let jit = self.jit.as_ref()?;
        if !self.jit_pending_regions.is_empty() {
            let finished = jit.finished();
            if finished != self.jit_finished_seen {
                self.jit_finished_seen = finished;
                self.poll_jit_regions();
            }
        }
        let jit = self.jit.as_ref()?;
        let hot_threshold = jit.config.hot_threshold;
        let entry = self.basic_block_cache[cache_index].as_ref()?;
        if entry.word_copy.is_some() {
            return None;
        }
        // The caller validated this entry's token just now.
        let (start_pc, memory_token) = (entry.start_pc, entry.memory_token);
        let index = Self::jit_table_index(start_pc);
        let slot = self.jit_table.get_mut(index)?;
        if slot.start_pc != start_pc
            || slot.len != block_len as u32
            || slot.memory_token != memory_token
        {
            if jit::diag::enabled() && slot.block_code != 0 && slot.block_code != jit::JIT_DECLINED
            {
                jit::diag::count(if slot.start_pc == start_pc {
                    jit::diag::Event::EntrySamePc
                } else {
                    jit::diag::Event::EntryCollision
                });
                if slot.region.is_some() {
                    jit::diag::count(jit::diag::Event::RegionLostWithEntry);
                }
            }
            *slot = PpcJitEntry {
                start_pc,
                len: block_len as u32,
                memory_token,
                ..PpcJitEntry::default()
            };
        }
        if slot.block_code == jit::JIT_DECLINED {
            return None;
        }
        if slot.block_code == 0 {
            let ready = if let Some(ticket) = &slot.ticket {
                let ready = jit::poll_ticket(ticket)?;
                slot.ticket = None;
                ready
            } else if slot.interpreter_runs < jit.config.threshold {
                slot.interpreter_runs += 1;
                return None;
            } else {
                match jit.request(start_pc, &entry.words[..block_len]) {
                    jit::JitRequest::Ready(ready) => ready,
                    jit::JitRequest::Pending(ticket) => {
                        slot.ticket = Some(ticket);
                        return None;
                    }
                }
            };
            match ready {
                Some(code) => {
                    slot.block_code = code;
                    slot.words = Some(entry.words[..block_len].into());
                    self.jit_stats.blocks_compiled += 1;
                }
                None => {
                    slot.block_code = jit::JIT_DECLINED;
                    self.jit_stats.blocks_rejected += 1;
                    return None;
                }
            }
        }
        let return_pc = self.import_call_stack.last().map(|frame| frame.return_pc);
        let block_end = start_pc.wrapping_add(block_len as u32 * 4);
        let epoch = mem.code_epoch().unwrap_or(0);
        let mut first = slot.block_code;
        slot.epoch = epoch;
        if let Some(region) = slot.region.as_deref_mut()
            && region.epoch != epoch
        {
            if Self::jit_region_current(mem, region, epoch) {
                jit::diag::count(jit::diag::Event::Revalidated);
            } else {
                jit::diag::count(jit::diag::Event::DroppedStale);
                // Formed under older code; form it again once hot.
                slot.region = None;
                slot.edges = [(0, 0); 2];
                slot.covered = false;
            }
        }
        (slot.code, slot.lo, slot.hi, slot.region_epoch) = (first, start_pc, block_end, 0);
        if let Some(region) = slot.region.as_deref()
            && region.code != 0
            && region.code != jit::JIT_DECLINED
        {
            (slot.code, slot.lo, slot.hi, slot.region_epoch) =
                (region.code, region.lo, region.hi, region.epoch);
            if limits.chain && Self::jit_region_clear(region, &limits, return_pc) {
                first = region.code;
            }
        }
        let real_mem: *mut M = mem;
        let (blocks, instructions, result) = jit::with_session(self, mem, |session| {
            let mut code = first;
            let mut epoch = epoch;
            let mut from = index;
            let mut total = 0u64;
            let mut blocks = 0u64;
            loop {
                // SAFETY: `code` was produced by the engine held in `cpu.jit`
                // for exactly the block (or region) starting at `cpu.pc`,
                // valid in code epoch `epoch`.
                let outcome = unsafe { session.run(code, limits.max_cycles - total, epoch) };
                blocks += 1;
                match outcome {
                    Ok(completed) => total += completed,
                    Err((pc, completed, result)) => {
                        total += completed;
                        return (
                            blocks,
                            total,
                            PpcBasicBlockRunResult::Stop {
                                pc,
                                completed: total,
                                result,
                            },
                        );
                    }
                }
                if !limits.chain || total >= limits.max_cycles {
                    break;
                }
                // SAFETY: `real_mem` is the memory the session's adapter wraps;
                // no native code runs (so nothing else touches it) during the
                // call, and using it directly avoids a dyn hop per block.
                match session.cpu().next_chain_block(
                    unsafe { &mut *real_mem },
                    from,
                    total,
                    &limits,
                    return_pc,
                    hot_threshold,
                ) {
                    Some((next, index, now)) => {
                        code = next;
                        from = index;
                        epoch = now;
                    }
                    None => break,
                }
            }
            (blocks, total, PpcBasicBlockRunResult::Advanced(total))
        });
        self.jit_stats.block_runs += blocks;
        self.jit_stats.instructions += instructions;
        Some(result)
    }

    /// Edge observations required for promotion (`0` disables promotion).
    #[inline]
    fn jit_hot_trigger(covered: bool, threshold: u32) -> u32 {
        if threshold == u32::MAX {
            0
        } else if covered {
            threshold.saturating_mul(PPC_JIT_COVERED_HOT_FACTOR).max(1)
        } else {
            threshold.max(1)
        }
    }

    /// Whether a region may run without the outer loop seeing the halt PC,
    /// an import trap or a native-return PC inside any of its blocks.
    fn jit_region_clear(
        region: &PpcJitRegion,
        limits: &PpcJitLimits,
        return_pc: Option<u32>,
    ) -> bool {
        Self::jit_span_clear(region.lo, region.hi, limits, return_pc)
            || region
                .spans
                .iter()
                .all(|&(lo, hi)| Self::jit_span_clear(lo, hi, limits, return_pc))
    }

    /// Whether native code covering guest code `[lo, hi)` may run without
    /// the outer loop seeing the halt PC, an import trap or a native-return
    /// PC inside it (their PCs do not change while native code runs).
    #[inline]
    fn jit_span_clear(lo: u32, hi: u32, limits: &PpcJitLimits, return_pc: Option<u32>) -> bool {
        let inside =
            |target: u32| (target & 3) == 0 && target.wrapping_sub(lo) < hi.wrapping_sub(lo);
        let trap_end = limits
            .trap_base
            .wrapping_add(limits.import_count.wrapping_mul(4));
        let traps = limits.import_count > 0 && limits.trap_base < hi && trap_end > lo;
        !(inside(limits.halt_pc) || traps || return_pc.is_some_and(inside))
    }

    /// Native entry point, JIT table index and current code epoch for the
    /// block at `self.pc` if the outer run loop would dispatch it to the JIT
    /// as well. Mirrors the checks the loop and `run_cached_basic_block` make
    /// between blocks. `from` is the entry of the block that just ran, whose
    /// links usually name the successor without touching its entry or asking
    /// the memory for a token.
    #[inline]
    fn next_chain_block<M: PpcMemory + ?Sized>(
        &mut self,
        mem: &mut M,
        from: usize,
        total: u64,
        limits: &PpcJitLimits,
        return_pc: Option<u32>,
        hot_threshold: u32,
    ) -> Option<(usize, usize, u64)> {
        let pc = self.pc;
        let trap_bytes = u64::from(limits.import_count) * 4;
        if pc & 3 != 0
            || pc == limits.halt_pc
            || u64::from(pc.wrapping_sub(limits.trap_base)) < trap_bytes
        {
            return None;
        }
        let epoch = mem.code_epoch().unwrap_or(0);
        // A region's entry is not necessarily its exiting block. Do not
        // attribute its outgoing transition to the entry.
        if self.jit_table[from].region_epoch == 0 {
            let slot = &mut self.jit_table[from];
            let edge = if let Some(edge) = slot
                .edges
                .iter_mut()
                .find(|edge| edge.1 != 0 && edge.0 == pc)
            {
                edge
            } else {
                let replace = usize::from(slot.edges[1].1 < slot.edges[0].1);
                slot.edges[replace] = (pc, 0);
                &mut slot.edges[replace]
            };
            edge.1 = edge.1.saturating_add(1);
        }
        let index = Self::jit_table_index(pc);
        let links = self.jit_table[from].links;
        let generation = self.jit_link_gen;
        let fresh = |link: &PpcJitLink| {
            epoch != 0 && link.pc == pc && link.epoch == epoch && link.generation == generation
        };
        let link = if fresh(&links[0]) {
            links[0]
        } else if fresh(&links[1]) {
            links[1]
        } else {
            let slot = &self.jit_table[index];
            if slot.start_pc != pc || slot.code == 0 {
                return None;
            }
            if epoch == 0 || slot.epoch != epoch {
                let token = slot.memory_token;
                if mem.global_instruction_cache_token() != Some(token)
                    && mem.instruction_cache_token(pc) != Some(token)
                    && !self.refresh_jit_token(mem, index, pc)
                {
                    return None;
                }
                self.jit_table[index].epoch = epoch;
            }
            let slot = &self.jit_table[index];
            // A region from an older epoch; the outer loop checks it again.
            if slot.region_epoch != 0 && slot.region_epoch != epoch {
                return None;
            }
            let link = PpcJitLink {
                pc,
                len: slot.len,
                code: slot.code,
                epoch,
                lo: slot.lo,
                hi: slot.hi,
                generation,
            };
            if epoch != 0 {
                let links = &mut self.jit_table[from].links;
                links[1] = links[0];
                links[0] = link;
            }
            link
        };
        if limits.max_cycles - total < u64::from(link.len) {
            return None;
        }
        // `[lo, hi)` is exact for a block and bounds a region's blocks.
        if !Self::jit_span_clear(link.lo, link.hi, limits, return_pc)
            && !self.jit_table[index]
                .region
                .as_deref()
                .is_some_and(|region| {
                    region.code == link.code && Self::jit_region_clear(region, limits, return_pc)
                })
        {
            return None;
        }
        let slot = &self.jit_table[from];
        if slot.region_epoch == 0
            && slot.region.is_none()
            && slot.edges.iter().any(|&(target, count)| {
                target == pc
                    && count >= Self::jit_hot_trigger(slot.covered, hot_threshold)
                    && hot_threshold != u32::MAX
            })
        {
            self.form_jit_region(mem, from, epoch);
        }
        Some((link.code, index, epoch))
    }

    /// The block at `pc` in JIT entry `index` carries an older token (a
    /// write elsewhere in its page renewed it): if memory still holds the
    /// words it was built from, move it (and its interpreter cache entry, if
    /// that has the same words) to the current token.
    #[cold]
    #[inline(never)]
    fn refresh_jit_token<M: PpcMemory + ?Sized>(
        &mut self,
        mem: &mut M,
        index: usize,
        pc: u32,
    ) -> bool {
        let Some(now) = mem.instruction_cache_token(pc) else {
            return false;
        };
        let slot = &self.jit_table[index];
        let Some(words) = slot.words.as_deref() else {
            return false;
        };
        let same = words.iter().enumerate().all(|(offset, &word)| {
            let at = pc.wrapping_add(offset as u32 * 4);
            mem.instruction_cache_token(at) == Some(now)
                && mem.read_instruction_u32_be(at) == Some(word)
        });
        if !same {
            return false;
        }
        let (old, len) = (slot.memory_token, slot.len);
        self.jit_table[index].memory_token = now;
        // Same token, so built from the same words.
        if let Some(entry) = self.basic_block_cache[Self::basic_block_cache_index(pc)].as_mut()
            && entry.start_pc == pc
            && u32::from(entry.len) == len
            && entry.memory_token == old
        {
            entry.memory_token = now;
        }
        true
    }

    /// Queue a region along the dominant successor path, validating each
    /// block against memory. Retry later if no useful path exists.
    #[cold]
    #[inline(never)]
    fn form_jit_region<M: PpcMemory + ?Sized>(&mut self, mem: &mut M, index: usize, epoch: u64) {
        let Some(jit) = self.jit.clone() else {
            return;
        };
        jit::diag::count(jit::diag::Event::FormAttempt);
        let root = &self.jit_table[index];
        if epoch == 0 || root.region.is_some() || root.len == 0 || root.code == 0 {
            self.jit_table[index].edges = [(0, 0); 2];
            return;
        }
        // The current token and words of the compiled block in entry `t`, if
        // memory still holds those words.
        let current = |cpu: &Self, mem: &mut M, t: usize| {
            let slot = &cpu.jit_table[t];
            let words = slot.words.as_deref()?;
            let token = Self::jit_words_current(mem, slot.start_pc, slot.memory_token, words)?;
            Some((token, words.to_vec()))
        };
        let (root_pc, root_len) = (root.start_pc, root.len);
        let Some(root_block) = current(self, mem, index) else {
            jit::diag::root_stale(root_pc);
            self.jit_table[index].edges = [(0, 0); 2];
            return;
        };
        // (table index, pc, token, words)
        let mut blocks = vec![(index, root_pc, root_block.0, root_block.1)];
        let mut instructions = root_len as usize;
        let mut next = 0;
        while next < blocks.len() {
            let slot = &self.jit_table[blocks[next].0];
            let dominant = slot
                .edges
                .iter()
                .filter(|edge| edge.1 > 0)
                .max_by_key(|edge| edge.1);
            for link in slot
                .links
                .into_iter()
                .filter(|link| dominant.is_some_and(|edge| edge.0 == link.pc))
            {
                if link.len == 0 {
                    continue;
                }
                let t = Self::jit_table_index(link.pc);
                let target = &self.jit_table[t];
                if target.start_pc != link.pc
                    || target.len != link.len
                    || blocks.iter().any(|block| block.0 == t)
                    || blocks.len() >= PPC_JIT_REGION_MAX_BLOCKS
                    || instructions + link.len as usize > PPC_JIT_REGION_MAX_INSTRUCTIONS
                {
                    continue;
                }
                let Some((token, words)) = current(self, mem, t) else {
                    continue;
                };
                blocks.push((t, link.pc, token, words));
                instructions += link.len as usize;
            }
            next += 1;
        }
        let mut spec: jit::RegionSpec = Vec::with_capacity(blocks.len());
        let (mut lo, mut hi) = (u32::MAX, 0u32);
        let mut spans = Vec::with_capacity(blocks.len());
        let mut region_blocks = Vec::with_capacity(blocks.len());
        let mut edges = 0;
        for (t, pc, token, words) in &blocks {
            let mut successors: Vec<usize> = Vec::new();
            for link in self.jit_table[*t].links {
                if link.len == 0 {
                    continue;
                }
                let position = blocks
                    .iter()
                    .position(|block| block.1 == link.pc && block.3.len() == link.len as usize);
                if let Some(position) = position
                    && !successors.contains(&position)
                {
                    successors.push(position);
                }
            }
            edges += successors.len();
            let end = pc.saturating_add(words.len() as u32 * 4);
            lo = lo.min(*pc);
            hi = hi.max(end);
            spans.push((*pc, end));
            region_blocks.push((*pc, *token, words.clone()));
            spec.push((*pc, words.clone(), successors));
        }
        if edges == 0 {
            jit::diag::count(jit::diag::Event::FormNoEdges);
            self.jit_table[index].edges = [(0, 0); 2];
            return;
        }
        jit::diag::formed(root_pc);
        for (t, ..) in &blocks {
            let slot = &mut self.jit_table[*t];
            slot.covered = true;
            slot.edges = [(0, 0); 2];
        }
        let (code, ticket) = match jit.request_region(spec) {
            jit::JitRequest::Ready(ready) => (ready.unwrap_or(jit::JIT_DECLINED), None),
            jit::JitRequest::Pending(ticket) => (0, Some(ticket)),
        };
        let pending = ticket.is_some();
        self.jit_table[index].region = Some(Box::new(PpcJitRegion {
            code,
            ticket,
            epoch,
            lo,
            hi,
            spans,
            blocks: region_blocks,
        }));
        if pending {
            self.jit_pending_regions.push(index);
        } else if code != jit::JIT_DECLINED {
            self.install_jit_region(index);
        }
    }

    /// Revalidate every block and adopt `epoch` if the region is unchanged.
    #[cold]
    #[inline(never)]
    fn jit_region_current<M: PpcMemory + ?Sized>(
        mem: &mut M,
        region: &mut PpcJitRegion,
        epoch: u64,
    ) -> bool {
        if epoch == 0 {
            return false;
        }
        for (pc, token, words) in region.blocks.iter_mut() {
            match Self::jit_words_current(mem, *pc, *token, words) {
                Some(now) => *token = now,
                None => return false,
            }
        }
        region.epoch = epoch;
        true
    }

    /// Return the current token if the token or instruction words still match.
    fn jit_words_current<M: PpcMemory + ?Sized>(
        mem: &mut M,
        pc: u32,
        token: u64,
        words: &[u32],
    ) -> Option<u64> {
        let global = mem.global_instruction_cache_token();
        if global == Some(token) {
            return Some(token);
        }
        let page = mem.instruction_cache_token(pc);
        if page == Some(token) {
            return Some(token);
        }
        let now = page.or(global)?;
        words
            .iter()
            .enumerate()
            .all(|(index, &word)| {
                mem.read_instruction_u32_be(pc.wrapping_add(index as u32 * 4)) == Some(word)
            })
            .then_some(now)
    }

    /// Pick up regions the background compiler has finished.
    #[cold]
    #[inline(never)]
    fn poll_jit_regions(&mut self) {
        let mut next = 0;
        while next < self.jit_pending_regions.len() {
            let index = self.jit_pending_regions[next];
            let region = self
                .jit_table
                .get_mut(index)
                .and_then(|slot| slot.region.as_mut())
                .filter(|region| region.ticket.is_some());
            let Some(region) = region else {
                self.jit_pending_regions.swap_remove(next);
                continue;
            };
            let Some(ready) = jit::poll_ticket(region.ticket.as_ref().expect("filtered")) else {
                next += 1;
                continue;
            };
            region.ticket = None;
            region.code = ready.unwrap_or(jit::JIT_DECLINED);
            if ready.is_none() {
                jit::diag::count(jit::diag::Event::Declined);
            }
            self.jit_pending_regions.swap_remove(next);
            if ready.is_some() {
                self.install_jit_region(index);
            }
        }
    }

    /// Point chaining into JIT entry `index` at its newly compiled region.
    fn install_jit_region(&mut self, index: usize) {
        self.jit_stats.regions_compiled += 1;
        jit::diag::count(jit::diag::Event::Installed);
        let slot = &mut self.jit_table[index];
        let Some(region) = slot.region.as_deref() else {
            return;
        };
        if slot.code != 0 {
            slot.code = region.code;
            slot.lo = region.lo;
            slot.hi = region.hi;
            slot.region_epoch = region.epoch;
        }
        // Links into this block still name its plain code; forget them all
        // so chaining picks the region up.
        self.jit_link_gen = self.jit_link_gen.wrapping_add(1);
    }
}
