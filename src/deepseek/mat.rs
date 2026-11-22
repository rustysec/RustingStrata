//! Quantised matrix view over the GGUF mmap with an f32 matvec (slow reference path).
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{GgmlType, dequant_row};
use rayon::prelude::*;
use std::sync::{Mutex, OnceLock};

/// Optional GPU offload of the non-expert matvecs (STRATA_DS_GPU=1). Weights stay resident in VRAM; the activation is
/// copied up and the result down per call, so everything else in the forward stays the CPU reference code.
/// STRATA_DS_BACKEND (cuda or vulkan; unset: CUDA if it loads, else Vulkan) and STRATA_DS_DEVICES (such as 0,1) pick the GPUs; with several,
/// the weights fill them in order and the expert budget is divided in proportion to free VRAM.
mod dev {
    use super::*;
    use crate::backend::{Backend, Buf, GpuTensor};
    use crate::weights::Tensor;
    /// Scratch slot for experts that missed the VRAM cache: copied up per use, so the CPU never dequantises them.
    pub const STAGE: usize = 16 << 20;
    pub const STAGED: u32 = u32::MAX;
    /// Floats in each of the two activation scratch buffers of a GPU.
    pub const SCRATCH: usize = 5_000_000;
    /// A tensor id holds its GPU in the top 8 bits (so STAGED, GPU 255, is never a real id) and its index there below.
    const SHIFT: u32 = 24;
    pub fn gpu_of(id: u32) -> usize { (id >> SHIFT) as usize }
    /// One card's side of the device. `bytes` counts everything it holds — plain weights and expert slots together — so
    /// that on a shared-memory device `budget` is one ceiling for both kinds (see `plan`).
    struct Part { be: Box<dyn Backend>, ts: Vec<GpuTensor>, arena: Arena, conv: usize, x: Buf, y: Buf, stage: Buf, bytes: usize, budget: usize, room: usize, unified: bool }
    pub struct Dev { parts: Vec<Part>, ex: std::collections::HashMap<usize, usize>, exs: Vec<(u32, usize, u64, f32)>, now: u64, cur: usize, lookups: u64 }
    static DEV: OnceLock<Option<Mutex<Dev>>> = OnceLock::new();
    /// What one GPU may hold in weights: the expert cache's budget, with the plain weights living inside that budget
    /// (`room` is what they get before the engine moves on to the next card). A discrete card takes `frac` of its free
    /// VRAM, or its share of `gb` when an absolute amount was asked for.
    ///
    /// An integrated GPU is not the same kind of memory: its device heap is system RAM, so a weight uploaded there is a
    /// second copy of bytes the model already holds in page cache, and "80% of what the card reports free" is a share of
    /// memory the page cache is asking for at the same time. So the budget here is a fraction of *system* RAM,
    /// `SHARED_MEM_FRACTION`, and less still when the box has that little free at this moment — a server already holding
    /// a model leaves a fraction of what `total_ram` says. An explicit request is clamped to what RAM could back at all,
    /// free space included, unless `STRATA_DS_GPU_NO_CAP=1` names the risk; either way `weights::device_claim` counts the
    /// budget with the mapped files, so an over-committed pair stops keeping two copies of every expert
    /// (`weights::cold` drops the host copy once the upload has read it).
    ///
    /// Byte accounting is *not* what the Hy4-flash reports on Strix Halo hit: those runs died with ~110 GB of the
    /// machine free, which is `Arena`'s problem, not this function's. The fraction is a conservative starting point
    /// that has never been confirmed on that box (nothing at any budget finished there yet), not a measurement — see
    /// docs/AMD.md for the ramp that measures it.
    ///
    /// Returns the budget and, when something clamped it, that ceiling (the load line says so). A `ram` of 0 means
    /// undetectable, which leaves the request alone rather than guessing small.
    pub(super) fn plan(f: usize, total: usize, frac: f64, gb: Option<f64>, unified: bool, ram: usize, avail: usize, uncapped: bool) -> (usize, Option<usize>) {
        let card = |v: f64| (v * f as f64 / total.max(1) as f64) as usize;
        // `all` on a discrete card is that card's own free heap; on a shared one it is deliberately left unbounded
        // here, because the ceiling below is the machine's, which is the number that means something there.
        let all = gb.is_some_and(|g| g.is_infinite());
        let want = match gb {
            Some(_) if all => if unified { usize::MAX } else { f.saturating_sub(56 << 20) },
            Some(g) => card(g * 1e9),
            None => ((f.saturating_sub(56 << 20)) as f64 * frac) as usize,
        };
        // With no ceiling to speak of (a discrete card, `STRATA_DS_GPU_NO_CAP`, or RAM this OS will not report), `all`
        // means the heap the card itself offered - never `usize::MAX`, which would print an absurd budget and hand the
        // arena an unlimited claim.
        if !unified || uncapped || ram == 0 { return (if all { f.saturating_sub(56 << 20) } else { want }, None); }
        // Left to itself a shared card takes `SHARED_MEM_FRACTION` of the machine. `STRATA_DS_GPU_EXPERTS=<GB>` reaches
        // past that, up to what RAM could back at all: past where the budget and the mapped files no longer fit together
        // (`weights::device_claim`) every byte uploaded to the card hands its file page back, so it raises how much the
        // GPU holds rather than how much is held twice. `STRATA_DS_GPU_NO_CAP=1` removes the ceiling as well. Free space
        // bounds an explicit request too: a card already holding another model cannot take 40 GB just because the
        // machine has 134 GB fitted.
        let free_cap = if avail > 0 { card(avail.saturating_sub(AVAIL_KEEP) as f64) } else { usize::MAX };
        let cap = match gb {
            None => card(ram as f64 * SHARED_MEM_FRACTION).min(free_cap),
            Some(_) => card(ram.saturating_sub(AVAIL_KEEP) as f64).min(free_cap),
        }.max(SHARED_MEM_MIN);
        (want.min(cap), (want > cap).then_some(cap))
    }

    /// The share of system RAM an integrated GPU may take for weights. See `plan`.
    const SHARED_MEM_FRACTION: f64 = 0.125;

    /// Kept back from the machine's free RAM when a shared-memory budget is sized from it: this process's activations
    /// and staging, and the room the OS needs to reclaim page cache without swapping.
    pub(super) const AVAIL_KEEP: usize = 2 << 30;

    /// The smallest shared-memory budget worth opening a device for: below it nearly every expert runs on the CPU
    /// anyway, which `STRATA_DS_GPU=0` then says out loud instead of doing silently.
    const SHARED_MEM_MIN: usize = 1 << 30;

    /// Weight memory carved out of a few large device buffers rather than one per matrix, because a run's failure
    /// tracked the *count* of buffers and not the bytes: Hy4-flash Q4_K_M has 555 non-expert matrices and 1.0 MB expert
    /// matrices (read off its tensor table), so at one buffer each a 14 GB cache is ~14000 of them and a 40 GB one
    /// ~40000, while both reports died with ~110 GB of the machine's RAM untouched. `Not enough memory for command
    /// submission` is ENOMEM from the amdgpu submit ioctl, which is what the driver returns when it runs out of its
    /// per-process objects as well as when it runs out of memory - and only the first of those is invisible in
    /// `free`. Slots are therefore handed out of `CHUNK`-sized arenas at 256-byte steps, so an entire cache costs
    /// `budget / CHUNK` buffers whatever the model. A chunk the driver will not allocate is its answer that the budget
    /// is too big, and the caller falls back to the CPU instead of queueing a submission on memory that is not there.
    /// The count this removes is the assumption it tests: if a Hy4 run still refuses submissions at 2 GB, the ceiling is
    /// bytes or something else, and docs/AMD.md's ramp is what tells them apart.
    struct Arena { chunk: usize, bufs: Vec<Buf>, next: u64, left: usize }

    /// Where `n` bytes of a chunk-based allocator go: `Some((addr, left))` taking them from the chunk that has `left`
    /// bytes at `next`, or `None` meaning a new chunk of `max(chunk, n)` is needed first. Pure, so the wrap can be
    /// tested without a device.
    pub(super) fn carve(next: u64, left: usize, n: usize) -> Option<(u64, usize)> {
        if left < n { return None; }
        Some((next, left - n))
    }

    impl Arena {
        fn new(chunk: usize) -> Arena { Arena { chunk, bufs: vec![], next: 0, left: 0 } }
        /// Device address of `n` bytes, or None when no new chunk could be allocated.
        fn take(&mut self, be: &dyn Backend, n: usize) -> Option<u64> {
            let n = n.next_multiple_of(256);
            if carve(self.next, self.left, n).is_none() { self.next_chunk(be, n)?; }
            let (a, left) = carve(self.next, self.left, n)?;
            self.next = a + n as u64;
            self.left = left;
            Some(a)
        }
        fn next_chunk(&mut self, be: &dyn Backend, n: usize) -> Option<()> {
            let b = be.alloc(self.chunk.max(n)).ok()?;
            self.next = b.addr();
            self.left = b.len();
            self.bufs.push(b);
            Some(())
        }
        fn buffers(&self) -> usize { self.bufs.len() }
    }

    /// One expert cache chunk. Small enough that the zeroing on allocate stays instant, large enough that a 40 GB
    /// cache is 40 buffers.
    const CHUNK: usize = 1 << 30;

    /// `STRATA_DS_GPU_EXPERTS` as bytes-per-GB: a number, or `all`/`inf` for "as much as this machine can back"
    /// (`plan`'s ceiling), which is how a whole expert set is made resident on a big integrated GPU.
    pub(super) fn gb_asked() -> Option<f64> {
        let v = std::env::var("STRATA_DS_GPU_EXPERTS").ok()?;
        if v.eq_ignore_ascii_case("all") || v.eq_ignore_ascii_case("inf") { return Some(f64::INFINITY); }
        v.parse().ok()
    }

    pub fn get() -> Option<&'static Mutex<Dev>> {
        DEV.get_or_init(|| {
            // Unset or 0: every matvec runs on the CPU.
            if std::env::var("STRATA_DS_GPU").map_or(true, |v| v == "0") { return None; }
            let kind = std::env::var("STRATA_DS_BACKEND").ok();
            let ids: Vec<usize> = std::env::var("STRATA_DS_DEVICES").ok()
                .map(|v| v.split(',').filter_map(|d| d.trim().parse().ok()).collect()).filter(|v: &Vec<usize>| !v.is_empty()).unwrap_or(vec![0]);
            // `STRATA_DS_GPU_EXPERTS` is the absolute weight budget in GB (plain weights and expert cache), or `all` for
            // as much as this machine can back - every expert the model has, if there is room, since the cache simply
            // stops filling once the budget is spent. Unset: `STRATA_DS_GPU_FRACTION` of free VRAM (80% by default, 90%
            // for GLM), which keeps the old defaults on a 12 GB card and uses most of an 80 GB one.
            let gb: Option<f64> = gb_asked();
            let frac: f64 = std::env::var("STRATA_DS_GPU_FRACTION").ok().and_then(|v| v.parse().ok()).unwrap_or(0.8);
            let mut bes = vec![];
            for d in ids {
                match crate::backend::open(kind.as_deref(), d) {
                    Ok(be) => bes.push(be),
                    Err(e) => { eprintln!("GPU {d}: {e:#}; running without it"); return None; }
                }
            }
            let free: Vec<usize> = bes.iter().map(|b| b.mem_info().map_or(1, |m| m.0.max(1))).collect();
            let total: usize = free.iter().sum();
            let uncapped = std::env::var_os("STRATA_DS_GPU_NO_CAP").is_some();
            let ram = crate::memory::total_ram().unwrap_or(0) as usize;
            let avail = crate::memory::available_ram().unwrap_or(0) as usize;
            let parts = bes.into_iter().zip(&free).enumerate().map(|(i, (be, &f))| {
                // Room for a prefill chunk of activations and outputs, so row batches reuse them instead of allocating per expert.
                let (x, y, stage) = (be.alloc(4 * SCRATCH).ok()?, be.alloc(4 * SCRATCH).ok()?, be.alloc(STAGE).ok()?);
                let unified = be.unified_memory();
                let (budget, capped) = plan(f, total, frac, gb, unified, ram, avail, uncapped);
                if unified {
                    // Which rule decided is said as plainly as the number, because it is the thing to quote in a report:
                    // a budget that did not move when asked is a different problem from one that was capped on purpose.
                    let all = gb.is_some_and(|g| g.is_infinite());
                    let why = match (gb.is_some(), uncapped, capped) {
                        (_, true, _) => format!("{:.1} GB of its heap with no ceiling (STRATA_DS_GPU_NO_CAP)", f as f64 / 1e9),
                        (true, _, Some(c)) if all => format!("all this machine can back: {:.0} GB free of {:.0} RAM, less what the run needs for itself", c as f64 / 1e9, ram as f64 / 1e9),
                        (true, _, Some(c)) => format!("asked for more than this machine can back; clamped to {:.0} GB", c as f64 / 1e9),
                        (true, _, None) => "as asked (STRATA_DS_GPU_EXPERTS)".into(),
                        (false, _, Some(_)) => "an eighth of RAM, or what is free now if less".into(),
                        (false, _, None) => "what the heap reports free, under the ceiling".into(),
                    };
                    eprintln!("GPU {}: shares system RAM with the CPU; {:.1} GB of weights on it ({})", i, budget as f64 / 1e9, why);
                }
                // The buffer arithmetic, because it is the number that has been killing these runs and nothing else
                // shows it: one driver object per matrix would be thousands of them for a cache this size, while an
                // arena costs one per CHUNK whatever the model. Say which the run is using.
                if i == 0 {
                    eprintln!("gpu buffers: weights come from {} MB arenas, ~{} buffers for a full cache (one buffer per matrix would be thousands at this budget)",
                        CHUNK >> 20, budget / CHUNK + 1);
                }
                // Room for the plain weights before moving on to the next GPU. On a discrete card that is what its own
                // memory has left once the expert cache took its share; on a shared one `budget` *is* the ceiling, since
                // both kinds of weight are copies of RAM and have to add up to at most that.
                let room = if unified { budget } else { f.saturating_sub(budget + (768 << 20)) };
                Some(Part { be, ts: vec![], arena: Arena::new(CHUNK), conv: 0, x, y, stage, bytes: 0, budget, room, unified })
            }).collect::<Option<Vec<_>>>()?;
            // Counted with the mapped files against RAM, because on a shared-memory device they are the same pages twice.
            let claim: usize = parts.iter().map(|p| p.budget).sum();
            crate::weights::device_claim(claim);
            if parts.iter().any(|p| p.unified) {
                // The two claims and what RAM makes of them, because the symptom of getting this wrong is a wedge three
                // lines later. "one copy" is `weights::cold` giving back each file page once the card has read it.
                let mapped = crate::weights::mapped_bytes() as f64 / 1e9;
                eprintln!("memory: {:.1} GB of weights on the card, {:.1} GB of mapped model files, {} RAM -> {}", claim as f64 / 1e9, mapped,
                    match crate::memory::total_ram() { Some(r) => format!("{:.1} GB", r as f64 / 1e9), None => "unknown".into() },
                    match (crate::weights::fits_ram(), std::env::var("STRATA_PRELOAD").ok()) {
                        (true, _) => "both fit, file pages stay".into(),
                        (false, Some(v)) if v != "0" => format!("one copy (forced by STRATA_PRELOAD={v})"),
                        (false, Some(_)) => "one copy (forced off by STRATA_PRELOAD=0)".into(),
                        (false, None) => "over RAM: one copy (an uploaded expert drops its file page)".into(),
                    });
            }
            Some(Mutex::new(Dev { parts, ex: Default::default(), exs: vec![], now: 0, cur: 0, lookups: 0 }))
        }).as_ref()
    }
    impl Dev {
        pub fn bytes(&self) -> usize { self.parts.iter().map(|p| p.bytes).sum() }
        pub fn info(&self) -> (&'static str, usize, usize) { (self.parts[0].be.name(), self.ex.len(), self.bytes()) }
        /// Driver buffers the weight memory costs: one per arena chunk plus one per matrix that had to be converted on
        /// the way up. This is the number a driver with a fixed table of buffer objects runs out of, so it is what a
        /// report on `Not enough memory for command submission` has to quote (`STRATA_DS_PROF` prints it per token).
        pub fn buffers(&self) -> usize { self.parts.iter().map(|p| p.arena.buffers() + p.conv).sum() }
        /// Bytes still available under the expert budget (plain weights already subtracted).
        pub fn budget_left(&self) -> usize { self.parts.iter().map(|p| p.budget.saturating_sub(p.bytes)).sum() }
        /// The GPU that takes `len` more bytes of plain weights: the current one while they fit, then the next; the last takes all.
        pub fn pick(&mut self, len: usize) -> usize {
            while self.cur + 1 < self.parts.len() && self.parts[self.cur].bytes + len > self.parts[self.cur].room { self.cur += 1; }
            self.cur
        }
        pub fn upload_on(&mut self, d: usize, data: &[u8], ty: GgmlType, rows: usize, cols: usize) -> Option<u32> {
            let p = &mut self.parts[d];
            let unified = p.unified;
            // A weight that needs no type conversion goes into the arena: one slot per matrix is what filled the
            // driver's table of buffer objects (`Arena`). The converted kinds (F16 as F32, Q4_0/Q5_0 as Q8_0) are host
            // work the backend does on the way up, so they keep their own buffer.
            let t = if crate::backend::upload_type(ty) == ty {
                let (arena, be) = (&mut p.arena, &*p.be);
                let a = arena.take(be, data.len())?;
                let dst = Buf::new(a, data.len(), ());
                be.htod(data, &dst, 0).ok()?;
                // On a shared-memory card this copy is the second copy of bytes the file already has in page cache, so
                // hand those pages back once the upload has read them - but only once budget and mapped files no longer
                // fit together (`weights::cold` checks `fits_ram`, which counts this budget). One byte held in one
                // place: from the slot now, from the file again if that slot is later overwritten or evicted.
                if unified { crate::weights::cold(data); }
                GpuTensor::new(ty, rows, cols, dst)
            } else {
                let t = crate::weights::upload_then_cold(data, unified, || p.be.upload(&Tensor { ty, cols, rows, data }).ok())?;
                p.conv += 1;  // its own driver buffer, not an arena slot: counted in `buffers`
                t
            };
            p.bytes += data.len();
            p.ts.push(t);
            Some((d as u32) << SHIFT | (p.ts.len() as u32 - 1))
        }
        pub fn upload(&mut self, data: &[u8], ty: GgmlType, rows: usize, cols: usize) -> Option<u32> {
            let d = self.pick(data.len());
            if !self.accepts(d, data.len()) { return None; }
            self.upload_on(d, data, ty, rows, cols)
        }
        /// Whether the card `pick` chose may take `len` more weight bytes. A discrete card ends the chain by taking what
        /// it is given (its memory is its own); on a shared-memory one the ceiling is system RAM for weights of every
        /// kind, so the answer there is the budget's, and a matrix that does not fit stays on the CPU. Over-running it
        /// is how an allocation the driver would refuse becomes a submission that wedges the card (see `plan`).
        pub(super) fn accepts(&self, d: usize, len: usize) -> bool { !self.parts[d].unified || self.parts[d].bytes + len <= self.parts[d].room }
        fn tensor(&self, i: u32) -> &GpuTensor { &self.parts[gpu_of(i)].ts[(i & (1 << SHIFT) - 1) as usize] }
        /// VRAM copy of one routed expert: uploaded on first use while under a GPU's STRATA_DS_GPU_EXPERTS share (GB), then the
        /// least recently used slot of the same shape is overwritten. Slots touched in the last 96 lookups are kept (the
        /// matrices of one layer's experts are in flight together). No slot to take: STAGED (copied up per use).
        /// `evict` false: a miss with no free room returns None, so the caller can run that expert on the CPU.
        /// Whether the expert that starts at `data` has a VRAM slot (no lookup is counted).
        pub fn resident(&self, data: &[u8]) -> bool { self.ex.contains_key(&(data.as_ptr() as usize)) }
        pub fn expert(&mut self, data: &[u8], ty: GgmlType, rows: usize, cols: usize, evict: bool) -> Option<u32> {
            let k = data.as_ptr() as usize;
            self.now += 1;
            self.lookups += 1;
            let now = self.now;
            if self.lookups % 4096 == 0 { for e in &mut self.exs { e.3 *= 0.5; } }
            if let Some(&p) = self.ex.get(&k) {
                self.exs[p].2 = now;
                self.exs[p].3 += 1.0;
                EX_LOOKUPS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(self.exs[p].0);
            }
            EX_LOOKUPS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let room = |p: &Part| (p.bytes + data.len() <= p.budget).then(|| p.budget - p.bytes);
            if let Some(d) = (0..self.parts.len()).filter_map(|d| room(&self.parts[d]).map(|r| (r, d))).max().map(|(_, d)| d) {
                let i = self.upload_on(d, data, ty, rows, cols)?;
                self.ex.insert(k, self.exs.len());
                self.exs.push((i, k, now, 1.0));
                return Some(i);
            }
            if !evict || self.parts.iter().all(|p| p.budget == 0) || data.len() > STAGE || std::env::var_os("STRATA_DS_NOEVICT").is_some() { return None; }
            let lfu = std::env::var_os("STRATA_DS_LFU").is_some();
            let victim = self.exs.iter().enumerate()
                .filter(|(_, e)| e.2 + 96 <= now && { let t = self.tensor(e.0); t.ty == ty && t.rows == rows && t.cols == cols })
                .min_by(|(_, a), (_, b)| if lfu { a.3.total_cmp(&b.3).then(a.2.cmp(&b.2)) } else { a.2.cmp(&b.2) })
                .map(|(p, _)| p);
            let Some(p) = victim else { return Some(STAGED) };
            if lfu && self.exs[p].3 > 4.0 { return Some(STAGED); }
            let (i, old, _, _) = self.exs[p];
            let (be, t) = (&self.parts[gpu_of(i)].be, self.tensor(i));
            let unified = self.parts[gpu_of(i)].unified;
            if be.htod(data, &Buf::new(t.ptr, data.len(), ()), 0).is_err() { return Some(STAGED); }
            // The new expert's pages were just read into cache by `htod`; the old slot's pages were dropped when it was
            // uploaded. Without this the cache fills with file pages of every expert that has ever been overwritten,
            // which is the same double claim as the first upload (and on an APU the same wedged driver).
            if unified { crate::weights::cold(data); }
            self.ex.remove(&old);
            self.ex.insert(k, p);
            self.exs[p] = (i, k, now, 1.0);
            Some(i)
        }
        /// A failed device call means the card is gone (an over-committed amdgpu refuses the submission and then reports
        /// `ERROR_DEVICE_LOST`). These calls run inside rayon workers, so unwinding is not an option: a second worker
        /// panicking mid-unwind aborts with buffers freed and submissions still in flight, which is why the process turns
        /// into a zombie and the kernel logs "VM memory stats ... non-zero when fini" instead of releasing its GPU VM.
        /// Exiting at once gives amdgpu its normal fini path and lets a service manager restart the job; the weights it
        /// was computing are readable on the CPU, so the run itself is not lost - lower `STRATA_DS_GPU_EXPERTS` (see docs/AMD.md).
        fn must<T>(r: anyhow::Result<T>) -> T {
            match r {
                Ok(v) => v,
                Err(e) => { eprintln!("GPU call failed ({e}); exiting with status 1 so the driver can reset the card"); std::process::exit(1); }
            }
        }
        pub fn gemv(&self, i: u32, x: &[f32], y: &mut [f32], staged: Option<(&[u8], GgmlType, usize, usize)>) {
            let p = &self.parts[if i == STAGED { 0 } else { gpu_of(i) }];
            Self::must(p.be.put(x, &p.x, 0));
            if let Some((d, ty, rows, cols)) = staged {
                Self::must(p.be.htod(d, &p.stage, 0));
                let w = GpuTensor::new(ty, rows, cols, Buf::new(p.stage.addr(), d.len(), ()));
                Self::must(p.be.gemv(&w, p.x.addr(), p.y.addr()));
            } else {
                Self::must(p.be.gemv(self.tensor(i), p.x.addr(), p.y.addr()));
            }
            let got: Vec<f32> = Self::must(p.be.get(&p.y, 0, y.len()));
            y.copy_from_slice(&got);
        }
        /// One gemv per matrix, all queued before a single readback per GPU. Job i reads x[i*xs..] (xs = 0: every
        /// job reads x) and writes its rows after job i-1's. Staged experts are uploaded once each into a temporary
        /// buffer, so a layer with both resident and staged experts still batches.
        pub fn batch_mats(&self, ms: &[Mat], x: &[f32], xs: usize) -> Option<Vec<f32>> {
            let on: Vec<usize> = ms.iter().map(|m| self.part_of(m)).collect::<Option<_>>()?;
            let first = *on.first()?;
            if on.iter().all(|&d| d == first) { return self.batch_on(first, ms, x, xs); }
            // ponytail: the GPUs run their batches one after another; queue all of them before the readbacks if
            // multi-GPU decode ever waits on this.
            let mut y = vec![0f32; ms.iter().map(|m| m.rows).sum()];
            for d in 0..self.parts.len() {
                let idx: Vec<usize> = (0..ms.len()).filter(|&i| on[i] == d).collect();
                if idx.is_empty() { continue; }
                let sub: Vec<Mat> = idx.iter().map(|&i| ms[i]).collect();
                let subx: Vec<f32> = if xs == 0 { x.to_vec() } else { idx.iter().flat_map(|&i| x[i * xs..(i + 1) * xs].iter().copied()).collect() };
                let r = self.batch_on(d, &sub, &subx, xs)?;
                let mut from = 0;
                for &i in &idx {
                    let to: usize = ms[..i].iter().map(|m| m.rows).sum();
                    y[to..to + ms[i].rows].copy_from_slice(&r[from..from + ms[i].rows]);
                    from += ms[i].rows;
                }
            }
            Some(y)
        }
        fn batch_on(&self, d: usize, ms: &[Mat], x: &[f32], xs: usize) -> Option<Vec<f32>> {
            let p = &self.parts[d];
            let nx = if xs == 0 { x.len() } else { ms.len() * xs };
            if nx > p.x.len() / 4 { return None; }
            let xsrc = &x[..nx];
            p.be.put(xsrc, &p.x, 0).ok()?;
            let mut owned = vec![];
            let mut off = 0;
            for (i, m) in ms.iter().enumerate() {
                if off + m.rows > p.y.len() / 4 { return None; }
                p.be.gemv(self.weight(p, m, &mut owned)?, p.x.addr() + (4 * i * xs) as u64, p.y.addr() + (4 * off) as u64).ok()?;
                off += m.rows;
            }
            let y = p.be.get(&p.y, 0, off).ok()?;
            drop(owned);
            Some(y)
        }
        /// The device tensor of `m`. A staged expert is uploaded into a buffer that `owned` keeps until the readback.
        fn weight<'b>(&'b self, p: &Part, m: &Mat, owned: &'b mut Vec<GpuTensor>) -> Option<&'b GpuTensor> {
            Some(match m.gpu? {
                STAGED => {
                    let b = p.be.alloc(m.data.len()).ok()?;
                    p.be.htod(m.data, &b, 0).ok()?;
                    owned.push(GpuTensor::new(m.ty, m.rows, m.cols, b));
                    owned.last().unwrap()
                }
                t => self.tensor(t),
            })
        }
        /// Inkling's routed experts of one token in one round trip: `gus[i]` x (n gates, then n ups), the activation
        /// on the device (`Backend::situ`), then `downs[i]` of that. The down outputs, one after another. None unless
        /// the matrices share one GPU that has the kernel.
        pub fn situ_experts(&self, gus: &[Mat], downs: &[Mat], x: &[f32], beta: f32, lb: f32) -> Option<Vec<f32>> {
            let d = self.part_of(gus.first()?)?;
            if gus.iter().chain(downs).any(|m| self.part_of(m) != Some(d)) { return None; }
            let (p, ne, nf, nl) = (&self.parts[d], gus.len(), downs[0].cols, downs[0].rows);
            if x.len() > p.x.len() / 4 || ne * (2 * nf + nl) > p.y.len() / 4 { return None; }
            p.be.put(x, &p.x, 0).ok()?;
            let (mut owned, at) = (vec![], |i: usize| p.y.addr() + (4 * i) as u64);
            for (i, m) in gus.iter().enumerate() { p.be.gemv(self.weight(p, m, &mut owned)?, p.x.addr(), at(i * 2 * nf)).ok()?; }
            p.be.situ(p.y.addr(), nf, ne, beta, lb).ok()?;
            for (i, m) in downs.iter().enumerate() { p.be.gemv(self.weight(p, m, &mut owned)?, at(i * 2 * nf), at(ne * 2 * nf + i * nl)).ok()?; }
            let y = p.be.get(&p.y, ne * 2 * nf, ne * nl).ok()?;
            drop(owned);
            Some(y)
        }
        fn part_of(&self, m: &Mat) -> Option<usize> {
            let i = m.gpu?;
            Some(if i == STAGED { 0 } else { gpu_of(i) })
        }
        /// W x[t] for every row of `xs` (one matrix, several activations). Uses the tensor-core GEMM when there are
        /// enough rows and the GEMV otherwise; staged weights are uploaded once for the whole batch.
        pub fn rows(&self, m: &Mat, xs: &[&[f32]]) -> Option<Vec<f32>> {
            let i = m.gpu?;
            let d = if i == STAGED { 0 } else { gpu_of(i) };
            let p = &self.parts[d];
            if xs.is_empty() { return Some(vec![]); }
            if xs.iter().any(|x| x.len() != m.cols) { return None; }
            let n = xs.len();
            let mut flat = Vec::with_capacity(n * m.cols);
            for x in xs { flat.extend_from_slice(x); }
            let mut owned = vec![];
            let w = if i == STAGED {
                let b = p.be.alloc(m.data.len()).ok()?;
                p.be.htod(m.data, &b, 0).ok()?;
                owned.push(GpuTensor::new(m.ty, m.rows, m.cols, b));
                &owned[0]
            } else { self.tensor(i) };
            // Reuse the scratch pair when the chunk fits; otherwise allocate a GEMM-sized pair for this call.
            let y = if n * m.cols <= p.x.len() / 4 && n * m.rows <= p.y.len() / 4 {
                p.be.put(&flat, &p.x, 0).ok()?;
                if n <= 4 {
                    p.be.gemvn(w, p.x.addr(), m.cols, p.y.addr(), m.rows, n).ok()?;
                } else {
                    p.be.gemm_dense(w, n, p.x.addr(), m.cols, p.y.addr(), m.rows, 0).ok()?;
                }
                p.be.get(&p.y, 0, n * m.rows).ok()?
            } else {
                let xb = p.be.zeros::<f32>(n * m.cols).ok()?;
                let yb = p.be.zeros::<f32>(n * m.rows).ok()?;
                p.be.put(&flat, &xb, 0).ok()?;
                p.be.gemm_dense(w, n, xb.addr(), m.cols, yb.addr(), m.rows, 0).ok()?;
                p.be.get(&yb, 0, n * m.rows).ok()?
            };
            drop(owned);
            Some(y)
        }
        /// The routed experts of a chunk of tokens, all on the device. `x`: one row of activations a token. Expert i
        /// (`gus[i]`: its gate rows then its up rows; `downs[i]`) runs on `pairs[i]`, a list of (token, slot of the
        /// expert among the `k` of that token, router weight). Returns per token the weighted sum of these experts'
        /// outputs: one upload and one readback of a row a token, whatever the number of pairs.
        /// None when a matrix is on another GPU or the device has no room for the chunk.
        pub fn situ_chunk(&self, gus: &[Mat], downs: &[Mat], pairs: &[Vec<(usize, usize, f32)>], x: &[f32], k: usize, beta: f32, lb: f32) -> Option<Vec<f32>> {
            use crate::backend::GEMM_BM;
            let d = self.part_of(gus.first()?)?;
            if gus.iter().chain(downs).any(|m| self.part_of(m) != Some(d)) { return None; }
            let (p, nf, nl) = (&self.parts[d], downs[0].cols, downs[0].rows);
            let (nt, np) = (x.len() / nl, pairs.iter().map(|v| v.len()).sum::<usize>());
            // GEMM tiles [weights, first pair | pairs << 32] of the gate and up matrices, then of the down ones;
            // then per pair its token, and per (token, slot) the router weight and the pair (weight 0: not here).
            let (mut owned, mut tiles, mut xrow) = (vec![], vec![], vec![]);
            let (mut wts, mut inv) = (vec![0f32; nt * k], vec![0i32; nt * k]);
            let mut shape = [(GgmlType::F32, 0, 0); 2];
            for (side, mats) in [gus, downs].into_iter().enumerate() {
                let mut m0 = 0;
                for (m, ps) in mats.iter().zip(pairs) {
                    let w = self.weight(p, m, &mut owned)?;
                    shape[side] = (w.ty, w.rows, w.cols);
                    for at in (0..ps.len()).step_by(GEMM_BM) {
                        tiles.extend([w.ptr, (m0 + at) as u64 | (GEMM_BM.min(ps.len() - at) as u64) << 32]);
                    }
                    m0 += ps.len();
                }
            }
            for &(t, j, w) in pairs.iter().flatten() {
                (wts[t * k + j], inv[t * k + j]) = (w, xrow.len() as i32);
                xrow.push(t as i32);
            }
            let (n_tiles, ints) = (tiles.len() / 4, 2 * tiles.len());
            if x.len() > p.x.len() / 4 || ints + np + 2 * nt * k > p.y.len() / 4 { return None; }
            // Zeroed: the sums, then per pair the gate and up rows, then per pair the down rows.
            let b = p.be.alloc(4 * (nt * nl + np * (2 * nf + nl))).ok()?;
            let (ia, ya) = (|i: usize| p.y.addr() + 4 * i as u64, |i: usize| b.addr() + 4 * i as u64);
            let (gate_up, down) = (nt * nl, nt * nl + np * 2 * nf);
            p.be.put(x, &p.x, 0).ok()?;
            p.be.put(&tiles, &p.y, 0).ok()?;
            p.be.put(&xrow, &p.y, ints).ok()?;
            p.be.put(&wts, &p.y, ints + np).ok()?;
            p.be.put(&inv, &p.y, ints + np + nt * k).ok()?;
            let [(ty, rows, cols), (dty, drows, dcols)] = shape;
            p.be.gemm(ty, rows, cols, ia(0), n_tiles, p.x.addr(), nl, nt, ia(ints), ya(gate_up), 2 * nf, 0).ok()?;
            p.be.situ(ya(gate_up), nf, np, beta, lb).ok()?;
            p.be.gemm(dty, drows, dcols, ia(4 * n_tiles), n_tiles, ya(gate_up), 2 * nf, np, 0, ya(down), nl, 0).ok()?;
            p.be.moe_combine_seq(b.addr(), ya(down), ia(ints + np), ia(ints + np + nt * k), k, 0, 0, nl, nt).ok()?;
            let y = p.be.get(&b, 0, nt * nl).ok()?;
            drop(owned);
            Some(y)
        }
        /// `rows` for several matrices on one chunk of `n` rows: matrix i reads input set `ms[i].1` of `x` (set after
        /// set). One upload of the sets and one readback for all matrices. None when a matrix is staged or on
        /// another GPU, or the chunk does not fit the scratch buffers.
        pub fn rows_sets(&self, ms: &[(Mat, usize)], x: &[f32], n: usize) -> Option<Vec<f32>> {
            let (d, cols) = (self.part_of(&ms.first()?.0)?, ms[0].0.cols);
            if ms.iter().any(|(m, j)| self.part_of(m) != Some(d) || m.gpu == Some(STAGED) || m.cols != cols || (j + 1) * n * cols > x.len()) { return None; }
            let p = &self.parts[d];
            if n <= 4 || x.len() > p.x.len() / 4 || ms.iter().map(|m| n * m.0.rows).sum::<usize>() > p.y.len() / 4 { return None; }
            p.be.put(x, &p.x, 0).ok()?;
            let mut yo = 0;
            for (m, j) in ms {
                p.be.gemm_dense(self.tensor(m.gpu?), n, p.x.addr() + (4 * j * n * cols) as u64, cols, p.y.addr() + 4 * yo as u64, m.rows, 0).ok()?;
                yo += n * m.rows;
            }
            p.be.get(&p.y, 0, yo).ok()
        }
        /// `rows` for several matrices: job i runs its matrix on its own `n` rows of `x` (after job i-1's rows).
        /// One upload and one readback for all jobs. None when the batch does not fit the scratch buffers.
        /// ponytail: None too when the weights are on different GPUs; batch per GPU if a multi-GPU prompt waits here.
        pub fn rows_many(&self, jobs: &[(Mat, usize)], x: &[f32]) -> Option<Vec<f32>> {
            let d = self.part_of(&jobs.first()?.0)?;
            if jobs.iter().any(|j| self.part_of(&j.0) != Some(d)) { return None; }
            let p = &self.parts[d];
            let (nx, ny): (usize, usize) = jobs.iter().fold((0, 0), |a, (m, n)| (a.0 + n * m.cols, a.1 + n * m.rows));
            if nx != x.len() || nx > p.x.len() / 4 || ny > p.y.len() / 4 { return None; }
            p.be.put(x, &p.x, 0).ok()?;
            let mut owned = vec![];
            let (mut xo, mut yo) = (0, 0);
            for (m, n) in jobs {
                let w = match m.gpu? {
                    STAGED => {
                        let b = p.be.alloc(m.data.len()).ok()?;
                        p.be.htod(m.data, &b, 0).ok()?;
                        owned.push(GpuTensor::new(m.ty, m.rows, m.cols, b));
                        owned.last().unwrap()
                    }
                    t => self.tensor(t),
                };
                let (xa, ya) = (p.x.addr() + 4 * xo as u64, p.y.addr() + 4 * yo as u64);
                if *n <= 4 {
                    p.be.gemvn(w, xa, m.cols, ya, m.rows, *n).ok()?;
                } else {
                    p.be.gemm_dense(w, *n, xa, m.cols, ya, m.rows, 0).ok()?;
                }
                xo += n * m.cols;
                yo += n * m.rows;
            }
            let y = p.be.get(&p.y, 0, yo).ok()?;
            drop(owned);
            Some(y)
        }
    }
}
/// Matvecs of several matrices on one activation (xs = 0) or on consecutive slices x[i*xs..] (xs > 0); outputs concatenated.
/// One GPU round trip when the matrices share a GPU (staged experts included). Otherwise each runs on its own, as
/// `Mat::matvec`.
pub fn matvec_many(ms: &[Mat], x: &[f32], xs: usize) -> Vec<f32> {
    if let Some(y) = matvec_many_opt(ms, x, xs) { return y; }
    let mut y = vec![0f32; ms.iter().map(|m| m.rows).sum()];
    let mut off = 0;
    for (i, m) in ms.iter().enumerate() {
        m.matvec(&x[i * xs..i * xs + m.cols], &mut y[off..off + m.rows]);
        off += m.rows;
    }
    y
}

/// The batched path of `matvec_many`, or None when the matrices cannot share one GPU batch (or batching is off).
pub fn matvec_many_opt(ms: &[Mat], x: &[f32], xs: usize) -> Option<Vec<f32>> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("STRATA_DS_BATCH").ok().as_deref() != Some("0")) { return None; }
    dev::get().and_then(|d| d.lock().unwrap().batch_mats(ms, x, xs))
}

/// `Dev::situ_experts`, or None where it does not apply.
/// `Dev::situ_chunk` on the open GPU path (None without one).
pub fn situ_chunk(gus: &[Mat], downs: &[Mat], pairs: &[Vec<(usize, usize, f32)>], x: &[f32], k: usize, beta: f32, lb: f32) -> Option<Vec<f32>> {
    dev::get().and_then(|d| d.lock().unwrap().situ_chunk(gus, downs, pairs, x, k, beta, lb))
}
pub fn situ_experts(gus: &[Mat], downs: &[Mat], x: &[f32], beta: f32, lb: f32) -> Option<Vec<f32>> {
    dev::get().and_then(|d| d.lock().unwrap().situ_experts(gus, downs, x, beta, lb))
}

/// W x[t] for each row of `xs` (all rows of one matrix). Batches on the GPU when the weight is resident.
pub fn matvec_rows(m: &Mat, xs: &[&[f32]]) -> Vec<f32> {
    // A host matrix does not take the device lock: its rows run while another thread holds the device.
    if let Some(y) = dev::get().filter(|_| m.gpu.is_some()).and_then(|d| d.lock().unwrap().rows(m, xs)) { return y; }
    let (n, mut y) = (xs.len(), vec![0f32; xs.len() * m.rows]);
    // On the CPU each weight row is read once for all the activations: the dots of `matvec`, in another order.
    if let Some(dot) = m.int_dot().filter(|_| n > 1 && (m.gpu.is_none() || dev::get().is_none()) && xs.iter().all(|x| x.len() == m.cols)) {
        let rb = m.ty.row_bytes(m.cols as u64);
        let xq: Vec<_> = xs.par_iter().map(|x| crate::cpu_moe::quantize_q8k(x)).collect();
        let mut by_row = vec![0f32; m.rows * n];
        let dot4 = crate::cpu_moe::int_dot4(m.ty);
        by_row.par_chunks_mut(32 * n).enumerate().for_each(|(c, o)| {
            for (k, o) in o.chunks_exact_mut(n).enumerate() {
                let w = &m.data[(c * 32 + k) * rb..(c * 32 + k + 1) * rb];
                let mut t = 0;
                if let Some(dot4) = dot4 {
                    while t + 4 <= n {
                        o[t..t + 4].copy_from_slice(&dot4(w, [&xq[t], &xq[t + 1], &xq[t + 2], &xq[t + 3]]));
                        t += 4;
                    }
                }
                for t in t..n { o[t] = dot(w, &xq[t]); }
            }
        });
        y.par_chunks_mut(m.rows).enumerate().for_each(|(t, y)| {
            for (r, v) in y.iter_mut().enumerate() { *v = by_row[r * n + t]; }
        });
        return y;
    }
    for (i, x) in xs.iter().enumerate() {
        m.matvec(x, &mut y[i * m.rows..(i + 1) * m.rows]);
    }
    y
}

/// `matvec_rows` of several matrices on one chunk of `n` tokens: matrix i reads input set `ms[i].1` of `x` (set after
/// set, a row a token). Outputs concatenated. On the GPU the sets go up once and the outputs come back in one copy.
pub fn matvec_rows_sets(ms: &[(Mat, usize)], x: &[f32], n: usize) -> Vec<f32> {
    if let Some(y) = dev::get().filter(|_| ms.iter().all(|m| m.0.gpu.is_some())).and_then(|d| d.lock().unwrap().rows_sets(ms, x, n)) { return y; }
    let w = n * ms[0].0.cols;
    ms.iter().flat_map(|(m, j)| matvec_rows(m, &x[j * w..(j + 1) * w].chunks_exact(m.cols).collect::<Vec<_>>())).collect()
}

/// `matvec_rows` for several matrices: job i runs its matrix on its own `n` rows of `x` (after job i-1's rows);
/// outputs concatenated. On the GPU each run of jobs that fits the scratch buffers is one round trip.
pub fn matvec_rows_many(jobs: &[(Mat, usize)], x: &[f32]) -> Vec<f32> {
    let size = |j: &(Mat, usize)| (j.1 * j.0.cols, j.1 * j.0.rows);
    let mut y = Vec::with_capacity(jobs.iter().map(|j| size(j).1).sum());
    let (mut i, mut at) = (0, 0);
    while i < jobs.len() {
        let (mut j, mut nx, mut ny) = (i + 1, size(&jobs[i]).0, size(&jobs[i]).1);
        while j < jobs.len() && nx + size(&jobs[j]).0 <= dev::SCRATCH && ny + size(&jobs[j]).1 <= dev::SCRATCH {
            nx += size(&jobs[j]).0;
            ny += size(&jobs[j]).1;
            j += 1;
        }
        match dev::get().and_then(|d| d.lock().unwrap().rows_many(&jobs[i..j], &x[at..at + nx])) {
            Some(r) => y.extend(r),
            None => {
                let mut o = at;
                for (m, n) in &jobs[i..j] {
                    y.extend(matvec_rows(m, &x[o..o + n * m.cols].chunks_exact(m.cols).collect::<Vec<_>>()));
                    o += n * m.cols;
                }
            }
        }
        (i, at) = (j, at + nx);
    }
    y
}
/// Expert cache lookups since the last PROF print: [hit, miss].
pub static EX_LOOKUPS: [std::sync::atomic::AtomicU64; 2] = [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];
pub fn gpu_bytes() -> usize { dev::get().map_or(0, |d| d.lock().unwrap().bytes()) }
/// Device buffers (arena chunks plus the odd converted upload) the weights now occupy. See `dev::Dev::buffers`.
pub fn gpu_buffers() -> usize { dev::get().map_or(0, |d| d.lock().unwrap().buffers()) }
/// Expert-budget bytes not yet used (0 without a GPU).
pub fn budget_left() -> usize { dev::get().map_or(0, |d| d.lock().unwrap().budget_left()) }
/// `STRATA_DS_GPU_WARM=1`: fill the expert cache at load instead of one lookup at a time. Implied by
/// `STRATA_DS_GPU_EXPERTS=all`, whose whole point is that everything is resident before the first token - and which is
/// also the answer to "I never see the experts load": unwarmed, the cache fills as tokens route experts, so the card
/// grows a few megabytes per token and the first hundred tokens of a session all pay for the uploads.
pub fn warm() -> bool {
    std::env::var("STRATA_DS_GPU_WARM").map_or(false, |v| v != "0") || dev::gb_asked().is_some_and(|g| g.is_infinite())
}

/// Fill the VRAM expert cache from a model's routed experts until the budget is spent: `layers` is one
/// `(gate, up, down)` triple per MoE layer, `n_expert` the experts each holds. Expert-major on purpose: a budget that
/// covers half the model then leaves half the experts of *every* layer resident rather than a few layers complete, and
/// a token routes across all of them. Nothing is evicted (a warm-up must not cost a slot to an expert the model has not
/// asked for), so an expert that does not fit is left exactly where it would otherwise be: read from the mapped file
/// when it is routed. Matrices are the same size within a model, so the first one that does not fit ends the pass.
/// Returns (matrices resident, bytes uploaded).
pub fn warm_experts(layers: &[(Mat, Mat, Mat)], n_expert: usize) -> (usize, usize) {
    let (t0, mut n, mut bytes, mut next) = (std::time::Instant::now(), 0usize, 0usize, 1usize << 30);
    for e in 0..n_expert {
        for (g, u, d) in layers {
            for m in [g.expert_host(e, n_expert), u.expert_host(e, n_expert), d.expert_host(e, n_expert)] {
                if dev::get().and_then(|dv| dv.lock().unwrap().expert(m.data, m.ty, m.rows, m.cols, false)).is_none() {
                    return (n, bytes);
                }
                n += 1;
                bytes += m.data.len();
            }
        }
        if bytes >= next {
            eprintln!("warm: {:.1} GB of experts resident, {n} matrices (experts 0..{} of {n_expert} in every layer), {:.0} s",
                bytes as f64 / 1e9, e + 1, t0.elapsed().as_secs_f64());
            next += 8 << 30;
        }
    }
    (n, bytes)
}
/// (backend, resident expert slots, bytes of all uploaded weights) when the GPU path is open.
pub fn dev_info() -> Option<(&'static str, usize, usize)> {
    dev::get().map(|d| d.lock().unwrap().info())
}

/// Row-major weight: `rows` rows of `cols` elements (ggml dims[0] = cols, dims[1] = rows).
/// A 3-D expert tensor is addressed with `expert(e)`.
#[derive(Clone, Copy)]
pub struct Mat<'a> {
    pub data: &'a [u8],
    pub ty: GgmlType,
    pub rows: usize,
    pub cols: usize,
    /// Routed-expert tensor: llama.cpp runs these on the CPU (Q8_K activations); the rest runs on CUDA (Q8_1 blocks of 32).
    pub cpu: bool,
    /// Index of the VRAM copy (STRATA_DS_GPU).
    pub gpu: Option<u32>,
}

impl<'a> Mat<'a> {
    pub fn load(g: &'a Gguf, name: &str) -> anyhow::Result<Self> {
        let t: &TensorInfo = g.tensors.get(name).ok_or_else(|| anyhow::anyhow!("missing tensor {name}"))?;
        let ty = GgmlType::from_u32(t.ty)?;
        let cols = t.dims[0] as usize;
        let rows = t.dims[1..].iter().product::<u64>() as usize;
        let cpu = name.contains("_exps");
        let data = g.bytes(t);
        let gpu = if cpu || t.dims.len() != 2 { None } else { dev::get().and_then(|d| d.lock().unwrap().upload(data, ty, rows, cols)) };
        Ok(Mat { data, ty, rows, cols, cpu, gpu })
    }

    /// Like `load` for a matrix used as `n` row groups (grouped output projection): each group gets its own VRAM copy.
    pub fn load_groups(g: &'a Gguf, name: &str, n: usize) -> anyhow::Result<Self> {
        let mut m = Self::load(g, name)?;
        m.gpu = None;
        if let Some(d) = dev::get() {
            let (gr, rb) = (m.rows / n, m.ty.row_bytes(m.cols as u64));
            let mut d = d.lock().unwrap();
            let gpu = d.pick(m.data.len());  // the groups get consecutive ids on one GPU
            // One matrix of `n` groups is `n` copies on the card: the budget says whether this machine can afford that.
            if !d.accepts(gpu, m.data.len()) { return Ok(m); }
            let ids: Vec<_> = (0..n).map(|i| d.upload_on(gpu, &m.data[i * gr * rb..(i + 1) * gr * rb], m.ty, gr, m.cols)).collect();
            if ids.iter().all(|i| i.is_some()) { m.gpu = ids[0]; }
        }
        Ok(m)
    }

    /// Row group `i` of `n` (see `load_groups`).
    pub fn group(&self, i: usize, n: usize) -> Mat<'a> {
        let rows = self.rows / n;
        let sz = rows * self.ty.row_bytes(self.cols as u64);
        Mat { data: &self.data[i * sz..(i + 1) * sz], rows, gpu: self.gpu.map(|b| b + i as u32), ..*self }
    }

    /// Slice of a [cols, rows, n_expert] tensor for expert e.
    pub fn expert(&self, e: usize, n_expert: usize) -> Mat<'a> {
        self.expert_opt(e, n_expert, true)
    }

    /// `expert` that never overwrites a VRAM slot when `evict` is false: an expert that is not resident and has no
    /// free room comes back host-only, for the CPU.
    pub fn expert_opt(&self, e: usize, n_expert: usize, evict: bool) -> Mat<'a> {
        let mut m = self.expert_host(e, n_expert);
        m.gpu = dev::get().and_then(|d| d.lock().unwrap().expert(m.data, m.ty, m.rows, m.cols, evict));
        m
    }

    /// This matrix copied to the GPU for each use (no VRAM cache slot). Unchanged without a GPU.
    pub fn staged(mut self) -> Self {
        if dev::get().is_some() { self.gpu = Some(dev::STAGED); }
        self
    }

    /// Expert e read from the mapped file, without a VRAM cache lookup.
    pub fn expert_host(&self, e: usize, n_expert: usize) -> Mat<'a> {
        let rows = self.rows / n_expert;
        let sz = rows * self.ty.row_bytes(self.cols as u64);
        Mat { data: &self.data[e * sz..(e + 1) * sz], ty: self.ty, rows, cols: self.cols, cpu: self.cpu, gpu: None }
    }

    /// Whether this matrix has no VRAM copy and must be read from the mapped file (CPU or staged expert).
    pub fn host_only(&self) -> bool { self.gpu.is_none() || self.gpu == Some(dev::STAGED) }

    /// Ask the kernel to read expert e's pages now (madvise WILLNEED), so all selected experts of a layer queue on the SSD together.
    #[cfg(target_os = "linux")]
    pub fn prefetch_expert(&self, e: usize, n_expert: usize) {
        unsafe extern "C" { fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32; }
        // A model that fits in RAM is read into the page cache at start (`weights::preload`) and stays there: once
        // those reads are queued the call would only cost its page-table walk, about 4 us per expert matrix.
        if crate::weights::preloaded() { return; }
        let sz = self.rows / n_expert * self.ty.row_bytes(self.cols as u64);
        // An expert with a VRAM slot runs from there: its host pages are not needed, and reading them would push
        // other experts out of the page cache.
        if dev::get().is_some_and(|d| d.lock().unwrap().resident(&self.data[e * sz..])) { return; }
        let a = self.data[e * sz..].as_ptr() as usize;
        let start = a & !4095;
        unsafe { madvise(start as *mut std::ffi::c_void, a + sz - start, 3) };
    }
    /// No madvise outside Linux (the MSVC linker has no such symbol, issue #21): the pages are read when touched.
    #[cfg(not(target_os = "linux"))]
    pub fn prefetch_expert(&self, _e: usize, _n_expert: usize) {}

    pub fn row_into(&self, r: usize, out: &mut [f32]) {
        let rb = self.ty.row_bytes(self.cols as u64);
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }

    /// y = W x
    /// The integer dot of the CPU path, when this matrix uses it. Integer Q8_K dots are several times faster than
    /// dequantising each row, and match llama.cpp's CPU path. `STRATA_CPU_REF=1` keeps the f32 reference for
    /// debugging and exact old numerics.
    fn int_dot(&self) -> Option<fn(&[u8], &[crate::cpu_moe::Q8K]) -> f32> {
        if (self.cpu || self.rows >= 512) && std::env::var_os("STRATA_CPU_REF").is_none() && self.cols % self.ty.block_elems() == 0 {
            crate::cpu_moe::int_dot(self.ty)
        } else {
            None
        }
    }

    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(y.len(), self.rows);
        if let (Some(i), Some(d)) = (self.gpu, dev::get()) {
            return d.lock().unwrap().gemv(i, x, y, (i == dev::STAGED).then_some((self.data, self.ty, self.rows, self.cols)));
        }
        let rb = self.ty.row_bytes(self.cols as u64);
        if let Some(dot) = self.int_dot() {
            let xq = crate::cpu_moe::quantize_q8k(x);
            y.par_chunks_mut(32).enumerate().for_each(|(c, ys)| {
                for (k, o) in ys.iter_mut().enumerate() {
                    let r = c * 32 + k;
                    *o = dot(&self.data[r * rb..(r + 1) * rb], &xq);
                }
            });
            return;
        }
        let xq = q8_roundtrip(self.ty, self.cpu, x);
        let x = &xq[..];
        y.par_iter_mut().enumerate().for_each_init(
            || vec![0f32; self.cols],
            |tmp, (r, o)| {
                dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], tmp);
                *o = tmp.iter().zip(x).map(|(a, b)| a * b).sum();
            },
        );
    }
}

/// ggml quantises the activation to Q8_K (k-quants, IQ1/IQ2) or Q8_0 (Q8_0, IQ4_NL) before a quantised dot.
/// Reproduce that rounding so the f32 reference tracks llama.cpp; other types see the activation unchanged.
fn q8_roundtrip(ty: GgmlType, cpu: bool, x: &[f32]) -> Vec<f32> {
    use GgmlType::*;
    let mut out = x.to_vec();
    match ty {
        Q5K | IQ1M | IQ2XXS | IQ2XS | IQ2S if cpu => {
            for b in out.chunks_exact_mut(256) {
                let (mut amax, mut max) = (0f32, 0f32);
                for &v in b.iter() {
                    if v.abs() > amax { amax = v.abs(); max = v; }
                }
                if amax == 0.0 { b.fill(0.0); continue; }
                let iscale = -127.0 / max;
                for v in b.iter_mut() { *v = (iscale * *v).round_ties_even().min(127.0) / iscale; }
            }
        }
        Q5K | IQ1M | IQ2XXS | IQ2XS | IQ2S | Q8_0 | IQ4NL => {
            for b in out.chunks_exact_mut(32) {
                let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                if d == 0.0 { continue; }
                let dh = half::f16::from_f32(d).to_f32();
                for v in b.iter_mut() { *v = (*v / d).round() * dh; }
            }
        }
        _ => {}
    }
    out
}

/// Small F32/BF16 tensor fully dequantised.
pub fn vec_f32(g: &Gguf, name: &str) -> anyhow::Result<Vec<f32>> {
    let m = Mat::load(g, name)?;
    let mut v = vec![0f32; m.rows * m.cols];
    for r in 0..m.rows {
        m.row_into(r, &mut v[r * m.cols..(r + 1) * m.cols]);
    }
    Ok(v)
}

#[cfg(test)]
mod budget_tests {
    use super::dev::{AVAIL_KEEP, carve, plan};
    const G: usize = 1 << 30;

    /// Slots come out of one arena chunk at a time: an address 256-byte aligned inside the current chunk while it has
    /// room, and None when a new chunk is needed. The point of the arena is that the number of driver buffers does not
    /// grow with the number of experts, so this must never hand out two overlapping ranges nor skip a chunk.
    #[test]
    fn an_arena_carves_slots_and_wraps_to_a_new_chunk() {
        assert_eq!(carve(0x1000, 1024, 256), Some((0x1000, 768)));
        assert_eq!(carve(0x1000, 200, 256), None, "no room left in the chunk: the caller takes a new one");
        assert_eq!(carve(0x1000, 0, 4096), None, "an empty chunk cannot serve a big matrix");
        // Exactly the last bytes of a chunk, then nothing.
        assert_eq!(carve(0x1000, 256, 256), Some((0x1000, 0)));
        // Walked as `Arena::take` walks it: consecutive slots, one new chunk each time the previous runs out.
        let (mut next, mut left) = (0u64, 1024usize);
        let mut got = vec![];
        for _ in 0..6 {
            if carve(next, left, 256).is_none() { (next, left) = (0x8000, 1024); }
            let (a, l) = carve(next, left, 256).unwrap();
            got.push(a);
            (next, left) = (a + 256, l);
        }
        assert_eq!(got, vec![0, 256, 512, 768, 0x8000, 0x8100], "four slots per 1 KB chunk, then the next chunk");
    }

    /// `all` (`STRATA_DS_GPU_EXPERTS=all`): as much as the machine can back on a shared card, the card's own heap on a
    /// discrete one, and never the absurd number an unclamped infinity would cast to.
    #[test]
    fn all_means_what_the_machine_can_back() {
        let inf = f64::INFINITY;
        let ram = 125 * G;
        // An idle box: the machine's size, less what the run needs for itself.
        assert_eq!(plan(83 * G, 83 * G, 0.8, Some(inf), true, ram, ram, false).0, ram - AVAIL_KEEP);
        // A box with 20 GB free cannot be asked for more than that, however the request is phrased.
        let (b, capped) = plan(83 * G, 83 * G, 0.8, Some(inf), true, ram, 20 * G, false);
        assert_eq!((b, capped), (20 * G - AVAIL_KEEP, Some(20 * G - AVAIL_KEEP)));
        // A discrete card takes the heap it was offered; there is no machine ceiling to report.
        assert_eq!(plan(24 * G, 24 * G, 0.8, Some(inf), false, ram, ram, false), (24 * G - (56 << 20), None));
        // Even with the ceiling switched off, `all` is the heap on the table and not `usize::MAX`.
        assert_eq!(plan(24 * G, 24 * G, 0.8, Some(inf), true, ram, ram, true).0, 24 * G - (56 << 20));
    }

    /// A discrete card keeps the old rule: `frac` of what it reports free, and an absolute request is taken as asked. Its
    /// own memory is not RAM the page cache needs, so neither the size of the machine nor its free space comes into it.
    #[test]
    fn a_discrete_card_keeps_the_fraction_of_its_own_memory() {
        let (b, capped) = plan(12 * G, 12 * G, 0.8, None, false, 32 * G, 4 * G, false);
        assert_eq!(capped, None);
        assert!((b as f64 - 0.8 * (12 * G - 56 * 1024 * 1024) as f64).abs() < 2.0, "{b}");
        assert_eq!(plan(12 * G, 12 * G, 0.8, Some(9.0), false, 32 * G, 1 * G, false).0, (9e9) as usize);
    }

    /// The Strix Halo box from the report: an 82.9 GiB "free" device heap on 125 GiB of RAM, empty enough that only the
    /// machine's size limits it. 80% of that heap (66 GB) is what over-committed it; an eighth of RAM, with Hy4-flash's
    /// ~90 GiB of experts in page cache beside it, is what ran.
    #[test]
    fn a_shared_heap_is_capped_at_a_fraction_of_ram() {
        let ram = 125 * G;
        let (b, capped) = plan(83 * G, 83 * G, 0.8, None, true, ram, ram, false);
        assert_eq!(capped, Some(ram / 8), "the ceiling is returned so the load line can say why the budget shrank");
        assert_eq!(b, ram / 8);
        assert!(b + 90 * G + 4 * G <= ram, "device copy plus the model's own pages have to fit in RAM");
    }

    /// A box already serving another model is the case that matters here: `total_ram` says 125 GiB while 20 are free.
    /// The budget comes down to what is actually available, less what this process needs for itself (`AVAIL_KEEP`).
    #[test]
    fn a_busy_box_limits_it_further() {
        let ram = 125 * G;
        let (b, capped) = plan(83 * G, 83 * G, 0.8, None, true, ram, 8 * G, false);
        assert_eq!(b, 8 * G - AVAIL_KEEP);
        assert_eq!(capped, Some(8 * G - AVAIL_KEEP));
        // With next to nothing free the floor keeps a token cache, so the card still does what it can.
        assert_eq!(plan(83 * G, 83 * G, 0.8, None, true, ram, 1 << 30, false).0, 1 << 30);
    }

    /// Asking for more than the default eighth is what raises how much of the model sits on the card, up to what RAM can
    /// back at all — beyond that even holding each byte once does not fit.
    #[test]
    fn an_explicit_budget_raises_the_default_ceiling() {
        let ram = 125 * G;
        // A request is not held down to the default eighth: it raises how much of the model the card holds, because
        // past the point where budget and file pages no longer fit RAM each uploaded byte hands its page back.
        assert_eq!(plan(83 * G, 83 * G, 0.8, Some(66.0), true, ram, ram, false).0, (66e9) as usize);
        // But a box already holding another model cannot take it just because the machine has the memory fitted: the
        // driver refuses the submission, so the request follows free space too.
        let (b, capped) = plan(83 * G, 83 * G, 0.8, Some(40.0), true, ram, 16 * G, false);
        assert_eq!((b, capped), (16 * G - AVAIL_KEEP, Some(16 * G - AVAIL_KEEP)));
        // Up to what RAM could back at all, then: a small machine keeps the pages around its own working set.
        let (b, capped) = plan(83 * G, 83 * G, 0.8, Some(50.0), true, 24 * G, 24 * G, false);
        assert_eq!((b, capped), (24 * G - AVAIL_KEEP, Some(24 * G - AVAIL_KEEP)));
        // The ceiling gone too, at the caller's own risk.
        assert_eq!(plan(83 * G, 83 * G, 0.8, Some(50.0), true, 24 * G, 24 * G, true).0, (50e9) as usize);

        // Two identical APUs split one ceiling between them by how much heap each had free.
        let (a, c) = plan(40 * G, 80 * G, 0.8, None, true, ram, ram, false);
        assert_eq!((a, c), (ram / 16, Some(ram / 16)));

        // Without a measurable RAM size nothing is clamped: an unknown must not silently mean "no memory".
        assert_eq!(plan(83 * G, 83 * G, 0.8, None, true, 0, 0, false).1, None);
    }
}
