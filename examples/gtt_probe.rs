//! What this device can actually back, measured rather than inferred:
//!   gtt_probe [max_gib=2] [step_mib=256] [--backend cuda|vulkan] [--device N] [--resident FILE]
//!   gtt_probe --bo-count N [--bo-mib M]     same question, asked with many small buffers instead of few big ones
//!
//! Every buffer stays allocated and a copy is submitted after each one, because an over-commit on an integrated GPU does
//! not fail the allocation - it fails the submission (`__radv/amdgpu: Not enough memory for command submission`) and the
//! card goes with it. The number printed at each step is the total held so far, which is what `STRATA_DS_GPU_EXPERTS`
//! competes with everything else for.
//!
//! `--resident FILE` maps a file first (a model shard) and asks the kernel to read it in, so the measurement happens with
//! that much page cache already taken: on an APU the weights in cache and the copies on the card are claims on one pool,
//! which is exactly the pair that has to be sized together. Run it twice, with and without `--resident`, and the gap says
//! whether the budget should follow free RAM or free VRAM.
//!
//! Start small (the 2 GiB default) and raise `max_gib` by a few at a time: past this box's real ceiling the driver may
//! not recover, so do not probe with a number you cannot afford to lose. Exit status is 0 either way - the answer is the
//! last line it prints, and a run that dies mid-way has itself answered the question.

use rusting_strata::backend;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let positional: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).filter(|a| a.parse::<f64>().is_ok()).collect();
    let max_gib: f64 = positional.first().and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let step_mib: usize = positional.get(1).and_then(|s| s.parse().ok()).unwrap_or(256) << 20;
    let (kind, device) = (flag("--backend"), flag("--device").map_or(0, |d| d.parse().unwrap_or(0)));

    if let Some(p) = flag("--resident") {
        // Map and read the shard in, from its own thread order: `weights::will_need` is what a load uses for this.
        let f = std::fs::File::open(&p)?;
        let m = unsafe { memmap2::Mmap::map(&f)? };
        println!("resident: reading {} GB of {} into the page cache", m.len() / 1_000_000_000, p);
        rusting_strata::weights::will_need(m.as_ptr() as usize, m.len());
        std::thread::sleep(std::time::Duration::from_secs(25)); // one readahead pass; the map lives until `m` drops
        println!("resident: {} GB mapped (page cache may show less; only clean pages count against us)", m.len() / 1_000_000_000);
    }

    let be = backend::open(kind.as_deref(), device)?;
    let (free, total) = be.mem_info().unwrap_or((0, 0));
    println!("device {} ({}, {}): heap {:.1} of {:.1} GB free{}", be.device(), be.name(), device, free as f64 / 1e9, total as f64 / 1e9,
        if be.unified_memory() { ", shares system RAM with the CPU" } else { ", its own memory" });
    let ram = rusting_strata::memory::total_ram().unwrap_or(0) as f64 / 1e9;
    let avail = rusting_strata::memory::available_ram().unwrap_or(0) as f64 / 1e9;
    println!("host: {ram:.1} GB RAM, {avail:.1} GB available now");

    let limit = (max_gib * (1u64 << 30) as f64) as usize;
    // Two ways a shared-memory card can run out, and they need different fixes: total bytes pinned, or the number of
    // separate buffers mapped into its VM (the engine takes one per distinct expert it has ever uploaded, which is how
    // `amdgpu_gem_va_update_vm ... -12` appears long before the byte budget is spent). Phase 1 asks for few large
    // buffers; phase 2 (`--bo-count`, implied by a small --bo-mib) asks for many small ones of the engine's own shape.
    if let Some(n_str) = flag("--bo-count") {
        let n: usize = n_str.parse().unwrap_or(0);
        let size_mib: usize = flag("--bo-mib").and_then(|v| v.parse().ok()).unwrap_or(4);
        return probe_buffers(be.as_ref(), n, size_mib << 20);
    }

    let mut held = vec![];
    let mut done = 0usize;
    while done + step_mib <= limit {
        match be.alloc(step_mib) {
            // Touch it: an allocation the driver has never written is not yet a promise about physical pages, and it is
            // the submission after it - not the alloc - that reports an amdgpu over-commit.
            Ok(b) => match be.memset(&b).and_then(|()| be.sync()) {
                Ok(()) => {
                    held.push(b);
                    done += step_mib;
                    if done % (1 << 30) == 0 || done + step_mib > limit { println!("ok: {:.1} GB held", done as f64 / 1e9); }
                }
                Err(e) => { println!("FAILED to submit at {:.2} GB total: {e}", (done + step_mib) as f64 / 1e9); break; }
            },
            Err(e) => { println!("FAILED to allocate at {:.2} GB total: {e}", (done + step_mib) as f64 / 1e9); break; }
        }
    }
    println!("one-buffer-at-a-time ceiling: {:.1} GB held in {} allocation(s)", done as f64 / 1e9, held.len());
    eprintln!("now run --bo-count 8000 --bo-mib <expert MiB> to see if the same number survives when it is many buffers.");
    println!("largest that still submitted: {:.1} GB. Set STRATA_DS_GPU_EXPERTS under it - two thirds, say - and remember", done as f64 / 1e9);
    println!("that on this engine the plain non-expert weights are counted against the same budget.");
    Ok(())
}

/// Hold `n` buffers of `size` bytes each, submitting a copy after every one, and report both how many survived and what
/// the driver thought was free when it stopped. Compare with the single-large-buffer phase: identical failure *bytes* at
/// wildly different buffer counts means the ceiling is VM mappings, not memory, and the engine must reuse buffers rather
/// than ask for a bigger budget.
fn probe_buffers(be: &dyn backend::Backend, n: usize, size: usize) -> anyhow::Result<()> {
    let mut held = vec![];
    while held.len() < n {
        match be.alloc(size) {
            Ok(b) => match be.memset(&b).and_then(|()| be.sync()) {
                Ok(()) => held.push(b),
                Err(e) => break_e("submit", held.len(), size, &e, be),
            },
            Err(e) => break_e("allocate", held.len(), size, &e, be),
        }
    }
    let g = |b: usize| (held.len() * b) as f64 / 1e9;
    println!("{} buffers of {} MiB held ({:.1} GB): {}", held.len(), size >> 20, g(size),
        if held.len() >= n { "no limit reached in this run" } else { "stopped here" });
    Ok(())
}

/// Reports the failure with what the heap still claims to have free - the number that says whether radv ran out of its
/// own carve-out or the kernel ran out of something it does not report through Vulkan.
fn break_e(what: &str, n: usize, size: usize, e: &anyhow::Error, be: &dyn backend::Backend) -> ! {
    let free = be.mem_info().map(|(f, _)| format!("{:.1} GB free", f as f64 / 1e9)).unwrap_or_default();
    println!("FAILED to {what} buffer #{} of {} MiB ({} held, {:.1} GB, {free}): {e}", n + 1, size >> 20, n, (n * size) as f64 / 1e9);
    std::process::exit(0);
}
