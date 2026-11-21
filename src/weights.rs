//! Named tensor views over the mmapped GGUF, page cache advice for them, and a reference parallel matvec.

use crate::config::ModelConfig;
use crate::gguf::Gguf;
use crate::quant::{GgmlType, dequant_row};
use anyhow::Context;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    pub ty: GgmlType,
    /// Elements per row (GGUF dims[0]).
    pub cols: usize,
    /// Product of the remaining dims; for 3-D expert tensors this is out_rows * n_expert.
    pub rows: usize,
    pub data: &'a [u8],
}

impl<'a> Tensor<'a> {
    fn row_bytes(&self) -> usize { self.ty.row_bytes(self.cols as u64) }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        let rb = self.row_bytes();
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }

    /// y[r] = row_r · x for all rows.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!((x.len(), y.len()), (self.cols, self.rows));
        y.par_chunks_mut(16).enumerate().for_each_init(|| vec![0f32; self.cols], |buf, (ci, ys)| {
            for (k, yv) in ys.iter_mut().enumerate() {
                self.row(ci * 16 + k, buf);
                *yv = buf.iter().zip(x).map(|(a, b)| a * b).sum();
            }
        });
    }

    /// out[t][r] = row_r · x[t] for all rows and tokens. `x` is `n_tokens * cols`, `out` is `n_tokens * rows`.
    pub fn matmul(&self, x: &[f32], n_tokens: usize, out: &mut [f32]) {
        assert_eq!((x.len(), out.len()), (n_tokens * self.cols, n_tokens * self.rows));
        // Row-outer: each weight row is decoded once, whatever n_tokens is. Blocks of rows that fit the L2 cache stay
        // there while every token passes over them.
        let block = (1 << 16) / self.cols.max(1) + 1;
        let mut by_row = vec![0f32; out.len()];
        by_row.par_chunks_mut(block * n_tokens).enumerate().for_each_init(|| vec![0f32; block * self.cols], |buf, (b, ys)| {
            let nr = ys.len() / n_tokens;
            for (r, w) in buf.chunks_exact_mut(self.cols).take(nr).enumerate() { self.row(b * block + r, w); }
            for (t, xt) in x.chunks_exact(self.cols).enumerate() {
                for (r, w) in buf.chunks_exact(self.cols).take(nr).enumerate() { ys[r * n_tokens + t] = dot(w, xt); }
            }
        });
        out.par_chunks_mut(self.rows).enumerate().for_each(|(t, ys)| {
            for (r, yv) in ys.iter_mut().enumerate() { *yv = by_row[r * n_tokens + t]; }
        });
    }

    /// Rows belonging to expert `e` of an [cols, out, n_expert] tensor.
    pub fn expert(&self, e: usize, n_expert: usize) -> Tensor<'a> {
        let rows = self.rows / n_expert;
        let rb = self.row_bytes();
        Tensor { rows, data: &self.data[e * rows * rb..(e + 1) * rows * rb], ..*self }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.rows * self.cols];
        for (r, out) in v.chunks_exact_mut(self.cols).enumerate() { self.row(r, out); }
        v
    }
}

pub struct Weights<'a> {
    pub g: &'a Gguf,
    pub aux_g: Option<&'a Gguf>,
    aliases: &'static [(&'static str, &'static str)],
    phys_layer: Vec<usize>,
    /// Loader-owned overrides: synthesized `ssm_beta`/`ssm_alpha` and LoRA-baked tensors.
    pub built: HashMap<String, Tensor<'a>>,
    /// Heap storage the baked tensors in `built` point into. Kept beside `built` so a reload frees it instead of
    /// leaking the F32 copy; `built` is cleared before `baked` so no tensor outlives its bytes.
    baked: HashMap<String, Box<[u8]>>,
    pub lora_baked: std::collections::HashSet<String>,
}

impl<'a> Weights<'a> {
    pub fn new(g: &'a Gguf) -> anyhow::Result<Weights<'a>> {
        Self::new_with_aux(g, None)
    }

    pub fn new_with_aux(g: &'a Gguf, aux_g: Option<&'a Gguf>) -> anyhow::Result<Weights<'a>> {
        let c = ModelConfig::from_gguf(g)?;
        let mut w = Weights {
            g,
            aux_g,
            aliases: c.aliases,
            phys_layer: c.phys_layer.clone(),
            built: HashMap::new(),
            baked: HashMap::new(),
            lora_baked: std::collections::HashSet::new(),
        };
        // llama.cpp src/models/qwen3next.cpp: per key head group, r = nv / nk beta rows, then r alpha rows.
        for il in 0..c.n_layer {
            let p = c.phys_layer.get(il).copied().unwrap_or(il);
            if p != il { continue; }
            let leak = |v: Vec<u8>| -> &'a [u8] { Box::leak(v.into_boxed_slice()) };
            if let Ok(qkvg) = w.layer(il, "attn_qkv_gate.weight") {
                let channels = c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state;
                let rb = qkvg.row_bytes();
                anyhow::ensure!(qkvg.rows == channels + c.ssm_d_inner,
                    "blk.{il}.attn_qkv_gate: {} rows, expected {}", qkvg.rows, channels + c.ssm_d_inner);
                let split_at = channels * rb;
                w.built.insert(format!("blk.{il}.attn_qkv.weight"), Tensor { rows: channels, data: &qkvg.data[..split_at], ..qkvg });
                w.built.insert(format!("blk.{il}.attn_gate.weight"), Tensor { rows: c.ssm_d_inner, data: &qkvg.data[split_at..], ..qkvg });
            }
            if let Ok(ba) = w.layer(il, "ssm_beta_alpha.weight") {
                let (nv, rb) = (c.ssm_dt_rank, ba.row_bytes());
                anyhow::ensure!(nv > 0 && ba.rows == 2 * nv, "blk.{il}.ssm_beta_alpha: {} rows for {nv}", ba.rows);
                let split_at = nv * rb;
                let half = Tensor { rows: nv, ..ba };
                w.built.insert(format!("blk.{il}.ssm_beta.weight"), Tensor { data: &ba.data[..split_at], ..half });
                w.built.insert(format!("blk.{il}.ssm_alpha.weight"), Tensor { data: &ba.data[split_at..], ..half });
            }
            let Ok(ba) = w.layer(il, "ssm_ba.weight") else { continue };
            let (nk, nv, rb) = (c.ssm_n_group, c.ssm_dt_rank, ba.row_bytes());
            anyhow::ensure!(nk > 0 && nv > 0 && nv % nk == 0 && ba.rows == 2 * nv && ba.data.len() == ba.rows * rb,
                "blk.{il}.ssm_ba: {} rows, {} bytes for group_count {nk}, time_step_rank {nv}", ba.rows, ba.data.len());
            let r = nv / nk;
            let (mut b, mut a) = (Vec::with_capacity(ba.data.len() / 2), Vec::with_capacity(ba.data.len() / 2));
            for grp in ba.data.chunks_exact(2 * r * rb) {
                b.extend_from_slice(&grp[..r * rb]);
                a.extend_from_slice(&grp[r * rb..]);
            }
            // ponytail: leaked, a few KB per layer per load; own the bytes in Weights if one process loads many models.
            let half = Tensor { rows: ba.rows / 2, ..ba };
            w.built.insert(format!("blk.{il}.ssm_beta.weight"), Tensor { data: leak(b), ..half });
            w.built.insert(format!("blk.{il}.ssm_alpha.weight"), Tensor { data: leak(a), ..half });
        }
        // Standalone EAGLE3 draft GGUF: remap layer-0 draft weights onto blk.{n_layer}.* so they never shadow layer 0.
        if let Some(ag) = aux_g.filter(|ag| ag.tensors.contains_key("fc.weight")) {
            let nl = c.n_layer;
            for (name, t) in &ag.tensors {
                if matches!(name.as_str(), "d2t" | "t2d" | "token_embd.weight") { continue; }
                let Ok(ty) = GgmlType::from_u32(t.ty) else { continue };
                let dst = match name.as_str() {
                    "fc.weight" => format!("blk.{nl}.eagle3.fc.weight"),
                    "output.weight" => format!("blk.{nl}.eagle3.output.weight"),
                    "output_norm.weight" => format!("blk.{nl}.nextn.shared_head_norm.weight"),
                    "blk.0.attn_norm.weight" => format!("blk.{nl}.nextn.enorm.weight"),
                    "blk.0.attn_norm_2.weight" => format!("blk.{nl}.nextn.hnorm.weight"),
                    "blk.0.ffn_norm.weight" => format!("blk.{nl}.post_attention_norm.weight"),
                    _ if name.starts_with("blk.0.") => format!("blk.{nl}.{}", &name["blk.0.".len()..]),
                    _ => continue,
                };
                w.built.insert(dst, Tensor {
                    ty,
                    cols: t.dims[0] as usize,
                    rows: t.dims[1..].iter().product::<u64>() as usize,
                    data: ag.bytes(t),
                });
            }
        }
        Ok(w)
    }

    /// `name` with the architecture's alias for a canonical suffix tried second.
    pub fn get(&self, name: &str) -> anyhow::Result<Tensor<'a>> {
        if let Some(t) = self.built.get(name) { return Ok(*t); }
        let find_t = |g: &'a Gguf| -> Option<(&'a crate::gguf::TensorInfo, &'a Gguf)> {
            if let Some(t) = g.tensors.get(name) {
                return Some((t, g));
            }
            if let Some(t) = self.aliases.iter().find_map(|&(c, a)| g.tensors.get(&format!("{}{a}", name.strip_suffix(c)?))) {
                return Some((t, g));
            }
            None
        };
        let (t, owner_g) = find_t(self.g)
            .or_else(|| self.aux_g.and_then(find_t))
            .with_context(|| format!("missing tensor {name}"))?;
        let ty = GgmlType::from_u32(t.ty).with_context(|| format!("tensor {name}"))?;
        Ok(Tensor {
            ty,
            cols: t.dims[0] as usize,
            rows: t.dims[1..].iter().product::<u64>() as usize,
            data: owner_g.bytes(t),
        })
    }

    pub fn layer(&self, il: usize, suffix: &str) -> anyhow::Result<Tensor<'a>> {
        let p = self.phys_layer.get(il).copied().unwrap_or(il);
        if p != il
            && let Ok(t) = self.get(&format!("blk.{il}.{suffix}"))
        {
            return Ok(t);
        }
        self.get(&format!("blk.{p}.{suffix}"))
    }

    /// Refuse files containing quant types this engine does not implement, except in the MTP (nextn) layers, blocks
    /// `n_layer` on: false if one of those has such a type, so that the model loads without MTP.
    /// Drop the LoRA-baked overrides and the heap copies they point into; synthesized `ssm_*` tensors stay.
    pub fn clear_baked(&mut self) {
        for name in std::mem::take(&mut self.lora_baked) { self.built.remove(&name); }
        self.baked.clear();
    }

    pub fn bake_lora(&mut self, loras: &[crate::model::Lora]) {
        for lora in loras {
            for (name, entry) in &lora.entries {
                if name.ends_with("ffn_down_exps.weight") { continue; }
                if entry.n_expert != 1 {
                    eprintln!("lora: target {name} has {} experts, expected 1; skipping", entry.n_expert);
                    continue;
                }
                let base = match self.get(name) { Ok(t) => t, Err(_) => { eprintln!("lora: target {name} not found, skipping"); continue; } };
                if base.cols != entry.in_dim || base.rows != entry.out_dim {
                    eprintln!("lora: target {name} is {}x{}, adapter is {}x{}; skipping", base.rows, base.cols, entry.out_dim, entry.in_dim);
                    continue;
                }
                let mut w = base.to_f32();
                let (a, b) = (&entry.a[0], &entry.b[0]);
                let (rank, in_dim, out_dim) = (entry.rank, entry.in_dim, entry.out_dim);
                let mult = if lora.alpha > 0.0 { lora.scale * lora.alpha / rank as f32 } else { lora.scale };
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        let mut d = 0f32;
                        for r in 0..rank { d += b[o * rank + r] * a[r * in_dim + i]; }
                        w[o * in_dim + i] += mult * d;
                    }
                }
                self.built.remove(name);
                self.baked.remove(name);
                let bytes: Box<[u8]> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
                // `bytes` lives in `self.baked` until `clear_baked`, which removes the tensor below first.
                let data: &'a [u8] = unsafe { std::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) };
                let t = Tensor { ty: crate::quant::GgmlType::F32, cols: base.cols, rows: base.rows, data };
                self.baked.insert(name.clone(), bytes);
                self.built.insert(name.clone(), t);
                self.lora_baked.insert(name.clone());
            }
        }
    }
    pub fn validate(&self, n_layer: usize) -> anyhow::Result<bool> {
        let mut nextn = true;
        let iter = self.g.tensors.values().chain(self.aux_g.into_iter().flat_map(|ag| ag.tensors.values()));
        for t in iter {
            if matches!(t.name.as_str(), "d2t" | "t2d") { continue; }
            if let Err(e) = GgmlType::from_u32(t.ty) {
                let il = t.name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|i| i.parse::<usize>().ok());
                if il.is_some_and(|il| il >= n_layer) { nextn = false; continue; }
                return Err(e).with_context(|| format!("tensor {}", t.name));
            }
        }
        Ok(nextn)
    }
}

#[cfg(target_os = "linux")]
fn madvise(p: usize, n: usize, advice: i32) {
    unsafe extern "C" { fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32; }
    unsafe { madvise(p as *mut _, n, advice) };
}
#[cfg(not(target_os = "linux"))]
fn madvise(_: usize, _: usize, _: i32) {}

/// Starts reading mapped bytes p..p + n into the page cache without waiting (MADV_WILLNEED).
pub fn will_need(p: usize, n: usize) {
    // One call reads at most one readahead window and silently skips the rest of its range (16 MB steps on a 4 MB
    // window read 25% of a file). The window is read_ahead_kb, 128 KB unless the system or the file system raised it.
    let (mut a, end) = (p & !4095, p + n);
    while a < end {
        let l = (end - a).min(128 << 10);
        madvise(a, l, 3);
        a += l;
    }
}

/// Bytes of the GGUF files this process has mapped now, not counting their row-read tables (`Gguf::table_ranges`): the
/// weights a token can ask for are what has to fit.
static MAPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes the GPU's weight budget will hold (`deepseek::mat::plan`, set once when the device opens). Counted with the
/// mapped files, because on an integrated GPU a device copy and a page-cache copy are the same RAM twice; together
/// they have to fit. A claim large enough not to is what turns `cold` back on for experts that live in VRAM.
static DEVICE_CLAIMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn device_claim(bytes: usize) {
    DEVICE_CLAIMED.store(bytes as u64, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the mapped model files fit in RAM with 4 GiB to spare: then their pages stay in the page cache (`cold` is
/// a no-op) and `preload` reads them. `STRATA_PRELOAD=0`/`1` overrides.
// ponytail: MemTotal, not what other programs left free; use MemAvailable if that decides wrongly.
pub fn fits_ram() -> bool {
    match std::env::var("STRATA_PRELOAD").ok().as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => { static RAM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new(); *RAM.get_or_init(crate::memory::total_ram) }.is_some_and(|t| MAPPED.load(std::sync::atomic::Ordering::Relaxed) + DEVICE_CLAIMED.load(std::sync::atomic::Ordering::Relaxed) + (4 << 30) <= t),
    }
}

/// Counts a newly mapped model file and, with `read`, when the files fit in RAM (`fits_ram`), a background thread starts
/// reading it into the page cache, so experts that run on the CPU do not wait for the drive on their first use
/// (issue #49). `skip` is the shard's row-read tables (`Gguf::table_ranges`, ascending by start): those bytes are neither
/// counted nor read in, because 59 GB of Engram rows touched once at start would push out the expert pages every token
/// reads. Returns the bytes added to the count, which `unmapped` takes off when the file closes.
pub fn preload(b: &[u8], skip: &[(usize, usize)], read: bool) -> u64 {
    // Everything but the tables: the bytes that should stay resident, as absolute addresses. MADV_WILLNEED on a range
    // unmapped in the meantime fails without a fault, so the thread below holds no borrow of `b`.
    let mut resident = vec![];
    let (base, mut cur) = (b.as_ptr() as usize, 0usize);
    for &(s, n) in skip {
        let (s, e) = (s.min(b.len()), s.saturating_add(n).min(b.len()));
        if e <= cur { continue; }  // a range already walked past, or one contained in the last
        if s > cur { resident.push((base + cur, s - cur)); }
        cur = e;
    }
    if cur < b.len() { resident.push((base + cur, b.len() - cur)); }
    // The count is these bytes and nothing else. It used to be `b.len() - resident`, which counted the tables that were
    // deliberately left out instead of the weights that were read in: `fits_ram` then said a 100 GB model fit anywhere,
    // so `cold` stopped dropping the host copy of experts the card had taken and every uploaded byte was held twice -
    // which on a shared-memory GPU is precisely the over-commit that runs the machine out of page-table memory.
    let keep = resident.iter().map(|&(_, n)| n as u64).sum::<u64>();
    MAPPED.fetch_add(keep, std::sync::atomic::Ordering::Relaxed);
    if read && fits_ram() {
        PRELOADING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::thread::spawn(move || {
            for (p, n) in resident { will_need(p, n); }
            PRELOADING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        });
    }
    keep
}

/// Bytes counted as needing to stay resident: the mapped model files less their row-read tables (`preload`). Said out
/// loud at load on a shared-memory GPU, where this and `device_claim` are two claims on one pool.
pub fn mapped_bytes() -> u64 {
    MAPPED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Takes the bytes of an unmapped model file off the count.
pub fn unmapped(n: usize) {
    MAPPED.fetch_sub(n as u64, std::sync::atomic::Ordering::Relaxed);
}

/// `preload` threads still queueing reads.
static PRELOADING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether every read `preload` asked for is queued: from then on a readahead request for a part of the model adds
/// nothing. False until then, and always when the model does not fit in RAM.
pub fn preloaded() -> bool {
    static FITS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    PRELOADING.load(std::sync::atomic::Ordering::Relaxed) == 0 && *FITS.get_or_init(fits_ram)
}

/// Upload `data`, then give its file pages back on a device that shares system RAM. The order is the point: the upload
/// reads every byte of `data`, so `cold` before it only drops pages for the copy to fault straight back in. That leaves
/// the GPU weights *and* their page-cache copies resident — 40 GB on the card beside 40 GB of file pages on a 134 GB
/// box is what makes amdgpu refuse a submission. A failed upload keeps the host copy (nothing was copied yet).
pub fn upload_then_cold<T>(data: &[u8], unified: bool, upload: impl FnOnce() -> Option<T>) -> Option<T> {
    upload_then(unified, upload, || cold(data))
}

/// The same ordering with the release step passed in, so the order can be tested without a GPU or a page-cache policy.
fn upload_then<T>(unified: bool, upload: impl FnOnce() -> Option<T>, release: impl FnOnce()) -> Option<T> {
    let t = upload()?;
    if unified { release(); }
    Some(t)
}

/// Drops the whole pages of `b` (MADV_DONTNEED): bytes now on the GPU, whose host copy competes with the experts that
/// missed VRAM for page cache. Expert files exceed the page cache, so pages read once at load would otherwise outlive
/// the experts a prompt reads from it. `STRATA_COLD_KEEP` only marks them reclaimable (MADV_COLD) instead. Nothing is
/// dropped when the model fits in RAM (`fits_ram`). Call this *after* the last host read of `b` (`upload_then_cold`).
pub fn cold(b: &[u8]) {
    if fits_ram() { return; }
    cold_pages(b);
}

fn cold_pages(b: &[u8]) {
    let (a, e) = ((b.as_ptr() as usize + 4095) & !4095, (b.as_ptr() as usize + b.len()) & !4095);
    if e > a { madvise(a, e - a, if std::env::var_os("STRATA_COLD_KEEP").is_some() { 20 } else { 4 }); }
}

/// Reads ranges of mapped weights into the page cache from a background thread (`will_need`), in the order sent, so
/// a cold read keeps many drive reads in flight instead of one page fault at a time. Dropping it joins the thread.
pub struct Prefetch { tx: Option<std::sync::mpsc::Sender<Vec<(usize, usize)>>>, th: Option<std::thread::JoinHandle<()>> }

impl Prefetch {
    pub fn start() -> Prefetch {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<(usize, usize)>>();
        let th = std::thread::spawn(move || for rs in rx { for (p, n) in rs { will_need(p, n); } });
        Prefetch { tx: Some(tx), th: Some(th) }
    }
    /// (address, bytes) ranges of mapped weights.
    pub fn send(&self, rs: Vec<(usize, usize)>) { let _ = self.tx.as_ref().unwrap().send(rs); }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(t) = self.th.take() { let _ = t.join(); }
    }
}

/// Dot product with 16 independent partial sums, so the compiler vectorizes it (a single running f32 sum cannot
/// be). 32 and 64 sums measured slower.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) { for i in 0..16 { acc[i] += x[i] * y[i]; } }
    acc.iter().sum::<f32>() + tail
}

#[cfg(test)]
mod tests {
    use super::{preload, upload_then};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// What `preload` returns is what stays resident: the shard less its row-read tables. Counted the other way round,
    /// `fits_ram` says a model that cannot fit does fit, which silences `cold` so every expert the card takes is held
    /// twice - the over-commit an integrated GPU answers by running out of page-table memory.
    #[test]
    fn only_the_non_table_bytes_are_counted() {
        let b = vec![0u8; 100];
        assert_eq!(preload(&b, &[], false), 100);
        assert_eq!(preload(&b, &[(10, 20)], false), 80, "a table in the middle");
        assert_eq!(preload(&b, &[(0, 40), (60, 40)], false), 20, "tables at both ends");

        // Clamped and merged: a shard whose ranges run past it must not make the count negative or double-count.
        assert_eq!(preload(&b, &[(95, 50)], false), 95);
        assert_eq!(preload(&b, &[(10, 30), (20, 10)], false), 70, "one range contained in the other");
        assert_eq!(preload(&b, &[(10, 30), (10, 30)], false), 70, "the same table listed twice");
    }

    /// An upload reads the mapped bytes. Releasing the file pages before that read just lets the copy fault them back
    /// in, so a 40 GB GPU cache sits beside 40 GB of page cache and amdgpu refuses a submission. The release must come
    /// after the upload, and a failed upload must leave the host copy alone (the bytes may still be needed on the CPU).
    #[test]
    fn file_pages_are_released_after_the_upload() {
        let released = AtomicBool::new(false);
        let got = upload_then(true,
            || { assert!(!released.load(Ordering::SeqCst), "pages dropped before the upload read them"); Some(7) },
            || released.store(true, Ordering::SeqCst));
        assert_eq!(got, Some(7));
        assert!(released.load(Ordering::SeqCst));

        let released = AtomicBool::new(false);
        assert_eq!(upload_then::<u8>(true, || None, || released.store(true, Ordering::SeqCst)), None);
        assert!(!released.load(Ordering::SeqCst), "a failed upload copied nothing: keep the file pages");

        // A discrete card has its own memory: nothing is released there.
        let released = AtomicBool::new(false);
        assert_eq!(upload_then(false, || Some(1), || released.store(true, Ordering::SeqCst)), Some(1));
        assert!(!released.load(Ordering::SeqCst));
    }

    /// The primitive behind that release, checked against the kernel rather than a spy: a mapped page faulted in by a
    /// read is gone from this mapping after MADV_DONTNEED, so the GPU copy really is the only one left.
    #[cfg(target_os = "linux")]
    #[test]
    fn cold_pages_are_no_longer_resident() {
        // mincore says whether a page is in the page cache, which MADV_DONTNEED on a shared file mapping need not evict.
        // What matters here is this process's mapping: its RSS drops, so the card's copy is not held in this address
        // space as well. `/proc/self/smaps` reports that per VMA.
        fn mapped_rss(p: usize) -> usize {
            let s = std::fs::read_to_string("/proc/self/smaps").unwrap();
            let mut here = false;
            for line in s.lines() {
                if line.as_bytes().first().is_some_and(u8::is_ascii_hexdigit) && line.contains('-') {
                    let (a, b) = line.split_whitespace().next().unwrap().split_once('-').unwrap();
                    let (a, b) = (usize::from_str_radix(a, 16).unwrap(), usize::from_str_radix(b, 16).unwrap());
                    here = a <= p && p < b;
                } else if here && line.starts_with("Rss:") {
                    return line.split_whitespace().nth(1).unwrap().parse::<usize>().unwrap() * 1024;
                }
            }
            0
        }
        if std::env::var_os("STRATA_COLD_KEEP").is_some() { return; } // MADV_COLD marks, it does not drop
        let p = std::env::temp_dir().join(format!("strata-cold-{}.bin", std::process::id()));
        let mut f = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(&p).unwrap();
        f.set_len(8192).unwrap();
        // Write real bytes: reading a hole maps the kernel's shared zero page, which mincore always calls resident.
        use std::io::Write;
        f.write_all(&[0x5a; 8192]).unwrap();
        f.sync_all().unwrap(); // a dirty page cannot be dropped by MADV_DONTNEED
        let m = unsafe { memmap2::Mmap::map(&f).unwrap() };
        assert_eq!(m[0], 0x5a); // fault the first page in
        assert!(mapped_rss(m.as_ptr() as usize) >= 4096, "the read populated this mapping");
        super::cold_pages(&m);
        assert_eq!(mapped_rss(m.as_ptr() as usize), 0, "MADV_DONTNEED left the page behind");
        std::fs::remove_file(&p).ok();
    }
}
