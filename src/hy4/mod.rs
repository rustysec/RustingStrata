//! Tencent Hy4-flash (`hy_v4_flash`): hyper-connection streams, gated MLA attention with sinks, latent MoE, Engram.
//! CPU forward that mirrors llama.cpp `src/models/hy-v4-flash.cpp`; matvecs and routed experts reuse `deepseek::mat`.
//! A file's `block_count` may count a NextN draft block at the end: this engine runs the trunk layers only.
pub mod chat;
pub mod forward;

use crate::deepseek::EngramConfig;
use crate::gguf::Gguf;
use anyhow::{Context, ensure};

pub const ARCH: &str = "hy_v4_flash";

#[derive(Debug, Clone)]
pub struct Hy4Config {
    /// Blocks in the file (`block_count`), the trunk ones plus the NextN draft block.
    pub n_layer_file: usize,
    /// Draft blocks at the end of the file (`nextn_predict_layers`). This forward runs none of them.
    pub n_nextn: usize,
    /// Transformer layers this forward runs: `n_layer_file - n_nextn`.
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    /// Heads per layer (32 on the sparse layers, 64 on the sliding-window ones).
    pub n_head: Vec<usize>,
    /// Sliding-window layers.
    pub swa: Vec<bool>,
    pub n_swa: usize,
    /// Per-head query/key width (192 no-position + `n_rot` rotary) and value width.
    pub key_len: usize,
    pub val_len: usize,
    pub kv_rank: usize,
    pub n_rot: usize,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub eps: f32,
    pub hc_eps: f32,
    pub hc_mag: f32,
    pub n_expert: usize,
    pub n_used: usize,
    pub exp_scale: f32,
    pub exp_norm: bool,
    pub clamp_exp: Vec<f32>,
    /// Leading layers with a dense FFN instead of the latent MoE.
    pub n_dense: usize,
    /// Dense-FFN hidden width (`feed_forward_length`), for `n_dense` layers' `ffn_*` matrices.
    pub n_ff: usize,
    /// Latent MoE widths: expert hidden (`expert_feed_forward_length`) and the shared bottleneck every routed expert
    /// reads and writes (`expert_latent_length`).
    pub n_ff_exp: usize,
    pub exp_latent: usize,
    /// Query LoRA rank of the two-stage Q projection.
    pub q_lora: usize,
    pub engram: Option<EngramConfig>,
    /// Sparse-layer indexer: positions per pooled group and the number of groups a position keeps (0: no indexer).
    pub kpool: usize,
    pub idx_top: usize,
    /// Indexer head count and per-head key width (`attention.indexer.head_count` / `.key_length`).
    pub idx_head: usize,
    pub idx_dim: usize,
    /// Layers with an indexer, from `attention.indexer.types`; empty when the file does not say which.
    pub idx_layer: Vec<bool>,
}

/// A per-layer array from the file, cut to the trunk layers this forward runs. Too few entries is an error; a draft
/// block's entry on top of them is dropped.
fn trunk<T>(mut v: Vec<T>, n_layer: usize, key: &str) -> anyhow::Result<Vec<T>> {
    ensure!(v.len() >= n_layer, "{key}: per-layer metadata does not cover {n_layer} trunk layers");
    v.truncate(n_layer);
    Ok(v)
}

impl Hy4Config {
    pub fn from_gguf(g: &Gguf) -> anyhow::Result<Self> {
        let k = |s: &str| format!("{ARCH}.{s}");
        let u = |s: &str| -> anyhow::Result<usize> { Ok(g.get(&k(s))?.as_u64().with_context(|| k(s))? as usize) };
        let f = |s: &str| -> anyhow::Result<f32> { g.get(&k(s))?.as_f32().with_context(|| k(s)) };
        let arr_u = |s: &str| -> anyhow::Result<Vec<u64>> {
            g.get(&k(s))?.as_arr().with_context(|| k(s))?.iter().map(|v| v.as_u64().with_context(|| k(s))).collect()
        };
        // `block_count` counts the NextN draft block, which this forward does not run: like llama.cpp, the trunk is
        // what is left once `nextn_predict_layers` (absent in a `--no-mtp` conversion) goes. The draft block has no
        // hyper-connection or Engram tensors of its own, so asking for them fails the load (`blk.N.hc_attn_fn.weight`).
        let n_layer_file = u("block_count")?;
        let n_nextn = g.get(&k("nextn_predict_layers")).ok().and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        ensure!(n_nextn <= n_layer_file, "nextn_predict_layers > block_count");
        let n_layer = n_layer_file - n_nextn;
        // Per-layer metadata lists one entry per block in the file, draft block included; keep the trunk's share of it.
        let n_head = trunk(arr_u("attention.head_count")?.into_iter().map(|v| v as usize).collect::<Vec<_>>(), n_layer, "attention.head_count")?;
        let swa = trunk(arr_u("attention.sliding_window_pattern")?.into_iter().map(|v| v != 0).collect::<Vec<_>>(), n_layer, "attention.sliding_window_pattern")?;
        let clamp_exp = trunk(g.get(&k("swiglu_clamp_exp"))?.as_arr().with_context(|| k("swiglu_clamp_exp"))?
            .iter().map(|v| v.as_f32().context("swiglu_clamp_exp")).collect::<anyhow::Result<Vec<_>>>()?, n_layer, "swiglu_clamp_exp")?;
        ensure!(u("hyper_connection.count")? == 4, "only 4 hyper-connection streams are supported");
        ensure!(u("expert_gating_func")? == 2 && u("expert_shared_count")? == 1 && u("attention.head_count_kv")? == 1, "unsupported Hy4 variant");
        let engram = if g.meta.contains_key(&k("ple.layers")) {
            let layers: Vec<usize> = arr_u("ple.layers")?.into_iter().map(|v| v as usize).collect();
            // `engram_index` numbers the trunk's layers, so an Engram listed on the draft block could never run.
            ensure!(layers.iter().all(|&l| l < n_layer), "engram layer is not a trunk layer");
            let (ngram, hpn) = (u("ple.ngram_size")?, u("ple.heads_per_ngram")?);
            let n_cols = (ngram - 1) * hpn;
            let (mult, offsets, primes) = (arr_u("ple.layer_multipliers")?, arr_u("ple.head_offsets")?, arr_u("ple.head_vocab_sizes")?);
            ensure!(mult.len() == layers.len() * ngram && offsets.len() == layers.len() * n_cols && primes.len() == offsets.len(), "engram metadata is inconsistent");
            let token_map: Vec<i32> = g.get(&k("engram.token_map"))?.as_arr().context("engram.token_map")?.iter()
                .map(|v| v.as_u64().map(|x| x as i32).or(if let crate::gguf::Value::I32(x) = v { Some(*x) } else { None }).context("token_map"))
                .collect::<anyhow::Result<_>>()?;
            let pad_id = u("engram.pad_token_id")?;
            ensure!(pad_id < token_map.len() && token_map[pad_id] >= 0, "engram pad token is not mapped");
            Some(EngramConfig { layers, ngram, heads_per_ngram: hpn, head_dim: u("embedding_length_per_layer_input")?, multipliers: mult, offsets, primes, pad_id, token_map })
        } else {
            None
        };
        let (mut kpool, mut idx_top, mut idx_layer) = (0, 0, vec![]);
        if g.meta.contains_key(&k("attention.indexer.top_k")) {
            kpool = u("attention.indexer.kpool")?;
            ensure!(kpool >= 2 && u("attention.indexer.top_k")? / kpool >= 2, "unsupported indexer pool size");
            ensure!(matches!(g.get(&k("attention.indexer.kpool_select_tail")), Ok(crate::gguf::Value::Bool(true))), "only the tail-selecting indexer is supported");
            // One group of the budget is the open tail group, which is always kept.
            idx_top = u("attention.indexer.top_k")? / kpool - 1;
            // Which layers have one is metadata rather than something to infer from the file, so that a missing tensor
            // is an error instead of a layer that quietly stops being sparse.
            if let Ok(v) = g.get(&k("attention.indexer.types")) {
                idx_layer = v.as_arr().with_context(|| k("attention.indexer.types"))?.iter()
                    .map(|e| matches!(e, crate::gguf::Value::Bool(true)) || e.as_u64().unwrap_or(0) != 0).collect();
                if !idx_layer.is_empty() { idx_layer = trunk(idx_layer, n_layer, "attention.indexer.types")?; }
            }
        }
        Ok(Hy4Config {
            n_layer_file,
            n_nextn,
            n_layer,
            n_embd: u("embedding_length")?,
            n_vocab: g.tensors.get("token_embd.weight").context("missing token_embd.weight")?.dims[1] as usize,
            n_head,
            swa,
            n_swa: u("attention.sliding_window")?,
            key_len: u("attention.key_length_mla")?,
            val_len: u("attention.value_length_mla")?,
            kv_rank: u("attention.kv_lora_rank")?,
            n_rot: u("rope.dimension_count")?,
            rope_base: f("rope.freq_base")?,
            rope_base_swa: f("rope.freq_base_swa")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            hc_eps: f("hyper_connection.epsilon")?,
            hc_mag: f("hyper_connection.magnitude")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            exp_scale: f("expert_weights_scale")?,
            exp_norm: u("expert_weights_norm")? != 0,
            clamp_exp,
            n_dense: u("leading_dense_block_count")?,
            n_ff: u("feed_forward_length")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            // Optional in llama.cpp's reader; a file without it has no bottleneck, so the experts are `n_ff_exp` wide.
            exp_latent: g.get(&k("expert_latent_length")).ok().and_then(|v| v.as_u64()).unwrap_or(u("expert_feed_forward_length")? as u64) as usize,
            q_lora: u("attention.q_lora_rank")?,
            engram,
            kpool,
            idx_top,
            idx_head: g.get(&k("attention.indexer.head_count")).ok().and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            idx_dim: g.get(&k("attention.indexer.key_length")).ok().and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            idx_layer,
        })
    }

    pub fn engram_index(&self, il: usize) -> Option<usize> {
        self.engram.as_ref()?.layers.iter().position(|&l| l == il)
    }

    /// Whether layer `il` runs the sparse indexer. False for a file that does not say which layers do (`idx_layer`).
    pub fn has_indexer(&self, il: usize) -> bool {
        self.kpool > 0 && self.idx_layer.get(il).copied().unwrap_or(false)
    }

    /// Every tensor the forward loads, with its dims in GGUF order (fastest first), as llama.cpp's `hy-v4-flash.cpp`
    /// creates them. The NextN draft block is not in it: this forward runs the trunk only, so a file that has one must
    /// not be asked for hyper-connection weights on the draft block.
    pub fn expected_tensors(&self) -> Vec<(String, Vec<u64>)> {
        let (e, h) = (self.n_embd as u64, 4u64); // `from_gguf` accepts only 4 hyper-connection streams
        let (ql, kr, kl, vl, nr) = (self.q_lora as u64, self.kv_rank as u64, self.key_len as u64, self.val_len as u64, self.n_rot as u64);
        let (ne, ffe, lat) = (self.n_expert as u64, self.n_ff_exp as u64, self.exp_latent as u64);
        let (nid, nih) = (self.idx_dim as u64, self.idx_head as u64);
        let mut v = vec![
            ("token_embd.weight".to_string(), vec![e, self.n_vocab as u64]),
            ("output_norm.weight".into(), vec![e]),
            ("output.weight".into(), vec![e, self.n_vocab as u64]),
            // The head mixes the streams itself: one weight per stream, not the pair a trunk layer carries.
            ("output_hc_fn.weight".into(), vec![h * e, h]),
            ("output_hc_base.weight".into(), vec![h]),
            ("output_hc_scale.weight".into(), vec![1]),
        ];
        if let Some(en) = &self.engram {
            // One row per bucket of every head; the file may carry more rows than the last bucket needs.
            let rows = en.offsets.iter().zip(&en.primes).map(|(o, p)| o + p).max().unwrap_or(0);
            v.push(("per_layer_token_embd.weight".into(), vec![en.head_dim as u64, rows]));
        }
        for il in 0..self.n_layer {
            let b = |n: &str| format!("blk.{il}.{n}");
            let mut t = |n: &str, d: Vec<u64>| v.push((b(n), d));
            let nh = self.n_head[il] as u64;
            t("attn_norm.weight", vec![e]);
            t("attn_sinks.weight", vec![nh]);
            t("attn_q_a.weight", vec![e, ql]);
            t("attn_q_a_norm.weight", vec![ql]);
            t("attn_q_b.weight", vec![ql, nh * kl]);
            t("attn_kv_a_mqa.weight", vec![e, kr + nr]);
            t("attn_kv_a_norm.weight", vec![kr]);
            t("attn_k_b.weight", vec![kl - nr, kr, nh]);
            t("attn_v_b.weight", vec![kr, vl, nh]);
            t("attn_gate.weight", vec![e, nh * vl]);
            t("attn_output.weight", vec![nh * vl, e]);
            t("post_attention_norm.weight", vec![e]);
            for s in ["attn", "ffn"] {
                t(&format!("hc_{s}_fn.weight"), vec![h * e, 2 * h]);
                t(&format!("hc_{s}_base.weight"), vec![2 * h]);
                t(&format!("hc_{s}_scale.weight"), vec![2]);
            }
            t("ffn_norm.weight", vec![e]);
            t("post_ffw_norm.weight", vec![e]);
            if il < self.n_dense {
                let ff = self.n_ff as u64;
                t("ffn_gate.weight", vec![e, ff]);
                t("ffn_down.weight", vec![ff, e]);
                t("ffn_up.weight", vec![e, ff]);
            } else {
                // The shared expert is one expert wide (`expert_shared_count`, checked by `from_gguf`).
                t("ffn_gate_inp.weight", vec![e, ne]);
                t("exp_probs_b.bias", vec![ne]);
                t("ffn_routed_down.weight", vec![e, lat]);
                t("ffn_routed_up.weight", vec![lat, e]);
                t("ffn_gate_exps.weight", vec![lat, ffe, ne]);
                t("ffn_up_exps.weight", vec![lat, ffe, ne]);
                t("ffn_down_exps.weight", vec![ffe, lat, ne]);
                t("ffn_gate_shexp.weight", vec![e, ffe]);
                t("ffn_down_shexp.weight", vec![ffe, e]);
                t("ffn_up_shexp.weight", vec![e, ffe]);
            }
            if self.has_indexer(il) {
                t("indexer.attn_k.weight", vec![e, nid]);
                t("indexer.k_norm.weight", vec![nid]);
                t("indexer_compressor_value.weight", vec![nid, nid]);
                t("indexer_compressor_gate.weight", vec![nid, nid]);
                t("indexer_compressor_ape.weight", vec![nid, self.kpool as u64]);
                t("indexer_compressor_norm.weight", vec![nid]);
                t("indexer.attn_q_b.weight", vec![ql, nid * nih]);
                t("indexer.proj.weight", vec![e, nih]);
            }
            if let (Some(_), Some(en)) = (self.engram_index(il), &self.engram) {
                let (ncol, hd) = ((en.ngram - 1) * en.heads_per_ngram, en.head_dim);
                t("engram_kv.weight", vec![(ncol * hd) as u64, e * (h + 1)]);
                t("engram_kv_b.weight", vec![e, h + 1]);
                t("engram_gate.weight", vec![e, h]);
                t("engram_hc_scale.weight", vec![e, h]);
                t("engram_lookup_norm.weight", vec![hd as u64, ncol as u64]);
                t("engram_pre_gamma.weight", vec![e]);
            }
        }
        v
    }

    /// Check every expected tensor is in the file with the expected dims and a quant type this project can read: every
    /// problem at once, instead of one `missing tensor` per run. Reading no tensor data, so it costs nothing on a
    /// 100 GB model.
    pub fn validate(&self, g: &Gguf) -> anyhow::Result<()> {
        let mut errs = vec![];
        for (name, dims) in self.expected_tensors() {
            match g.tensors.get(&name) {
                None => errs.push(format!("missing tensor {name}")),
                Some(t) => {
                    // The Engram table is indexed by bucket, so it only needs at least that many rows.
                    let ok = if name == "per_layer_token_embd.weight" { t.dims[0] == dims[0] && t.dims[1] >= dims[1] } else { t.dims == dims };
                    if !ok { errs.push(format!("{name}: dims {:?}, expected {:?}", t.dims, dims)); }
                    if let Err(e) = crate::quant::GgmlType::from_u32(t.ty) { errs.push(format!("{name}: {e}")); }
                }
            }
        }
        ensure!(errs.is_empty(), "{} tensor problems:\n{}", errs.len(), errs.join("\n"));
        Ok(())
    }
}
