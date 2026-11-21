//! `Hy4Config` over a synthetic GGUF: the trunk layer count of a file whose `block_count` includes the NextN draft
//! block (the Hy4-flash Q4_K_M release has 43 blocks, 42 trunk + 1 draft; a conversion without a draft head has none),
//! and what `validate` makes of a tensor table. Neither reads tensor data, so this runs on a 100 GB model in seconds —
//! which is the point: `missing tensor blk.42.hc_attn_fn.weight` should be a check failure, not a load failure.
use rusting_strata::gguf::Gguf;
use rusting_strata::hy4::Hy4Config;

const U32T: u32 = 4;
const F32T: u32 = 6;
const BOOLT: u32 = 7;
const STRT: u32 = 8;
const ARRT: u32 = 9;

/// The widths of the synthetic model. Every pair the forward multiplies is a different number, so an expected dim in
/// the wrong order shows up as a dimension mismatch instead of passing by coincidence.
const E: usize = 4; // embedding_length, and the 4 hyper-connection streams make hc * n_embd 16
const VOCAB: u64 = 6;
const QL: u64 = 3; // attention.q_lora_rank
const KL: usize = 6; // attention.key_length_mla (of which NR is rotary)
const VL: u64 = 2; // attention.value_length_mla
const KR: u64 = 5; // attention.kv_lora_rank
const NR: usize = 2; // rope.dimension_count
const FF: u64 = 11; // feed_forward_length, the dense layer's FFN
const FFE: u64 = 7; // expert_feed_forward_length
const LAT: u64 = 9; // expert_latent_length
const NE: u64 = 8; // expert_count
const NIDX: u64 = 5; // attention.indexer.key_length
const NHEADX: u64 = 2; // attention.indexer.head_count
const KPOOL: usize = 2; // attention.indexer.kpool
const HD: u64 = 3; // embedding_length_per_layer_input, one Engram row
const NGRAM: usize = 3; // ple.ngram_size
const HPN: usize = 2; // ple.heads_per_ngram

fn w_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

/// Appends metadata key/value pairs and counts them, so the header's `kv_count` cannot drift from the body. Writing a
/// key twice is how a test changes one value: the parser keeps the last of what it reads.
struct Meta { b: Vec<u8>, n: u64 }

impl Meta {
    fn new() -> Self { Meta { b: Vec::new(), n: 0 } }
    fn key(&mut self, k: &str, t: u32) { w_str(&mut self.b, k); self.b.extend(t.to_le_bytes()); self.n += 1; }
    fn u32(&mut self, k: &str, v: u32) { self.key(k, U32T); self.b.extend(v.to_le_bytes()); }
    fn f32(&mut self, k: &str, v: f32) { self.key(k, F32T); self.b.extend(v.to_le_bytes()); }
    /// An array of scalars: element type, length, values.
    fn arr<T>(&mut self, k: &str, elem: u32, vals: &[T], put: impl Fn(&mut Vec<u8>, &T)) {
        self.key(k, ARRT);
        self.b.extend(elem.to_le_bytes());
        self.b.extend((vals.len() as u64).to_le_bytes());
        for v in vals { put(&mut self.b, v); }
    }
    fn u32s(&mut self, k: &str, vals: &[u32]) { self.arr(k, U32T, vals, |b, v| b.extend(v.to_le_bytes())); }
    fn bools(&mut self, k: &str, vals: &[bool]) { self.arr(k, BOOLT, vals, |b, v| b.push(u8::from(*v))); }
}

/// Hy4-flash metadata for `n_all` blocks of which the last `n_nextn` are NextN draft blocks; `None` leaves
/// `nextn_predict_layers` out altogether. Per-layer arrays cover all `n_all` blocks, like the release files' do.
fn meta(n_all: usize, n_nextn: Option<usize>) -> Meta {
    let f = |s: &str| format!("hy_v4_flash.{s}");
    let mut m = Meta::new();
    m.key("general.architecture", STRT);
    w_str(&mut m.b, "hy_v4_flash");
    m.u32(&f("block_count"), n_all as u32);
    if let Some(n) = n_nextn { m.u32(&f("nextn_predict_layers"), n as u32); }
    // Sliding-window layers have one head, the sparse ones two: enough to tell a layer's count from its neighbour's.
    let swa = (0..n_all).map(|il| il % 4 != 1 && il % 4 != 2).collect::<Vec<_>>();
    m.u32s(&f("attention.head_count"), &(0..n_all).map(|il| if swa[il] { 1 } else { 2 }).collect::<Vec<_>>());
    m.bools(&f("attention.sliding_window_pattern"), &swa);
    m.arr(&f("swiglu_clamp_exp"), F32T, &vec![10.0f32; n_all], |b, v| b.extend(v.to_le_bytes()));
    // The indexer's layers: layer 1 and every fourth after it, so a file that expects one elsewhere is visible.
    m.bools(&f("attention.indexer.types"), &(0..n_all).map(|il| il == 1 || il % 4 == 3).collect::<Vec<_>>());
    // One Engram module, on trunk layer 1 (`ple.layers` numbers blocks, as the forward's `engram_index` expects).
    m.u32s(&f("ple.layers"), &[1]);
    m.u32(&f("ple.ngram_size"), NGRAM as u32);
    m.u32(&f("ple.heads_per_ngram"), HPN as u32);
    m.u32(&f("embedding_length_per_layer_input"), HD as u32);
    let ncol = ((NGRAM - 1) * HPN) as u32;
    m.u32s(&f("ple.layer_multipliers"), &[7, 11, 13]);
    m.u32s(&f("ple.head_offsets"), &(0..ncol).map(|c| c * 8).collect::<Vec<_>>());
    m.u32s(&f("ple.head_vocab_sizes"), &vec![8; ncol as usize]);
    m.u32(&f("engram.pad_token_id"), 0);
    m.u32s(&f("engram.token_map"), &vec![1; VOCAB as usize]);
    m.u32(&f("hyper_connection.count"), 4);
    m.f32(&f("hyper_connection.epsilon"), 1e-6);
    m.f32(&f("hyper_connection.magnitude"), 2.0);
    m.u32(&f("expert_gating_func"), 2);
    m.u32(&f("expert_shared_count"), 1);
    m.u32(&f("attention.head_count_kv"), 1);
    m.u32(&f("embedding_length"), E as u32);
    m.u32(&f("feed_forward_length"), FF as u32);
    m.u32(&f("expert_feed_forward_length"), FFE as u32);
    m.u32(&f("expert_latent_length"), LAT as u32);
    m.u32(&f("attention.q_lora_rank"), QL as u32);
    m.u32(&f("attention.sliding_window"), 512);
    m.u32(&f("attention.key_length_mla"), KL as u32);
    m.u32(&f("attention.value_length_mla"), VL as u32);
    m.u32(&f("attention.kv_lora_rank"), KR as u32);
    m.u32(&f("rope.dimension_count"), NR as u32);
    m.f32(&f("rope.freq_base"), 1e7);
    m.f32(&f("rope.freq_base_swa"), 1e4);
    m.f32(&f("attention.layer_norm_rms_epsilon"), 1e-5);
    m.u32(&f("expert_count"), NE as u32);
    m.u32(&f("expert_used_count"), 8);
    m.f32(&f("expert_weights_scale"), 3.998);
    m.u32(&f("expert_weights_norm"), 1);
    m.u32(&f("leading_dense_block_count"), 1);
    // `top_k / kpool - 1` groups are kept besides the open one, so two is the smallest useful budget.
    m.u32(&f("attention.indexer.top_k"), (KPOOL * 3) as u32);
    m.u32(&f("attention.indexer.kpool"), KPOOL as u32);
    m.key(&f("attention.indexer.kpool_select_tail"), BOOLT);
    m.b.push(1);
    m.u32(&f("attention.indexer.head_count"), NHEADX as u32);
    m.u32(&f("attention.indexer.key_length"), NIDX as u32);
    m
}

/// The offset a GGUF writer gives tensor `i`: consecutive 32-byte-aligned runs, in the order they are listed.
fn offsets(tensors: &[(String, Vec<u64>)]) -> Vec<u64> {
    let mut off = 0u64;
    tensors.iter().map(|(_, d)| {
        let o = off;
        off = (4 * d.iter().product::<u64>()).next_multiple_of(32) + o;
        o
    }).collect()
}

/// A GGUF with `m`'s metadata and one F32 tensor per entry, filled with 1.0 bytes: nothing here reads the data, it only
/// has to be there for `Gguf::open`'s truncation check. Pass `[]` for a file with the vocabulary tensor alone.
fn gguf(m: &Meta, tensors: &[(String, Vec<u64>)]) -> Vec<u8> {
    let offs = offsets(tensors);
    let mut b = Vec::new();
    b.extend(b"GGUF");
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend(m.n.to_le_bytes());
    b.extend(&m.b);
    for ((name, dims), off) in tensors.iter().zip(&offs) {
        w_str(&mut b, name);
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims { b.extend(d.to_le_bytes()); }
        b.extend(0u32.to_le_bytes()); // F32
        b.extend(off.to_le_bytes());
    }
    while b.len() % 32 != 0 { b.push(0); }
    // The data section holds every tensor at its own offset, each run padded so the next one starts aligned.
    let end = offs.last().map(|o| 4 * tensors.last().unwrap().1.iter().product::<u64>() + o).unwrap_or(0) as usize;
    b.resize(b.len() + end, 0);
    b
}

fn open(bytes: &[u8], tag: &str) -> Gguf {
    let p = std::env::temp_dir().join(format!("strata-{}-{tag}.gguf", std::process::id()));
    std::fs::write(&p, bytes).unwrap();
    let g = Gguf::open(&p).unwrap();
    std::fs::remove_file(&p).ok();
    g
}

/// `Hy4Config` of a file with just enough metadata and the one tensor its vocabulary comes from.
fn cfg(m: &Meta, tag: &str) -> Hy4Config {
    Hy4Config::from_gguf(&open(&gguf(m, &[("token_embd.weight".into(), vec![E as u64, VOCAB])]), tag)).unwrap()
}

/// A whole tensor table for `c`: everything its own list asks for. Used as the base for the broken tables below.
fn table(c: &Hy4Config) -> Vec<(String, Vec<u64>)> { c.expected_tensors() }

/// `block_count` counts the NextN draft block, and the draft layer has no hyper-connection weights: a forward that ran
/// all 43 blocks of the Q4_K_M release would ask for `blk.42.hc_attn_fn.weight`, which no Hy4 file has ever had.
#[test]
fn draft_block_is_not_a_trunk_layer() {
    let c = cfg(&meta(43, Some(1)), "hy4-draft");
    assert_eq!((c.n_layer_file, c.n_nextn, c.n_layer), (43, 1, 42));
    // The per-layer arrays are cut to the trunk: the draft block's entry on top of them is dropped.
    assert_eq!((c.n_head.len(), c.swa.len(), c.clamp_exp.len()), (42, 42, 42));
    assert_eq!((c.n_head[0], c.n_head[1]), (1, 2)); // one head on a window layer, two on a sparse one
    assert!(c.swa[3] && !c.swa[1] && !c.swa[2]);
    assert_eq!(c.n_vocab, VOCAB as usize);
    // And nothing the forward loads lives on the draft block.
    assert!(table(&c).iter().all(|(n, _)| !n.starts_with("blk.42.")), "the draft block is in the table");
}

/// A conversion without a draft head has no `nextn_predict_layers` key; declaring it as 0 means the same thing. Either
/// way every block in the file is a trunk layer.
#[test]
fn file_without_draft_block_keeps_every_layer() {
    for (n_nextn, tag) in [(None, "hy4-nomtp"), (Some(0), "hy4-zero")] {
        let c = cfg(&meta(6, n_nextn), tag);
        assert_eq!((c.n_layer_file, c.n_nextn, c.n_layer, c.swa.len()), (6, 0, 6, 6));
    }
}

/// More draft blocks than blocks is malformed; per-layer metadata has to cover the trunk; an Engram layer on the draft
/// block could never be applied.
#[test]
fn rejects_inconsistent_layer_counts() {
    let mut m = meta(5, Some(6));
    let e = Hy4Config::from_gguf(&open(&gguf(&m, &[("token_embd.weight".into(), vec![E as u64, VOCAB])]), "hy4-bad-nextn")).err().unwrap();
    assert!(format!("{e:#}").contains("nextn_predict_layers > block_count"), "{e:#}");

    m = meta(5, Some(1)); // four trunk layers, arrays with three entries
    m.u32s("hy_v4_flash.attention.head_count", &[2, 2, 2]);
    let e = Hy4Config::from_gguf(&open(&gguf(&m, &[("token_embd.weight".into(), vec![E as u64, VOCAB])]), "hy4-short-arrays")).err().unwrap();
    assert!(format!("{e:#}").contains("attention.head_count: per-layer metadata does not cover 4 trunk layers"), "{e:#}");

    m = meta(3, Some(1)); // two trunk layers, so block 2 is the draft one
    m.u32s("hy_v4_flash.ple.layers", &[2]);
    let e = Hy4Config::from_gguf(&open(&gguf(&m, &[("token_embd.weight".into(), vec![E as u64, VOCAB])]), "hy4-ple-on-draft")).err().unwrap();
    assert!(format!("{e:#}").contains("engram layer is not a trunk layer"), "{e:#}");
}

/// The tensor list follows the file's own rules per layer: hyper-connections on every trunk layer, a dense FFN only on
/// the leading one, an indexer only where `attention.indexer.types` says so, and Engram weights only on its own layer.
#[test]
fn expected_tensors_follow_the_layer_kinds() {
    let c = cfg(&meta(6, Some(2)), "hy4-kinds"); // trunk 0..3
    assert_eq!((c.n_layer, c.n_dense), (4, 1));
    assert!(c.has_indexer(1) && c.has_indexer(3) && !c.has_indexer(0) && !c.has_indexer(2));
    let t = table(&c);
    let names: Vec<&str> = t.iter().map(|(n, _)| n.as_str()).collect();
    let dims = |n: &str| t.iter().find(|(k, _)| k == n).map(|(_, d)| d.clone());
    // Hyper-connections mix the 4 streams into a pair of weights per stream: hc*n_embd by 2*hc.
    assert_eq!(dims("blk.3.hc_attn_fn.weight"), Some(vec![16, 8]));
    assert_eq!(dims("blk.3.hc_attn_base.weight"), Some(vec![8]));
    assert_eq!(dims("blk.3.hc_ffn_scale.weight"), Some(vec![2]));
    // The head takes one weight per stream, not a pair.
    assert_eq!(dims("output_hc_fn.weight"), Some(vec![16, 4]));
    // MLA: the no-position key part is `key_length_mla - n_rot` wide, one group of `kv_lora_rank` rows per head. A
    // sliding-window layer here has one head, a sparse one two.
    assert_eq!(dims("blk.1.attn_q_b.weight"), Some(vec![QL, 2 * KL as u64]));
    assert_eq!(dims("blk.3.attn_q_b.weight"), Some(vec![QL, 1 * KL as u64]));
    assert_eq!(dims("blk.1.attn_k_b.weight"), Some(vec![(KL - NR) as u64, KR, 2]));
    assert_eq!(dims("blk.3.attn_v_b.weight"), Some(vec![KR, VL, 1]));
    // The latent MoE reads and writes its bottleneck; the dense layer has a plain FFN and no routing at all.
    assert_eq!(dims("blk.1.ffn_routed_up.weight"), Some(vec![LAT, E as u64]));
    assert_eq!(dims("blk.1.ffn_gate_exps.weight"), Some(vec![LAT, FFE, NE]));
    assert_eq!(dims("blk.1.ffn_down_exps.weight"), Some(vec![FFE, LAT, NE]));
    assert_eq!(dims("blk.0.ffn_up.weight"), Some(vec![E as u64, FF]));
    assert!(!names.contains(&"blk.0.ffn_gate_exps.weight") && names.contains(&"blk.1.ffn_gate_exps.weight"));
    // The indexer and the Engram, on their layers only.
    assert!(names.contains(&"blk.1.indexer.proj.weight") && !names.contains(&"blk.2.indexer.proj.weight"));
    assert_eq!(dims("blk.1.indexer_compressor_ape.weight"), Some(vec![NIDX, KPOOL as u64]));
    assert_eq!(dims("blk.1.engram_kv.weight"), Some(vec![((NGRAM - 1) * HPN) as u64 * HD, (E * 5) as u64]));
    assert_eq!(dims("blk.1.engram_lookup_norm.weight"), Some(vec![HD, ((NGRAM - 1) * HPN) as u64]));
    assert!(!names.contains(&"blk.3.engram_kv.weight"));
}

/// A complete table validates; one missing, mis-dimensioned or unreadably-typed tensor is reported by name, and every
/// problem in the file comes back at once instead of one per run.
#[test]
fn validate_reports_the_whole_table_at_once() {
    let c = cfg(&meta(6, Some(2)), "hy4-validate");
    let good = table(&c);
    assert!(c.validate(&open(&gguf(&meta(6, Some(2)), &good), "hy4-ok")).is_ok());

    let mut bad = good.clone();
    for drop in ["blk.1.exp_probs_b.bias", "blk.3.hc_ffn_fn.weight"] {
        bad.retain(|(n, _)| n != drop);
    }
    // The draft block's own tensors are in the file and not asked for: they must not be reported either way.
    bad.push(("blk.5.nextn.eh_proj.weight".into(), vec![8, 4]));
    let e = c.validate(&open(&gguf(&meta(6, Some(2)), &bad), "hy4-missing")).err().unwrap();
    let msg = format!("{e:#}");
    assert!(msg.contains("missing tensor blk.1.exp_probs_b.bias"), "{msg}");
    assert!(msg.contains("missing tensor blk.3.hc_ffn_fn.weight"), "{msg}");
    assert_eq!(msg.matches("missing tensor").count(), 2, "{msg}");

    // Wrong dims: the transposed form of a real matrix, which is how a renamed tensor tends to arrive. Layer 3 has one
    // head, so its output projection is [n_head * value_length, n_embd] and not the other way round.
    let mut bad = good.clone();
    bad.iter_mut().find(|(n, _)| n == "blk.3.attn_output.weight").unwrap().1 = vec![E as u64, VL];
    let e = c.validate(&open(&gguf(&meta(6, Some(2)), &bad), "hy4-dims")).err().unwrap();
    assert!(format!("{e:#}").contains("blk.3.attn_output.weight: dims [4, 2], expected [2, 4]"), "{e:#}");

    // A type this project has no layout for is a load failure later, so it is one here too. Q8_1 (id 9) has none:
    // rewriting the id in the tensor table is how `validate` gets to see it.
    let bytes = rewritten_type(&gguf(&meta(6, Some(2)), &good), "blk.1.ffn_down_exps.weight", 9);
    let e = c.validate(&open(&bytes, "hy4-type")).err().unwrap();
    assert!(format!("{e:#}").contains("blk.1.ffn_down_exps.weight: unsupported quant type Q8_1"), "{e:#}");
}

/// The same GGUF with one tensor's type id replaced. The type is the u32 right after that tensor's dims, in table order.
fn rewritten_type(bytes: &[u8], name: &str, ty: u32) -> Vec<u8> {
    let mut out = bytes.to_vec();
    // Find the name as it appears in the tensor table (a length-prefixed string), then walk past dims to the type.
    let nb = name.as_bytes();
    let mut pat = nb.len().to_le_bytes().to_vec();
    pat.extend(nb);
    let at = out.windows(pat.len()).position(|w| w == pat).expect("name in the tensor table");
    let mut p = at + pat.len();
    let nd = u32::from_le_bytes(out[p..p + 4].try_into().unwrap()) as usize;
    p += 4 + 8 * nd;
    out[p..p + 4].copy_from_slice(&ty.to_le_bytes());
    out
}
