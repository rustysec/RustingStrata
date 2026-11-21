//! Gated by STRATA_HY4_MODEL (path to shard 1 of Hy4-flash). Skips when unset.
use rusting_strata::{gguf::Gguf, hy4::forward::Hy4Model, tokenizer::Tokenizer};

#[test]
fn first_token_after_france_prompt_is_paris() {
    let Ok(p) = std::env::var("STRATA_HY4_MODEL") else { eprintln!("STRATA_HY4_MODEL unset, skipping"); return };
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let mut m = Hy4Model::new(&g).unwrap();
    let logits = m.prefill(&tok.encode("The capital of France is").unwrap());
    let top = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
    assert_eq!(tok.decode(&[top]).unwrap(), " Paris", "llama.cpp greedy picks ' Paris'");
}

/// The CPU path (STRATA_DS_GPU unset) runs the same kernels either way, so the logits agree closely.
#[test]
fn chunked_prefill_matches_token_by_token() {
    let Ok(p) = std::env::var("STRATA_HY4_MODEL") else { eprintln!("STRATA_HY4_MODEL unset, skipping"); return };
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    // More tokens than the widest layer has heads, so a chunk's per-head projections run too.
    let toks = tok.encode(&"The capital of France is Paris. ".repeat(12)).unwrap();
    assert!(toks.len() > 65, "{} tokens", toks.len());
    let (mut a, mut b) = (Hy4Model::new(&g).unwrap(), Hy4Model::new(&g).unwrap());
    let chunked = a.prefill(&toks);
    let mut stepped = vec![];
    for &t in &toks { stepped = b.step(t); }
    let worst = chunked.iter().zip(&stepped).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(worst < 1e-3, "chunked prefill logits differ by {worst}");
}

/// The tensor table of a real release, checked the way `check` does it: no weight is read, so this takes seconds on a
/// 165 GB model instead of loading one. Run it with `STRATA_PRELOAD=0` to keep the shard's page-cache readahead off a
/// machine that is busy — the check itself needs no bytes from the data section.
#[test]
fn tensor_table_of_the_release_file_is_complete() {
    let Ok(p) = std::env::var("STRATA_HY4_MODEL") else { eprintln!("STRATA_HY4_MODEL unset, skipping"); return };
    let g = Gguf::open(std::path::Path::new(&p)).unwrap();
    let c = rusting_strata::hy4::Hy4Config::from_gguf(&g).unwrap();
    // The Q4_K_M release ships 43 blocks with `nextn_predict_layers` 1: the draft block is not a trunk layer, and it
    // has no hyper-connection weights to ask for.
    assert_eq!(c.n_layer_file - c.n_nextn, c.n_layer);
    assert_eq!((c.n_head.len(), c.swa.len(), c.clamp_exp.len()), (c.n_layer, c.n_layer, c.n_layer));
    c.validate(&g).unwrap();
    // The indexer is where the metadata puts it, which on a real file is also where its tensors are.
    for il in 0..c.n_layer {
        let ix = format!("blk.{il}.indexer.attn_k.weight");
        let en = format!("blk.{il}.engram_kv.weight");
        assert_eq!(c.has_indexer(il), g.tensors.contains_key(&ix), "layer {il}: indexer");
        assert_eq!(c.engram_index(il).is_some(), g.tensors.contains_key(&en), "layer {il}: Engram");
    }
    // And what the loader counts against RAM: everything but the row-read table. The number is printed at load by
    // `deepseek::mat`, and a run whose report disagrees with this either used another file or has an accounting bug -
    // which is the difference between "the budget is too big" and "the budget was measured wrong".
    let shard1 = std::fs::metadata(std::path::Path::new(&p)).unwrap().len();
    let table = g.tensors["per_layer_token_embd.weight"].dims[0] * g.tensors["per_layer_token_embd.weight"].dims[1];
    assert_eq!(g.table_bytes, table, "the Engram table is the only row-read tensor here");
    let mapped = rusting_strata::weights::mapped_bytes();
    assert!(mapped >= shard1 && mapped < shard1 * 105 / 100, "{mapped} counted for a {shard1} byte first shard plus a header-sized second one");
}
