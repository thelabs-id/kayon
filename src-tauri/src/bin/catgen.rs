//! catgen — build-time catalog entry generator (CAT-6).
//!
//! Given a Hugging Face repo and a list of GGUF quant files, catgen fetches each file,
//! computes its real SHA-256 and exact byte size, and reads the GGUF header for the
//! auto-derivable `arch` block (architecture, block_count, head counts, embedding length,
//! key/value length, context length). Curation-judgment fields — capabilities.reasoning,
//! runtimeArgs, runtimeMinVersion — are scaffolded empty for human review, never guessed.
//!
//! Because checksums are computed from the very files they describe, they cannot drift.
//! This is intended to run in CI against a PR-reviewed catalog repo; the app itself never
//! calls it. Pseudocode-faithful to the spec, not a stub.
//!
//! Usage:
//!   cargo run --bin catgen -- <repo> <id> <family> <params> <license> <quant.gguf>[,<quant2.gguf>...]
//! Example:
//!   cargo run --bin catgen -- bartowski/SmolLM2-135M-Instruct-GGUF smollm2-135m SmolLM2 135M Apache-2.0 \
//!       SmolLM2-135M-Instruct-Q4_K_M.gguf,SmolLM2-135M-Instruct-Q8_0.gguf

use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[path = "../gguf/mod.rs"]
mod gguf;

fn hf_resolve_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo, file)
}

fn quant_label_from_filename(file: &str) -> String {
    // e.g. Model-Q4_K_M.gguf -> Q4_K_M
    let stem = file.strip_suffix(".gguf").unwrap_or(file);
    stem.rsplit('-').next().unwrap_or(stem).to_string()
}

fn meta_u32(h: &gguf::GgufHeader, arch: &str, key: &str) -> Option<u32> {
    h.metadata.get(&format!("{arch}.{key}")).and_then(|v| v.as_u32())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 6 {
        eprintln!("usage: catgen <repo> <id> <family> <params> <license> <file1.gguf[,file2.gguf...]>");
        std::process::exit(2);
    }
    let (repo, id, family, params, license) =
        (&args[0], &args[1], &args[2], &args[3], &args[4]);
    let files: Vec<&str> = args[5].split(',').collect();

    let client = reqwest::Client::builder()
        .user_agent("Kayon-catgen/0.1")
        .build()
        .unwrap();

    let tmp_dir = std::env::temp_dir().join("kayon-catgen");
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let mut quants = Vec::new();
    let mut arch_json: Option<serde_json::Value> = None;

    for file in &files {
        let url = hf_resolve_url(repo, file);
        eprintln!("fetching {url}");
        let tmp = tmp_dir.join(file);
        // Stream to disk while hashing incrementally — GGUF quants can be several GB, so never
        // buffer the whole file in memory (would OOM/thrash in CI).
        let (sha256, size) = {
            use futures_util::StreamExt;
            use std::io::Write;
            let resp = client.get(&url).send().await.expect("http get");
            if !resp.status().is_success() {
                eprintln!("  HTTP {} for {file}", resp.status());
                std::process::exit(1);
            }
            let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp).expect("create temp"));
            let mut hasher = Sha256::new();
            let mut size: u64 = 0;
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.expect("stream chunk");
                out.write_all(&chunk).expect("write chunk");
                hasher.update(&chunk);
                size += chunk.len() as u64;
            }
            out.flush().expect("flush");
            (format!("{:x}", hasher.finalize()), size)
        };
        eprintln!("  sha256={sha256} bytes={size}");

        if arch_json.is_none() {
            let h = gguf::parse_gguf_header(&tmp).expect("parse gguf header");
            let architecture = gguf::arch_from_header(&h).unwrap_or_else(|| "unknown".into());
            arch_json = Some(serde_json::json!({
                "architecture": architecture,
                "blockCount": meta_u32(&h, &architecture, "block_count"),
                "headCount": meta_u32(&h, &architecture, "attention.head_count"),
                // Fall back to head_count for non-GQA models that omit head_count_kv, so the
                // generated (required u32) field is never null and the signed catalog deserializes.
                "headCountKv": meta_u32(&h, &architecture, "attention.head_count_kv")
                    .or_else(|| meta_u32(&h, &architecture, "attention.head_count")),
                "embeddingLength": meta_u32(&h, &architecture, "embedding_length"),
                "contextLength": meta_u32(&h, &architecture, "context_length"),
                "keyLength": meta_u32(&h, &architecture, "attention.key_length"),
                "valueLength": meta_u32(&h, &architecture, "attention.value_length"),
                "attentionType": null,          // curation judgment — review before setting
                "runtimeMinVersion": null,      // curation judgment
            }));
        }

        quants.push(serde_json::json!({
            "label": quant_label_from_filename(file),
            "bytes": size,
            "sha256": sha256,
            "source": url,
            "runtimeArgs": [],                  // curation judgment (e.g. --jinja)
        }));
        let _ = std::fs::remove_file(&tmp);
    }

    let entry = serde_json::json!({
        "id": id,
        "family": family,
        "params": params,
        "license": license,
        "capabilities": { "tools": false, "reasoning": false, "vision": false }, // scaffolded
        "arch": arch_json.unwrap(),
        "quants": quants,
    });

    // Emit the entry on stdout for a human to paste into the PR-reviewed catalog repo.
    println!("{}", serde_json::to_string_pretty(&entry).unwrap());
}
