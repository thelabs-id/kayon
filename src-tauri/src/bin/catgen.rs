//! catgen — generates + signs the **bundled** catalog (the offline anchor, CAT-5). At runtime Kayon
//! discovers the live catalog from Hugging Face itself (CAT-7, `kayon::discovery`); catgen exists to
//! bake a signed starting catalog into the binary so the app works offline and on first launch
//! before any discovery has run.
//!
//! How it pins checksums WITHOUT downloading the multi-GB files: Hugging Face stores GGUFs in Git
//! LFS, whose object id *is* the file's SHA-256. HF's metadata returns that `oid` for free. The
//! `arch` block still needs the header, so catgen (and discovery) range-fetch only the first few MB.
//!
//! Usage:
//!   cargo run --bin catgen -- auto [per_author] [author,author...]  # discover from HF + sign
//!   cargo run --bin catgen -- manifest catalog/catalog.source.json  # generate + sign from a manifest
//!   cargo run --bin catgen -- one <repo> <id> <family> <params> <license> <file.gguf>  # ad-hoc entry

use ed25519_dalek::{Signer, SigningKey};
use kayon::{catalog, discovery, gguf};
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn hf_resolve_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo, file)
}

/// Real SHA-256 (the Git LFS oid) + exact byte size, straight from HF metadata — no file download.
async fn hf_checksum_size(client: &reqwest::Client, repo: &str, file: &str) -> anyhow::Result<(String, u64)> {
    let url = format!("https://huggingface.co/api/models/{}/paths-info/main", repo);
    let body = serde_json::json!({ "paths": [file] });
    let resp = client.post(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("HF paths-info {} for {}/{}", resp.status(), repo, file);
    }
    let arr: serde_json::Value = resp.json().await?;
    let entry = arr.as_array().and_then(|a| a.first())
        .ok_or_else(|| anyhow::anyhow!("{}/{} not found on HF", repo, file))?;
    let lfs = entry.get("lfs")
        .ok_or_else(|| anyhow::anyhow!("{}/{} is not an LFS file (no checksum)", repo, file))?;
    let oid = lfs.get("oid").and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("no lfs.oid for {}/{}", repo, file))?;
    let size = lfs.get("size").and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("no lfs.size for {}/{}", repo, file))?;
    Ok((oid.to_string(), size))
}

/// Derive the GGUF `arch` block by range-fetching just the header (grows the window if a model has
/// an unusually large tokenizer). Never downloads tensor data.
async fn fetch_arch(client: &reqwest::Client, repo: &str, file: &str) -> anyhow::Result<serde_json::Value> {
    let url = hf_resolve_url(repo, file);
    for window in [8_000_000u64, 32_000_000, 96_000_000] {
        let resp = client.get(&url).header("Range", format!("bytes=0-{}", window - 1)).send().await?;
        if !resp.status().is_success() && resp.status().as_u16() != 206 {
            anyhow::bail!("HF header fetch {} for {}/{}", resp.status(), repo, file);
        }
        let bytes = resp.bytes().await?;
        let tmp = std::env::temp_dir().join(format!("kayon-hdr-{}", file.replace(['/', '\\'], "_")));
        std::fs::write(&tmp, &bytes)?;
        match gguf::parse_gguf_header(&tmp) {
            Ok(h) => {
                let _ = std::fs::remove_file(&tmp);
                let architecture = gguf::arch_from_header(&h).unwrap_or_else(|| "unknown".into());
                let m = |k: &str| h.metadata.get(&format!("{architecture}.{k}")).and_then(|v| v.as_u32());
                return Ok(serde_json::json!({
                    "architecture": architecture,
                    "blockCount": m("block_count"),
                    "headCount": m("attention.head_count"),
                    "headCountKv": m("attention.head_count_kv").or_else(|| m("attention.head_count")),
                    "embeddingLength": m("embedding_length"),
                    "contextLength": m("context_length"),
                    "keyLength": m("attention.key_length"),
                    "valueLength": m("attention.value_length"),
                    "attentionType": gguf::attention_type(&h),
                    "runtimeMinVersion": null
                }));
            }
            Err(_) => { let _ = std::fs::remove_file(&tmp); /* header not fully in window — grow it */ }
        }
    }
    anyhow::bail!("could not parse GGUF header for {}/{} within 96 MB", repo, file)
}

/// Sign `catalog.json` in place → `catalog.json.sig`, using the same key resolution as `catsign`
/// (KAYON_CATALOG_SEED hex env, else the gitignored catalog/signing.key).
fn sign_catalog() -> anyhow::Result<()> {
    let sk: SigningKey = if let Ok(hex_seed) = std::env::var("KAYON_CATALOG_SEED") {
        let bytes = hex::decode(hex_seed.trim())?;
        SigningKey::from_bytes(&bytes.as_slice().try_into().map_err(|_| anyhow::anyhow!("seed must be 32 bytes"))?)
    } else {
        let key_path = crate_root().join("catalog").join("signing.key");
        let bytes = std::fs::read(&key_path).map_err(|_| anyhow::anyhow!(
            "no signing key: set KAYON_CATALOG_SEED or run `catsign sign` once to create {}", key_path.display()))?;
        SigningKey::from_bytes(&bytes.as_slice().try_into().map_err(|_| anyhow::anyhow!("key file must be 32 bytes"))?)
    };
    let json_path = crate_root().join("catalog").join("catalog.json");
    let sig_path = crate_root().join("catalog").join("catalog.json.sig");
    let data = std::fs::read(&json_path)?;
    std::fs::write(&sig_path, sk.sign(&data).to_bytes())?;
    println!("signed {} (verifying key {})", json_path.display(), hex::encode(sk.verifying_key().to_bytes()));
    Ok(())
}

/// Discover the bundled catalog live from Hugging Face (same code path Kayon runs at launch) and
/// write + sign it. This is how the offline-anchor catalog is refreshed.
async fn generate_auto(per_author: usize, authors: Vec<String>) -> anyhow::Result<()> {
    // Local runs bump the existing catalog's revision by 1; CI/scripts can pin via env.
    let revision: u64 = std::env::var("KAYON_CATALOG_REVISION").ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::fs::read(crate_root().join("catalog").join("catalog.json")).ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|v| v.get("revision").and_then(|r| r.as_u64()))
                .unwrap_or(0) + 1
        });
    eprintln!("· discovering from {authors:?} (top {per_author} by downloads)");
    let seed = catalog::empty_catalog();
    let mut cat = discovery::discover_catalog(None, &authors, per_author, revision, &seed).await?;
    // The bundled catalog is the signed offline anchor (CAT-5), so mark its source accordingly and
    // pin a deterministic generatedAt for reproducible signed bytes.
    cat.source = "generated".to_string();
    cat.generated_at = "1970-01-01T00:00:00Z".parse().unwrap();
    for e in &cat.entries {
        eprintln!("  + {} ({} {}, {} quant(s))", e.id, e.family, e.params, e.quants.len());
    }
    let out = crate_root().join("catalog").join("catalog.json");
    std::fs::write(&out, serde_json::to_string_pretty(&cat)? + "\n")?;
    println!("wrote {} ({} models discovered + pinned, rev {})", out.display(), cat.entries.len(), revision);
    sign_catalog()
}

async fn generate_from_manifest(path: &str) -> anyhow::Result<()> {
    let src: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let client = reqwest::Client::builder().user_agent("Kayon-catgen/1.0").build()?;
    let models = src["models"].as_array().ok_or_else(|| anyhow::anyhow!("manifest has no models[]"))?;

    let mut entries = Vec::new();
    for m in models {
        let repo = m["repo"].as_str().ok_or_else(|| anyhow::anyhow!("model missing repo"))?;
        let id = m["id"].as_str().unwrap_or("?");
        let quant_defs = m["quants"].as_array().ok_or_else(|| anyhow::anyhow!("{} has no quants", id))?;
        eprintln!("· {id} ({repo})");

        let first_file = quant_defs[0]["file"].as_str().unwrap();
        let arch = fetch_arch(&client, repo, first_file).await?;
        eprintln!("    arch = {}", arch["architecture"]);

        let mut quants = Vec::new();
        for q in quant_defs {
            let file = q["file"].as_str().unwrap();
            let (sha256, bytes) = hf_checksum_size(&client, repo, file).await?;
            eprintln!("    {} → sha256 {}… ({} bytes)", q["label"].as_str().unwrap_or("?"), &sha256[..16], bytes);
            quants.push(serde_json::json!({
                "label": q["label"], "bytes": bytes, "sha256": sha256,
                "source": hf_resolve_url(repo, file),
                "runtimeArgs": q.get("runtimeArgs").cloned().unwrap_or(serde_json::json!([]))
            }));
        }
        entries.push(serde_json::json!({
            "id": m["id"], "family": m["family"], "params": m["params"], "license": m["license"],
            "capabilities": m["capabilities"], "arch": arch, "quants": quants
        }));
    }

    let catalog = serde_json::json!({
        "schemaVersion": src["schemaVersion"], "revision": src["revision"],
        "generatedAt": "1970-01-01T00:00:00Z", "source": "generated",
        "verifiedSignature": null, "entries": entries
    });
    let out = crate_root().join("catalog").join("catalog.json");
    std::fs::write(&out, serde_json::to_string_pretty(&catalog)? + "\n")?;
    println!("wrote {} ({} models, all checksums pinned)", out.display(), models.len());
    sign_catalog()
}

/// Ad-hoc single entry to stdout (paths-info checksum + header arch), for quickly scaffolding one
/// model before adding it to a manifest.
async fn generate_one(a: &[String]) -> anyhow::Result<()> {
    let (repo, id, family, params, license, file) = (&a[0], &a[1], &a[2], &a[3], &a[4], &a[5]);
    let client = reqwest::Client::builder().user_agent("Kayon-catgen/1.0").build()?;
    let (sha256, bytes) = hf_checksum_size(&client, repo, file).await?;
    let arch = fetch_arch(&client, repo, file).await?;
    let entry = serde_json::json!({
        "id": id, "family": family, "params": params, "license": license,
        "capabilities": { "tools": false, "reasoning": false, "vision": false },
        "arch": arch,
        "quants": [{ "label": file.strip_suffix(".gguf").and_then(|s| s.rsplit('-').next()).unwrap_or("Q?"),
                     "bytes": bytes, "sha256": sha256, "source": hf_resolve_url(repo, file), "runtimeArgs": [] }]
    });
    println!("{}", serde_json::to_string_pretty(&entry)?);
    Ok(())
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(|s| s.as_str()).unwrap_or("");
    let result = match mode {
        "auto" => {
            let per_author: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20);
            let authors: Vec<String> = args.get(2)
                .map(|s| s.split(',').map(str::to_string).collect())
                .unwrap_or_else(|| discovery::TRUSTED_AUTHORS.iter().map(|s| s.to_string()).collect());
            generate_auto(per_author, authors).await
        }
        "manifest" => {
            let p = args.get(1).cloned().unwrap_or_else(|| "catalog/catalog.source.json".into());
            generate_from_manifest(&p).await
        }
        "one" if args.len() >= 7 => generate_one(&args[1..]).await,
        _ => {
            eprintln!("usage:\n  catgen auto [per_author] [author,author...]   # discover from HF + sign\n  catgen manifest <source.json>                 # generate + sign from a manifest\n  catgen one <repo> <id> <family> <params> <license> <file.gguf>");
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("catgen error: {e:#}");
        std::process::exit(1);
    }
}
