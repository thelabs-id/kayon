//! catgen — catalog generator (CAT-6). Turns the human-curated `catalog.source.json` into a fully
//! pinned, signed `catalog.json` — no `TODO_PLACEHOLDER` checksums, and without downloading the
//! multi-GB model files.
//!
//! How it pins checksums WITHOUT downloading: Hugging Face stores GGUFs in Git LFS, and Git LFS
//! uses the file's SHA-256 as its object id. HF's `paths-info` API returns that `lfs.oid` (= the
//! real SHA-256) and the exact byte size for free. The GGUF `arch` block still needs the header, so
//! catgen range-fetches only the first few MB of each model (the header lives at the start).
//!
//! Usage:
//!   cargo run --bin catgen -- auto [limit] [author,author...]  # DISCOVER models from HF + sign
//!   cargo run --bin catgen -- manifest catalog/catalog.source.json   # generate + sign from a manifest
//!   cargo run --bin catgen -- one <repo> <id> <family> <params> <license> <file.gguf>  # ad-hoc entry
//!
//! `auto` mode needs no hand-curation: it queries Hugging Face for the most-downloaded GGUF repos by
//! trusted quantizers (default: bartowski), derives each model's identity from the repo, pins the
//! real checksum + size from HF's LFS metadata, derives the arch from a range-fetched header, and
//! signs the catalog. Run it on a schedule (GitHub Action) to keep the catalog fresh automatically.

use ed25519_dalek::{Signer, SigningKey};
use std::collections::HashSet;
use std::path::PathBuf;

/// Trusted GGUF quantizers to discover from. bartowski is the de-facto community standard; add
/// official orgs here if their file naming is single-file and consistent.
const TRUSTED_AUTHORS: &[&str] = &["bartowski"];
/// Quant labels we surface (one entry per model gets these if present as single files).
const WANTED_QUANTS: &[&str] = &["Q4_K_M", "Q8_0"];
/// Known families to label a model from its repo name.
const FAMILIES: &[&str] = &[
    "Llama", "Qwen", "Gemma", "Phi", "Mistral", "DeepSeek", "SmolLM", "StableLM", "Yi",
    "Falcon", "Command", "Nemotron", "Granite", "Hermes", "Codestral", "Mixtral",
];

#[path = "../gguf/mod.rs"]
mod gguf;

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
/// (KAYON_CATALOG_SEED hex env, else the gitignored catalog/signing.key). Fully pinned + signed in
/// one command.
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

        // arch is identical across quants — derive it once from the first quant's header.
        let first_file = quant_defs[0]["file"].as_str().unwrap();
        let arch = fetch_arch(&client, repo, first_file).await?;
        eprintln!("    arch = {}", arch["architecture"]);

        let mut quants = Vec::new();
        for q in quant_defs {
            let file = q["file"].as_str().unwrap();
            let (sha256, bytes) = hf_checksum_size(&client, repo, file).await?;
            eprintln!("    {} → sha256 {}… ({} bytes)", q["label"].as_str().unwrap_or("?"), &sha256[..16], bytes);
            quants.push(serde_json::json!({
                "label": q["label"],
                "bytes": bytes,
                "sha256": sha256,
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
        "schemaVersion": src["schemaVersion"],
        "revision": src["revision"],
        "generatedAt": "1970-01-01T00:00:00Z",
        "source": "generated",
        "verifiedSignature": null,
        "entries": entries
    });
    let out = crate_root().join("catalog").join("catalog.json");
    std::fs::write(&out, serde_json::to_string_pretty(&catalog)? + "\n")?;
    println!("wrote {} ({} models, all checksums pinned)", out.display(), models.len());
    sign_catalog()?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// auto mode — discover models from Hugging Face with zero hand-curation.
// ---------------------------------------------------------------------------------------------

/// A repo returned by HF search, before we decide whether we can build an entry from it.
struct HfRepo {
    id: String,
    downloads: u64,
    tags: Vec<String>,
}

/// Query HF for a trusted author's most-downloaded GGUF repos.
async fn hf_list_gguf_repos(client: &reqwest::Client, author: &str, limit: usize) -> anyhow::Result<Vec<HfRepo>> {
    let url = format!(
        "https://huggingface.co/api/models?author={author}&filter=gguf&sort=downloads&direction=-1&limit={limit}"
    );
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("HF model search {} for author {}", resp.status(), author);
    }
    let arr: serde_json::Value = resp.json().await?;
    let mut out = Vec::new();
    for m in arr.as_array().cloned().unwrap_or_default() {
        let id = match m.get("id").and_then(|v| v.as_str()) { Some(s) => s.to_string(), None => continue };
        let downloads = m.get("downloads").and_then(|v| v.as_u64()).unwrap_or(0);
        let tags = m.get("tags").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        out.push(HfRepo { id, downloads, tags });
    }
    Ok(out)
}

/// Root-level files of a repo, with LFS metadata (oid = SHA-256, size). Non-recursive: Kayon only
/// adopts single-file quants that live at the repo root.
async fn hf_root_gguf_files(client: &reqwest::Client, repo: &str) -> anyhow::Result<Vec<(String, String, u64)>> {
    let url = format!("https://huggingface.co/api/models/{}/tree/main", repo);
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("HF tree {} for {}", resp.status(), repo);
    }
    let arr: serde_json::Value = resp.json().await?;
    let mut out = Vec::new();
    for f in arr.as_array().cloned().unwrap_or_default() {
        let path = match f.get("path").and_then(|v| v.as_str()) { Some(s) => s, None => continue };
        if !path.to_ascii_lowercase().ends_with(".gguf") { continue; }
        // Skip split/multi-part quants ("-00001-of-00003.gguf") — Kayon adopts single files only.
        if path.to_ascii_lowercase().contains("-of-") { continue; }
        let lfs = match f.get("lfs") { Some(l) => l, None => continue };
        let oid = match lfs.get("oid").and_then(|v| v.as_str()) { Some(s) => s.to_string(), None => continue };
        let size = match lfs.get("size").and_then(|v| v.as_u64()) { Some(s) => s, None => continue };
        out.push((path.to_string(), oid, size));
    }
    Ok(out)
}

/// The billions-of-params figure parsed from a repo name, e.g. "Llama-3.2-3B-Instruct" → 3.0.
/// Returns (display, billions). Looks for the last `<number>B` token.
fn parse_params(name: &str) -> Option<(String, f64)> {
    // Scan tokens split on '-', '_', '.', ' ' but keep the "3B"/"3.2B" shape intact by regexing chars.
    let bytes = name.as_bytes();
    let mut best: Option<(String, f64)> = None;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_digit() {
            let start = i;
            while i < bytes.len() && ((bytes[i] as char).is_ascii_digit() || bytes[i] == b'.') { i += 1; }
            // A params token ends in 'B'/'b' immediately after the number.
            if i < bytes.len() && (bytes[i] == b'B' || bytes[i] == b'b') {
                // Guard against 'Bit'/'Base' etc. — next char must be a boundary.
                let after = bytes.get(i + 1).map(|&b| b as char);
                let boundary = matches!(after, None | Some('-') | Some('_') | Some('.') | Some(' '));
                if boundary {
                    let num_str = &name[start..i];
                    if let Ok(v) = num_str.parse::<f64>() {
                        // Prefer the largest plausible model-size token (avoids "3.2" version numbers).
                        if best.as_ref().map(|(_, b)| v > *b).unwrap_or(true) {
                            best = Some((format!("{}B", num_str), v));
                        }
                    }
                }
            }
        } else {
            i += 1;
        }
    }
    best
}

/// The model family, matched case-insensitively against the known list.
fn parse_family(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    FAMILIES.iter().find(|f| lower.contains(&f.to_ascii_lowercase())).map(|f| f.to_string())
}

/// The SPDX-ish license from HF tags (`license:apache-2.0` → "apache-2.0").
fn parse_license(tags: &[String]) -> String {
    tags.iter()
        .find_map(|t| t.strip_prefix("license:").map(str::to_string))
        .unwrap_or_else(|| "see model card".into())
}

/// A stable catalog id from a model name: lowercased, non-alphanumerics collapsed to single dashes.
fn slugify(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    let mut prev_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            s.push('-');
            prev_dash = true;
        }
    }
    s.trim_matches('-').to_string()
}

/// Strip HF's trailing "-GGUF" and a leading "org_" prefix bartowski adds (e.g. "google_gemma-2-9b").
fn clean_model_name(repo_id: &str) -> String {
    let name = repo_id.rsplit('/').next().unwrap_or(repo_id);
    let name = name.strip_suffix("-GGUF").or_else(|| name.strip_suffix("-gguf")).unwrap_or(name);
    // A single leading "Org_" segment (no dashes before the underscore) is the quantizer's org tag.
    if let Some(idx) = name.find('_') {
        if !name[..idx].contains('-') {
            return name[idx + 1..].to_string();
        }
    }
    name.to_string()
}

/// Try to assemble a fully-pinned catalog entry from one discovered repo. Returns None (with a log
/// line) when the model can't be identified or has no standard single-file quant — discovery skips
/// it rather than emitting a half-known entry.
async fn try_build_entry(client: &reqwest::Client, repo: &HfRepo) -> Option<serde_json::Value> {
    let name = clean_model_name(&repo.id);
    let (params_display, params_b) = match parse_params(&name) {
        Some(p) => p,
        None => { eprintln!("  skip {} — no params in name", repo.id); return None; }
    };
    // v1 is single-GPU consumer hardware: anything past ~34B never fits, so don't list it.
    if params_b > 34.0 { eprintln!("  skip {} — {} too large for v1", repo.id, params_display); return None; }
    let family = match parse_family(&name) {
        Some(f) => f,
        None => { eprintln!("  skip {} — unknown family", repo.id); return None; }
    };
    let license = parse_license(&repo.tags);
    let id = slugify(&name);

    let files = match hf_root_gguf_files(client, &repo.id).await {
        Ok(f) => f,
        Err(e) => { eprintln!("  skip {} — tree fetch failed: {e}", repo.id); return None; }
    };
    // Pick the wanted quants that exist as single root files.
    let mut quants = Vec::new();
    for label in WANTED_QUANTS {
        let needle = format!("-{}.gguf", label.to_ascii_lowercase());
        if let Some((path, oid, size)) = files.iter().find(|(p, _, _)| p.to_ascii_lowercase().ends_with(&needle)) {
            quants.push(serde_json::json!({
                "label": label,
                "bytes": size,
                "sha256": oid,
                "source": hf_resolve_url(&repo.id, path),
                "runtimeArgs": []
            }));
        }
    }
    if quants.is_empty() {
        eprintln!("  skip {} — no {:?} single-file quant", repo.id, WANTED_QUANTS);
        return None;
    }

    // arch is identical across quants — derive it once from the first quant's file.
    let first_file = quants[0]["source"].as_str().unwrap().rsplit('/').next().unwrap();
    let arch = match fetch_arch(client, &repo.id, first_file).await {
        Ok(a) => a,
        Err(e) => { eprintln!("  skip {} — arch derive failed: {e}", repo.id); return None; }
    };

    eprintln!("  + {id} ({family} {params_display}, {} quant(s), {} dl)", quants.len(), repo.downloads);
    Some(serde_json::json!({
        "id": id, "family": family, "params": params_display, "license": license,
        // Capabilities aren't reliably machine-derivable from HF metadata; default conservative.
        // A curated overlay can enrich these later without blocking automated discovery.
        "capabilities": { "tools": false, "reasoning": false, "vision": false },
        "arch": arch, "quants": quants
    }))
}

/// Discover models from the trusted authors, build pinned entries, write + sign the catalog.
async fn discover_and_generate(authors: &[String], per_author: usize, revision: u64) -> anyhow::Result<()> {
    let client = reqwest::Client::builder().user_agent("Kayon-catgen/1.0").build()?;
    let mut entries = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for author in authors {
        eprintln!("· discovering {author} (top {per_author} by downloads)");
        let repos = hf_list_gguf_repos(&client, author, per_author).await?;
        for repo in repos {
            if let Some(entry) = try_build_entry(&client, &repo).await {
                let id = entry["id"].as_str().unwrap_or("").to_string();
                if seen.insert(id) {
                    entries.push(entry);
                }
            }
        }
    }

    if entries.is_empty() {
        anyhow::bail!("discovered 0 usable models — refusing to write an empty catalog");
    }

    let catalog = serde_json::json!({
        "schemaVersion": 1,
        "revision": revision,
        "generatedAt": "1970-01-01T00:00:00Z",
        "source": "generated",
        "verifiedSignature": null,
        "entries": entries
    });
    let out = crate_root().join("catalog").join("catalog.json");
    std::fs::write(&out, serde_json::to_string_pretty(&catalog)? + "\n")?;
    println!("wrote {} ({} models discovered + pinned, rev {})", out.display(), catalog["entries"].as_array().unwrap().len(), revision);
    sign_catalog()?;
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
                .unwrap_or_else(|| TRUSTED_AUTHORS.iter().map(|s| s.to_string()).collect());
            // Revision must strictly increase for Kayon to adopt (CAT-5). CI sets a monotonic
            // minute-stamp via env; a local run defaults to bumping the existing catalog's rev by 1.
            let revision: u64 = std::env::var("KAYON_CATALOG_REVISION").ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    let existing = std::fs::read(crate_root().join("catalog").join("catalog.json")).ok()
                        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                        .and_then(|v| v.get("revision").and_then(|r| r.as_u64()))
                        .unwrap_or(0);
                    existing + 1
                });
            discover_and_generate(&authors, per_author, revision).await
        }
        "manifest" => {
            let p = args.get(1).cloned().unwrap_or_else(|| "catalog/catalog.source.json".into());
            generate_from_manifest(&p).await
        }
        "one" if args.len() >= 7 => generate_one(&args[1..]).await,
        _ => {
            eprintln!("usage:\n  catgen auto [per_author] [author,author...]   # discover from HF + sign\n  catgen manifest <source.json>                 # generate + sign the full catalog\n  catgen one <repo> <id> <family> <params> <license> <file.gguf>");
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("catgen error: {e:#}");
        std::process::exit(1);
    }
}

/// Ad-hoc single entry to stdout (paths-info checksum + header arch), for quickly scaffolding one
/// model before adding it to the source manifest.
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
