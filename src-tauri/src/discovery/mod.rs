//! CAT-7 — runtime catalog discovery from Hugging Face.
//!
//! Kayon builds its model catalog live from Hugging Face on launch: it queries the most-downloaded
//! GGUF repos by a trusted allow-list of quantizers, pins each quant's real checksum (Git-LFS `oid`
//! *is* the SHA-256) and byte size from HF metadata, and derives the §7 `arch` block from a
//! range-fetched header. No hand-curation, no GitHub round-trip, no multi-GB downloads.
//!
//! Trust model (see spec §5 / CAT-7): the download origin is Hugging Face (CAT-2), and every
//! download is checksum-gated against the pinned `oid` (DL-3), so a corrupted/substituted file is
//! caught. What runtime discovery trades away vs. the bundled *signed* catalog is the Kayon-author
//! signature: discovered entries are pinned to HF's published hash rather than signed by Kayon's
//! key. Because HF is already the download origin, this collapses the trust to a single party
//! instead of adding one — but it is a real change from "trust rides on the signature", so it is
//! documented, and the bundled signed catalog remains the offline anchor and arch cache.
//!
//! Every outbound request is logged at egress (PRIV-5) when a database handle is supplied.

use crate::db::Database;
use crate::ipc::{ArchBlock, Capabilities, Catalog, CatalogEntry, Quant};
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Trusted GGUF quantizers to discover from. bartowski is the de-facto community standard for
/// single-file, consistently-named GGUF quants. Adding an author here is the one curation lever.
pub const TRUSTED_AUTHORS: &[&str] = &["bartowski"];
/// Quant labels we surface per model when present as single root files.
pub const WANTED_QUANTS: &[&str] = &["Q4_K_M", "Q8_0"];
/// Known families used to label a model from its repo name; discovery skips anything it can't match.
pub const FAMILIES: &[&str] = &[
    "Llama", "Qwen", "Gemma", "Phi", "Mistral", "DeepSeek", "SmolLM", "StableLM", "Yi",
    "Falcon", "Command", "Nemotron", "Granite", "Hermes", "Codestral", "Mixtral",
];
/// v1 is single consumer GPU: anything past this many billion params never fits, so don't list it.
pub const MAX_PARAMS_B: f64 = 34.0;

fn client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder().user_agent("Kayon-discovery/1.0").build()?)
}

fn log(db: Option<&Arc<Database>>, url: &str, bytes_in: u64, status: Option<u16>, note: Option<String>) {
    if let Some(db) = db {
        crate::telemetry::log_network_request_full(db, "GET", url, "catalog", 0, bytes_in, status, note);
    }
}

/// A repo returned by HF search, before we decide whether we can build an entry from it. HF already
/// sorts by downloads server-side, so we keep only what we need to build the entry.
struct HfRepo {
    id: String,
    tags: Vec<String>,
}

async fn hf_list_gguf_repos(
    client: &reqwest::Client,
    db: Option<&Arc<Database>>,
    author: &str,
    limit: usize,
) -> Result<Vec<HfRepo>> {
    let url = format!(
        "https://huggingface.co/api/models?author={author}&filter=gguf&sort=downloads&direction=-1&limit={limit}"
    );
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => { log(db, &url, 0, None, Some(format!("failed: {e}"))); return Err(e.into()); }
    };
    let status = resp.status().as_u16();
    let bytes = resp.bytes().await?;
    log(db, &url, bytes.len() as u64, Some(status), None);
    if status != 200 {
        return Err(anyhow!("HF model search {} for author {}", status, author));
    }
    let arr: serde_json::Value = serde_json::from_slice(&bytes)?;
    let mut out = Vec::new();
    for m in arr.as_array().cloned().unwrap_or_default() {
        let id = match m.get("id").and_then(|v| v.as_str()) { Some(s) => s.to_string(), None => continue };
        let tags = m.get("tags").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        out.push(HfRepo { id, tags });
    }
    Ok(out)
}

/// Root-level GGUF files of a repo, with LFS metadata: (path, oid = SHA-256, size). Non-recursive
/// and split-file-skipping: Kayon adopts single-file quants at the repo root only.
async fn hf_root_gguf_files(
    client: &reqwest::Client,
    db: Option<&Arc<Database>>,
    repo: &str,
) -> Result<Vec<(String, String, u64)>> {
    let url = format!("https://huggingface.co/api/models/{}/tree/main", repo);
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => { log(db, &url, 0, None, Some(format!("failed: {e}"))); return Err(e.into()); }
    };
    let status = resp.status().as_u16();
    let bytes = resp.bytes().await?;
    log(db, &url, bytes.len() as u64, Some(status), None);
    if status != 200 {
        return Err(anyhow!("HF tree {} for {}", status, repo));
    }
    let arr: serde_json::Value = serde_json::from_slice(&bytes)?;
    let mut out = Vec::new();
    for f in arr.as_array().cloned().unwrap_or_default() {
        let path = match f.get("path").and_then(|v| v.as_str()) { Some(s) => s, None => continue };
        let lower = path.to_ascii_lowercase();
        if !lower.ends_with(".gguf") || lower.contains("-of-") { continue; }
        let lfs = match f.get("lfs") { Some(l) => l, None => continue };
        let oid = match lfs.get("oid").and_then(|v| v.as_str()) { Some(s) => s.to_string(), None => continue };
        let size = match lfs.get("size").and_then(|v| v.as_u64()) { Some(s) => s, None => continue };
        out.push((path.to_string(), oid, size));
    }
    Ok(out)
}

fn hf_resolve_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", repo, file)
}

/// Derive the §7 `arch` block by range-fetching only the GGUF header (grows the window if a model
/// has an unusually large tokenizer). Never downloads tensor data. Returns None if the header
/// doesn't carry the fields honest fit needs.
async fn fetch_arch(
    client: &reqwest::Client,
    db: Option<&Arc<Database>>,
    repo: &str,
    file: &str,
) -> Result<ArchBlock> {
    let url = hf_resolve_url(repo, file);
    for window in [8_000_000u64, 32_000_000, 96_000_000] {
        let resp = client.get(&url).header("Range", format!("bytes=0-{}", window - 1)).send().await;
        let resp = match resp {
            Ok(r) => r,
            Err(e) => { log(db, &url, 0, None, Some(format!("failed: {e}"))); return Err(e.into()); }
        };
        let status = resp.status().as_u16();
        if status != 200 && status != 206 {
            log(db, &url, 0, Some(status), None);
            return Err(anyhow!("HF header fetch {} for {}/{}", status, repo, file));
        }
        let bytes = resp.bytes().await?;
        log(db, &url, bytes.len() as u64, Some(status), None);
        let tmp = std::env::temp_dir().join(format!("kayon-disc-{}", file.replace(['/', '\\'], "_")));
        std::fs::write(&tmp, &bytes)?;
        match crate::gguf::parse_gguf_header(&tmp) {
            Ok(h) => {
                let _ = std::fs::remove_file(&tmp);
                let architecture = crate::gguf::arch_from_header(&h).unwrap_or_else(|| "unknown".into());
                let m = |k: &str| h.metadata.get(&format!("{architecture}.{k}")).and_then(|v| v.as_u32());
                let (block_count, head_count, embedding_length, context_length) =
                    match (m("block_count"), m("attention.head_count"), m("embedding_length"), m("context_length")) {
                        (Some(b), Some(h), Some(e), Some(c)) => (b, h, e, c),
                        // Missing a load-bearing fit field — can't build an honest verdict, so skip.
                        _ => return Err(anyhow!("{}/{} header missing required arch fields", repo, file)),
                    };
                // Resolve every metadata read before moving `architecture` out (the closure borrows it).
                let head_count_kv = m("attention.head_count_kv").unwrap_or(head_count);
                let key_length = m("attention.key_length");
                let value_length = m("attention.value_length");
                let attention_type = crate::gguf::attention_type(&h);
                return Ok(ArchBlock {
                    architecture,
                    block_count,
                    head_count,
                    head_count_kv,
                    embedding_length,
                    context_length,
                    key_length,
                    value_length,
                    attention_type,
                    runtime_min_version: None,
                });
            }
            Err(_) => { let _ = std::fs::remove_file(&tmp); /* header not fully in window — grow it */ }
        }
    }
    Err(anyhow!("could not parse GGUF header for {}/{} within 96 MB", repo, file))
}

// ---- name / metadata heuristics ----

/// The billions-of-params figure parsed from a repo name (e.g. "…-3B-Instruct" → "3B", 3.0). Picks
/// the largest `<number>B` token so a version like "3.2" never masquerades as the size.
pub fn parse_params(name: &str) -> Option<(String, f64)> {
    let bytes = name.as_bytes();
    let mut best: Option<(String, f64)> = None;
    let mut i = 0;
    while i < bytes.len() {
        if (bytes[i] as char).is_ascii_digit() {
            let start = i;
            while i < bytes.len() && ((bytes[i] as char).is_ascii_digit() || bytes[i] == b'.') { i += 1; }
            if i < bytes.len() && (bytes[i] == b'B' || bytes[i] == b'b') {
                let after = bytes.get(i + 1).map(|&b| b as char);
                if matches!(after, None | Some('-') | Some('_') | Some('.') | Some(' ')) {
                    if let Ok(v) = name[start..i].parse::<f64>() {
                        if best.as_ref().map(|(_, b)| v > *b).unwrap_or(true) {
                            best = Some((format!("{}B", &name[start..i]), v));
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

pub fn parse_family(name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    FAMILIES.iter().find(|f| lower.contains(&f.to_ascii_lowercase())).map(|f| f.to_string())
}

fn parse_license(tags: &[String]) -> String {
    tags.iter().find_map(|t| t.strip_prefix("license:").map(str::to_string))
        .unwrap_or_else(|| "see model card".into())
}

pub fn slugify(name: &str) -> String {
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

/// Strip HF's trailing "-GGUF" and a leading "Org_" prefix bartowski adds (e.g. "google_gemma-2-9b").
pub fn clean_model_name(repo_id: &str) -> String {
    let name = repo_id.rsplit('/').next().unwrap_or(repo_id);
    let name = name.strip_suffix("-GGUF").or_else(|| name.strip_suffix("-gguf")).unwrap_or(name);
    if let Some(idx) = name.find('_') {
        if !name[..idx].contains('-') {
            return name[idx + 1..].to_string();
        }
    }
    name.to_string()
}

/// Build an oid → ArchBlock map from an existing catalog, so already-known files don't need their
/// header re-fetched. The bundled signed catalog seeds this, making steady-state launches cheap.
fn arch_cache(seed: &Catalog) -> HashMap<String, ArchBlock> {
    let mut map = HashMap::new();
    for e in &seed.entries {
        for q in &e.quants {
            map.insert(q.sha256.clone(), e.arch.clone());
        }
    }
    map
}

/// Assemble a fully-pinned entry from one discovered repo, reusing cached arch when the oid is
/// already known. Returns None (skipping, honestly) when the model can't be identified or lacks a
/// standard single-file quant.
async fn try_build_entry(
    client: &reqwest::Client,
    db: Option<&Arc<Database>>,
    repo: &HfRepo,
    cache: &HashMap<String, ArchBlock>,
) -> Option<CatalogEntry> {
    let name = clean_model_name(&repo.id);
    let (params_display, params_b) = parse_params(&name)?;
    if params_b > MAX_PARAMS_B { return None; }
    let family = parse_family(&name)?;
    let license = parse_license(&repo.tags);
    let id = slugify(&name);

    let files = hf_root_gguf_files(client, db, &repo.id).await.ok()?;
    let mut quants = Vec::new();
    for label in WANTED_QUANTS {
        let needle = format!("-{}.gguf", label.to_ascii_lowercase());
        if let Some((path, oid, size)) = files.iter().find(|(p, _, _)| p.to_ascii_lowercase().ends_with(&needle)) {
            quants.push(Quant {
                label: (*label).to_string(),
                bytes: *size,
                sha256: oid.clone(),
                source: hf_resolve_url(&repo.id, path),
                runtime_args: vec![],
            });
        }
    }
    if quants.is_empty() { return None; }

    // arch is identical across quants — reuse from cache if this oid is already known, else fetch.
    let arch = match cache.get(&quants[0].sha256) {
        Some(a) => a.clone(),
        None => {
            let first_file = quants[0].source.rsplit('/').next().unwrap();
            match fetch_arch(client, db, &repo.id, first_file).await {
                Ok(a) => a,
                Err(_) => return None,
            }
        }
    };

    Some(CatalogEntry {
        id,
        family,
        params: params_display,
        license,
        // Curation-judgment fields are never guessed (CAT-6): default conservative. A curated
        // overlay may enrich them later without blocking automated discovery.
        capabilities: Capabilities { tools: false, reasoning: false, vision: false },
        arch,
        quants,
    })
}

/// Discover models live from Hugging Face and return a fully-pinned in-memory catalog. `seed` (the
/// currently active catalog) is used only as an arch cache — its checksums are re-pinned fresh from
/// HF. `db` enables PRIV-5 egress logging (pass None from CLI contexts).
pub async fn discover_catalog(
    db: Option<&Arc<Database>>,
    authors: &[String],
    per_author: usize,
    revision: u64,
    seed: &Catalog,
) -> Result<Catalog> {
    let client = client()?;
    let cache = arch_cache(seed);
    let mut entries: Vec<CatalogEntry> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for author in authors {
        let repos = hf_list_gguf_repos(&client, db, author, per_author).await?;
        for repo in repos {
            if let Some(entry) = try_build_entry(&client, db, &repo, &cache).await {
                if seen.insert(entry.id.clone()) {
                    entries.push(entry);
                }
            }
        }
    }

    if entries.is_empty() {
        return Err(anyhow!("discovered 0 usable models — not replacing the current catalog"));
    }

    Ok(Catalog {
        schema_version: crate::catalog::SUPPORTED_SCHEMA_VERSION,
        revision,
        generated_at: chrono::Utc::now(),
        entries,
        source: "huggingface".to_string(),
        // Not Kayon-signed: discovered entries are pinned to HF's published hash (CAT-7). The
        // download checksum gate (DL-3) still verifies every byte against this pinned oid.
        verified_signature: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_params_ignoring_version_numbers() {
        assert_eq!(parse_params("Llama-3.2-3B-Instruct").unwrap().0, "3B");
        assert_eq!(parse_params("Qwen2.5-14B-Instruct").unwrap().0, "14B");
        assert_eq!(parse_params("Mistral-7B-Instruct-v0.3").unwrap().0, "7B");
        assert!(parse_params("DeepSeek-Coder-V2-Lite-Instruct").is_none());
    }

    #[test]
    fn cleans_repo_names() {
        assert_eq!(clean_model_name("bartowski/Llama-3.2-3B-Instruct-GGUF"), "Llama-3.2-3B-Instruct");
        assert_eq!(clean_model_name("bartowski/google_gemma-2-9b-it-GGUF"), "gemma-2-9b-it");
    }

    #[test]
    fn slugifies() {
        assert_eq!(slugify("Llama-3.2-3B-Instruct"), "llama-3-2-3b-instruct");
    }

    #[test]
    fn family_matched_case_insensitively() {
        assert_eq!(parse_family("qwen2.5-7b-instruct").as_deref(), Some("Qwen"));
        assert!(parse_family("some-unknown-model-7b").is_none());
    }
}
