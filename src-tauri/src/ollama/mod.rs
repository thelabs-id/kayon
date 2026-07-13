use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

use crate::ipc::OllamaModel;

pub fn ollama_store_dir() -> Option<PathBuf> {
    if let Ok(val) = std::env::var("OLLAMA_MODELS") {
        let p = PathBuf::from(val);
        if p.exists() {
            return Some(p);
        }
    }
    dirs::home_dir().map(|h| h.join(".ollama").join("models"))
}

pub fn discover_ollama_models(library_path: &str) -> Result<Vec<OllamaModel>> {
    let store = match ollama_store_dir() {
        Some(s) => s,
        None => return Ok(vec![]),
    };
    // Ollama registry namespace. Manifests live at
    //   manifests/registry.ollama.ai/<namespace>/<name>/<tag>
    // where <tag> is a *file* (JSON), not a directory (OLL-2).
    let manifests_dir = store.join("manifests").join("registry.ollama.ai");
    if !manifests_dir.exists() {
        return Ok(vec![]);
    }

    let mut models = Vec::new();
    let mut seen_digests: std::collections::HashSet<String> = std::collections::HashSet::new();
    let library_vol = volume_id(library_path);

    for manifest_path in walk_manifest_files(&manifests_dir) {
        // Derive name + tag from the path: the filename is the tag; the immediate
        // parent directory is the model name.
        let tag = match manifest_path.file_name().and_then(|s| s.to_str()) {
            Some(t) => t.to_string(),
            None => continue,
        };
        let name = manifest_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let manifest_data = match std::fs::read_to_string(&manifest_path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let manifest: serde_json::Value = match serde_json::from_str(&manifest_data) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let layers = match manifest["layers"].as_array() {
            Some(l) => l,
            None => continue,
        };

        for layer in layers {
            let media_type = layer["mediaType"].as_str().unwrap_or("");
            if media_type != "application/vnd.ollama.image.model" {
                continue;
            }
            let digest = layer["digest"].as_str().unwrap_or("").to_string();
            if digest.is_empty() {
                continue;
            }
            let hex_digest = digest.replace("sha256:", "");
            // Dedupe when multiple tags point at the same blob (OLL-5).
            if !seen_digests.insert(hex_digest.clone()) {
                break;
            }
            let blob_path = store.join("blobs").join(format!("sha256-{}", hex_digest));
            let size = layer["size"].as_u64().unwrap_or(0);

            let blob_vol = volume_id(&blob_path.to_string_lossy());
            let same_volume = library_vol.is_some() && library_vol == blob_vol;
            let blob_exists = blob_path.exists();

            let arch = read_arch_from_blob(&blob_path);
            let needs_newer_runtime = match arch.as_deref() {
                Some(a) => !crate::runtime::RuntimeManager::supported_runtime_archs()
                    .iter()
                    .any(|s| s.eq_ignore_ascii_case(a)),
                None => false,
            };

            let adopt_reason = if !blob_exists {
                Some("blob not found on disk".into())
            } else if !same_volume {
                Some("cross-volume: copy or relocate needed".into())
            } else if needs_newer_runtime {
                Some(format!(
                    "adoptable, but architecture '{}' needs a newer llama.cpp runtime to load",
                    arch.clone().unwrap_or_default()
                ))
            } else {
                None
            };

            models.push(OllamaModel {
                name: name.clone(),
                tag: tag.clone(),
                digest: hex_digest,
                size_bytes: size,
                blob_path: blob_path.to_string_lossy().to_string(),
                architecture: arch,
                families: vec![],
                parameter_size: None,
                quantization: None,
                same_volume_as_library: same_volume,
                // Cross-volume/blob-missing block adoption here; a newer-runtime need does not
                // (OLL-6: adopt but flag), so it stays adoptable.
                adoptable: blob_exists && same_volume,
                adopt_reason,
                needs_newer_runtime,
            });
            break;
        }
    }

    Ok(models)
}

pub fn adopt_model(
    blob_path: &str,
    library_path: &str,
    model_name: &str,
    tag: &str,
    digest: &str,
    _size: u64,
) -> Result<String> {
    let blob = Path::new(blob_path);
    if !blob.exists() {
        return Err(anyhow!("blob not found: {}", blob_path));
    }

    // OLL-3 checksum gate: the blob's content must actually hash to the manifest digest before
    // it enters the library. We never trust the caller-supplied digest or the filename alone —
    // that's what makes "record the digest as the checksum for free" honest.
    let expected = digest.trim().to_lowercase().replace("sha256:", "");
    // The digest MUST be a well-formed SHA-256; a malformed one can't gate anything, so adoption
    // fails rather than silently linking an unverified blob (OLL-3 / §5 checksum gate).
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(anyhow!("invalid Ollama digest '{}' — expected a 64-char SHA-256", digest));
    }
    let computed = hash_file(blob)?;
    if computed != expected {
        return Err(anyhow!(
            "blob hash mismatch: manifest digest {} but content hashes to {} — not adopting",
            expected, computed
        ));
    }

    let dest = PathBuf::from(library_path).join(format!("{}-{}.gguf", model_name, tag));
    // Adopt in place via hard link — zero bytes copied. On a cross-volume attempt this returns
    // an OS error (links can't span volumes); the caller surfaces the copy/relocate choice
    // (OLL-4) rather than silently copying.
    if dest.exists() {
        return Ok(dest.to_string_lossy().to_string());
    }
    std::fs::hard_link(blob, &dest)
        .map_err(|e| anyhow!("hard link failed (cross-volume? use copy/relocate): {}", e))?;
    Ok(dest.to_string_lossy().to_string())
}

fn hash_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let f = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::with_capacity(1 << 20, f);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_arch_from_blob(path: &Path) -> Option<String> {
    if !path.exists() { return None; }
    crate::gguf::parse_gguf_header(path).ok()
        .and_then(|h| crate::gguf::arch_from_header(&h))
}

fn volume_id(path: &str) -> Option<String> {
    let p = Path::new(path);
    p.components().next().map(|c| c.as_os_str().to_string_lossy().to_string())
}

/// Recursively collect every regular file under `dir` — each is a candidate Ollama
/// manifest (a tag file). Depth-bounded to avoid pathological trees.
fn walk_manifest_files(dir: &Path) -> Vec<PathBuf> {
    fn recurse(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if depth == 0 {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    recurse(&path, depth - 1, out);
                } else if path.is_file() {
                    out.push(path);
                }
            }
        }
    }
    let mut results = Vec::new();
    recurse(dir, 8, &mut results);
    results
}
