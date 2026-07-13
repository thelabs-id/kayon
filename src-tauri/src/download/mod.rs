use anyhow::{anyhow, Result};
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::db::Database;
use crate::ipc::*;

pub struct DownloadManager {
    client: reqwest::Client,
}

impl DownloadManager {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Kayon/0.1.0")
            .build()
            .expect("failed to build reqwest client");
        Self { client }
    }

    pub async fn start_download(
        &self,
        db: &Arc<Database>,
        model_id: &str,
        quant_label: &str,
        url: &str,
        target_path: &str,
        total_bytes: u64,
        sha256_expected: &str,
    ) -> Result<DownloadState> {
        // Honesty guard: never spend bandwidth on a catalog entry whose checksum has not
        // been pinned yet (CAT-6 generator scaffolds these as placeholders). Without a real
        // pinned hash the DL-3 gate can't verify, so we refuse up front instead of
        // downloading gigabytes only to quarantine them.
        let sha_norm = sha256_expected.trim().to_lowercase();
        if sha_norm.is_empty() || sha_norm.contains("todo") || sha_norm.contains("placeholder") || sha_norm.len() != 64 {
            return Err(anyhow!(
                "checksum not pinned for this entry — the catalog generator (CAT-6) must populate a real SHA-256 before download is allowed"
            ));
        }
        let free = free_disk_for_path(target_path)?;
        if free < total_bytes {
            return Err(anyhow!(
                "Insufficient disk: need {} bytes, {} available",
                total_bytes, free
            ));
        }
        let path = Path::new(target_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let received = if path.exists() {
            std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };
        let state = DownloadState {
            id: uuid::Uuid::new_v4().to_string(),
            model_id: model_id.to_string(),
            quant_label: quant_label.to_string(),
            url: url.to_string(),
            target_path: target_path.to_string(),
            total_bytes,
            received_bytes: received,
            sha256_expected: sha256_expected.to_string(),
            status: DownloadStatus::Active,
            started_at: Utc::now(),
            updated_at: Utc::now(),
            error: None,
            throughput_bps: 0,
            eta_seconds: None,
        };
        db.insert_download(&state)?;
        Ok(state)
    }

    pub async fn resume_download(
        &self,
        db: &Arc<Database>,
        download_id: &str,
        progress_tx: Option<mpsc::Sender<DownloadProgress>>,
    ) -> Result<()> {
        let state = db.get_download(download_id)?
            .ok_or_else(|| anyhow!("download not found"))?;

        // Only stream from the network when there are still bytes to fetch. A file that is
        // already fully on disk (e.g. resumed after a full download) skips straight to
        // verification instead of issuing a Range request that would 416.
        let need_network = state.total_bytes == 0 || state.received_bytes < state.total_bytes;
        let mut total_received = state.received_bytes;

        if need_network {
            let mut headers = reqwest::header::HeaderMap::new();
            if state.received_bytes > 0 {
                let range = format!("bytes={}-", state.received_bytes);
                headers.insert("Range", range.parse()?);
            }

            let resp = self.client.get(&state.url).headers(headers).send().await?;
            let http_status = resp.status().as_u16();
            // PRIV-5: log the request at the point it actually fires, with its real status.
            crate::telemetry::log_network_request_full(
                db, "GET", &state.url, "download", 0, 0, Some(http_status), None,
            );
            if !resp.status().is_success() && http_status != 206 {
                let msg = format!("HTTP {}", resp.status());
                db.set_download_error(download_id, &msg)?;
                return Err(anyhow!(msg));
            }

            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&state.target_path)?;

            let mut last_report = std::time::Instant::now();
            let start = std::time::Instant::now();

            let mut stream = resp.bytes_stream();
            use futures_util::StreamExt;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                file.write_all(&chunk)?;
                let n = chunk.len() as u64;
                total_received += n;

                if last_report.elapsed().as_millis() > 500 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let throughput = if elapsed > 0.0 {
                        ((total_received - state.received_bytes) as f64 / elapsed) as u64
                    } else {
                        0
                    };
                    let remaining = state.total_bytes.saturating_sub(total_received);
                    let eta = if throughput > 0 { Some(remaining / throughput) } else { None };

                    let progress = DownloadProgress {
                        id: download_id.to_string(),
                        model_id: state.model_id.clone(),
                        quant_label: state.quant_label.clone(),
                        bytes: total_received,
                        total_bytes: state.total_bytes,
                        percent: if state.total_bytes > 0 { (total_received as f32 / state.total_bytes as f32) * 100.0 } else { 0.0 },
                        throughput_bps: throughput,
                        eta_seconds: eta,
                        status: DownloadStatus::Active,
                    };
                    db.update_download_progress(download_id, total_received, &DownloadStatus::Active, throughput, eta)?;
                    if let Some(tx) = &progress_tx {
                        let _ = tx.send(progress).await;
                    }
                    last_report = std::time::Instant::now();
                }
            }
            file.flush()?;
        }

        // Verify by hashing the whole file from disk — correct for both fresh and resumed
        // downloads (a partial-stream hash would miss the pre-existing prefix). DL-3.
        let computed = hash_file(&state.target_path)?;
        if computed != state.sha256_expected {
            let quarantine = format!("{}.quarantine", state.target_path);
            let _ = std::fs::rename(&state.target_path, &quarantine);
            db.set_download_status(download_id, &DownloadStatus::Quarantined)?;
            return Err(anyhow!("SHA-256 mismatch: expected {}, got {} (quarantined)", state.sha256_expected, computed));
        }

        db.update_download_progress(download_id, total_received, &DownloadStatus::Completed, 0, None)?;

        // Enter the verified file into the library (DL-4, LIB-2). Idempotent on path.
        let existing = db.find_installed_by_path(&state.target_path)?;
        let model = InstalledModel {
            id: existing.map(|m| m.id).unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model_id: state.model_id.clone(),
            quant_label: state.quant_label.clone(),
            path: state.target_path.clone(),
            bytes: state.total_bytes,
            sha256: computed,
            source: ModelSource::Downloaded,
            installed_at: Utc::now(),
            ollama_tag: None,
            ollama_digest: None,
        };
        db.insert_installed_model(&model)?;
        Ok(())
    }

    pub async fn cancel_download(&self, db: &Arc<Database>, download_id: &str) -> Result<()> {
        db.set_download_status(download_id, &DownloadStatus::Cancelled)?;
        Ok(())
    }
}

fn hash_file(path: &str) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn free_disk_for_path(path: &str) -> Result<u64> {
    let p = Path::new(path);
    let root = p.ancestors().last().unwrap_or(p);
    let root_str = root.to_string_lossy();
    for disk in sysinfo::Disks::new_with_refreshed_list().list() {
        let mount = disk.mount_point().to_string_lossy();
        if root_str.starts_with(&*mount) || mount.starts_with(&*root_str) {
            return Ok(disk.available_space());
        }
    }
    Ok(0)
}
