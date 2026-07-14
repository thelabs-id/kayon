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
    // Ids of downloads the user has asked to STOP (cancel or pause); the streaming loop polls this
    // every chunk and aborts, so a stopped transfer never later completes and installs the model.
    // The endpoint sets the row's status (Cancelled/Paused) — the loop only stops (DL).
    cancelled: std::sync::Mutex<std::collections::HashSet<String>>,
    // One async lock per download id, held by a drive task for its whole lifetime. A resume waits
    // on it so it can't spawn a second writer while the paused task is still blocked in the stream
    // (which would let two tasks append to the same file and corrupt the partial download).
    drive_locks: std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<tokio::sync::Mutex<()>>>>,
}

impl DownloadManager {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Kayon/0.1.0")
            .build()
            .expect("failed to build reqwest client");
        Self {
            client,
            cancelled: std::sync::Mutex::new(std::collections::HashSet::new()),
            drive_locks: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    fn stop_requested(&self, id: &str) -> bool {
        self.cancelled.lock().unwrap().contains(id)
    }

    fn drive_lock(&self, id: &str) -> std::sync::Arc<tokio::sync::Mutex<()>> {
        self.drive_locks.lock().unwrap().entry(id.to_string()).or_default().clone()
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
    ) -> Result<(DownloadState, bool)> {
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
        // Don't start a second transfer for a target that's already downloading — two tasks
        // appending to the same GGUF would corrupt it. Return the in-flight row instead.
        if let Some(existing) = db.list_downloads()?.into_iter().find(|d| {
            d.target_path == target_path
                && matches!(d.status, DownloadStatus::Active | DownloadStatus::Queued)
        }) {
            return Ok((existing, false)); // already downloading — caller must not spawn again
        }
        let path = Path::new(target_path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Bytes already on disk from a prior partial transfer count toward the model — the
        // pre-flight only needs free space for the REMAINING bytes (CAT-4, DL-1).
        let mut received = if path.exists() {
            std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };
        // A file larger than the expected size is stale/corrupt: discard it now and treat this as a
        // fresh download, so the pre-flight checks the FULL size instead of a bogus zero remaining.
        if total_bytes > 0 && received > total_bytes {
            let _ = std::fs::remove_file(path);
            received = 0;
        }
        let remaining = total_bytes.saturating_sub(received);
        let free = free_disk_for_path(target_path)?;
        if free < remaining {
            return Err(anyhow!(
                "Insufficient disk: need {} more bytes, {} available",
                remaining, free
            ));
        }
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
        Ok((state, true))
    }

    /// Drive a download to completion, ensuring the row never gets stuck `Active` on an
    /// unexpected error (network drop, disk error). On failure, flip a still-active row to failed
    /// so a later Start re-spawns it, without clobbering a cancelled/quarantined status.
    pub async fn drive(&self, db: &Arc<Database>, download_id: &str) {
        if let Err(e) = self.resume_download(db, download_id, None).await {
            if let Ok(Some(d)) = db.get_download(download_id) {
                if matches!(d.status, DownloadStatus::Active | DownloadStatus::Queued) {
                    let _ = db.set_download_error(download_id, &e.to_string());
                }
            }
            log::error!("download {} failed: {}", download_id, e);
        }
    }

    pub async fn resume_download(
        &self,
        db: &Arc<Database>,
        download_id: &str,
        progress_tx: Option<mpsc::Sender<DownloadProgress>>,
    ) -> Result<()> {
        // Serialize drives of this id: wait for any prior task (e.g. one paused but still blocked in
        // the stream) to fully return before we touch the file, so two tasks never append at once.
        let lock = self.drive_lock(download_id);
        let _guard = lock.lock().await;

        // Now that the previous task has exited, clear any pause/cancel stop signal so this task's
        // streaming loop isn't immediately aborted. (A fresh start isn't in the set — harmless.)
        self.cancelled.lock().unwrap().remove(download_id);

        let state = db.get_download(download_id)?
            .ok_or_else(|| anyhow!("download not found"))?;

        // Reconcile the resume offset with the ACTUAL bytes on disk, not the last persisted
        // progress row. After a crash the file may hold chunks written after the last 500 ms DB
        // update; resuming from the stale DB offset would re-request already-present bytes and
        // duplicate them, failing the checksum. The file length is the source of truth (DL-1).
        let disk_len = std::fs::metadata(&state.target_path).map(|m| m.len()).unwrap_or(0);
        let mut resume_from = disk_len;
        if state.total_bytes > 0 && resume_from > state.total_bytes {
            // File is longer than expected (corrupt/over-written) — start clean.
            let _ = std::fs::remove_file(&state.target_path);
            resume_from = 0;
        }

        // Only stream from the network when there are still bytes to fetch. A file that is
        // already fully on disk (e.g. resumed after a full download) skips straight to
        // verification instead of issuing a Range request that would 416.
        let need_network = state.total_bytes == 0 || resume_from < state.total_bytes;
        let mut total_received = resume_from;

        if need_network {
            let mut headers = reqwest::header::HeaderMap::new();
            if resume_from > 0 {
                let range = format!("bytes={}-", resume_from);
                headers.insert("Range", range.parse()?);
            }

            let resp = match self.client.get(&state.url).headers(headers).send().await {
                Ok(r) => r,
                Err(e) => {
                    // PRIV-5: account for the outbound attempt even when it fails before a response
                    // (DNS/TLS/connect timeout), so the network log never omits a request we made.
                    crate::telemetry::log_network_request_full(
                        db, "GET", &state.url, "download", 0, 0, None, Some(format!("failed: {}", e)),
                    );
                    db.set_download_error(download_id, &e.to_string())?;
                    return Err(e.into());
                }
            };
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

            // If we asked to resume (received > 0) but the server ignored Range and returned a
            // full 200 body, appending would duplicate the prefix and fail the checksum. Restart
            // cleanly from byte 0 instead (DL-1 correctness).
            let restart_from_zero = resume_from > 0 && http_status == 200;
            let mut file = if restart_from_zero {
                total_received = 0;
                db.update_download_progress(download_id, 0, &DownloadStatus::Active, 0, None)?;
                std::fs::OpenOptions::new()
                    .create(true).write(true).truncate(true)
                    .open(&state.target_path)?
            } else {
                std::fs::OpenOptions::new()
                    .create(true).append(true)
                    .open(&state.target_path)?
            };

            let mut last_report = std::time::Instant::now();
            let start = std::time::Instant::now();
            let session_start_bytes = total_received; // bytes already on disk this session

            let mut stream = resp.bytes_stream();
            use futures_util::StreamExt;
            while let Some(chunk) = stream.next().await {
                // Cooperative stop: if the user cancelled or paused this download, stop streaming so
                // it never later marks itself completed/installed. The partial file is left on disk
                // for a resume (DL-1); the endpoint already set the row's status (Cancelled/Paused).
                if self.stop_requested(download_id) {
                    return Ok(());
                }
                let chunk = chunk?;
                file.write_all(&chunk)?;
                let n = chunk.len() as u64;
                total_received += n;

                // Never write past the catalog-signed size: a compromised/misconfigured origin
                // streaming more than `total_bytes` would blow the disk budget reserved by the
                // pre-flight. Stop and fail rather than let it run away (the checksum would also
                // reject it, but we don't want the extra bytes on disk in the first place).
                if state.total_bytes > 0 && total_received > state.total_bytes {
                    db.set_download_error(download_id, "origin sent more than the signed size")?;
                    return Err(anyhow!(
                        "origin sent more than the signed size ({} > {}) — aborted",
                        total_received, state.total_bytes
                    ));
                }

                if last_report.elapsed().as_millis() > 500 {
                    let elapsed = start.elapsed().as_secs_f64();
                    let throughput = if elapsed > 0.0 {
                        (total_received.saturating_sub(session_start_bytes) as f64 / elapsed) as u64
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

        // If stopped during the transfer or verification window, do not install (DL). Status was
        // already set by the pause/cancel endpoint.
        if self.stop_requested(download_id) {
            return Ok(());
        }

        // Verify by hashing the whole file from disk — correct for both fresh and resumed
        // downloads (a partial-stream hash would miss the pre-existing prefix). DL-3. Runs on the
        // blocking pool so hashing a multi-GB file doesn't stall the async runtime.
        let hash_path = state.target_path.clone();
        let computed = tokio::task::spawn_blocking(move || hash_file(&hash_path))
            .await
            .map_err(|e| anyhow!("hash task panicked: {}", e))??;
        if computed != state.sha256_expected {
            let quarantine = format!("{}.quarantine", state.target_path);
            let _ = std::fs::rename(&state.target_path, &quarantine);
            db.set_download_status(download_id, &DownloadStatus::Quarantined)?;
            return Err(anyhow!("SHA-256 mismatch: expected {}, got {} (quarantined)", state.sha256_expected, computed));
        }

        db.update_download_progress(download_id, total_received, &DownloadStatus::Completed, 0, None)?;

        // Enter the verified file into the library (DL-4, LIB-2). Idempotent on path.
        let existing = db.find_installed_by_path(&state.target_path)?;
        let architecture = crate::gguf::parse_gguf_header(std::path::Path::new(&state.target_path))
            .ok()
            .and_then(|h| crate::gguf::arch_from_header(&h));
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
            architecture,
            needs_newer_runtime: false,
        };
        db.insert_installed_model(&model)?;
        Ok(())
    }

    pub async fn cancel_download(&self, db: &Arc<Database>, download_id: &str) -> Result<()> {
        // Signal the in-flight streaming task to stop, then mark the row cancelled.
        self.cancelled.lock().unwrap().insert(download_id.to_string());
        db.set_download_status(download_id, &DownloadStatus::Cancelled)?;
        Ok(())
    }

    /// Pause an in-flight download: signal the streaming loop to stop and mark the row Paused. The
    /// partial bytes stay on disk (writes are unbuffered), so `resume_download` continues from the
    /// file length via a Range request — same mechanism as the DL-1 restart resume.
    pub async fn pause_download(&self, db: &Arc<Database>, download_id: &str) -> Result<()> {
        self.cancelled.lock().unwrap().insert(download_id.to_string());
        db.set_download_status(download_id, &DownloadStatus::Paused)?;
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
