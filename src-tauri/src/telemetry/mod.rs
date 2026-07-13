use chrono::Utc;
use std::sync::Arc;

use crate::db::Database;
use crate::ipc::*;

pub struct TelemetryManager {
    enabled: bool,
}

impl TelemetryManager {
    pub fn new(db: &Database) -> Self {
        let enabled = db.get_preference("telemetry_enabled")
            .map(|v| v == "true")
            .unwrap_or(false);
        Self { enabled }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn toggle(&mut self, db: &Database, enabled: bool) -> Result<(), String> {
        self.enabled = enabled;
        db.set_preference("telemetry_enabled", if enabled { "true" } else { "false" })
            .map_err(|e| e.to_string())
    }

    pub fn status(&self, db: &Database) -> TelemetryStatus {
        let preview = db.get_preference("last_preview_payload");
        let preview_at = db.get_preference("last_preview_at")
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc));
        TelemetryStatus {
            enabled: self.enabled,
            last_preview_payload: preview,
            last_preview_at: preview_at,
        }
    }

    pub fn preview_payload(&self, db: &Database) -> TelemetryPreview {
        let payload = serde_json::json!({
            "app": "kayon",
            "version": "0.1.0",
            "timestamp": Utc::now().to_rfc3339(),
            "event": "session_start",
        }).to_string();

        let _ = db.set_preference("last_preview_payload", &payload);
        let _ = db.set_preference("last_preview_at", &Utc::now().to_rfc3339());

        TelemetryPreview {
            endpoint: "https://telemetry.kayon.app/v1/events".into(),
            payload: payload.clone(),
            byte_size: payload.len(),
            shown_at: Utc::now(),
        }
    }
}

pub fn log_network_request(db: &Arc<Database>, method: &str, url: &str, purpose: &str) {
    log_network_request_full(db, method, url, purpose, 0, 0, None, None);
}

/// Record an actual outbound request with its result. This is the single place the net log
/// is written for egress (PRIV-5): callers invoke it at the point a request truly fires,
/// with the observed status and byte counts, so the log accounts for real traffic — never
/// intent that was refused before a socket opened.
#[allow(clippy::too_many_arguments)]
pub fn log_network_request_full(
    db: &Arc<Database>,
    method: &str,
    url: &str,
    purpose: &str,
    bytes_out: u64,
    bytes_in: u64,
    status: Option<u16>,
    note: Option<String>,
) {
    let entry = NetworkLogEntry {
        id: 0,
        ts: Utc::now(),
        method: method.to_string(),
        url: url.to_string(),
        purpose: purpose.to_string(),
        bytes_out,
        bytes_in,
        status,
        note,
    };
    let _ = db.insert_net_log(&entry);
}
