use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::ipc::{
    ChatMessage, ChatSession, ChatSessionSummary, DownloadState, DownloadStatus, InstalledModel,
    ModelSource,
};

pub struct Database {
    conn: Mutex<Connection>,
    path: PathBuf,
}

impl Database {
    pub fn open() -> Result<Self> {
        let dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".kayon");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("kayon.db");
        let conn = Connection::open(&path).context("opening kayon.db")?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;",
        )?;
        let db = Self { conn: Mutex::new(conn), path };
        db.init_tables()?;
        Ok(db)
    }

    fn init_tables(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS installed_models (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                quant_label TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                source TEXT NOT NULL CHECK(source IN ('downloaded','adopted')),
                installed_at TEXT NOT NULL,
                ollama_tag TEXT,
                ollama_digest TEXT,
                architecture TEXT
            );
            CREATE TABLE IF NOT EXISTS downloads (
                id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                quant_label TEXT NOT NULL,
                url TEXT NOT NULL,
                target_path TEXT NOT NULL,
                total_bytes INTEGER NOT NULL,
                received_bytes INTEGER NOT NULL DEFAULT 0,
                sha256_expected TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error TEXT,
                throughput_bps INTEGER NOT NULL DEFAULT 0,
                eta_seconds INTEGER
            );
            CREATE TABLE IF NOT EXISTS prefs (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS net_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                method TEXT NOT NULL,
                url TEXT NOT NULL,
                purpose TEXT NOT NULL,
                bytes_out INTEGER NOT NULL DEFAULT 0,
                bytes_in INTEGER NOT NULL DEFAULT 0,
                status INTEGER,
                note TEXT
            );
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                quant_label TEXT NOT NULL,
                context_length INTEGER NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                gen_tokens INTEGER NOT NULL,
                prompt_eval_tok_per_s REAL NOT NULL,
                gen_tok_per_s REAL NOT NULL,
                warm INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                run_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS catalog_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model_id TEXT,
                system_prompt TEXT NOT NULL,
                temperature REAL NOT NULL,
                top_p REAL NOT NULL,
                max_tokens INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                reasoning TEXT,
                ordinal INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                ON chat_messages (session_id, ordinal);",
        )?;
        // Migration for DBs created before the architecture column existed. Ignore the error if
        // the column is already present.
        let _ = conn.execute("ALTER TABLE installed_models ADD COLUMN architecture TEXT", []);
        Ok(())
    }

    pub fn path(&self) -> &PathBuf { &self.path }

    // ---- Chat sessions (RUN-5) ----

    pub fn create_chat_session(&self, s: &ChatSession) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO chat_sessions
             (id, title, model_id, system_prompt, temperature, top_p, max_tokens, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                s.id, s.title, s.model_id, s.system_prompt, s.temperature as f64, s.top_p as f64,
                s.max_tokens, s.created_at.to_rfc3339(), s.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn list_chat_sessions(&self) -> Result<Vec<ChatSessionSummary>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT s.id, s.title, s.model_id, s.updated_at,
                    (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id)
             FROM chat_sessions s ORDER BY s.updated_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(ChatSessionSummary {
                id: r.get(0)?,
                title: r.get(1)?,
                model_id: r.get(2)?,
                updated_at: parse_dt(r.get::<_, String>(3)?),
                message_count: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn get_chat_session(&self, id: &str) -> Result<Option<ChatSession>> {
        let conn = self.conn.lock().unwrap();
        let s = conn.query_row(
            "SELECT id, title, model_id, system_prompt, temperature, top_p, max_tokens, created_at, updated_at
             FROM chat_sessions WHERE id = ?1",
            params![id],
            |r| Ok(ChatSession {
                id: r.get(0)?,
                title: r.get(1)?,
                model_id: r.get(2)?,
                system_prompt: r.get(3)?,
                temperature: r.get::<_, f64>(4)? as f32,
                top_p: r.get::<_, f64>(5)? as f32,
                max_tokens: r.get(6)?,
                created_at: parse_dt(r.get::<_, String>(7)?),
                updated_at: parse_dt(r.get::<_, String>(8)?),
            }),
        ).optional()?;
        Ok(s)
    }

    pub fn get_chat_messages(&self, session_id: &str) -> Result<Vec<ChatMessage>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, session_id, role, content, reasoning, ordinal, created_at
             FROM chat_messages WHERE session_id = ?1 ORDER BY ordinal ASC",
        )?;
        let rows = stmt.query_map(params![session_id], |r| {
            Ok(ChatMessage {
                id: r.get(0)?,
                session_id: r.get(1)?,
                role: r.get(2)?,
                content: r.get(3)?,
                reasoning: r.get(4)?,
                ordinal: r.get(5)?,
                created_at: parse_dt(r.get::<_, String>(6)?),
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    /// Append a message and bump the session's `updated_at`. The ordinal is assigned as the next
    /// slot in the session, so ordering is stable regardless of clock resolution. Returns the
    /// assigned ordinal so callers report the true position rather than a placeholder.
    pub fn append_chat_message(&self, m: &ChatMessage) -> Result<i64> {
        let conn = self.conn.lock().unwrap();
        let next: i64 = conn.query_row(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM chat_messages WHERE session_id = ?1",
            params![m.session_id],
            |r| r.get(0),
        )?;
        conn.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, reasoning, ordinal, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![m.id, m.session_id, m.role, m.content, m.reasoning, next, m.created_at.to_rfc3339()],
        )?;
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ?2 WHERE id = ?1",
            params![m.session_id, Utc::now().to_rfc3339()],
        )?;
        Ok(next)
    }

    pub fn rename_chat_session(&self, id: &str, title: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE chat_sessions SET title = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, title, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Persist a session's per-conversation settings (system prompt + sampling params).
    pub fn update_chat_session_settings(
        &self, id: &str, system_prompt: &str, temperature: f32, top_p: f32, max_tokens: i64,
        model_id: Option<&str>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE chat_sessions
             SET system_prompt = ?2, temperature = ?3, top_p = ?4, max_tokens = ?5, model_id = ?6,
                 updated_at = ?7
             WHERE id = ?1",
            params![id, system_prompt, temperature as f64, top_p as f64, max_tokens, model_id,
                    Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn delete_chat_session(&self, id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?1", params![id])?;
        conn.execute("DELETE FROM chat_sessions WHERE id = ?1", params![id])?;
        Ok(())
    }

    pub fn get_preference(&self, key: &str) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT value FROM prefs WHERE key = ?1",
            params![key],
            |row| row.get(0),
        ).ok()
    }

    pub fn set_preference(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO prefs (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn get_catalog_meta(&self, key: &str) -> Option<String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT value FROM catalog_meta WHERE key = ?1",
            params![key],
            |row| row.get(0),
        ).ok()
    }

    pub fn set_catalog_meta(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO catalog_meta (key, value) VALUES (?1, ?2)",
            params![key, value],
        )?;
        Ok(())
    }

    pub fn insert_installed_model(&self, m: &InstalledModel) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let src = match m.source {
            ModelSource::Downloaded => "downloaded",
            ModelSource::Adopted => "adopted",
        };
        conn.execute(
            "INSERT OR REPLACE INTO installed_models
             (id, model_id, quant_label, path, bytes, sha256, source, installed_at, ollama_tag, ollama_digest, architecture)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                m.id, m.model_id, m.quant_label, m.path,
                m.bytes as i64, m.sha256, src,
                m.installed_at.to_rfc3339(),
                m.ollama_tag, m.ollama_digest, m.architecture,
            ],
        )?;
        Ok(())
    }

    fn map_installed(row: &rusqlite::Row<'_>) -> rusqlite::Result<InstalledModel> {
        let source_str: String = row.get(6)?;
        let source = match source_str.as_str() {
            "adopted" => ModelSource::Adopted,
            _ => ModelSource::Downloaded,
        };
        Ok(InstalledModel {
            id: row.get(0)?,
            model_id: row.get(1)?,
            quant_label: row.get(2)?,
            path: row.get(3)?,
            bytes: row.get::<_, i64>(4)? as u64,
            sha256: row.get(5)?,
            source,
            installed_at: parse_dt(row.get::<_, String>(7)?),
            ollama_tag: row.get(8)?,
            ollama_digest: row.get(9)?,
            architecture: row.get(10)?,
            needs_newer_runtime: false, // computed in the library listing, not stored
        })
    }

    pub fn list_installed_models(&self) -> Result<Vec<InstalledModel>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, model_id, quant_label, path, bytes, sha256, source, installed_at, ollama_tag, ollama_digest, architecture
             FROM installed_models ORDER BY installed_at DESC",
        )?;
        let rows = stmt.query_map([], Self::map_installed)?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn get_installed_model(&self, id: &str) -> Result<Option<InstalledModel>> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row(
            "SELECT id, model_id, quant_label, path, bytes, sha256, source, installed_at, ollama_tag, ollama_digest, architecture
             FROM installed_models WHERE id = ?1",
            params![id],
            Self::map_installed,
        ).optional()?)
    }

    pub fn find_installed_by_path(&self, path: &str) -> Result<Option<InstalledModel>> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row(
            "SELECT id, model_id, quant_label, path, bytes, sha256, source, installed_at, ollama_tag, ollama_digest, architecture
             FROM installed_models WHERE path = ?1",
            params![path],
            Self::map_installed,
        ).optional()?)
    }

    pub fn update_installed_path(&self, id: &str, new_path: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE installed_models SET path = ?2 WHERE id = ?1",
            params![id, new_path],
        )?;
        Ok(())
    }

    pub fn remove_installed_model(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute("DELETE FROM installed_models WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }

    fn status_to_str(s: &DownloadStatus) -> &'static str {
        match s {
            DownloadStatus::Queued => "queued",
            DownloadStatus::Active => "active",
            DownloadStatus::Paused => "paused",
            DownloadStatus::Completed => "completed",
            DownloadStatus::Failed => "failed",
            DownloadStatus::Cancelled => "cancelled",
            DownloadStatus::Quarantined => "quarantined",
        }
    }

    fn str_to_status(s: &str) -> DownloadStatus {
        match s {
            "active" => DownloadStatus::Active,
            "paused" => DownloadStatus::Paused,
            "completed" => DownloadStatus::Completed,
            "failed" => DownloadStatus::Failed,
            "cancelled" => DownloadStatus::Cancelled,
            "quarantined" => DownloadStatus::Quarantined,
            _ => DownloadStatus::Queued,
        }
    }

    fn map_download(row: &rusqlite::Row<'_>) -> rusqlite::Result<DownloadState> {
        Ok(DownloadState {
            id: row.get(0)?,
            model_id: row.get(1)?,
            quant_label: row.get(2)?,
            url: row.get(3)?,
            target_path: row.get(4)?,
            total_bytes: row.get::<_, i64>(5)? as u64,
            received_bytes: row.get::<_, i64>(6)? as u64,
            sha256_expected: row.get(7)?,
            status: Self::str_to_status(&row.get::<_, String>(8)?),
            started_at: parse_dt(row.get::<_, String>(9)?),
            updated_at: parse_dt(row.get::<_, String>(10)?),
            error: row.get(11)?,
            throughput_bps: row.get::<_, i64>(12)? as u64,
            eta_seconds: row.get::<_, Option<i64>>(13)?.map(|v| v as u64),
        })
    }

    pub fn insert_download(&self, d: &DownloadState) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO downloads
             (id, model_id, quant_label, url, target_path, total_bytes, received_bytes,
              sha256_expected, status, started_at, updated_at, error, throughput_bps, eta_seconds)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                d.id, d.model_id, d.quant_label, d.url, d.target_path,
                d.total_bytes as i64, d.received_bytes as i64,
                d.sha256_expected, Self::status_to_str(&d.status),
                d.started_at.to_rfc3339(), d.updated_at.to_rfc3339(),
                d.error, d.throughput_bps as i64,
                d.eta_seconds.map(|v| v as i64),
            ],
        )?;
        Ok(())
    }

    pub fn update_download_progress(
        &self, id: &str, received: u64, status: &DownloadStatus,
        throughput: u64, eta: Option<u64>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE downloads SET received_bytes = ?2, status = ?3, throughput_bps = ?4,
                eta_seconds = ?5, updated_at = ?6 WHERE id = ?1",
            params![
                id, received as i64, Self::status_to_str(status),
                throughput as i64, eta.map(|v| v as i64),
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn set_download_status(&self, id: &str, status: &DownloadStatus) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE downloads SET status = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, Self::status_to_str(status), Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn set_download_error(&self, id: &str, err: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE downloads SET status = 'failed', error = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, err, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn list_downloads(&self) -> Result<Vec<DownloadState>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, model_id, quant_label, url, target_path, total_bytes, received_bytes,
                    sha256_expected, status, started_at, updated_at, error, throughput_bps, eta_seconds
             FROM downloads ORDER BY started_at DESC",
        )?;
        let results: Vec<DownloadState> = stmt.query_map([], Self::map_download)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(results)
    }

    pub fn get_download(&self, id: &str) -> Result<Option<DownloadState>> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row(
            "SELECT id, model_id, quant_label, url, target_path, total_bytes, received_bytes,
                    sha256_expected, status, started_at, updated_at, error, throughput_bps, eta_seconds
             FROM downloads WHERE id = ?1",
            params![id],
            Self::map_download,
        ).optional()?)
    }

    pub fn remove_download(&self, id: &str) -> Result<bool> {
        let conn = self.conn.lock().unwrap();
        let n = conn.execute("DELETE FROM downloads WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }

    pub fn insert_net_log(&self, e: &crate::ipc::NetworkLogEntry) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO net_log (ts, method, url, purpose, bytes_out, bytes_in, status, note)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                e.ts.to_rfc3339(), e.method, e.url, e.purpose,
                e.bytes_out as i64, e.bytes_in as i64,
                e.status.map(|s| s as i32), e.note,
            ],
        )?;
        Ok(())
    }

    pub fn list_net_log(&self) -> Result<Vec<crate::ipc::NetworkLogEntry>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, ts, method, url, purpose, bytes_out, bytes_in, status, note
             FROM net_log ORDER BY id DESC LIMIT 500",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(crate::ipc::NetworkLogEntry {
                id: row.get(0)?,
                ts: parse_dt(row.get::<_, String>(1)?),
                method: row.get(2)?,
                url: row.get(3)?,
                purpose: row.get(4)?,
                bytes_out: row.get::<_, i64>(5)? as u64,
                bytes_in: row.get::<_, i64>(6)? as u64,
                status: row.get::<_, Option<i32>>(7)?.map(|s| s as u16),
                note: row.get(8)?,
            })
        })?;
        Ok(rows.collect::<std::result::Result<Vec<_>, _>>()?)
    }

    pub fn insert_benchmark(&self, r: &crate::ipc::BenchmarkResult) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO benchmark_runs
             (model_id, quant_label, context_length, prompt_tokens, gen_tokens,
              prompt_eval_tok_per_s, gen_tok_per_s, warm, duration_ms, run_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                r.model_id, r.quant_label,
                r.context_length as i64, r.prompt_tokens as i64, r.gen_tokens as i64,
                r.prompt_eval_tok_per_s, r.gen_tok_per_s,
                if r.warm { 1 } else { 0 },
                r.duration_ms as i64, r.run_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn latest_benchmark(&self, model_id: &str, quant_label: &str) -> Result<Option<crate::ipc::BenchmarkResult>> {
        let conn = self.conn.lock().unwrap();
        Ok(conn.query_row(
            "SELECT model_id, quant_label, context_length, prompt_tokens, gen_tokens,
                    prompt_eval_tok_per_s, gen_tok_per_s, warm, duration_ms, run_at
             FROM benchmark_runs WHERE model_id = ?1 AND quant_label = ?2
             ORDER BY id DESC LIMIT 1",
            params![model_id, quant_label],
            |row| {
                Ok(crate::ipc::BenchmarkResult {
                    model_id: row.get(0)?,
                    quant_label: row.get(1)?,
                    context_length: row.get::<_, i64>(2)? as u32,
                    prompt_tokens: row.get::<_, i64>(3)? as u32,
                    gen_tokens: row.get::<_, i64>(4)? as u32,
                    prompt_eval_tok_per_s: row.get(5)?,
                    gen_tok_per_s: row.get(6)?,
                    warm: row.get::<_, i64>(7)? != 0,
                    duration_ms: row.get::<_, i64>(8)? as u64,
                    run_at: parse_dt(row.get::<_, String>(9)?),
                })
            },
        ).optional()?)
    }
}

fn parse_dt(s: String) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(&s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}