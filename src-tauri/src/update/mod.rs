//! UPD — self-update.
//!
//! The only egress Kayon performs *about itself*, so it is held to the telemetry standard rather
//! than the convenience standard (OD-13):
//!
//! - the check is **off-switchable** (UPD-1) and does nothing at all when disabled;
//! - every check and download is written to the **network log** (PRIV-5) — an update path exempt
//!   from the log would quietly make that log a lie;
//! - a found version is **offered, never fetched** (UPD-2), the same rule models live under (FR-3);
//! - the artifact's **signature is the boundary** (UPD-3): this downloads and runs an installer, so
//!   the plugin's minisign check against the baked-in public key is the only thing between an
//!   update channel and arbitrary code execution. TLS is not a substitute (CAT-2).
//!
//! The updater needs a Tauri `AppHandle`, which exists only in the desktop app. Under the headless
//! `server` binary this module reports `supported: false` and does nothing, rather than pretending.

use std::sync::{Arc, Mutex, OnceLock};

use crate::db::Database;
use serde::Serialize;

/// Preference key for the automatic launch check (UPD-1). Absent means on.
pub const PREF_AUTO_CHECK: &str = "update.auto_check";

/// Where the app's `AppHandle` is parked for the API layer. Set once, in the Tauri `setup()`.
static APP: OnceLock<tauri::AppHandle> = OnceLock::new();

pub fn set_app_handle(handle: tauri::AppHandle) {
    let _ = APP.set(handle);
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateState {
    /// False under the headless server binary: there is no updater to drive.
    pub supported: bool,
    pub checking: bool,
    pub downloading: bool,
    /// Set once a newer version has been announced (UPD-2 — announced, not fetched).
    pub available: Option<String>,
    pub notes: Option<String>,
    pub current: String,
    /// Downloaded and verified; the UI offers "Relaunch to update" from here.
    pub ready: bool,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    /// Surfaced rather than swallowed (UPD-5).
    pub error: Option<String>,
    /// False when the user switched automatic checking off (UPD-1).
    pub auto_check: bool,
}

static STATE: OnceLock<Arc<Mutex<UpdateState>>> = OnceLock::new();

pub fn state() -> Arc<Mutex<UpdateState>> {
    STATE.get_or_init(|| Arc::new(Mutex::new(UpdateState::default()))).clone()
}

/// The running app's version — the one the updater compares against, from `tauri.conf.json`.
///
/// Deliberately not `CARGO_PKG_VERSION`: the crate is still 0.1.0 while the app is 1.4.x, so that
/// constant would have shown the user a version Kayon has never shipped.
fn current_version() -> String {
    APP.get()
        .map(|a| a.package_info().version.to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// A snapshot for the API, with the live values that aren't cached in the struct.
pub fn snapshot(db: &Arc<Database>) -> UpdateState {
    let mut s = state().lock().unwrap().clone();
    s.supported = APP.get().is_some();
    s.auto_check = auto_check_enabled(db);
    s.current = current_version();
    s
}

pub fn auto_check_enabled(db: &Arc<Database>) -> bool {
    db.get_preference(PREF_AUTO_CHECK).as_deref() != Some("false")
}

pub fn set_auto_check(db: &Arc<Database>, on: bool) {
    let _ = db.set_preference(PREF_AUTO_CHECK, if on { "true" } else { "false" });
}

/// The manifest URL, for the network log. Read from the same config the plugin uses so the log
/// can never name a different host than the one actually contacted.
fn endpoint() -> String {
    APP.get()
        .and_then(|app| {
            let cfg = app.config();
            cfg.plugins
                .0
                .get("updater")
                .and_then(|v| v.get("endpoints"))
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .unwrap_or_else(|| "(no updater endpoint configured)".to_string())
}

/// UPD-1: ask whether a newer version exists. Returns the version when one is announced.
///
/// `manual` distinguishes a Settings click from the launch check: the off switch governs the
/// automatic one only, since asking for a check *is* consent to make the request.
pub async fn check(db: &Arc<Database>, manual: bool) -> Result<Option<String>, String> {
    if !manual && !auto_check_enabled(db) {
        return Ok(None); // switched off: make no request at all (UPD-1)
    }
    let Some(app) = APP.get() else {
        return Err("updates are not available in this build".into());
    };
    {
        let st = state();
        let mut s = st.lock().unwrap();
        if s.checking {
            return Ok(s.available.clone());
        }
        s.checking = true;
        s.error = None;
    }

    use tauri_plugin_updater::UpdaterExt;
    let url = endpoint();
    let result = async {
        let updater = app.updater().map_err(|e| e.to_string())?;
        updater.check().await.map_err(|e| e.to_string())
    }
    .await;

    let st = state();
    let mut s = st.lock().unwrap();
    s.checking = false;
    match result {
        Ok(Some(update)) => {
            let v = update.version.clone();
            // PRIV-5: the check is real egress and is logged like any other request.
            crate::telemetry::log_network_request_full(
                db, "GET", &url, "update check", 0, 0, Some(200),
                Some(format!("update available: {v}")),
            );
            // Hold the announced update itself. Everything downstream acts on *this* object, so the
            // thing the user was offered is the thing they download and install — and no later step
            // re-contacts the network behind their back.
            if s.available.as_deref() != Some(v.as_str()) {
                // A different version than any bytes we may be holding: those are now stale.
                PENDING.lock().unwrap().take();
                s.ready = false;
            }
            ANNOUNCED.lock().unwrap().replace(update.clone());
            s.available = Some(v.clone());
            s.notes = update.body.clone();
            Ok(Some(v))
        }
        Ok(None) => {
            crate::telemetry::log_network_request_full(
                db, "GET", &url, "update check", 0, 0, Some(200),
                Some("already up to date".into()),
            );
            // Nothing on offer any more — which also covers an update being *withdrawn* after we
            // announced or downloaded it. Drop the offer and the bytes together, or install could
            // still run a version that has since been pulled.
            ANNOUNCED.lock().unwrap().take();
            PENDING.lock().unwrap().take();
            s.available = None;
            s.notes = None;
            s.ready = false;
            Ok(None)
        }
        Err(e) => {
            // UPD-5: a failed check is surfaced and logged, never a startup blocker.
            crate::telemetry::log_network_request_full(
                db, "GET", &url, "update check", 0, 0, None, Some(format!("failed: {e}")),
            );
            s.error = Some(e.clone());
            Err(e)
        }
    }
}

/// UPD-2/UPD-3: download the announced update, on the user's say-so. The plugin verifies the
/// artifact's signature against the baked-in public key and refuses anything that fails.
pub async fn download(db: &Arc<Database>) -> Result<(), String> {
    if APP.get().is_none() {
        return Err("updates are not available in this build".into());
    }
    // Act on the announced update, never a fresh check: re-checking here would be a second,
    // unlogged request, and could silently swap the version the user agreed to for another one.
    let update = ANNOUNCED.lock().unwrap().clone().ok_or("no update has been announced")?;
    {
        let st = state();
        let mut s = st.lock().unwrap();
        if s.downloading {
            return Ok(());
        }
        s.downloading = true;
        s.ready = false;
        s.error = None;
        s.downloaded_bytes = 0;
        s.total_bytes = None;
    }

    // PRIV-5 names the URL the bytes actually come from — the artifact, not the manifest. Logging
    // the manifest URL here would put a host in the log that this request never contacted.
    let url = update.download_url.to_string();
    let version = update.version.clone();
    let st = state();
    let out = update
        .download(
            |chunk, total| {
                let mut s = st.lock().unwrap();
                s.downloaded_bytes += chunk as u64;
                s.total_bytes = total;
            },
            || {},
        )
        .await
        .map_err(|e| e.to_string());

    let mut s = st.lock().unwrap();
    s.downloading = false;
    match out {
        Ok(bytes) => {
            crate::telemetry::log_network_request_full(
                db, "GET", &url, "update download", 0, bytes.len() as u64, Some(200),
                Some(format!("downloaded and signature-verified: {version}")),
            );
            // Keep the version alongside the bytes so install can prove they belong together.
            PENDING.lock().unwrap().replace((version, bytes));
            s.ready = true;
            Ok(())
        }
        Err(e) => {
            crate::telemetry::log_network_request_full(
                db, "GET", &url, "update download", 0, s.downloaded_bytes, None,
                Some(format!("failed: {e}")),
            );
            s.error = Some(e.clone());
            Err(e)
        }
    }
}

/// The update the user was offered. Download and install both act on this exact object.
static ANNOUNCED: Mutex<Option<tauri_plugin_updater::Update>> = Mutex::new(None);
/// The verified installer bytes and the version they are, held until the user chooses to relaunch.
static PENDING: Mutex<Option<(String, Vec<u8>)>> = Mutex::new(None);

/// UPD-2: apply the downloaded update and restart. Only ever reached by an explicit click.
///
/// Makes no network request: the bytes are already here and already signature-verified. A check at
/// this point would egress unlogged, ignore the off switch, and risk installing something other
/// than what was downloaded.
pub fn install_and_relaunch() -> Result<(), String> {
    let Some(app) = APP.get() else {
        return Err("updates are not available in this build".into());
    };
    let update = ANNOUNCED.lock().unwrap().clone().ok_or("no update has been announced")?;
    let (version, bytes) = PENDING.lock().unwrap().take().ok_or("no update has been downloaded")?;
    // The bytes must be the ones for the update still on offer. If a later check moved the target,
    // refuse rather than install a version the user is no longer looking at.
    if version != update.version {
        return Err(format!(
            "downloaded {version} but {} is now the announced update; download it again",
            update.version
        ));
    }
    update.install(bytes).map_err(|e| e.to_string())?;
    app.restart();
}
