pub mod db;
pub mod probe;
pub mod gguf;
pub mod fit;
pub mod catalog;
pub mod discovery;
pub mod download;
pub mod library;
pub mod ollama;
pub mod runtime;
pub mod telemetry;
pub mod ipc;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json, sse::{Event, Sse}},
    routing::{get, post, delete},
    Router,
};
use futures_util::stream::Stream;
use std::convert::Infallible;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::IntervalStream;
use std::time::Duration;

use ipc::*;

#[derive(Clone)]
struct AppState {
    db: Arc<db::Database>,
    dl: Arc<download::DownloadManager>,
    rt: Arc<runtime::RuntimeManager>,
}

/// Start the local API server (the same Axum app the browser build serves) on a background
/// thread with its own Tokio runtime, so the Tauri event loop can own the main thread.
pub fn start_api_server() {
    std::thread::spawn(|| {
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async {

    let db = Arc::new(db::Database::open().expect("failed to open database"));
    let _ = library::init_library_dir();
    let dl = Arc::new(download::DownloadManager::new());
    let rt = Arc::new(runtime::RuntimeManager::new());

    let state = AppState { db, dl, rt };

    // DL-1: resume any downloads that were mid-flight when the app last exited. Their partial
    // files and rows are persisted, so we re-drive them (Range-resumed) on startup.
    {
        let db = state.db.clone();
        let dl = state.dl.clone();
        tokio::spawn(async move {
            if let Ok(downloads) = db.list_downloads() {
                for d in downloads.into_iter().filter(|d| {
                    matches!(d.status, DownloadStatus::Active | DownloadStatus::Queued)
                }) {
                    let (db, dl, id) = (db.clone(), dl.clone(), d.id.clone());
                    tokio::spawn(async move {
                        dl.drive(&db, &id).await;
                    });
                }
            }
        });
    }

    // CAT-7: discover the model catalog live from Hugging Face on launch, so the list stays current
    // without any user action or GitHub round-trip. This runs in the background — the bundled signed
    // catalog renders instantly and is transparently replaced when discovery lands. It is an
    // explicitly allowed network call (PRIV-1: "catalog updates"), every HF request is logged at
    // egress (PRIV-5), and it is independently controllable (`catalog_auto_refresh=off`). It never
    // downloads model weights — only small JSON + each new model's GGUF header. Failure is non-fatal.
    {
        let db = state.db.clone();
        tokio::spawn(async move {
            if db.get_preference("catalog_auto_refresh").as_deref() == Some("off") {
                log::info!("catalog discovery disabled by preference; using bundled/cached catalog");
                return;
            }
            match run_discovery(&db).await {
                Ok(n) => log::info!("catalog discovered from Hugging Face on launch: {} models", n),
                Err(e) => log::warn!("catalog discovery on launch failed (keeping current): {}", e),
            }
        });
    }

    let app = Router::new()
        .route("/api/hardware", get(hardware))
        .route("/api/hardware/stream", get(hardware_stream))
        .route("/api/catalog", get(get_catalog))
        .route("/api/catalog/refresh", post(refresh_catalog))
        .route("/api/catalog/status", get(catalog_status))
        .route("/api/fit/verdicts", get(all_verdicts))
        .route("/api/fit/verdict/{model_id}/{quant_label}", get(verdict))
        .route("/api/library", get(library_list))
        .route("/api/library/dir", get(library_dir_info))
        .route("/api/library/relocate", post(library_relocate))
        .route("/api/fit/local/{id}", get(local_verdict))
        .route("/api/library/delete/{id}", post(delete_model))
        .route("/api/downloads", get(list_downloads))
        .route("/api/downloads/start", post(start_download))
        .route("/api/downloads/{id}/cancel", delete(cancel_download))
        .route("/api/ollama/models", get(ollama_models))
        .route("/api/ollama/adopt", post(ollama_adopt))
        .route("/api/runtime/start", post(runtime_start))
        .route("/api/runtime/load/{id}", post(runtime_load))
        .route("/api/runtime/stop", post(runtime_stop))
        .route("/api/runtime/status", get(runtime_status))
        .route("/api/runtime/benchmark", post(benchmark))
        .route("/api/privacy/network-log", get(network_log))
        .route("/api/privacy/telemetry/status", get(telemetry_status))
        .route("/api/privacy/telemetry/toggle", post(telemetry_toggle))
        .route("/api/privacy/telemetry/preview", get(telemetry_preview))
        .route("/api/prefs/{key}", get(get_pref).put(set_pref))
        .route("/api/chat/sessions", get(list_chat_sessions).post(create_chat_session))
        .route("/api/chat/sessions/{id}", get(get_chat_session).delete(delete_chat_session))
        .route("/api/chat/sessions/{id}/rename", post(rename_chat_session))
        .route("/api/chat/sessions/{id}/settings", post(update_chat_settings))
        .route("/api/chat/sessions/{id}/messages", post(append_chat_message))
        .fallback(static_handler)
        // Defence in depth against CSRF to the loopback control API: CORS blocks reading
        // responses and preflighted requests, but a malicious page can still fire "simple"
        // cross-site POSTs (no custom headers). This middleware rejects any mutating request
        // whose Origin isn't a Kayon origin or whose Sec-Fetch-Site marks it cross-site.
        .layer(axum::middleware::from_fn(csrf_guard))
        // Tight CORS: the UI is served same-origin from this port, so only the Kayon origins are
        // allowed. This stops arbitrary websites the user visits from issuing preflighted
        // POST/DELETE calls to the unauthenticated local-control API (delete, download, adopt,
        // launch). Same-origin requests from the served UI are unaffected.
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin([
                    "http://127.0.0.1:9518".parse().unwrap(),
                    "http://localhost:9518".parse().unwrap(),
                    "http://127.0.0.1:3000".parse().unwrap(),
                    "http://localhost:3000".parse().unwrap(),
                    "http://tauri.localhost".parse().unwrap(),
                    "https://tauri.localhost".parse().unwrap(),
                ])
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any),
        )
        .with_state(state);

    // Bind loopback only: this is a private, local-control API with no auth (PRIV-2). It must
    // never be reachable from the LAN — every endpoint (delete, download, adopt, launch) is
    // local-user-only by design.
    let addr = SocketAddr::from(([127, 0, 0, 1], 9518));
            // Non-fatal bind: if the port is already taken (e.g. a stray second launch that raced
            // the single-instance guard), log and exit this thread quietly instead of panicking.
            let listener = match tokio::net::TcpListener::bind(addr).await {
                Ok(l) => l,
                Err(e) => {
                    log::warn!("Kayon API server could not bind {} ({}) — another instance is running", addr, e);
                    return;
                }
            };
            log::info!("Kayon server listening on http://{}", addr);
            if let Err(e) = axum::serve(listener, app).await {
                log::error!("Kayon API server stopped: {}", e);
            }
        });
    });
}

/// Desktop entry point: run the Tauri window (WebView2) that loads the bundled UI, backed by the
/// local API server on 127.0.0.1:9518. A single-instance guard ensures a second launch focuses the
/// existing window instead of starting a second server (which would fail to bind the port).
pub fn run() {
    use tauri::Manager;
    let _ = env_logger::try_init();
    tauri::Builder::default()
        // Must be the FIRST plugin: when a second copy is launched, this fires in the ALREADY-
        // running instance and the second process exits. Focus/restore the existing window.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
        }))
        .setup(|_app| {
            // Only the primary instance reaches setup(), so only it starts the API server — no
            // port contention. The UI polls at 1 Hz and shows data as soon as the server is up.
            start_api_server();
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Kayon");
}

const ALLOWED_ORIGINS: &[&str] = &[
    "http://127.0.0.1:9518", "http://localhost:9518",
    "http://127.0.0.1:3000", "http://localhost:3000",
    // Tauri WebView2 window origin(s) — the desktop app calls the local API from here.
    "http://tauri.localhost", "https://tauri.localhost",
];

/// Reject cross-site mutating requests to the loopback control API (CSRF defence). Safe methods
/// (GET/HEAD/OPTIONS) pass. For mutating methods we reject when `Sec-Fetch-Site` is cross-site, or
/// when an `Origin`/`Referer` is present that is not a Kayon origin. Non-browser clients (curl,
/// the app's own IPC) send no Origin and are allowed.
async fn csrf_guard(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = req.method();
    let mutating = !matches!(method, &axum::http::Method::GET | &axum::http::Method::HEAD | &axum::http::Method::OPTIONS);
    if mutating {
        let headers = req.headers();
        if let Some(site) = headers.get("sec-fetch-site").and_then(|v| v.to_str().ok()) {
            if site == "cross-site" || site == "same-site" {
                return (StatusCode::FORBIDDEN, "cross-site request rejected").into_response();
            }
        }
        let origin_ok = |val: Option<&str>| match val {
            None => true, // non-browser client (no Origin) — allowed
            Some(o) => ALLOWED_ORIGINS.iter().any(|a| o == *a),
        };
        let origin = headers.get("origin").and_then(|v| v.to_str().ok());
        if !origin_ok(origin) {
            return (StatusCode::FORBIDDEN, "disallowed origin").into_response();
        }
    }
    next.run(req).await
}

fn ok_json<T: serde::Serialize>(data: T) -> Json<ApiResponse<T>> {
    Json(ApiResponse::ok(data))
}

fn err_json(msg: &str) -> (StatusCode, Json<ApiResponse<()>>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(msg)))
}

async fn hardware(State(_s): State<AppState>) -> impl IntoResponse {
    match probe::probe_machine() {
        Ok(machine) => ok_json(machine).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn hardware_stream(State(_s): State<AppState>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = IntervalStream::new(tokio::time::interval(Duration::from_secs(1)))
        .map(|_| {
            let tel = probe::poll_gpu_telemetry(0);
            let data = serde_json::to_string(&tel).unwrap_or_default();
            Ok(Event::default().data(data))
        });
    Sse::new(stream)
}

async fn get_catalog(State(_s): State<AppState>) -> impl IntoResponse {
    match catalog::get_active_catalog() {
        Ok(c) => ok_json(c).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

/// Whether a catalog discovery pass is currently running, so the UI can show a "finding the best
/// models for your GPU" indicator (CAT-7 discovery is a background process on launch + on refresh).
static DISCOVERING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Sets DISCOVERING for the lifetime of a discovery pass and clears it on drop (even on error/early
/// return), so the flag can never get stuck on.
struct DiscoveryGuard;
impl DiscoveryGuard {
    fn new() -> Self { DISCOVERING.store(true, std::sync::atomic::Ordering::SeqCst); Self }
}
impl Drop for DiscoveryGuard {
    fn drop(&mut self) { DISCOVERING.store(false, std::sync::atomic::Ordering::SeqCst); }
}

async fn catalog_status() -> impl IntoResponse {
    let cat = catalog::get_active_catalog().ok();
    ok_json(serde_json::json!({
        "discovering": DISCOVERING.load(std::sync::atomic::Ordering::SeqCst),
        "source": cat.as_ref().map(|c| c.source.clone()),
        "revision": cat.as_ref().map(|c| c.revision),
    })).into_response()
}

/// Run one live discovery pass against Hugging Face and cache the result (CAT-7). The currently
/// active catalog seeds the arch cache so already-known models don't re-fetch their headers. Every
/// HF request is logged at egress (PRIV-5). Returns the number of models discovered.
async fn run_discovery(db: &Arc<db::Database>) -> anyhow::Result<usize> {
    let _guard = DiscoveryGuard::new();
    let seed = catalog::get_active_catalog().unwrap_or_else(|_| catalog::empty_catalog());
    // Monotonic revision so a fresh discovery supersedes the cache/bundled seed (get_active_catalog
    // guards on revision). Unix seconds is monotonic and dwarfs the small bundled revisions.
    let revision = chrono::Utc::now().timestamp().max(0) as u64;
    let authors: Vec<String> = discovery::TRUSTED_AUTHORS.iter().map(|s| s.to_string()).collect();
    let discovered = discovery::discover_catalog(Some(db), &authors, 20, revision, &seed).await?;
    let n = discovered.entries.len();
    catalog::save_discovered_catalog(&discovered)?;
    Ok(n)
}

async fn refresh_catalog(State(s): State<AppState>) -> impl IntoResponse {
    // Re-discover from Hugging Face on demand, then return the now-active catalog (CAT-7).
    match run_discovery(&s.db).await {
        Ok(_) => match catalog::get_active_catalog() {
            Ok(c) => ok_json(c).into_response(),
            Err(e) => err_json(&e.to_string()).into_response(),
        },
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct VerdictQuery {
    /// Target context length. Falls back to each model's native context when absent (FIT-4).
    ctx: Option<u32>,
    /// KV cache bytes-per-element: 2 = f16 (default, OD-1), 1 = q8_0 knob.
    kv_type_bytes: Option<u8>,
}

async fn all_verdicts(
    State(_s): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<VerdictQuery>,
) -> impl IntoResponse {
    let catalog = match catalog::get_active_catalog() {
        Ok(c) => c,
        Err(e) => return err_json(&e.to_string()).into_response(),
    };
    let kv = q.kv_type_bytes.unwrap_or(2).clamp(1, 2);
    let mut verdicts = Vec::new();
    for entry in &catalog.entries {
        let ctx = q.ctx.unwrap_or(entry.arch.context_length);
        for quant in &entry.quants {
            let v = fit::evaluate_remote(
                &entry.id, &quant.label, quant.bytes,
                &entry.arch, ctx, kv,
            );
            verdicts.push(v);
        }
    }
    ok_json(verdicts).into_response()
}

async fn verdict(
    State(_s): State<AppState>,
    Path((model_id, quant_label)): Path<(String, String)>,
    axum::extract::Query(q): axum::extract::Query<VerdictQuery>,
) -> impl IntoResponse {
    let catalog = match catalog::get_active_catalog() {
        Ok(c) => c,
        Err(e) => return err_json(&e.to_string()).into_response(),
    };
    let kv = q.kv_type_bytes.unwrap_or(2).clamp(1, 2);
    for entry in &catalog.entries {
        if entry.id == model_id {
            let ctx = q.ctx.unwrap_or(entry.arch.context_length);
            for quant in &entry.quants {
                if quant.label == quant_label {
                    let v = fit::evaluate_remote(
                        &entry.id, &quant.label, quant.bytes,
                        &entry.arch, ctx, kv,
                    );
                    return ok_json(v).into_response();
                }
            }
        }
    }
    err_json("model/quant not found").into_response()
}

async fn local_verdict(
    State(s): State<AppState>,
    Path(id): Path<String>,
    axum::extract::Query(q): axum::extract::Query<VerdictQuery>,
) -> impl IntoResponse {
    let model = match s.db.get_installed_model(&id) {
        Ok(Some(m)) => m,
        Ok(None) => return err_json("installed model not found").into_response(),
        Err(e) => return err_json(&e.to_string()).into_response(),
    };
    let kv = q.kv_type_bytes.unwrap_or(2).clamp(1, 2);
    let ctx = q.ctx.unwrap_or(4096);
    // FIT-1: exact local verdict from the GGUF on disk supersedes the remote approximation.
    match fit::evaluate_local(&model.model_id, &model.quant_label, &model.path, ctx, kv) {
        Ok(v) => ok_json(v).into_response(),
        Err(e) => err_json(&format!("could not read GGUF for local verdict: {}", e)).into_response(),
    }
}

async fn library_dir_info(State(_s): State<AppState>) -> impl IntoResponse {
    ok_json(library::library_dir().to_string_lossy().to_string()).into_response()
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct RelocateReq { path: String }

async fn library_relocate(
    State(s): State<AppState>,
    Json(req): Json<RelocateReq>,
) -> impl IntoResponse {
    // LIB-1 move-in-place migration; on the blocking pool since it moves/copies model files.
    let db = s.db.clone();
    let path = req.path.clone();
    let res = tokio::task::spawn_blocking(move || library::relocate_library(&db, &path))
        .await
        .unwrap_or_else(|e| Err(anyhow::anyhow!("relocate task panicked: {}", e)));
    match res {
        Ok(moved) => ok_json(serde_json::json!({
            "movedFiles": moved,
            "libraryDir": library::library_dir().to_string_lossy(),
        })).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn library_list(State(s): State<AppState>) -> impl IntoResponse {
    match library::list_installed(&s.db) {
        Ok(mut models) => {
            // Recompute the OLL-6 runtime gate for the listing so unsupported-arch models stay
            // marked "needs newer runtime" and the UI won't present a Load & Chat that fails.
            let supported = runtime::RuntimeManager::supported_runtime_archs();
            for m in &mut models {
                m.needs_newer_runtime = match m.architecture.as_deref() {
                    Some(a) => !supported.iter().any(|s| s.eq_ignore_ascii_case(a)),
                    None => false,
                };
            }
            ok_json(models).into_response()
        }
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn delete_model(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match library::delete_model(&s.db, &id, true) {
        Ok(_) => ok_json(true).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn list_downloads(State(s): State<AppState>) -> impl IntoResponse {
    match s.db.list_downloads() {
        Ok(d) => ok_json(d).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn start_download(
    State(s): State<AppState>,
    Json(req): Json<DownloadStartReq>,
) -> impl IntoResponse {
    // Trust rides on the signed catalog, not the caller (CAT-2, OD-2). Resolve the download
    // origin, exact size, and pinned SHA-256 from the verified catalog by model_id + quant —
    // never from client-supplied values. No arbitrary-URL downloads in v1.
    let catalog = match catalog::get_active_catalog() {
        Ok(c) => c,
        Err(e) => return err_json(&e.to_string()).into_response(),
    };
    let quant = catalog.entries.iter()
        .find(|e| e.id == req.model_id)
        .and_then(|e| e.quants.iter().find(|q| q.label == req.quant_label));
    let quant = match quant {
        Some(q) => q,
        None => return err_json("no such model/quant in the verified catalog").into_response(),
    };

    let target = library::deterministic_path(&req.model_id, &req.quant_label);
    match s.dl.start_download(
        &s.db, &req.model_id, &req.quant_label,
        &quant.source, &target, quant.bytes, &quant.sha256,
    ).await {
        Ok((state, is_new)) => {
            if is_new {
                let id = state.id.clone();
                let db = s.db.clone();
                let dl = s.dl.clone();
                tokio::spawn(async move {
                    dl.drive(&db, &id).await;
                });
            }
            ok_json(state).into_response()
        }
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn cancel_download(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match s.dl.cancel_download(&s.db, &id).await {
        Ok(_) => ok_json(true).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn ollama_models(State(_s): State<AppState>) -> impl IntoResponse {
    let lib_path = library::deterministic_path("probe", "probe");
    match ollama::discover_ollama_models(&lib_path) {
        Ok(models) => ok_json(models).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn ollama_adopt(
    State(s): State<AppState>,
    Json(req): Json<OllamaAdoptReq>,
) -> impl IntoResponse {
    let lib_dir = library::library_dir();
    let _ = std::fs::create_dir_all(&lib_dir);

    // Re-resolve the adoption target from the discovered Ollama manifests by (name, tag). The
    // client only names which model to adopt; the trusted blob path + digest come from the
    // server's manifest scan, never from the request body (defence against arbitrary-path linking).
    let discovered = match ollama::discover_ollama_models(&library::deterministic_path("probe", "probe")) {
        Ok(models) => models.into_iter().find(|m| m.name == req.name && m.tag == req.tag),
        Err(e) => return err_json(&e.to_string()).into_response(),
    };
    let m = match discovered {
        Some(m) => m,
        None => return err_json("no matching Ollama model found for that name:tag").into_response(),
    };
    let mode = if m.same_volume_as_library {
        ollama::AdoptMode::Link
    } else {
        // OLL-4: never silently copy across volumes. Surface both choices unless the user has
        // explicitly opted into copy this time.
        match req.mode.as_deref() {
            Some("copy") => ollama::AdoptMode::Copy,
            _ => return err_json(&format!(
                "cross-volume: the Ollama store for {}:{} is on a different drive than the library. \
                 Choose to copy the blob into the library (~{} bytes, disk pre-flight applies) by \
                 re-sending with mode=\"copy\", or relocate the library onto the Ollama volume (OLL-4).",
                m.name, m.tag, m.size_bytes
            )).into_response(),
        }
    };

    // Adoption hashes the blob (OLL-3 gate) which is CPU/IO-bound for multi-GB models; run it on
    // the blocking pool so it never stalls the async runtime handling other requests.
    let (blob_path, lib_str, name, tag, digest, size) = (
        m.blob_path.clone(), lib_dir.to_string_lossy().to_string(),
        m.name.clone(), m.tag.clone(), m.digest.clone(), m.size_bytes,
    );
    let adopt_result = tokio::task::spawn_blocking(move || {
        ollama::adopt_model(&blob_path, &lib_str, &name, &tag, &digest, size, mode)
    }).await.unwrap_or_else(|e| Err(anyhow::anyhow!("adoption task panicked: {}", e)));
    match adopt_result {
        Ok((path, _copied)) => {
            let model = InstalledModel {
                id: uuid::Uuid::new_v4().to_string(),
                model_id: format!("{}:{}", m.name, m.tag),
                quant_label: m.quantization.clone().unwrap_or("unknown".into()),
                path: path.clone(),
                bytes: m.size_bytes,
                sha256: m.digest.clone(),
                source: ModelSource::Adopted,
                installed_at: chrono::Utc::now(),
                ollama_tag: Some(format!("{}:{}", m.name, m.tag)),
                ollama_digest: Some(m.digest.clone()),
                architecture: m.architecture.clone(),
                needs_newer_runtime: m.needs_newer_runtime,
            };
            if let Err(e) = s.db.insert_installed_model(&model) {
                return err_json(&e.to_string()).into_response();
            }
            ok_json(model).into_response()
        }
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn runtime_start(
    State(s): State<AppState>,
    Json(req): Json<RuntimeStartReq>,
) -> impl IntoResponse {
    let port = runtime::RuntimeManager::find_available_port();
    match s.rt.start(
        &req.model_path, &req.model_id, &req.quant_label,
        req.n_gpu_layers, req.context_length, port,
        req.kv_type_bytes.unwrap_or(2), &req.runtime_args,
    ) {
        Ok(_) => {
            spawn_health_wait(s.rt.clone(), port);
            ok_json(s.rt.status()).into_response()
        }
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

/// Poll llama-server `/health` until ready (RUN-1), then keep watching liveness. If the sidecar
/// dies after startup (OOM on first prompt, killed, crash), flip the status to Error with the log
/// tail instead of reporting Running on a dead port. Port-guarded so a switch/stop ends the watch.
fn spawn_health_wait(rt: Arc<runtime::RuntimeManager>, port: u16) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let url = format!("http://127.0.0.1:{}/health", port);
        let mut ready = false;
        for _ in 0..20 {
            if let Ok(resp) = reqwest::get(&url).await {
                if resp.status().is_success() {
                    rt.mark_running(port);
                    ready = true;
                    break;
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        if !ready {
            let tail = rt.log_tail(8);
            rt.mark_error(port, &format!("health check timeout. llama-server log tail:\n{}", tail));
            return;
        }
        // Liveness watch: this is still the active port, and the sidecar stops responding.
        let mut misses = 0;
        loop {
            tokio::time::sleep(Duration::from_secs(3)).await;
            if rt.status().port != Some(port) {
                return; // switched or stopped — nothing to watch
            }
            let alive = reqwest::get(&url).await.map(|r| r.status().is_success()).unwrap_or(false);
            misses = if alive { 0 } else { misses + 1 };
            if misses >= 2 {
                let tail = rt.log_tail(8);
                rt.mark_error(port, &format!("llama-server exited unexpectedly. log tail:\n{}", tail));
                return;
            }
        }
    });
}

/// Load an installed model with the settings the fit engine computed (§7: "fit must match the
/// runtime it predicts"). n_gpu_layers comes from the local verdict; runtimeArgs come from the
/// model's catalog entry (e.g. `--jinja`, RUN-1). This is the one-press LIB-4 path.
async fn runtime_load(
    State(s): State<AppState>,
    Path(id): Path<String>,
    axum::extract::Query(q): axum::extract::Query<VerdictQuery>,
) -> impl IntoResponse {
    let model = match s.db.get_installed_model(&id) {
        Ok(Some(m)) => m,
        Ok(None) => return err_json("installed model not found").into_response(),
        Err(e) => return err_json(&e.to_string()).into_response(),
    };
    let ctx = q.ctx.unwrap_or(4096);
    let kv = q.kv_type_bytes.unwrap_or(2).clamp(1, 2);

    // OLL-6 runtime gate, recomputed on load: refuse to launch a model whose GGUF architecture the
    // bundled llama-server can't load, rather than starting a server that will fail. This catches
    // adopted models flagged "needs newer runtime" at discovery.
    if let Ok(h) = crate::gguf::parse_gguf_header(std::path::Path::new(&model.path)) {
        if let Some(arch) = crate::gguf::arch_from_header(&h) {
            let loadable = runtime::RuntimeManager::supported_runtime_archs()
                .iter().any(|a| a.eq_ignore_ascii_case(&arch));
            if !loadable {
                return err_json(&format!(
                    "architecture '{}' needs a newer llama.cpp runtime than the bundled one (OLL-6) — not launching",
                    arch
                )).into_response();
            }
        }
    }

    // Computed offload from the exact local verdict (single-sourced with the runtime).
    let verdict = fit::evaluate_local(&model.model_id, &model.quant_label, &model.path, ctx, kv);
    // Honest refusal: don't start llama-server for a model the fit engine says can't fit at all.
    if let Ok(v) = &verdict {
        if v.verdict == VerdictKind::ExceedsMachine {
            return err_json(&format!(
                "'{}' exceeds this machine's VRAM + RAM at ctx {} — not launching. {}",
                model.model_id, ctx, v.explainability
            )).into_response();
        }
    }
    let n_gpu_layers = verdict.as_ref().map(|v| v.n_gpu_layers).unwrap_or(999);

    // Catalog-derived launch settings (RUN-1): runtimeArgs, plus the runtimeMinVersion gate — a
    // signed entry can require a newer llama.cpp than the bundled one even when the arch is known.
    let cat_entry = catalog::get_active_catalog().ok()
        .and_then(|c| c.entries.into_iter().find(|e| e.id == model.model_id));
    if let Some(min) = cat_entry.as_ref().and_then(|e| e.arch.runtime_min_version.clone()) {
        match runtime::RuntimeManager::bundled_runtime_version() {
            Some(bundled) if runtime_version_satisfies(&bundled, &min) => {}
            Some(bundled) => return err_json(&format!(
                "'{}' needs llama.cpp runtime >= {} but the bundled runtime is {} — needs newer runtime (RUN-1)",
                model.model_id, min, bundled
            )).into_response(),
            None => return err_json(&format!(
                "'{}' declares runtimeMinVersion {} but the bundled runtime version is unknown — refusing to launch (set KAYON_RUNTIME_VERSION)",
                model.model_id, min
            )).into_response(),
        }
    }
    let runtime_args = cat_entry.as_ref()
        .and_then(|e| e.quants.iter().find(|qu| qu.label == model.quant_label).map(|qu| qu.runtime_args.clone()))
        .unwrap_or_default();

    let port = runtime::RuntimeManager::find_available_port();
    match s.rt.start(&model.path, &model.model_id, &model.quant_label, n_gpu_layers, ctx, port, kv, &runtime_args) {
        Ok(_) => {
            spawn_health_wait(s.rt.clone(), port);
            ok_json(s.rt.status()).into_response()
        }
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn runtime_stop(State(s): State<AppState>) -> impl IntoResponse {
    match s.rt.stop() {
        Ok(_) => ok_json(s.rt.status()).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn runtime_status(State(s): State<AppState>) -> impl IntoResponse {
    ok_json(s.rt.status()).into_response()
}

async fn benchmark(
    State(s): State<AppState>,
    Json(req): Json<BenchmarkReq>,
) -> impl IntoResponse {
    let status = s.rt.status();
    if status.kind != RuntimeStatusKind::Running {
        return err_json("no model is running — load a model first (HW-5 benchmark needs a live runtime)").into_response();
    }
    let port = match status.port {
        Some(p) => p,
        None => return err_json("runtime has no port").into_response(),
    };

    // HW-5 / OD-7: fixed prompt + fixed generation length, declared context, warm run only.
    // A neutral *continuation* seed (not an instruction) is used on purpose: an instruction prompt
    // makes an instruct model greedily emit an end-of-generation token immediately (and `ignore_eos`
    // only skips the EOS token, not every EOG token), which would collapse the generation to ~1
    // token. This seed reliably generates the full N_PREDICT tokens across models, so the throughput
    // number is real. It also carries enough tokens for a stable prompt-eval measurement.
    const PROMPT: &str = "Once upon a time in a small village near the mountains there lived";
    const N_PREDICT: u32 = 128;
    let url = format!("http://127.0.0.1:{}/completion", port);
    let client = reqwest::Client::new();

    let run = |warm_up: bool| {
        let client = client.clone();
        let url = url.clone();
        async move {
            let body = serde_json::json!({
                "prompt": PROMPT,
                "n_predict": if warm_up { 8 } else { N_PREDICT },
                "temperature": 0.0,
                "cache_prompt": false,
                // A throughput benchmark must measure a full generation. Without this, a model that
                // emits EOS after ~1 token (predicted_ms ≈ 0) yields a nonsensical rate.
                "ignore_eos": true,
            });
            client.post(&url).json(&body).send().await.ok()?.json::<serde_json::Value>().await.ok()
        }
    };

    // Discard the cold run, then measure the warm run.
    let _ = run(true).await;
    let started = std::time::Instant::now();
    let resp = match run(false).await {
        Some(v) => v,
        None => return err_json("benchmark request to llama-server failed").into_response(),
    };
    let duration_ms = started.elapsed().as_millis() as u64;

    // Compute rates ourselves from token counts + milliseconds rather than trusting llama-server's
    // `*_per_second`, which reports a nonsensical ~1e6 when a phase takes ~0ms (e.g. a 1-token EOS).
    // The `ignore_eos` above forces a full run so this is real; the ms floor keeps us honest if a
    // phase still comes back near-zero — we report 0 (unavailable) rather than a fabricated number.
    let t = &resp["timings"];
    let prompt_ms = t["prompt_ms"].as_f64().unwrap_or(0.0);
    let predicted_ms = t["predicted_ms"].as_f64().unwrap_or(0.0);
    let prompt_tokens = t["prompt_n"].as_u64().unwrap_or(0) as u32;
    let gen_tokens = t["predicted_n"].as_u64().unwrap_or(0) as u32;
    // Require a handful of tokens and a non-trivial elapsed time before reporting a rate, so a
    // model that stops early can't yield a noisy or fabricated tokens/sec — report 0 (unavailable).
    let rate = |toks: u32, ms: f64, min: u32| if ms >= 1.0 && toks >= min { (1000.0 * toks as f64 / ms) as f32 } else { 0.0 };
    let prompt_eval_tps = rate(prompt_tokens, prompt_ms, 2);
    let gen_tps = rate(gen_tokens, predicted_ms, 8);

    let result = BenchmarkResult {
        model_id: req.model_id.clone(),
        quant_label: req.quant_label.clone(),
        context_length: req.context_length,
        prompt_tokens,
        gen_tokens,
        prompt_eval_tok_per_s: prompt_eval_tps,
        gen_tok_per_s: gen_tps,
        warm: true,
        duration_ms,
        run_at: chrono::Utc::now(),
    };
    let _ = s.db.insert_benchmark(&result);
    ok_json(result).into_response()
}

async fn network_log(State(s): State<AppState>) -> impl IntoResponse {
    match s.db.list_net_log() {
        Ok(log) => ok_json(log).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn telemetry_status(State(s): State<AppState>) -> impl IntoResponse {
    let mgr = telemetry::TelemetryManager::new(&s.db);
    ok_json(mgr.status(&s.db)).into_response()
}

async fn telemetry_toggle(
    State(s): State<AppState>,
    Json(body): Json<ToggleReq>,
) -> impl IntoResponse {
    let mut mgr = telemetry::TelemetryManager::new(&s.db);
    match mgr.toggle(&s.db, body.enabled) {
        Ok(_) => ok_json(mgr.status(&s.db)).into_response(),
        Err(e) => err_json(&e).into_response(),
    }
}

async fn telemetry_preview(State(s): State<AppState>) -> impl IntoResponse {
    let mgr = telemetry::TelemetryManager::new(&s.db);
    ok_json(mgr.preview_payload(&s.db)).into_response()
}

async fn get_pref(
    State(s): State<AppState>,
    Path(key): Path<String>,
) -> impl IntoResponse {
    ok_json(s.db.get_preference(&key)).into_response()
}

async fn set_pref(
    State(s): State<AppState>,
    Path(key): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let val = body.as_str().unwrap_or(&body.to_string()).to_string();
    match s.db.set_preference(&key, &val) {
        Ok(_) => ok_json(true).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

// ---- Chat sessions (RUN-5): local-only conversation history ----

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateSessionBody {
    title: Option<String>,
    model_id: Option<String>,
    system_prompt: Option<String>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<i64>,
}

async fn list_chat_sessions(State(s): State<AppState>) -> impl IntoResponse {
    match s.db.list_chat_sessions() {
        Ok(v) => ok_json(v).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn create_chat_session(
    State(s): State<AppState>,
    Json(body): Json<CreateSessionBody>,
) -> impl IntoResponse {
    let now = chrono::Utc::now();
    let session = ipc::ChatSession {
        id: uuid::Uuid::new_v4().to_string(),
        title: body.title.filter(|t| !t.trim().is_empty()).unwrap_or_else(|| "New chat".into()),
        model_id: body.model_id,
        system_prompt: body.system_prompt.unwrap_or_default(),
        temperature: body.temperature.unwrap_or(0.7),
        top_p: body.top_p.unwrap_or(0.95),
        max_tokens: body.max_tokens.unwrap_or(2048),
        created_at: now,
        updated_at: now,
    };
    match s.db.create_chat_session(&session) {
        Ok(_) => ok_json(session).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn get_chat_session(State(s): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match s.db.get_chat_session(&id) {
        Ok(Some(session)) => match s.db.get_chat_messages(&id) {
            Ok(messages) => ok_json(ipc::ChatSessionDetail { session, messages }).into_response(),
            Err(e) => err_json(&e.to_string()).into_response(),
        },
        Ok(None) => err_json("session not found").into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn delete_chat_session(State(s): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    match s.db.delete_chat_session(&id) {
        Ok(_) => ok_json(true).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct RenameBody { title: String }

async fn rename_chat_session(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(body): Json<RenameBody>,
) -> impl IntoResponse {
    match s.db.rename_chat_session(&id, &body.title) {
        Ok(_) => ok_json(true).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct SettingsBody {
    system_prompt: String,
    temperature: f32,
    top_p: f32,
    max_tokens: i64,
    model_id: Option<String>,
}

async fn update_chat_settings(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(b): Json<SettingsBody>,
) -> impl IntoResponse {
    match s.db.update_chat_session_settings(&id, &b.system_prompt, b.temperature, b.top_p, b.max_tokens, b.model_id.as_deref()) {
        Ok(_) => ok_json(true).into_response(),
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppendMessageBody {
    role: String,
    content: String,
    reasoning: Option<String>,
}

async fn append_chat_message(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(b): Json<AppendMessageBody>,
) -> impl IntoResponse {
    let mut msg = ipc::ChatMessage {
        id: uuid::Uuid::new_v4().to_string(),
        session_id: id,
        role: b.role,
        content: b.content,
        reasoning: b.reasoning,
        ordinal: 0, // reassigned below to the true slot returned by the DB
        created_at: chrono::Utc::now(),
    };
    match s.db.append_chat_message(&msg) {
        Ok(ordinal) => { msg.ordinal = ordinal; ok_json(msg).into_response() }
        Err(e) => err_json(&e.to_string()).into_response(),
    }
}

async fn static_handler(
    State(_s): State<AppState>,
    req: axum::extract::Request,
) -> impl IntoResponse {
    let path = req.uri().path();
    if !path.starts_with("/api/") {
        let dist_dir = ui_dist_dir();
        // Serve built assets (js/css/svg) directly; SPA-fallback everything else to index.html.
        let rel = path.trim_start_matches('/');
        if !rel.is_empty() && path != "/" {
            let asset = dist_dir.join(rel);
            if asset.is_file() {
                let ct = content_type_for(&asset);
                if let Ok(bytes) = std::fs::read(&asset) {
                    return (StatusCode::OK, [("content-type", ct)], bytes).into_response();
                }
            }
        }
        let index = dist_dir.join("index.html");
        if index.is_file() {
            let html = std::fs::read_to_string(&index).unwrap_or_else(|_| FALLBACK_HTML.to_string());
            return (StatusCode::OK, [("content-type", "text/html")], html).into_response();
        }
    }
    (StatusCode::OK, [("content-type", "text/html")], FALLBACK_HTML.to_string()).into_response()
}

/// Whether the bundled llama.cpp version satisfies a catalog entry's `runtimeMinVersion`. Compares
/// the numeric build id (e.g. "b4321" -> 4321); when a value can't be parsed we fail open to
/// launching rather than blocking a valid model on a format quirk.
fn runtime_version_satisfies(bundled: &str, required: &str) -> bool {
    let num = |s: &str| s.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse::<u64>().ok();
    match (num(bundled), num(required)) {
        (Some(b), Some(r)) => b >= r,
        _ => true,
    }
}

/// Resolve the built UI directory. Order: `KAYON_UI_DIR` override, then `dist/` next to the
/// installed executable (packaged layout), then the dev source tree. This keeps the app UI
/// available in packaged builds where the source checkout isn't present.
fn ui_dist_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("KAYON_UI_DIR") {
        if !d.trim().is_empty() {
            return std::path::PathBuf::from(d);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let packaged = dir.join("dist");
            if packaged.is_dir() {
                return packaged;
            }
        }
    }
    catalog::crate_root().join("..").join("src").join("dist")
}

fn content_type_for(p: &std::path::Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()) {
        Some("js") | Some("mjs") => "text/javascript",
        Some("css") => "text/css",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("json") => "application/json",
        Some("woff2") => "font/woff2",
        Some("html") => "text/html",
        _ => "application/octet-stream",
    }
}

const FALLBACK_HTML: &str = r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Kayon</title>
<style>
body{font-family:system-ui,sans-serif;background:#faf9f5;color:#1a1a1a;display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}
.box{text-align:center;padding:40px}
.box h1{font-weight:600;margin-bottom:8px}
.box p{color:#666;font-size:14px}
.box a{color:#E0916B;text-decoration:none}
</style></head><body><div class="box">
<h1>Kayon backend is running</h1>
<p>Frontend not built yet. Run <code>npm run dev</code> in src/ for dev mode.</p>
<p>API available at <a href="/api/hardware">/api/hardware</a></p>
</div></body></html>"#;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct DownloadStartReq {
    model_id: String,
    quant_label: String,
    // url / totalBytes / sha256 are intentionally NOT accepted from the client: the server
    // resolves them from the verified catalog so the trust model can't be bypassed (CAT-2).
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct OllamaAdoptReq {
    // Only the identity is accepted; the trusted blob path + digest are re-resolved server-side
    // from the Ollama manifest scan (never trusted from the client).
    name: String,
    tag: String,
    // For cross-volume stores the user must explicitly choose "copy" (OLL-4). Same-volume hard-
    // links need no mode. Absent + cross-volume → the handler returns the choice rather than
    // silently copying gigabytes.
    #[serde(default)]
    mode: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeStartReq {
    model_path: String,
    model_id: String,
    quant_label: String,
    n_gpu_layers: i32,
    context_length: u32,
    #[serde(default)]
    kv_type_bytes: Option<u8>,
    #[serde(default)]
    runtime_args: Vec<String>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct BenchmarkReq {
    model_id: String,
    quant_label: String,
    context_length: u32,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToggleReq {
    enabled: bool,
}
