mod db;
mod probe;
mod gguf;
mod fit;
mod catalog;
mod download;
mod library;
mod ollama;
mod runtime;
mod telemetry;
mod ipc;

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

#[tokio::main]
async fn main() {
    env_logger::init();

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
                        if let Err(e) = dl.resume_download(&db, &id, None).await {
                            log::error!("resume of download {} failed: {}", id, e);
                        }
                    });
                }
            }
        });
    }

    let app = Router::new()
        .route("/api/hardware", get(hardware))
        .route("/api/hardware/stream", get(hardware_stream))
        .route("/api/catalog", get(get_catalog))
        .route("/api/catalog/refresh", post(refresh_catalog))
        .route("/api/fit/verdicts", get(all_verdicts))
        .route("/api/fit/verdict/{model_id}/{quant_label}", get(verdict))
        .route("/api/library", get(library_list))
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
                ])
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any),
        )
        .with_state(state);

    // Bind loopback only: this is a private, local-control API with no auth (PRIV-2). It must
    // never be reachable from the LAN — every endpoint (delete, download, adopt, launch) is
    // local-user-only by design.
    let addr = SocketAddr::from(([127, 0, 0, 1], 9518));
    log::info!("Kayon server listening on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

const ALLOWED_ORIGINS: &[&str] = &[
    "http://127.0.0.1:9518", "http://localhost:9518",
    "http://127.0.0.1:3000", "http://localhost:3000",
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

async fn refresh_catalog(State(s): State<AppState>) -> impl IntoResponse {
    match catalog::get_active_catalog() {
        Ok(local) => {
            // fetch_remote_catalog logs both GETs at egress and verifies the signature (PRIV-5,
            // CAT-5). We cache the exact verified bytes only when the revision is strictly newer.
            match catalog::fetch_remote_catalog(&s.db).await {
                Ok((remote, json_bytes, sig_bytes)) => {
                    if catalog::maybe_update_catalog(&local, &remote) {
                        let _ = catalog::save_local_catalog_raw(&json_bytes, &sig_bytes);
                        ok_json(remote).into_response()
                    } else {
                        ok_json(local).into_response()
                    }
                }
                Err(e) => err_json(&e.to_string()).into_response(),
            }
        }
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

async fn library_list(State(s): State<AppState>) -> impl IntoResponse {
    match library::list_installed(&s.db) {
        Ok(models) => ok_json(models).into_response(),
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
        Ok(state) => {
            let id = state.id.clone();
            let db = s.db.clone();
            let dl = s.dl.clone();
            tokio::spawn(async move {
                if let Err(e) = dl.resume_download(&db, &id, None).await {
                    log::error!("download failed: {}", e);
                }
            });
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
    let mode = if m.same_volume_as_library { ollama::AdoptMode::Link } else { ollama::AdoptMode::Copy };

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

/// Poll llama-server `/health` until ready (RUN-1), marking the runtime running or errored.
fn spawn_health_wait(rt: Arc<runtime::RuntimeManager>, port: u16) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(2)).await;
        let url = format!("http://127.0.0.1:{}/health", port);
        for _ in 0..20 {
            if let Ok(resp) = reqwest::get(&url).await {
                if resp.status().is_success() {
                    rt.mark_running(port);
                    return;
                }
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
        let tail = rt.log_tail(8);
        rt.mark_error(port, &format!("health check timeout. llama-server log tail:\n{}", tail));
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
    let n_gpu_layers = verdict.as_ref().map(|v| v.n_gpu_layers).unwrap_or(999);

    // runtimeArgs from the catalog entry, if this model came from the catalog (RUN-1).
    let runtime_args = catalog::get_active_catalog().ok()
        .and_then(|c| c.entries.iter()
            .find(|e| e.id == model.model_id)
            .and_then(|e| e.quants.iter().find(|qu| qu.label == model.quant_label).map(|qu| qu.runtime_args.clone())))
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
    const PROMPT: &str = "Explain what a GPU is in two sentences.";
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

    // llama-server returns a `timings` object with prompt/predicted tokens and per-second rates.
    let t = &resp["timings"];
    let prompt_eval_tps = t["prompt_per_second"].as_f64().unwrap_or(0.0) as f32;
    let gen_tps = t["predicted_per_second"].as_f64().unwrap_or(0.0) as f32;
    let prompt_tokens = t["prompt_n"].as_u64().unwrap_or(0) as u32;
    let gen_tokens = t["predicted_n"].as_u64().unwrap_or(N_PREDICT as u64) as u32;

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

async fn static_handler(
    State(_s): State<AppState>,
    req: axum::extract::Request,
) -> impl IntoResponse {
    let path = req.uri().path();
    if !path.starts_with("/api/") {
        let dist_dir = catalog::crate_root().join("..").join("src").join("dist");
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
