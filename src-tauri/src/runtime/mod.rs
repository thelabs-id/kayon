use anyhow::{anyhow, Result};
use chrono::Utc;
use std::collections::VecDeque;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

use crate::ipc::*;

pub struct RuntimeManager {
    child: Mutex<Option<Child>>,
    status: Mutex<RuntimeStatus>,
    // Recent sidecar log lines (drained from stdout/stderr) for surfacing on a health timeout.
    logs: Arc<Mutex<VecDeque<String>>>,
}

impl RuntimeManager {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            logs: Arc::new(Mutex::new(VecDeque::new())),
            status: Mutex::new(RuntimeStatus {
                kind: RuntimeStatusKind::Stopped,
                model_id: None,
                quant_label: None,
                pid: None,
                port: None,
                context_length: 2048,
                n_gpu_layers: 0,
                started_at: None,
                message: None,
                supports_tools: false,
            }),
        }
    }

    pub fn status(&self) -> RuntimeStatus {
        self.status.lock().unwrap().clone()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn start(
        &self,
        model_path: &str,
        model_id: &str,
        quant_label: &str,
        n_gpu_layers: i32,
        context_length: u32,
        port: u16,
        kv_type_bytes: u8,
        runtime_args: &[String],
    ) -> Result<()> {
        let mut child_guard = self.child.lock().unwrap();
        // RUN-2 strict single active model: cleanly unload the current sidecar before loading the
        // next, so one-press Load & Chat (LIB-4) is an atomic switch, not an error.
        if let Some(mut existing) = child_guard.take() {
            let _ = existing.kill();
            let _ = existing.wait();
        }

        let mut cmd = Command::new(Self::llama_server_binary());
        cmd.arg("-m").arg(model_path)
            .arg("-ngl").arg(n_gpu_layers.to_string())
            .arg("-c").arg(context_length.to_string())
            .arg("--host").arg("127.0.0.1")
            .arg("--port").arg(port.to_string())
            .arg("--metrics")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // §7 single-sourcing: if the fit verdict assumed q8_0 KV (kv_type_bytes == 1), the runtime
        // MUST launch with the matching cache type, or it would run a different memory config than
        // the verdict predicted (OD-1). f16 (== 2) is the default and needs no flags.
        if kv_type_bytes == 1 {
            cmd.arg("--cache-type-k").arg("q8_0").arg("--cache-type-v").arg("q8_0");
        }

        // runtimeArgs are per-model launch flags (e.g. --jinja), but they must NOT override the
        // configuration the fit engine single-sourced (-ngl/-c/cache-type) or the loopback binding
        // (--host/--port) or the model path (-m). Drop any reserved flag (and its value) so a
        // catalog/client arg can't silently change the memory config or expose the API (§7, PRIV).
        // `-ub`/`-b` are reserved for the same reason as -ngl/-c: the fit engine's compute-buffer
        // model is `n_ubatch × (2·n_embd + 3·n_ff) × 4 + n_ubatch × C × 2` at llama.cpp's *default*
        // n_ubatch (§7). Letting an arg raise it would make the runtime allocate more than the
        // verdict promised — a silent OOM against a "fits" the user was shown.
        //
        // `--parallel` is reserved because the §7 context term assumes llama-server's default
        // `kv_unified = true`, i.e. `n_ctx_seq == n_ctx`. Passing it flips unified KV off and
        // silently re-cuts every slot's context to `n_ctx / n_parallel` — the verdict would then
        // describe a context the user does not actually get.
        const RESERVED: &[&str] = &[
            "-ngl", "--n-gpu-layers", "-c", "--ctx-size", "--cache-type-k", "--cache-type-v",
            "--host", "--port", "-m", "--model",
            "-ub", "--ubatch-size", "-b", "--batch-size",
            "-np", "--parallel",
        ];
        let mut skip_value = false;
        for arg in runtime_args {
            if skip_value {
                skip_value = false;
                continue;
            }
            let flag = arg.split('=').next().unwrap_or(arg);
            if RESERVED.contains(&flag) {
                if !arg.contains('=') {
                    skip_value = true; // "-flag value" form — also drop the following value
                }
                log::warn!("ignoring reserved runtimeArg that would override fit/binding: {}", arg);
                continue;
            }
            cmd.arg(arg);
        }

        // TOOL-2: detect tool-calling from the model's GGUF chat template. When supported, launch
        // llama-server with --jinja so it uses the embedded template and actually parses/serializes
        // OpenAI tool calls — without --jinja, `tool_calls` are never emitted.
        let supports_tools = crate::gguf::parse_gguf_header(std::path::Path::new(model_path))
            .ok()
            .map(|h| crate::gguf::template_supports_tools(&h))
            .unwrap_or(false);
        if supports_tools {
            cmd.arg("--jinja");
        }

        // Windows: llama-server is a console-subsystem exe, so spawning it normally pops a console
        // window on screen. We already capture its stdout/stderr (piped + drained below), so no
        // console is needed — CREATE_NO_WINDOW (0x08000000) starts it fully hidden.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }

        let mut child = cmd.spawn().map_err(|e| anyhow!("failed to start llama-server: {}", e))?;
        let pid = child.id();

        // Drain stdout+stderr on background threads. Without this, a child that writes enough log
        // output fills the OS pipe buffer and blocks — never reaching /health or stalling mid-gen.
        self.logs.lock().unwrap().clear();
        if let Some(out) = child.stdout.take() {
            Self::drain_stream(out, self.logs.clone());
        }
        if let Some(err) = child.stderr.take() {
            Self::drain_stream(err, self.logs.clone());
        }

        *child_guard = Some(child);

        let mut status = self.status.lock().unwrap();
        *status = RuntimeStatus {
            kind: RuntimeStatusKind::Starting,
            model_id: Some(model_id.to_string()),
            quant_label: Some(quant_label.to_string()),
            pid: Some(pid),
            port: Some(port),
            context_length,
            n_gpu_layers,
            started_at: Some(Utc::now()),
            message: Some("waiting for /health".into()),
            supports_tools,
        };

        Ok(())
    }

    /// Spawn a thread that reads a child pipe line-by-line into the capped log ring buffer.
    fn drain_stream<R: std::io::Read + Send + 'static>(stream: R, logs: Arc<Mutex<VecDeque<String>>>) {
        std::thread::spawn(move || {
            use std::io::BufRead;
            let reader = std::io::BufReader::new(stream);
            for line in reader.lines().map_while(Result::ok) {
                let mut buf = logs.lock().unwrap();
                buf.push_back(line);
                while buf.len() > 200 {
                    buf.pop_front();
                }
            }
        });
    }

    /// Last few captured sidecar log lines (for surfacing on a health-check timeout).
    pub fn log_tail(&self, n: usize) -> String {
        let buf = self.logs.lock().unwrap();
        let start = buf.len().saturating_sub(n);
        buf.iter().skip(start).cloned().collect::<Vec<_>>().join("\n")
    }

    // Port-guarded so a stale health probe from a previous model (since stopped or switched)
    // can't overwrite the status of the current/next runtime.
    pub fn mark_running(&self, port: u16) {
        let mut status = self.status.lock().unwrap();
        if status.port == Some(port) {
            status.kind = RuntimeStatusKind::Running;
            status.message = Some("ready".into());
        }
    }

    pub fn mark_error(&self, port: u16, msg: &str) {
        let mut status = self.status.lock().unwrap();
        if status.port == Some(port) {
            status.kind = RuntimeStatusKind::Error;
            status.message = Some(msg.to_string());
        }
    }

    pub fn stop(&self) -> Result<()> {
        let mut child_guard = self.child.lock().unwrap();
        if let Some(mut child) = child_guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let mut status = self.status.lock().unwrap();
        *status = RuntimeStatus {
            kind: RuntimeStatusKind::Stopped,
            model_id: None,
            quant_label: None,
            pid: None,
            port: None,
            context_length: 0,
            n_gpu_layers: 0,
            started_at: None,
            message: None,
            supports_tools: false,
        };
        Ok(())
    }

    /// Architectures the bundled llama-server build can load (OLL-6 version gate).
    /// This is deliberately broader than the fit engine's supported-standard-*attention*
    /// set — a model can be loadable yet fit-unverifiable (e.g. a hybrid). Versioned
    /// alongside the shipped `llama-server` binary.
    pub fn supported_runtime_archs() -> &'static [&'static str] {
        &[
            "llama", "mistral", "mixtral", "gemma", "gemma2", "gemma3", "qwen2", "qwen2moe",
            "qwen3", "phi2", "phi3", "phi", "starcoder2", "command-r", "cohere2", "deepseek2",
            "stablelm", "gptneox", "falcon", "mpt", "bloom", "baichuan", "internlm2", "orion",
            "olmo", "granite", "granitemoe", "nemotron", "exaone",
        ]
    }

    /// Version of the bundled llama.cpp runtime (RUN-1 / catalog `runtimeMinVersion` gate), or None
    /// if unknown. Real builds inject `KAYON_RUNTIME_VERSION` at package time. When it's unknown we
    /// fail CLOSED against a runtimeMinVersion requirement rather than launch a model that may need
    /// a newer runtime — better to block than to start-and-fail.
    pub fn bundled_runtime_version() -> Option<String> {
        std::env::var("KAYON_RUNTIME_VERSION").ok().filter(|s| !s.trim().is_empty())
    }

    /// Resolve the `llama-server` binary (RUN-1). The runtime is bundled as a Tauri sidecar so it
    /// works out of the box with no user setup. Resolution order:
    ///   1. installed layout: `<exe_dir>/resources/binaries/llama/llama-server.exe` (Tauri resources)
    ///   2. `<exe_dir>/binaries/llama/llama-server.exe` (portable layout)
    ///   3. dev tree: `<crate>/binaries/llama/llama-server.exe`
    ///   4. `KAYON_LLAMA_SERVER` env override (for a custom/CUDA build)
    ///   5. `llama-server.exe` on PATH
    pub fn llama_server_binary() -> String {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                for cand in [
                    dir.join("resources").join("binaries").join("llama").join("llama-server.exe"),
                    dir.join("binaries").join("llama").join("llama-server.exe"),
                    dir.join("llama-server.exe"),
                ] {
                    if cand.is_file() {
                        return cand.to_string_lossy().to_string();
                    }
                }
            }
        }
        let dev = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("binaries").join("llama").join("llama-server.exe");
        if dev.is_file() {
            return dev.to_string_lossy().to_string();
        }
        // Explicit override (e.g. a custom CUDA build) or PATH as a last resort.
        if let Ok(p) = std::env::var("KAYON_LLAMA_SERVER") {
            if !p.trim().is_empty() {
                return p;
            }
        }
        "llama-server.exe".to_string()
    }

    pub fn find_available_port() -> u16 {
        use std::net::TcpListener;
        TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .map(|addr| addr.port())
            .unwrap_or(8080)
    }
}
