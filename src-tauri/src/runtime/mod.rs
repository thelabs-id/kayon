use anyhow::{anyhow, Result};
use chrono::Utc;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use crate::ipc::*;

pub struct RuntimeManager {
    child: Mutex<Option<Child>>,
    status: Mutex<RuntimeStatus>,
}

impl RuntimeManager {
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
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
        if child_guard.is_some() {
            return Err(anyhow!("a model is already running — stop it first (RUN-2 strict single)"));
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

        for arg in runtime_args {
            cmd.arg(arg);
        }

        let child = cmd.spawn().map_err(|e| anyhow!("failed to start llama-server: {}", e))?;
        let pid = child.id();

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
        };

        Ok(())
    }

    pub fn mark_running(&self) {
        let mut status = self.status.lock().unwrap();
        status.kind = RuntimeStatusKind::Running;
        status.message = Some("ready".into());
    }

    pub fn mark_error(&self, msg: &str) {
        let mut status = self.status.lock().unwrap();
        status.kind = RuntimeStatusKind::Error;
        status.message = Some(msg.to_string());
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

    /// Resolve the `llama-server` binary (RUN-1). Order: `KAYON_LLAMA_SERVER` env override,
    /// else the bundled sidecar under the crate's `binaries/` dir, else the name on PATH.
    pub fn llama_server_binary() -> String {
        if let Ok(p) = std::env::var("KAYON_LLAMA_SERVER") {
            if !p.trim().is_empty() {
                return p;
            }
        }
        let bundled = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("binaries")
            .join("llama-server.exe");
        if bundled.is_file() {
            return bundled.to_string_lossy().to_string();
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
