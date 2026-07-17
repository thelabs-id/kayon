use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub pci_id: Option<String>,
    pub architecture: Option<String>,
    pub compute_capability: Option<String>,
    pub driver_version: Option<String>,
    pub cuda_version: Option<String>,
    pub total_vram_bytes: u64,
    pub telemetry: GpuTelemetry,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuTelemetry {
    pub vram_used_bytes: u64,
    pub vram_free_bytes: u64,
    pub utilization_percent: f32,
    pub temperature_c: f32,
    pub power_watts: f32,
    pub core_clock_mhz: u32,
    pub mem_clock_mhz: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuInfo {
    pub brand: String,
    pub core_count: usize,
    pub thread_count: usize,
    pub frequency_mhz: u64,
    pub usage_percent: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RamInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub used_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiskInfo {
    pub mount: String,
    pub total_bytes: u64,
    pub free_bytes: u64,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OsInfo {
    pub name: String,
    pub version: String,
    pub kernel_version: Option<String>,
    pub host_name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MachineProfile {
    pub gpus: Vec<GpuInfo>,
    pub primary_gpu_index: Option<usize>,
    pub cpu: CpuInfo,
    pub ram: RamInfo,
    pub disks: Vec<DiskInfo>,
    pub os: OsInfo,
    pub probed_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerdictKind {
    FitsFully,
    FitsTight,
    GpuCpuSplit,
    CpuOnly,
    ExceedsMachine,
    UnverifiedArch,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerdictBreakdown {
    pub weights_bytes: u64,
    pub kv_bytes: u64,
    pub compute_buffer_bytes: u64,
    pub runtime_overhead_bytes: u64,
    pub vram_avail_bytes: u64,
    pub ram_avail_bytes: u64,
    pub headroom_display_bytes: u64,
    pub headroom_os_bytes: u64,
    pub comfort_margin_bytes: u64,
    pub total_need_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FitVerdict {
    pub model_id: String,
    pub quant_label: String,
    pub context_length: u32,
    pub kv_type_bytes: u8,
    pub verdict: VerdictKind,
    pub n_gpu_layers: i32,
    pub per_block_bytes: Option<u64>,
    pub breakdown: Option<VerdictBreakdown>,
    pub explainability: String,
    pub computed_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Capabilities {
    #[serde(default)]
    pub tools: bool,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub vision: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArchBlock {
    pub architecture: String,
    pub block_count: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub embedding_length: u32,
    pub context_length: u32,
    pub key_length: Option<u32>,
    pub value_length: Option<u32>,
    /// `{arch}.vocab_size` from the GGUF. Sizes the compute buffer, which is dominated by the
    /// output logits and therefore tracks vocabulary, not parameter count (see `fit::COMPUTE_*`).
    /// `None` when the header didn't carry it — the fit engine then falls back conservatively.
    #[serde(default)]
    pub vocab_size: Option<u32>,
    pub attention_type: Option<String>,
    pub runtime_min_version: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Quant {
    pub label: String,
    pub bytes: u64,
    pub sha256: String,
    pub source: String,
    pub runtime_args: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogEntry {
    pub id: String,
    pub family: String,
    pub params: String,
    pub license: String,
    pub capabilities: Capabilities,
    pub arch: ArchBlock,
    pub quants: Vec<Quant>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Catalog {
    pub schema_version: u32,
    pub revision: u64,
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub entries: Vec<CatalogEntry>,
    pub source: String,
    pub verified_signature: Option<String>,
}

// ---- Chat sessions (RUN-5): local-only conversation history ----

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatMessage {
    pub id: String,
    pub session_id: String,
    /// "user" | "assistant" | "system".
    pub role: String,
    pub content: String,
    /// Reasoning segment, when the model emitted one (RUN-3).
    pub reasoning: Option<String>,
    /// TOOL-7: JSON array of the tool calls made in this turn (name, args, status, result), so the
    /// saved transcript remains auditable — what was approved/executed — after a reload.
    #[serde(default)]
    pub tools: Option<String>,
    /// Monotonic position within the session, so ordering never depends on timestamp ties.
    pub ordinal: i64,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatSession {
    pub id: String,
    pub title: String,
    /// The model this session was last chatting with (informational; the active runtime decides).
    pub model_id: Option<String>,
    pub system_prompt: String,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: i64,
    /// TOOL family: optional attached workspace folder (absolute path). Scopes filesystem/code tools.
    #[serde(default)]
    pub workspace: Option<String>,
    /// TOOL-5: per-session Web toggle (off by default).
    #[serde(default)]
    pub web_enabled: bool,
    /// TOOL-6: auto-approve side-effectful tools (write_file/code) for this session (off by default).
    #[serde(default)]
    pub auto_approve: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// A session plus its messages, returned when reopening a chat.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatSessionDetail {
    #[serde(flatten)]
    pub session: ChatSession,
    pub messages: Vec<ChatMessage>,
}

/// Lightweight row for the session list (most-recent-first).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatSessionSummary {
    pub id: String,
    pub title: String,
    pub model_id: Option<String>,
    pub message_count: i64,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    Downloaded,
    Adopted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    pub id: String,
    pub model_id: String,
    pub quant_label: String,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    pub source: ModelSource,
    pub installed_at: chrono::DateTime<chrono::Utc>,
    pub ollama_tag: Option<String>,
    pub ollama_digest: Option<String>,
    /// GGUF architecture, recorded at install time (for the OLL-6 runtime gate in the library).
    #[serde(default)]
    pub architecture: Option<String>,
    /// Computed for the library listing: the bundled runtime can't load this architecture.
    #[serde(default)]
    pub needs_newer_runtime: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DownloadStatus {
    Queued,
    Active,
    Paused,
    Completed,
    Failed,
    Cancelled,
    Quarantined,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadState {
    pub id: String,
    pub model_id: String,
    pub quant_label: String,
    pub url: String,
    pub target_path: String,
    pub total_bytes: u64,
    pub received_bytes: u64,
    pub sha256_expected: String,
    pub status: DownloadStatus,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub error: Option<String>,
    pub throughput_bps: u64,
    pub eta_seconds: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadProgress {
    pub id: String,
    pub model_id: String,
    pub quant_label: String,
    pub bytes: u64,
    pub total_bytes: u64,
    pub percent: f32,
    pub throughput_bps: u64,
    pub eta_seconds: Option<u64>,
    pub status: DownloadStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeStatusKind {
    Stopped,
    Starting,
    Running,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeStatus {
    pub kind: RuntimeStatusKind,
    pub model_id: Option<String>,
    pub quant_label: Option<String>,
    pub pid: Option<u32>,
    pub port: Option<u16>,
    pub context_length: u32,
    pub n_gpu_layers: i32,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub message: Option<String>,
    /// Whether the loaded model's GGUF chat template supports tool calling (TOOL-2). Detected at
    /// load; drives whether the chat UI offers tools and whether the agent loop advertises them.
    #[serde(default)]
    pub supports_tools: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkResult {
    pub model_id: String,
    pub quant_label: String,
    pub context_length: u32,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub prompt_eval_tok_per_s: f32,
    pub gen_tok_per_s: f32,
    pub warm: bool,
    pub duration_ms: u64,
    pub run_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaModel {
    pub name: String,
    pub tag: String,
    pub digest: String,
    pub size_bytes: u64,
    pub blob_path: String,
    pub architecture: Option<String>,
    pub families: Vec<String>,
    pub parameter_size: Option<String>,
    pub quantization: Option<String>,
    pub same_volume_as_library: bool,
    pub adoptable: bool,
    pub adopt_reason: Option<String>,
    /// OLL-6: blob loads a GGUF whose architecture the bundled runtime cannot load.
    /// Still adoptable, but flagged so the UI never presents a model that will fail to load.
    pub needs_newer_runtime: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkLogEntry {
    pub id: i64,
    pub ts: chrono::DateTime<chrono::Utc>,
    pub method: String,
    pub url: String,
    pub purpose: String,
    pub bytes_out: u64,
    pub bytes_in: u64,
    pub status: Option<u16>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryStatus {
    pub enabled: bool,
    pub last_preview_payload: Option<String>,
    pub last_preview_at: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TelemetryPreview {
    pub endpoint: String,
    pub payload: String,
    pub byte_size: usize,
    pub shown_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ApiResponse<T> {
    pub ok: bool,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn ok(data: T) -> Self {
        Self {
            ok: true,
            data: Some(data),
            error: None,
        }
    }

    pub fn err(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            data: None,
            error: Some(msg.into()),
        }
    }
}