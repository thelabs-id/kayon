// The Rust API server always listens on 127.0.0.1:9518. In the browser build the UI is served
// from the same origin (relative works); inside the Tauri window the origin is tauri.localhost,
// so we address the server absolutely. Using the absolute URL is correct in both.
const BASE = (typeof window !== 'undefined' && (window as any).__TAURI_INTERNALS__)
  ? 'http://127.0.0.1:9518'
  : ''

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  // Never throw: a thrown fetch/parse error in a caller without try/catch becomes a silent
  // no-op (e.g. a mutating request rejected by the CSRF guard returns a 403 *plain-text* body,
  // not JSON). Instead, always resolve to an ApiResponse-shaped value so callers can surface the
  // failure. Endpoints that return a non-ApiResponse payload (few) still parse via the happy path.
  try {
    const resp = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...opts?.headers },
      ...opts,
    })
    const text = await resp.text()
    try {
      return JSON.parse(text) as T
    } catch {
      // Non-JSON body (error middleware, proxy, etc.). Represent it as a failed ApiResponse.
      return { ok: false, error: text || `HTTP ${resp.status}` } as unknown as T
    }
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : 'network error' } as unknown as T
  }
}

export interface ApiResponse<T> {
  ok: boolean
  data?: T
  error?: string
}

export interface GpuInfo {
  name: string
  pciId?: string
  architecture?: string
  computeCapability?: string
  driverVersion?: string
  cudaVersion?: string
  totalVramBytes: number
  telemetry: GpuTelemetry
}

export interface GpuTelemetry {
  vramUsedBytes: number
  vramFreeBytes: number
  utilizationPercent: number
  temperatureC: number
  powerWatts: number
  coreClockMhz: number
  memClockMhz: number
}

export interface CpuInfo {
  brand: string
  coreCount: number
  threadCount: number
  frequencyMhz: number
  usagePercent: number
}

export interface RamInfo {
  totalBytes: number
  availableBytes: number
  usedBytes: number
}

export interface DiskInfo {
  mount: string
  totalBytes: number
  freeBytes: number
  kind: string
}

export interface OsInfo {
  name: string
  version: string
  kernelVersion?: string
  hostName: string
}

export interface MachineProfile {
  gpus: GpuInfo[]
  primaryGpuIndex?: number
  cpu: CpuInfo
  ram: RamInfo
  disks: DiskInfo[]
  os: OsInfo
  probedAt: string
}

export type VerdictKind = 'FITS_FULLY' | 'FITS_TIGHT' | 'GPU_CPU_SPLIT' | 'CPU_ONLY' | 'EXCEEDS_MACHINE' | 'UNVERIFIED_ARCH'

export interface VerdictBreakdown {
  weightsBytes: number
  kvBytes: number
  computeBufferBytes: number
  cudaOverheadBytes: number
  vramAvailBytes: number
  ramAvailBytes: number
  headroomDisplayBytes: number
  headroomOsBytes: number
  comfortMarginBytes: number
  totalNeedBytes: number
}

export interface FitVerdict {
  modelId: string
  quantLabel: string
  contextLength: number
  kvTypeBytes: number
  verdict: VerdictKind
  nGpuLayers: number
  perBlockBytes?: number
  breakdown?: VerdictBreakdown
  explainability: string
  computedAt: string
}

export interface Capabilities {
  tools: boolean
  reasoning: boolean
  vision: boolean
}

export interface ArchBlock {
  architecture: string
  blockCount: number
  headCount: number
  headCountKv: number
  embeddingLength: number
  contextLength: number
  keyLength?: number
  valueLength?: number
  attentionType?: string
  runtimeMinVersion?: string
}

export interface Quant {
  label: string
  bytes: number
  sha256: string
  source: string
  runtimeArgs: string[]
}

export interface CatalogEntry {
  id: string
  family: string
  params: string
  license: string
  capabilities: Capabilities
  arch: ArchBlock
  quants: Quant[]
}

export interface Catalog {
  schemaVersion: number
  revision: number
  generatedAt: string
  entries: CatalogEntry[]
  source: string
  verifiedSignature?: string
}

export interface InstalledModel {
  id: string
  modelId: string
  quantLabel: string
  path: string
  bytes: number
  sha256: string
  source: 'downloaded' | 'adopted'
  installedAt: string
  ollamaTag?: string
  ollamaDigest?: string
  architecture?: string
  needsNewerRuntime?: boolean
}

export interface DownloadState {
  id: string
  modelId: string
  quantLabel: string
  url: string
  targetPath: string
  totalBytes: number
  receivedBytes: number
  sha256Expected: string
  status: string
  startedAt: string
  updatedAt: string
  error?: string
  throughputBps: number
  etaSeconds?: number
}

export interface RuntimeStatus {
  kind: 'stopped' | 'starting' | 'running' | 'error'
  modelId?: string
  quantLabel?: string
  pid?: number
  port?: number
  contextLength: number
  nGpuLayers: number
  startedAt?: string
  message?: string
  /** TOOL-2: whether the loaded model's GGUF chat template supports tool calling. */
  supportsTools?: boolean
}

export interface OllamaModel {
  name: string
  tag: string
  digest: string
  sizeBytes: number
  blobPath: string
  architecture?: string
  sameVolumeAsLibrary: boolean
  adoptable: boolean
  adoptReason?: string
  needsNewerRuntime: boolean
}

export interface NetworkLogEntry {
  id: number
  ts: string
  method: string
  url: string
  purpose: string
  bytesOut: number
  bytesIn: number
  status?: number
  note?: string
}

export interface TelemetryStatus {
  enabled: boolean
  lastPreviewPayload?: string
  lastPreviewAt?: string
}

export interface BenchmarkResult {
  modelId: string
  quantLabel: string
  contextLength: number
  promptTokens: number
  genTokens: number
  promptEvalTokPerS: number
  genTokPerS: number
  warm: boolean
  durationMs: number
  runAt: string
}

// Chat sessions (RUN-5): local-only conversation history.
export interface ChatMessage {
  id: string
  sessionId: string
  role: 'user' | 'assistant' | 'system'
  content: string
  reasoning?: string
  /** TOOL-7: JSON array of the tool calls made in this turn, so history stays auditable. */
  tools?: string
  ordinal: number
  createdAt: string
}

export interface ChatSession {
  id: string
  title: string
  modelId?: string
  systemPrompt: string
  temperature: number
  topP: number
  maxTokens: number
  workspace?: string
  webEnabled: boolean
  autoApprove: boolean
  createdAt: string
  updatedAt: string
}

export interface ChatSessionDetail extends ChatSession {
  messages: ChatMessage[]
}

export interface ChatSessionSummary {
  id: string
  title: string
  modelId?: string
  messageCount: number
  updatedAt: string
}

export interface ChatSettings {
  systemPrompt: string
  temperature: number
  topP: number
  maxTokens: number
  modelId?: string
  workspace?: string
  webEnabled?: boolean
  autoApprove?: boolean
}

export const api = {
  hardware: () => apiFetch<ApiResponse<MachineProfile>>('/api/hardware'),
  catalog: () => apiFetch<ApiResponse<Catalog>>('/api/catalog'),
  catalogRefresh: () => apiFetch<ApiResponse<Catalog>>('/api/catalog/refresh', { method: 'POST' }),
  catalogStatus: () => apiFetch<ApiResponse<{ discovering: boolean; source?: string; revision?: number }>>('/api/catalog/status'),
  verdicts: (ctx?: number, kvTypeBytes?: number) => {
    const p = new URLSearchParams()
    if (ctx) p.set('ctx', String(ctx))
    if (kvTypeBytes) p.set('kvTypeBytes', String(kvTypeBytes))
    const qs = p.toString()
    return apiFetch<ApiResponse<FitVerdict[]>>(`/api/fit/verdicts${qs ? `?${qs}` : ''}`)
  },
  verdict: (id: string, q: string) => apiFetch<ApiResponse<FitVerdict>>(`/api/fit/verdict/${id}/${q}`),
  library: () => apiFetch<ApiResponse<InstalledModel[]>>('/api/library'),
  libraryDir: () => apiFetch<ApiResponse<string>>('/api/library/dir'),
  relocateLibrary: (path: string) =>
    apiFetch<ApiResponse<{ movedFiles: number; libraryDir: string }>>('/api/library/relocate', { method: 'POST', body: JSON.stringify({ path }) }),
  localVerdict: (id: string, ctx?: number, kvTypeBytes?: number) => {
    const p = new URLSearchParams()
    if (ctx) p.set('ctx', String(ctx))
    if (kvTypeBytes) p.set('kvTypeBytes', String(kvTypeBytes))
    const qs = p.toString()
    return apiFetch<ApiResponse<FitVerdict>>(`/api/fit/local/${id}${qs ? `?${qs}` : ''}`)
  },
  deleteModel: (id: string) => apiFetch<ApiResponse<boolean>>(`/api/library/delete/${id}`, { method: 'POST' }),
  downloads: () => apiFetch<ApiResponse<DownloadState[]>>('/api/downloads'),
  // Downloads are resolved server-side from the verified catalog; the client only names the entry.
  startDownload: (body: { modelId: string; quantLabel: string }) =>
    apiFetch<ApiResponse<DownloadState>>('/api/downloads/start', { method: 'POST', body: JSON.stringify(body) }),
  cancelDownload: (id: string) => apiFetch<ApiResponse<boolean>>(`/api/downloads/${id}/cancel`, { method: 'DELETE' }),
  pauseDownload: (id: string) => apiFetch<ApiResponse<boolean>>(`/api/downloads/${id}/pause`, { method: 'POST' }),
  resumeDownload: (id: string) => apiFetch<ApiResponse<boolean>>(`/api/downloads/${id}/resume`, { method: 'POST' }),
  ollamaModels: () => apiFetch<ApiResponse<OllamaModel[]>>('/api/ollama/models'),
  // Server re-resolves the blob + digest from the manifest; the client only names the model.
  // `mode: "copy"` opts into a cross-volume copy (OLL-4); same-volume adoptions ignore it.
  ollamaAdopt: (body: { name: string; tag: string; mode?: 'copy' }) =>
    apiFetch<ApiResponse<InstalledModel>>('/api/ollama/adopt', { method: 'POST', body: JSON.stringify(body) }),
  runtimeStart: (body: { modelPath: string; modelId: string; quantLabel: string; nGpuLayers: number; contextLength: number; runtimeArgs: string[] }) =>
    apiFetch<ApiResponse<RuntimeStatus>>('/api/runtime/start', { method: 'POST', body: JSON.stringify(body) }),
  // One-press load (LIB-4): server computes n_gpu_layers from the local verdict and pulls
  // runtimeArgs from the catalog, so the runtime matches the fit engine (§7).
  runtimeLoad: (installedId: string, ctx?: number, kvTypeBytes?: number) => {
    const p = new URLSearchParams()
    if (ctx) p.set('ctx', String(ctx))
    if (kvTypeBytes) p.set('kvTypeBytes', String(kvTypeBytes))
    const qs = p.toString()
    return apiFetch<ApiResponse<RuntimeStatus>>(`/api/runtime/load/${installedId}${qs ? `?${qs}` : ''}`, { method: 'POST' })
  },
  runtimeStop: () => apiFetch<ApiResponse<RuntimeStatus>>('/api/runtime/stop', { method: 'POST' }),
  runtimeStatus: () => apiFetch<ApiResponse<RuntimeStatus>>('/api/runtime/status'),
  benchmark: (body: { modelId: string; quantLabel: string; contextLength: number }) =>
    apiFetch<ApiResponse<BenchmarkResult>>('/api/runtime/benchmark', { method: 'POST', body: JSON.stringify(body) }),
  networkLog: () => apiFetch<ApiResponse<NetworkLogEntry[]>>('/api/privacy/network-log'),
  telemetryStatus: () => apiFetch<ApiResponse<TelemetryStatus>>('/api/privacy/telemetry/status'),
  telemetryToggle: (enabled: boolean) => apiFetch<ApiResponse<TelemetryStatus>>('/api/privacy/telemetry/toggle', { method: 'POST', body: JSON.stringify({ enabled }) }),
  telemetryPreview: () => apiFetch<ApiResponse<{ endpoint: string; payload: string; byteSize: number }>>('/api/privacy/telemetry/preview'),

  // Chat sessions (RUN-5)
  chatSessions: () => apiFetch<ApiResponse<ChatSessionSummary[]>>('/api/chat/sessions'),
  createChatSession: (body: Partial<ChatSettings> & { title?: string }) =>
    apiFetch<ApiResponse<ChatSession>>('/api/chat/sessions', { method: 'POST', body: JSON.stringify(body) }),
  chatSession: (id: string) => apiFetch<ApiResponse<ChatSessionDetail>>(`/api/chat/sessions/${id}`),
  appendChatMessage: (id: string, body: { role: string; content: string; reasoning?: string; tools?: string }) =>
    apiFetch<ApiResponse<ChatMessage>>(`/api/chat/sessions/${id}/messages`, { method: 'POST', body: JSON.stringify(body) }),
  renameChatSession: (id: string, title: string) =>
    apiFetch<ApiResponse<boolean>>(`/api/chat/sessions/${id}/rename`, { method: 'POST', body: JSON.stringify({ title }) }),
  updateChatSettings: (id: string, body: ChatSettings) =>
    apiFetch<ApiResponse<boolean>>(`/api/chat/sessions/${id}/settings`, { method: 'POST', body: JSON.stringify(body) }),
  deleteChatSession: (id: string) =>
    apiFetch<ApiResponse<boolean>>(`/api/chat/sessions/${id}`, { method: 'DELETE' }),

  // TOOL family: the agent loop is a streaming (SSE) POST, so callers use fetch() directly against
  // this absolute URL rather than the JSON apiFetch helper.
  agentUrl: () => `${BASE}/api/chat/agent`,
  // TOOL-6: resolve a pending side-effect confirmation (Approve/Deny).
  toolDecision: (callId: string, approved: boolean) =>
    apiFetch<ApiResponse<boolean>>('/api/tools/decision', { method: 'POST', body: JSON.stringify({ callId, approved }) }),
  // Attach a file into the session's workspace (auto-created if no folder is attached).
  attachFile: (sessionId: string, name: string, contentBase64: string) =>
    apiFetch<ApiResponse<{ name: string; bytes: number }>>(`/api/chat/sessions/${sessionId}/files`, { method: 'POST', body: JSON.stringify({ name, contentBase64 }) }),
  // List files in the session workspace (attached files + model-created artifacts).
  listWorkspace: (sessionId: string) =>
    apiFetch<ApiResponse<{ auto: boolean; files: { name: string; bytes: number; isDir: boolean }[] }>>(`/api/chat/sessions/${sessionId}/workspace`),
}
