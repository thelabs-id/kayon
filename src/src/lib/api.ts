const BASE = ''

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...opts?.headers },
    ...opts,
  })
  return resp.json()
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

export const api = {
  hardware: () => apiFetch<ApiResponse<MachineProfile>>('/api/hardware'),
  catalog: () => apiFetch<ApiResponse<Catalog>>('/api/catalog'),
  catalogRefresh: () => apiFetch<ApiResponse<Catalog>>('/api/catalog/refresh', { method: 'POST' }),
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
}
