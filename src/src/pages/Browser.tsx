import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type CatalogEntry, type DownloadState, type FitVerdict, type MachineProfile, type Quant } from '../lib/api'
import { VerdictChip } from '../components/icons'

const CTX = [2048, 4096, 8192, 16384, 32768]
const fmtB = (b: number) => b < 1024 ** 3 ? `${(b / 1024 ** 2).toFixed(0)} MB` : `${(b / 1024 ** 3).toFixed(1)} GB`
const g = (b: number) => (b / 1024 ** 3).toFixed(1)
const order: Record<string, number> = { FITS_FULLY: 0, FITS_TIGHT: 1, GPU_CPU_SPLIT: 2, CPU_ONLY: 3, UNVERIFIED_ARCH: 4, EXCEEDS_MACHINE: 5 }
const isPinned = (s: string) => /^[0-9a-f]{64}$/i.test((s || '').trim())
const dlPct = (d: DownloadState) => d.totalBytes > 0 ? Math.min(100, Math.floor((d.receivedBytes / d.totalBytes) * 100)) : 0

interface DlActions { onPause: (id: string) => void; onResume: (id: string) => void; onCancel: (id: string) => void; goLibrary: () => void }

// Inline download state for a model|quant — a live bar with pause/resume/cancel while in flight (so
// it never looks stuck on "Starting"), a library link when done, or the failure reason.
function DlProgress({ d, onPause, onResume, onCancel, goLibrary }: { d: DownloadState } & DlActions) {
  if (d.status === 'completed') return <button className="btn btn-line btn-sm" onClick={goLibrary}>In library ✓</button>
  if (d.status === 'cancelled' || d.status === 'failed' || d.status === 'quarantined')
    return <span className="mono" style={{ fontSize: 12, color: d.status === 'cancelled' ? 'var(--faint)' : 'var(--danger)' }}>{d.status === 'cancelled' ? 'cancelled' : (d.status === 'quarantined' ? 'checksum failed' : (d.error || 'failed'))}</span>
  const pct = dlPct(d)
  const paused = d.status === 'paused'
  return (
    <div className="dlprog">
      <div className="dlbar mini"><div className="dlfill" style={{ width: pct + '%', opacity: paused ? 0.5 : 1 }} /></div>
      <div className="dlprogrow">
        <span className="mono faint" style={{ fontSize: 11 }}>
          {paused ? `paused · ${pct}%`
            : d.status === 'queued' || d.totalBytes === 0 ? 'Starting…'
            : `${pct}% · ${fmtB(d.receivedBytes)}/${fmtB(d.totalBytes)} · ${(d.throughputBps / 1024 ** 2).toFixed(1)} MB/s${d.etaSeconds != null ? ` · ${d.etaSeconds}s` : ''}`}
        </span>
        <div className="dlbtns">
          {paused
            ? <button className="dlbtn" title="Resume" onClick={() => onResume(d.id)}>▶</button>
            : <button className="dlbtn" title="Pause" onClick={() => onPause(d.id)}>⏸</button>}
          <button className="dlbtn danger" title="Cancel" onClick={() => onCancel(d.id)}>✕</button>
        </div>
      </div>
    </div>
  )
}

function QuantRow({ q, v, ctxLabel, vramAvail, open, onToggle, onDownload, busy, dl, dlActions, installed }: {
  q: Quant; v?: FitVerdict; ctxLabel: string; vramAvail: string
  open: boolean; onToggle: () => void; onDownload: () => void; busy: boolean
  dl?: DownloadState; dlActions: DlActions; installed: boolean
}) {
  const bd = v?.breakdown
  const pinned = isPinned(q.sha256)
  return (
    <>
      <div className="qrow" onClick={onToggle}>
        <span className="qname">{q.label}</span>
        <span className="qsize">{fmtB(q.bytes)}</span>
        {v ? <VerdictChip v={v.verdict} /> : <span />}
        <span className="mono faint" style={{ fontSize: 12, textAlign: 'right' }}>
          {v && v.verdict !== 'EXCEEDS_MACHINE' && v.verdict !== 'UNVERIFIED_ARCH' && `${v.nGpuLayers} layers · `}
          {open ? '▾' : '▸'}
        </span>
      </div>
      {open && (
        <div className="breakdown">
          {bd ? <>
            <div className="bkbar">
              <span className="bkseg" style={{ width: `${(bd.weightsBytes / bd.totalNeedBytes) * 100}%`, background: 'var(--iris)' }} />
              <span className="bkseg" style={{ width: `${(bd.kvBytes / bd.totalNeedBytes) * 100}%`, background: 'var(--amber)' }} />
              <span className="bkseg" style={{ width: `${((bd.computeBufferBytes + bd.cudaOverheadBytes) / bd.totalNeedBytes) * 100}%`, background: 'var(--v-cpu)' }} />
            </div>
            weights {g(bd.weightsBytes)} + KV@{ctxLabel} {g(bd.kvBytes)} + buffers {g(bd.computeBufferBytes + bd.cudaOverheadBytes)} = <b style={{ color: 'var(--ink)' }}>{g(bd.totalNeedBytes)} GB</b> vs {vramAvail} GB available
          </> : (v?.explainability ?? '')}
          <div style={{ marginTop: 10 }}>
            {dl
              ? <DlProgress d={dl} {...dlActions} />
              : installed
                ? <button className="btn btn-line btn-sm" onClick={(e) => { e.stopPropagation(); dlActions.goLibrary() }}>In library ✓</button>
                : v?.verdict === 'EXCEEDS_MACHINE'
                  ? <span className="faint">Won't fit on this machine.</span>
                  : !pinned
                    ? <span className="faint" title="Checksum not yet pinned by the catalog generator (CAT-6).">Checksum pending — not yet downloadable.</span>
                    : <button className="btn btn-iris btn-sm" disabled={busy} onClick={(e) => { e.stopPropagation(); onDownload() }}>{busy ? 'Starting…' : `Download · ${fmtB(q.bytes)}`}</button>}
          </div>
        </div>
      )}
    </>
  )
}

function ModelCard({ entry, vmap, ctxLabel, vramAvail, lead, openQ, setOpenQ, download, busyKey, dlMap, dlActions, installed }: {
  entry: CatalogEntry; vmap: Map<string, FitVerdict>; ctxLabel: string; vramAvail: string; lead?: boolean
  openQ: string | null; setOpenQ: (k: string | null) => void; download: (e: CatalogEntry, q: Quant) => void; busyKey: string | null
  dlMap: Map<string, DownloadState>; dlActions: DlActions; installed: Set<string>
}) {
  const caps = [entry.capabilities.tools && 'tools', entry.capabilities.reasoning && 'reasoning', entry.capabilities.vision && 'vision'].filter(Boolean) as string[]
  // Best downloadable quant = pinned checksum + a runnable verdict, ranked by fit then size.
  const dlQuant = entry.quants
    .filter(q => isPinned(q.sha256))
    .map(q => ({ q, v: vmap.get(`${entry.id}|${q.label}`) }))
    .filter(x => x.v && x.v.verdict !== 'EXCEEDS_MACHINE')
    .sort((a, b) => (order[a.v!.verdict] ?? 9) - (order[b.v!.verdict] ?? 9))[0]?.q
  const anyPinned = entry.quants.some(q => isPinned(q.sha256))
  const busyThis = entry.quants.some(q => busyKey === `${entry.id}|${q.label}`)
  // A download in flight (or just finished) for any of this model's quants — drives the inline
  // progress shown in place of the Install button.
  const entryDl = entry.quants.map(q => dlMap.get(`${entry.id}|${q.label}`)).find(Boolean)
  return (
    <div className={`mcard ${lead ? 'lead' : ''}`}>
      {lead && <div className="leadbanner"><svg className="kmk" viewBox="0 0 64 64" width="15" height="15"><path className="ko" d="M32 7 C23 18 18 24 18 34 C18 45 24 51 32 57 C40 51 46 45 46 34 C46 24 41 18 32 7 Z" style={{ stroke: 'var(--iris)' }} /></svg><span style={{ color: 'var(--iris)', fontWeight: 600 }}>Best pick for your machine</span><span className="muted">— the most capable model that fits, computed from real free VRAM.</span></div>}
      <div className="mhead">
        <div>
          <div className="mname">{entry.id}</div>
          <div className="mmeta">
            <span className="tag mono">{entry.params}</span>
            <span className="tag mono">{entry.family}</span>
            {caps.map(c => <span key={c} className="tag">{c}</span>)}
          </div>
        </div>
        {entryDl
          ? <DlProgress d={entryDl} {...dlActions} />
          : dlQuant && installed.has(`${entry.id}|${dlQuant.label}`)
          ? <button className="btn btn-line btn-sm" onClick={dlActions.goLibrary}>In library ✓</button>
          : dlQuant
          ? <button className={`btn ${lead ? 'btn-iris' : 'btn-line'} btn-sm`} disabled={busyThis} onClick={() => download(entry, dlQuant)}>{busyThis ? 'Starting…' : `Install · ${dlQuant.label} · ${fmtB(dlQuant.bytes)}`}</button>
          : anyPinned
            ? <span className="mono faint" style={{ fontSize: 12, whiteSpace: 'nowrap' }}>won't fit</span>
            : <span className="tag" title="The catalog generator (CAT-6) hasn't pinned a real SHA-256 for this model yet, so it isn't downloadable in this build." style={{ color: 'var(--v-unv)', borderColor: 'color-mix(in oklab, var(--v-unv) 40%, var(--line2))' }}>checksum pending</span>}
      </div>
      <div className="qtable">
        {entry.quants.map(q => {
          const key = `${entry.id}|${q.label}`
          return <QuantRow key={key} q={q} v={vmap.get(key)} ctxLabel={ctxLabel} vramAvail={vramAvail} open={openQ === key} onToggle={() => setOpenQ(openQ === key ? null : key)} onDownload={() => download(entry, q)} busy={busyKey === key} dl={dlMap.get(key)} dlActions={dlActions} installed={installed.has(key)} />
        })}
      </div>
    </div>
  )
}

// Last-loaded browser data, kept at module scope so navigating away and back restores it instantly.
// Without this, a remount resets catalog/verdicts/downloads to empty, and an in-flight download
// briefly shows the Install button again — looking like it restarted from 0% even though the backend
// download never stopped.
const bcache: {
  catalog: CatalogEntry[]; catMeta: { source: string; verified?: string }; revision: number
  verdicts: FitVerdict[]; downloads: DownloadState[]; installed: Set<string>; ctx: number; kv: boolean
} = { catalog: [], catMeta: { source: '' }, revision: 0, verdicts: [], downloads: [], installed: new Set(), ctx: 4096, kv: false }

export default function Browser({ machine, goLibrary }: { machine: MachineProfile | null; goLibrary: () => void }) {
  const [catalog, setCatalog] = useState<CatalogEntry[]>(bcache.catalog)
  const [catMeta, setCatMeta] = useState<{ source: string; verified?: string }>(bcache.catMeta)
  const [verdicts, setVerdicts] = useState<FitVerdict[]>(bcache.verdicts)
  const [ctx, setCtx] = useState(bcache.ctx)
  const [kv, setKv] = useState(bcache.kv)
  const [openQ, setOpenQ] = useState<string | null>(null)
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const [loading, setLoading] = useState(bcache.catalog.length === 0)
  const [discovering, setDiscovering] = useState(false)
  const [downloads, setDownloads] = useState<DownloadState[]>(bcache.downloads)
  const [installed, setInstalled] = useState<Set<string>>(bcache.installed)
  const [reloadNonce, setReloadNonce] = useState(0)

  const loadedRevRef = useRef(bcache.revision)
  const load = async () => {
    if (catalog.length === 0) setLoading(true)
    const [c, v] = await Promise.all([api.catalog(), api.verdicts(ctx, kv ? 1 : 2)])
    if (c.ok && c.data) { bcache.catalog = c.data.entries; bcache.catMeta = { source: c.data.source, verified: c.data.verifiedSignature }; bcache.revision = c.data.revision; loadedRevRef.current = c.data.revision; setCatalog(bcache.catalog); setCatMeta(bcache.catMeta) }
    if (v.ok && v.data) { bcache.verdicts = v.data; setVerdicts(v.data) }
    setLoading(false)
  }
  // reloadNonce lets the (mount-only) poll below request a reload that runs with the CURRENT ctx/kv,
  // rather than a stale closure — otherwise a discovery-completion reload could overwrite verdicts
  // with ones computed for the fit knobs that were selected when the interval was created.
  useEffect(() => { bcache.ctx = ctx; bcache.kv = kv; load() }, [ctx, kv, reloadNonce])

  const pollDownloads = async () => { const d = await api.downloads(); if (d.ok && d.data) { bcache.downloads = d.data; setDownloads(d.data) } }
  // Which model|quant are actually installed — the source of truth for "In library", NOT a stale
  // completed download row (which survives a delete and would otherwise suppress re-install).
  const pollLibrary = async () => {
    const l = await api.library()
    if (l.ok && l.data) { const s = new Set(l.data.map(m => `${m.modelId}|${m.quantLabel}`)); bcache.installed = s; setInstalled(s) }
  }
  // Poll the background catalog-discovery status (CAT-7), in-flight downloads, and the library, so
  // the page can show "finding the best models…" while discovery runs, live progress on each
  // install, and an accurate installed state.
  const pollStatus = async () => {
    const s = await api.catalogStatus()
    if (!s.ok || !s.data) return
    setDiscovering(!!s.data.discovering)
    // Reload catalog + verdicts (with the CURRENT fit knobs) whenever a newer revision has landed.
    // Comparing the revision — rather than watching the discovering flag go true→false — is robust
    // to a discovery pass that starts and finishes between (or before) our poll ticks.
    if (s.data.revision != null && s.data.revision > loadedRevRef.current) {
      loadedRevRef.current = s.data.revision // optimistic: avoid re-bumping before load() catches up
      setReloadNonce(n => n + 1)
    }
  }
  useEffect(() => {
    pollDownloads(); pollLibrary(); pollStatus() // run once immediately, then on an interval
    const iv = setInterval(() => { pollStatus(); pollDownloads(); pollLibrary() }, 1200)
    return () => clearInterval(iv)
  }, [])

  const vmap = useMemo(() => { const m = new Map<string, FitVerdict>(); for (const v of verdicts) m.set(`${v.modelId}|${v.quantLabel}`, v); return m }, [verdicts])

  const score = (e: CatalogEntry) => Math.max(-1, ...e.quants.map(q => { const v = vmap.get(`${e.id}|${q.label}`); if (!v) return -1; return (99 - (order[v.verdict] ?? 99)) * 1e15 + q.bytes }))
  const sorted = useMemo(() => [...catalog].sort((a, b) => score(b) - score(a)), [catalog, vmap])
  const lead = sorted[0]
  const rest = sorted.slice(1)

  const gpu = machine?.gpus?.[0]
  const vramAvail = gpu ? g(Math.max(0, gpu.telemetry.vramFreeBytes - Math.max(1024 ** 3, gpu.totalVramBytes * 0.1))) : '0'
  const ctxLabel = ctx >= 1024 ? `${(ctx / 1024).toFixed(0)}k` : `${ctx}`

  // Only *live* downloads (in-flight / paused) drive the inline progress on a card. Rows are
  // newest-first, so keep the first such per key. Terminal rows are skipped: cancelled/failed let
  // the card fall back to Install (retry), and "completed" is intentionally NOT here — installed
  // state comes from the library (a completed row survives a delete and would block re-install).
  const dlMap = useMemo(() => {
    const m = new Map<string, DownloadState>()
    const shown = (s: string) => s === 'active' || s === 'queued' || s === 'paused'
    for (const d of downloads) { const k = `${d.modelId}|${d.quantLabel}`; if (!m.has(k) && shown(d.status)) m.set(k, d) }
    return m
  }, [downloads])

  const download = async (entry: CatalogEntry, q: Quant) => {
    const key = `${entry.id}|${q.label}`
    setBusyKey(key)
    const r = await api.startDownload({ modelId: entry.id, quantLabel: q.label })
    setBusyKey(null)
    if (!r.ok) { alert('Download refused: ' + (r.error || 'unknown')); return }
    pollDownloads() // surface progress inline immediately; the poll keeps it live
  }

  const dlActions: DlActions = {
    onPause: async (id) => { await api.pauseDownload(id); pollDownloads() },
    onResume: async (id) => { await api.resumeDownload(id); pollDownloads() },
    onCancel: async (id) => { await api.cancelDownload(id); pollDownloads() },
    goLibrary,
  }

  return (
    <div className="cinner">
      <div className="pagehead">
        <div>
          <p className="eyebrow">{catMeta.source === 'huggingface' ? 'Catalog · live from Hugging Face · checksum-pinned' : catMeta.verified === 'verified' ? 'Catalog · bundled · signed &amp; verified' : 'Catalog'}</p>
          <h1 className="ptitle">What actually fits</h1>
          <p className="psub">Every quant carries an honest verdict for <span className="iris">this</span> GPU — weights + KV cache + compute buffers vs. real free VRAM. Not <span className="mono">file_size &lt; VRAM</span>.</p>
        </div>
      </div>

      <div className="ctrlbar">
        <div className="ctrl">
          <span className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>CONTEXT</span>
          <div className="seg">{CTX.map(c => <button key={c} className={`segb ${ctx === c ? 'on' : ''}`} onClick={() => setCtx(c)}>{c >= 1024 ? `${c / 1024}k` : c}</button>)}</div>
        </div>
        <div className="ctrl">
          <span className="mono" style={{ fontSize: 12, color: 'var(--muted)' }}>q8_0 KV CACHE</span>
          <div className={`tgl ${kv ? 'on' : ''}`} onClick={() => setKv(v => !v)} />
          <span className="mono faint" style={{ fontSize: 11 }}>{kv ? 'halves KV' : 'f16 default'}</span>
        </div>
        <div style={{ flex: 1 }} />
        <span className="mono faint" style={{ fontSize: 11 }}>VRAM_avail {vramAvail} GB · headroom 1.0 GB</span>
      </div>

      {(discovering || (loading && catalog.length === 0)) && (
        <div className="findbar">
          <span className="findspin" />
          <span>{discovering
            ? 'Finding the best models for your GPU — discovering from Hugging Face…'
            : 'Computing fit for your GPU…'}</span>
        </div>
      )}

      {lead && <ModelCard entry={lead} vmap={vmap} ctxLabel={ctxLabel} vramAvail={vramAvail} lead openQ={openQ} setOpenQ={setOpenQ} download={download} busyKey={busyKey} dlMap={dlMap} dlActions={dlActions} installed={installed} />}
      <div style={{ marginTop: 22 }}>
        {rest.map(m => <ModelCard key={m.id} entry={m} vmap={vmap} ctxLabel={ctxLabel} vramAvail={vramAvail} openQ={openQ} setOpenQ={setOpenQ} download={download} busyKey={busyKey} dlMap={dlMap} dlActions={dlActions} installed={installed} />)}
      </div>
    </div>
  )
}
