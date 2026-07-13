import { useEffect, useMemo, useState } from 'react'
import { api, type CatalogEntry, type FitVerdict, type MachineProfile, type Quant } from '../lib/api'
import { VerdictChip } from '../components/icons'

const CTX = [2048, 4096, 8192, 16384, 32768]
const fmtB = (b: number) => b < 1024 ** 3 ? `${(b / 1024 ** 2).toFixed(0)} MB` : `${(b / 1024 ** 3).toFixed(1)} GB`
const g = (b: number) => (b / 1024 ** 3).toFixed(1)
const order: Record<string, number> = { FITS_FULLY: 0, FITS_TIGHT: 1, GPU_CPU_SPLIT: 2, CPU_ONLY: 3, UNVERIFIED_ARCH: 4, EXCEEDS_MACHINE: 5 }
const isPinned = (s: string) => /^[0-9a-f]{64}$/i.test((s || '').trim())

function QuantRow({ q, v, ctxLabel, vramAvail, open, onToggle, onDownload, busy }: {
  q: Quant; v?: FitVerdict; ctxLabel: string; vramAvail: string
  open: boolean; onToggle: () => void; onDownload: () => void; busy: boolean
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
            {v?.verdict === 'EXCEEDS_MACHINE'
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

function ModelCard({ entry, vmap, ctxLabel, vramAvail, lead, openQ, setOpenQ, download, busyKey }: {
  entry: CatalogEntry; vmap: Map<string, FitVerdict>; ctxLabel: string; vramAvail: string; lead?: boolean
  openQ: string | null; setOpenQ: (k: string | null) => void; download: (e: CatalogEntry, q: Quant) => void; busyKey: string | null
}) {
  const caps = [entry.capabilities.tools && 'tools', entry.capabilities.reasoning && 'reasoning', entry.capabilities.vision && 'vision'].filter(Boolean) as string[]
  const bestV = entry.quants.map(q => vmap.get(`${entry.id}|${q.label}`)).filter(Boolean).sort((a, b) => (order[a!.verdict] ?? 9) - (order[b!.verdict] ?? 9))[0]
  const pinnedQ = entry.quants.find(q => isPinned(q.sha256))
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
        {lead && pinnedQ
          ? <button className="btn btn-iris" onClick={() => download(entry, pinnedQ)}>Install · {fmtB(pinnedQ.bytes)}</button>
          : bestV && <span className="mono faint" style={{ fontSize: 12, whiteSpace: 'nowrap' }}>best: {bestV.verdict.replace(/_/g, ' ').toLowerCase()}</span>}
      </div>
      <div className="qtable">
        {entry.quants.map(q => {
          const key = `${entry.id}|${q.label}`
          return <QuantRow key={key} q={q} v={vmap.get(key)} ctxLabel={ctxLabel} vramAvail={vramAvail} open={openQ === key} onToggle={() => setOpenQ(openQ === key ? null : key)} onDownload={() => download(entry, q)} busy={busyKey === key} />
        })}
      </div>
    </div>
  )
}

export default function Browser({ machine, goLibrary }: { machine: MachineProfile | null; goLibrary: () => void }) {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [verdicts, setVerdicts] = useState<FitVerdict[]>([])
  const [ctx, setCtx] = useState(4096)
  const [kv, setKv] = useState(false)
  const [openQ, setOpenQ] = useState<string | null>(null)
  const [busyKey, setBusyKey] = useState<string | null>(null)

  const load = async () => {
    const [c, v] = await Promise.all([api.catalog(), api.verdicts(ctx, kv ? 1 : 2)])
    if (c.ok && c.data) setCatalog(c.data.entries)
    if (v.ok && v.data) setVerdicts(v.data)
  }
  useEffect(() => { load() }, [ctx, kv])

  const vmap = useMemo(() => { const m = new Map<string, FitVerdict>(); for (const v of verdicts) m.set(`${v.modelId}|${v.quantLabel}`, v); return m }, [verdicts])

  const score = (e: CatalogEntry) => Math.max(-1, ...e.quants.map(q => { const v = vmap.get(`${e.id}|${q.label}`); if (!v) return -1; return (99 - (order[v.verdict] ?? 99)) * 1e15 + q.bytes }))
  const sorted = useMemo(() => [...catalog].sort((a, b) => score(b) - score(a)), [catalog, vmap])
  const lead = sorted[0]
  const rest = sorted.slice(1)

  const gpu = machine?.gpus?.[0]
  const vramAvail = gpu ? g(Math.max(0, gpu.telemetry.vramFreeBytes - Math.max(1024 ** 3, gpu.totalVramBytes * 0.1))) : '0'
  const ctxLabel = ctx >= 1024 ? `${(ctx / 1024).toFixed(0)}k` : `${ctx}`

  const download = async (entry: CatalogEntry, q: Quant) => {
    const key = `${entry.id}|${q.label}`
    setBusyKey(key)
    const r = await api.startDownload({ modelId: entry.id, quantLabel: q.label })
    setBusyKey(null)
    if (!r.ok) alert('Download refused: ' + (r.error || 'unknown'))
    else goLibrary()
  }

  return (
    <div className="cinner">
      <div className="pagehead">
        <div>
          <p className="eyebrow">Catalog · signed &amp; verified</p>
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

      {lead && <ModelCard entry={lead} vmap={vmap} ctxLabel={ctxLabel} vramAvail={vramAvail} lead openQ={openQ} setOpenQ={setOpenQ} download={download} busyKey={busyKey} />}
      <div style={{ marginTop: 22 }}>
        {rest.map(m => <ModelCard key={m.id} entry={m} vmap={vmap} ctxLabel={ctxLabel} vramAvail={vramAvail} openQ={openQ} setOpenQ={setOpenQ} download={download} busyKey={busyKey} />)}
      </div>
    </div>
  )
}
