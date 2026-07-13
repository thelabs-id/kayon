import { useEffect, useMemo, useState } from 'react'
import { api, type CatalogEntry, type FitVerdict } from '../lib/api'
import VerdictBadge from '../components/VerdictBadge'

function fmt(b: number): string { return b < 1024**3 ? `${(b/1024**2).toFixed(0)} MB` : `${(b/1024**3).toFixed(2)} GB` }

// A checksum is "pinned" only when it's a real 64-hex-char SHA-256 (CAT-6). Placeholders are
// not downloadable — the fit verdict still shows, but download waits on a generator run.
function isPinned(sha: string): boolean { return /^[0-9a-f]{64}$/i.test(sha.trim()) }

const order: Record<string, number> = { FITS_FULLY:0, FITS_TIGHT:1, GPU_CPU_SPLIT:2, CPU_ONLY:3, UNVERIFIED_ARCH:4, EXCEEDS_MACHINE:5 }

export default function Browser() {
  const [catalog, setCatalog] = useState<CatalogEntry[]>([])
  const [verdicts, setVerdicts] = useState<FitVerdict[]>([])
  const [ctx, setCtx] = useState(4096)
  const [kvQuant, setKvQuant] = useState(false)
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState<string|null>(null)

  const load = async () => {
    const [c,v] = await Promise.all([api.catalog(), api.verdicts(ctx, kvQuant ? 1 : 2)])
    if (c.ok && c.data) setCatalog(c.data.entries)
    if (v.ok && v.data) setVerdicts(v.data)
    setLoading(false)
  }
  useEffect(() => { load() }, [ctx, kvQuant])

  const vMap = useMemo(() => {
    const m = new Map<string, FitVerdict>()
    for (const v of verdicts) m.set(`${v.modelId}|${v.quantLabel}`, v)
    return m
  }, [verdicts])

  // Rank each model by its BEST quant (lowest order index across all quants), so "best fit leads"
  // holds even when the first-listed quant doesn't fit but a lighter one does (CAT-3).
  const bestRank = (e: CatalogEntry) =>
    Math.min(...e.quants.map(q => order[vMap.get(`${e.id}|${q.label}`)?.verdict || ''] ?? 99), 99)
  const sorted = useMemo(() => [...catalog].sort((a,b) => bestRank(a) - bestRank(b)),
    [catalog, vMap])

  const download = async (entry: CatalogEntry, q: {label:string;bytes:number;sha256:string;source:string}) => {
    setDownloading(`${entry.id}-${q.label}`)
    const r = await api.startDownload({ modelId: entry.id, quantLabel: q.label })
    if (!r.ok) alert('Download refused: ' + (r.error || 'unknown'))
    setTimeout(() => { setDownloading(null); load() }, 1000)
  }

  if (loading) return <div className="card"><div className="empty-state"><div className="empty-state-title">Loading catalog...</div></div></div>

  return (
    <div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:20}}>
        <h1 className="page-title" style={{margin:0}}>Model Browser</h1>
        <div style={{display:'flex',gap:12,alignItems:'center'}}>
          <span className="text-sm text-muted">Context:</span>
          <select className="input" style={{width:120,padding:'6px 10px'}} value={ctx} onChange={e=>setCtx(+e.target.value)}>
            {[2048,4096,8192,16384,32768].map(v=><option key={v} value={v}>{v.toLocaleString()}</option>)}
          </select>
          <label className="text-sm text-muted" style={{display:'flex',gap:6,alignItems:'center',cursor:'pointer'}} title="q8_0 KV cache ≈ halves KV memory; often flips a Tight/Split verdict (OD-1)">
            <input type="checkbox" checked={kvQuant} onChange={e=>setKvQuant(e.target.checked)}/>
            q8_0 KV
          </label>
        </div>
      </div>
      <div style={{padding:'10px 16px',background:'var(--bg-card)',borderRadius:'var(--radius-md)',fontSize:13,color:'var(--text-muted)',marginBottom:16}}>
        Best fit for your machine leads. Verdicts come from the fit engine (weights + KV + buffers + headroom), never naive file-size comparison.
      </div>
      {sorted.map(entry => {
        const isBest = entry.quants.some(q => vMap.get(`${entry.id}|${q.label}`)?.verdict === 'FITS_FULLY')
        return (
          <div key={entry.id} className="model-card" style={{marginBottom:16}}>
            <div className="model-card-header">
              <div>
                <div className="model-card-name">{entry.id}</div>
                <div className="model-card-family">{entry.family} | {entry.params} | {entry.license}</div>
              </div>
              <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
                {isBest && <span className="badge badge-accent">Best Fit</span>}
                {entry.capabilities.tools && <span className="badge badge-info">tools</span>}
                {entry.capabilities.reasoning && <span className="badge badge-info">reasoning</span>}
              </div>
            </div>
            <table className="table" style={{marginTop:12}}>
              <thead><tr><th>Quant</th><th>Size</th><th>Verdict</th><th>n_gpu_layers</th><th>Explanation</th><th/></tr></thead>
              <tbody>
                {entry.quants.map(q => {
                  const v = vMap.get(`${entry.id}|${q.label}`)
                  return (
                    <tr key={q.label}>
                      <td><span className="quant-chip">{q.label}</span></td>
                      <td className="mono">{fmt(q.bytes)}</td>
                      <td>{v && <VerdictBadge verdict={v.verdict} explainability={v.explainability} nGpuLayers={v.nGpuLayers}/>}</td>
                      <td className="mono">{v?.nGpuLayers ?? '-'}</td>
                      <td className="text-xs text-muted" style={{maxWidth:300}}>{v?.explainability}</td>
                      <td>
                        {v?.verdict === 'EXCEEDS_MACHINE' ? (
                          <span className="text-xs text-muted">Won't fit</span>
                        ) : !isPinned(q.sha256) ? (
                          <span className="text-xs text-muted" title="The catalog generator (CAT-6) has not pinned a real SHA-256 for this entry yet.">Checksum pending</span>
                        ) : (
                          <button className="btn btn-primary btn-sm" disabled={downloading===`${entry.id}-${q.label}`} onClick={()=>download(entry,q)}>
                            {downloading===`${entry.id}-${q.label}` ? 'Starting...' : 'Download'}
                          </button>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )
      })}
    </div>
  )
}
