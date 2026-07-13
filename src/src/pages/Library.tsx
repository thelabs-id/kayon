import { useEffect, useState } from 'react'
import { api, type InstalledModel, type DownloadState, type FitVerdict } from '../lib/api'
import Modal from '../components/Modal'
import ProgressBar from '../components/ProgressBar'
import VerdictBadge from '../components/VerdictBadge'

function fmt(b: number): string { return b < 1024**3 ? `${(b/1024**2).toFixed(0)} MB` : `${(b/1024**3).toFixed(2)} GB` }

export default function Library() {
  const [models, setModels] = useState<InstalledModel[]>([])
  const [downloads, setDownloads] = useState<DownloadState[]>([])
  const [verdicts, setVerdicts] = useState<Record<string, FitVerdict>>({})
  const [confirm, setConfirm] = useState<InstalledModel|null>(null)
  const [busy, setBusy] = useState(false)

  const load = async () => {
    const [m,d] = await Promise.all([api.library(), api.downloads()])
    if (m.ok && m.data) {
      setModels(m.data)
      // FIT-1/LIB-2: exact local verdict read from each installed GGUF on disk.
      for (const model of m.data) {
        api.localVerdict(model.id, 4096, 2).then(r => {
          if (r.ok && r.data) setVerdicts(v => ({ ...v, [model.id]: r.data! }))
        })
      }
    }
    if (d.ok && d.data) setDownloads(d.data)
  }
  useEffect(() => { load(); const iv = setInterval(load, 2000); return () => clearInterval(iv) }, [])

  const active = downloads.filter(d => d.status === 'active' || d.status === 'queued')

  const del = async () => {
    if (!confirm) return
    setBusy(true)
    await api.deleteModel(confirm.id)
    setBusy(false)
    setConfirm(null)
    load()
  }

  const loadModel = async (m: InstalledModel) => {
    const r = await api.runtimeStart({ modelPath:m.path, modelId:m.modelId, quantLabel:m.quantLabel, nGpuLayers:999, contextLength:2048, runtimeArgs:[] })
    if (!r.ok) alert('Failed: ' + r.error)
    else window.location.hash = ''
  }

  return (
    <div>
      <h1 className="page-title">Library</h1>
      {active.length > 0 && (
        <div className="card">
          <div className="card-title" style={{marginBottom:12}}>Active Downloads</div>
          {active.map(d => (
            <div key={d.id} style={{marginBottom:12}}>
              <div style={{display:'flex',justifyContent:'space-between',marginBottom:4}}>
                <span className="font-medium">{d.modelId} <span className="text-muted">({d.quantLabel})</span></span>
                <span className="mono text-xs text-muted">{fmt(d.receivedBytes)} / {fmt(d.totalBytes)}</span>
              </div>
              <ProgressBar percent={d.totalBytes>0?(d.receivedBytes/d.totalBytes*100):0} throughputBps={d.throughputBps} etaSeconds={d.etaSeconds} status={d.status}/>
            </div>
          ))}
        </div>
      )}

      {models.length === 0 ? (
        <div className="card"><div className="empty-state">
          <div className="empty-state-title">No models installed yet</div>
          <div className="empty-state-text">Visit the Browser to find models, or adopt from Ollama in Settings.</div>
        </div></div>
      ) : (
        <div className="card">
          <table className="table">
            <thead><tr><th>Model</th><th>Quant</th><th>Source</th><th>Size</th><th>Fit (local)</th><th>Checksum</th><th/></tr></thead>
            <tbody>
              {models.map(m => (
                <tr key={m.id}>
                  <td className="font-medium">{m.modelId}</td>
                  <td><span className="quant-chip">{m.quantLabel}</span></td>
                  <td><span className={`badge ${m.source==='adopted'?'badge-accent':'badge-info'}`}>{m.source}</span></td>
                  <td className="mono">{fmt(m.bytes)}</td>
                  <td>{verdicts[m.id]
                    ? <VerdictBadge verdict={verdicts[m.id].verdict} showTooltip explainability={verdicts[m.id].explainability} nGpuLayers={verdicts[m.id].nGpuLayers}/>
                    : <span className="text-xs text-muted">…</span>}</td>
                  <td className="mono text-xs">{m.sha256.substring(0,16)}...</td>
                  <td style={{display:'flex',gap:6}}>
                    <button className="btn btn-primary btn-sm" onClick={()=>loadModel(m)}>Load & Chat</button>
                    <button className="btn btn-ghost btn-sm" onClick={()=>setConfirm(m)} title="Delete">Del</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <Modal open={!!confirm} onClose={()=>setConfirm(null)} title="Confirm Delete"
        actions={<button className="btn btn-danger btn-sm" disabled={busy} onClick={del}>{busy?'Deleting...':'Yes, delete'}</button>}>
        {confirm && <div>
          Delete <strong>{confirm.modelId}</strong> ({confirm.quantLabel})?<br/><br/>
          {confirm.source==='adopted'
            ? 'This removes the hard link. Ollama\'s blob is preserved.'
            : 'The file will be deleted. Re-download to use again.'}
        </div>}
      </Modal>
    </div>
  )
}
