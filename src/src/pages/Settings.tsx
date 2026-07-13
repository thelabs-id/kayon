import { useEffect, useState } from 'react'
import { api, type OllamaModel, type NetworkLogEntry, type TelemetryStatus } from '../lib/api'

function fmt(b: number): string { return b < 1024**3 ? `${(b/1024**2).toFixed(0)} MB` : `${(b/1024**3).toFixed(2)} GB` }

export default function Settings() {
  const [ollamaModels, setOllama] = useState<OllamaModel[]>([])
  const [netLog, setNetLog] = useState<NetworkLogEntry[]>([])
  const [telemetry, setTelemetry] = useState<TelemetryStatus|null>(null)
  const [preview, setPreview] = useState<{endpoint:string;payload:string;byteSize:number}|null>(null)
  const [adopting, setAdopting] = useState<string|null>(null)

  const load = async () => {
    const [o,n,t] = await Promise.all([api.ollamaModels(), api.networkLog(), api.telemetryStatus()])
    if (o.ok && o.data) setOllama(o.data)
    if (n.ok && n.data) setNetLog(n.data)
    if (t.ok && t.data) setTelemetry(t.data)
  }
  useEffect(() => { load() }, [])

  const toggle = async () => { if (!telemetry) return; const r = await api.telemetryToggle(!telemetry.enabled); if (r.ok && r.data) setTelemetry(r.data) }
  const showPreview = async () => { const r = await api.telemetryPreview(); if (r.ok && r.data) setPreview(r.data) }

  const adopt = async (m: OllamaModel) => {
    setAdopting(`${m.name}:${m.tag}`)
    const r = await api.ollamaAdopt({ name:m.name, tag:m.tag, digest:m.digest, blobPath:m.blobPath, sizeBytes:m.sizeBytes })
    setAdopting(null)
    if (!r.ok) alert('Adopt failed: '+r.error)
    load()
  }

  return (
    <div>
      <h1 className="page-title">Settings</h1>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Privacy & Telemetry</span>
          <span className={`badge ${telemetry?.enabled?'badge-warning':'badge-success'}`}>{telemetry?.enabled?'ENABLED':'OFF (default)'}</span>
        </div>
        <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:'8px 0'}}>
          <div>
            <div className="font-medium">Send anonymous telemetry</div>
            <div className="text-sm text-muted" style={{marginTop:4}}>Off by default. Literal payload shown before anything leaves.</div>
          </div>
          <button className={`toggle ${telemetry?.enabled?'active':''}`} onClick={toggle}/>
        </div>
        {telemetry?.enabled && <div style={{marginTop:12}}>
          <button className="btn btn-secondary btn-sm" onClick={showPreview}>Show literal payload</button>
          {preview && <pre className="mono" style={{fontSize:11,overflow:'auto',margin:'8px 0 0',padding:8,background:'#0a0a0a',color:'#a8e6a8',borderRadius:4}}>
            {preview.payload}
          </pre>}
        </div>}
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Adopt from Ollama</span>
          <span className="badge badge-neutral">{ollamaModels.length} detected</span>
        </div>
        <div className="text-sm text-muted" style={{marginBottom:12}}>NTFS hard links — zero bytes copied.</div>
        {ollamaModels.length === 0 ? (
          <div className="text-sm text-muted">No Ollama models found. Install Ollama and pull models.</div>
        ) : (
          <table className="table">
            <thead><tr><th>Model</th><th>Size</th><th>Status</th><th/></tr></thead>
            <tbody>
              {ollamaModels.map(m => (
                <tr key={`${m.name}:${m.tag}`}>
                  <td className="font-medium">{m.name}:{m.tag}</td>
                  <td className="mono">{fmt(m.sizeBytes)}</td>
                  <td>{m.adoptable
                    ? (m.needsNewerRuntime
                        ? <span className="badge badge-warning" title={m.adoptReason||''}>Needs newer runtime</span>
                        : <span className="badge badge-success">Ready</span>)
                    : <span className="badge badge-warning" title={m.adoptReason||''}>{m.adoptReason||'Not adoptable'}</span>}</td>
                  <td><button className="btn btn-primary btn-sm" disabled={!m.adoptable||adopting===`${m.name}:${m.tag}`} onClick={()=>adopt(m)}>{adopting===`${m.name}:${m.tag}`?'Adopting...':'Adopt'}</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <span className="card-title">Network Log</span>
          <button className="btn btn-ghost btn-sm" onClick={load}>Refresh</button>
        </div>
        <div className="text-sm text-muted" style={{marginBottom:12}}>Every outbound request. Local IPC not shown.</div>
        {netLog.length === 0 ? <div className="text-sm text-muted">No requests yet.</div> : (
          <table className="table">
            <thead><tr><th>Time</th><th>Method</th><th>URL</th><th>Purpose</th></tr></thead>
            <tbody>
              {netLog.slice(0,30).map(n => (
                <tr key={n.id}>
                  <td className="mono text-xs">{new Date(n.ts).toLocaleTimeString()}</td>
                  <td><span className="badge badge-info">{n.method}</span></td>
                  <td className="mono text-xs" style={{maxWidth:400,overflow:'hidden',textOverflow:'ellipsis'}}>{n.url}</td>
                  <td>{n.purpose}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <div className="card-title">About</div>
        <div style={{marginTop:8}}>
          <div className="font-medium">Kayon v0.1.0</div>
          <div className="text-sm text-muted" style={{marginTop:4}}>Honest, private, local LLM workstation. Windows + NVIDIA. NVML probe. ed25519-signed catalog. llama.cpp runtime. Ollama adoption.</div>
        </div>
      </div>
    </div>
  )
}
