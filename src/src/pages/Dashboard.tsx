import { useEffect, useState } from 'react'
import { api, type MachineProfile } from '../lib/api'

function fmt(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024**2) return `${(bytes/1024).toFixed(1)} KB`
  if (bytes < 1024**3) return `${(bytes/1024**2).toFixed(1)} MB`
  return `${(bytes/1024**3).toFixed(2)} GB`
}

function Meter({ label, used, total }: { label: string; used: number; total: number }) {
  const pct = total > 0 ? (used / total * 100) : 0
  const color = pct > 80 ? 'var(--danger)' : pct > 50 ? 'var(--warning)' : 'var(--gpu-green)'
  return (
    <div className="meter">
      <div className="meter-header">
        <span className="meter-label">{label}</span>
        <span className="meter-value mono">{fmt(used)} / {fmt(total)}</span>
      </div>
      <div className="meter-bar">
        <div className="meter-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [machine, setMachine] = useState<MachineProfile | null>(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    const load = () => api.hardware().then(r => {
      if (r.ok && r.data) setMachine(r.data)
      else setErr(r.error || 'No GPU')
    }).catch(e => setErr(e.message))
    load()
    const iv = setInterval(load, 1000) // HW-2/HW-4: 1 Hz telemetry
    return () => clearInterval(iv)
  }, [])

  const gpu = machine?.gpus?.[0]

  return (
    <div>
      <h1 className="page-title">Dashboard</h1>
      {err && !machine ? (
        <div className="card"><div className="empty-state">
          <div className="empty-state-icon">!</div>
          <div className="empty-state-title">No supported GPU</div>
          <div className="empty-state-text">{err}. Verdicts fall back to RAM-only.</div>
        </div></div>
      ) : !machine ? (
        <div className="card"><div className="empty-state">
          <div className="empty-state-title">Probing hardware...</div>
        </div></div>
      ) : (
        <>
          {gpu ? (
            <div className="card">
              <div className="card-header">
                <div>
                  <div className="card-title">{gpu.name}</div>
                  <div className="text-sm text-muted" style={{marginTop:4}}>
                    {gpu.architecture || 'Unknown'} | Compute {gpu.computeCapability || '?'} | Driver {gpu.driverVersion || '?'} | CUDA {gpu.cudaVersion || '?'}
                  </div>
                </div>
              </div>
              <div style={{display:'grid', gridTemplateColumns:'160px 1fr', gap:24, alignItems:'center'}}>
                <div style={{textAlign:'center'}}>
                  <svg width="140" height="140" viewBox="0 0 140 140">
                    <circle cx="70" cy="70" r="55" fill="none" stroke="var(--border)" strokeWidth="10"/>
                    <circle cx="70" cy="70" r="55" fill="none" stroke="var(--accent)" strokeWidth="10"
                      strokeDasharray={2*Math.PI*55}
                      strokeDashoffset={2*Math.PI*55*(1 - gpu.telemetry.vramUsedBytes/gpu.totalVramBytes)}
                      transform="rotate(-90 70 70)" strokeLinecap="round"
                      style={{transition:'stroke-dashoffset 0.5s'}}/>
                    <text x="70" y="68" textAnchor="middle" fontSize="24" fontWeight="600" fill="var(--text)" fontFamily="var(--font-mono)">
                      {((gpu.telemetry.vramUsedBytes/gpu.totalVramBytes)*100).toFixed(0)}%
                    </text>
                    <text x="70" y="86" textAnchor="middle" fontSize="10" fill="var(--text-muted)">VRAM</text>
                  </svg>
                </div>
                <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:16}}>
                  <div><span className="text-sm text-muted">Utilization</span><div className="mono" style={{fontSize:18,fontWeight:600}}>{gpu.telemetry.utilizationPercent.toFixed(1)}%</div></div>
                  <div><span className="text-sm text-muted">Temperature</span><div className="mono" style={{fontSize:18,fontWeight:600}}>{gpu.telemetry.temperatureC.toFixed(0)}C</div></div>
                  <div><span className="text-sm text-muted">Power</span><div className="mono" style={{fontSize:18,fontWeight:600}}>{gpu.telemetry.powerWatts.toFixed(1)} W</div></div>
                  <div><span className="text-sm text-muted">Core Clock</span><div className="mono" style={{fontSize:18,fontWeight:600}}>{gpu.telemetry.coreClockMhz} MHz</div></div>
                  <div><span className="text-sm text-muted">Mem Clock</span><div className="mono" style={{fontSize:18,fontWeight:600}}>{gpu.telemetry.memClockMhz} MHz</div></div>
                  <div><span className="text-sm text-muted">VRAM Free</span><div className="mono" style={{fontSize:18,fontWeight:600}}>{fmt(gpu.telemetry.vramFreeBytes)}</div></div>
                </div>
              </div>
            </div>
          ) : (
            <div className="card"><div className="empty-state">
              <div className="empty-state-title">No GPU detected</div>
              <div className="empty-state-text">Verdicts will be RAM-based.</div>
            </div></div>
          )}
          <div className="grid grid-3">
            <div className="card">
              <div className="card-title">CPU</div>
              <div className="text-sm text-muted" style={{margin:'8px 0'}}>{machine.cpu.brand}</div>
              <div className="meter">
                <div className="meter-header">
                  <span className="meter-label">Utilization</span>
                  <span className="meter-value mono">{machine.cpu.usagePercent.toFixed(0)}%</span>
                </div>
                <div className="meter-bar">
                  <div className="meter-fill" style={{
                    width: `${Math.min(100, machine.cpu.usagePercent)}%`,
                    background: machine.cpu.usagePercent > 80 ? 'var(--danger)' : machine.cpu.usagePercent > 50 ? 'var(--warning)' : 'var(--gpu-green)'
                  }}/>
                </div>
              </div>
              <div className="text-xs text-muted" style={{marginTop:8}}>{machine.cpu.threadCount} threads | {machine.cpu.frequencyMhz} MHz</div>
            </div>
            <div className="card">
              <div className="card-title">Memory</div>
              <div style={{marginTop:12}}><Meter label="RAM" used={machine.ram.usedBytes} total={machine.ram.totalBytes}/></div>
            </div>
            <div className="card">
              <div className="card-title">OS</div>
              <div className="text-sm" style={{marginTop:8}}>{machine.os.name} {machine.os.version}</div>
              <div className="text-xs text-muted" style={{marginTop:4}}>Host: {machine.os.hostName}</div>
            </div>
          </div>
          <div className="card">
            <div className="card-title" style={{marginBottom:12}}>Storage</div>
            {machine.disks.map((d,i) => (
              <div key={i} style={{marginBottom:12}}>
                <Meter label={`${d.mount} (${d.kind})`} used={d.totalBytes - d.freeBytes} total={d.totalBytes}/>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
