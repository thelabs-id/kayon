import { useEffect, useState } from 'react'
import { api, type NetworkLogEntry, type TelemetryStatus } from '../lib/api'

export default function Privacy() {
  const [tele, setTele] = useState<TelemetryStatus | null>(null)
  const [netlog, setNetlog] = useState<NetworkLogEntry[]>([])
  const [payload, setPayload] = useState<{ endpoint: string; payload: string; byteSize: number } | null>(null)
  const [pending, setPending] = useState(false) // showing payload, awaiting confirm to enable

  const load = async () => {
    const [t, n] = await Promise.all([api.telemetryStatus(), api.networkLog()])
    if (t.ok && t.data) setTele(t.data)
    if (n.ok && n.data) setNetlog(n.data)
  }
  useEffect(() => { load(); const iv = setInterval(load, 2000); return () => clearInterval(iv) }, [])

  const toggle = async () => {
    if (!tele) return
    if (tele.enabled) { const r = await api.telemetryToggle(false); if (r.ok && r.data) setTele(r.data); setPending(false); return }
    const r = await api.telemetryPreview()
    if (r.ok && r.data) { setPayload(r.data); setPending(true) }
  }
  const confirmSend = async () => { const r = await api.telemetryToggle(true); if (r.ok && r.data) setTele(r.data); setPending(false) }
  const keepOff = () => setPending(false)

  const enabled = tele?.enabled || pending
  const teleCount = netlog.filter(n => n.purpose === 'telemetry').length

  return (
    <div className="cinner">
      <div className="pagehead">
        <div>
          <p className="eyebrow">Private by construction</p>
          <h1 className="ptitle">Network &amp; telemetry</h1>
          <p className="psub">Every outbound request passes through one instrumented client and lands in this log. Nothing else can open a socket. Telemetry is off until you turn it on — and you see the exact bytes first.</p>
        </div>
      </div>

      <div className="panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 20, marginBottom: 20, flexWrap: 'wrap' }}>
        <div className="fx ac gap12">
          <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 3l7 3v5c0 4.5-3 8.5-7 10-4-1.5-7-5.5-7-10V6z" /></svg>
          <div>
            <div style={{ fontWeight: 600 }}>Anonymous telemetry</div>
            <div className="mono faint" style={{ fontSize: 12, marginTop: 2 }}>{tele?.enabled ? 'enabled — sends session events' : 'off by default · nothing leaves the machine'}</div>
          </div>
        </div>
        <div className="fx ac gap12">
          <span className="mono faint" style={{ fontSize: 11 }}>OFF</span>
          <div className={`tgl ${enabled ? 'on' : ''}`} onClick={toggle} />
          <span className="mono faint" style={{ fontSize: 11 }}>ON</span>
        </div>
      </div>

      {pending && payload && (
        <div className="panel" style={{ marginBottom: 20, borderColor: 'color-mix(in oklab, var(--amber) 40%, var(--line))' }}>
          <div className="fx ac jb" style={{ marginBottom: 12 }}>
            <span className="mkey" style={{ color: 'var(--amber)' }}>Literal payload · this is exactly what would be sent</span>
            <span className="tag" style={{ color: 'var(--amber)', borderColor: 'color-mix(in oklab, var(--amber) 40%, var(--line2))' }}>held — not sent</span>
          </div>
          <div className="payload">{payload.payload}</div>
          <div className="fx gap12" style={{ marginTop: 14 }}>
            <button className="btn btn-iris btn-sm" onClick={confirmSend}>Confirm &amp; enable telemetry</button>
            <button className="btn btn-line btn-sm" onClick={keepOff}>Keep it off</button>
          </div>
        </div>
      )}

      <div className="fx ac jb" style={{ marginBottom: 12 }}>
        <span className="mono" style={{ fontSize: 11, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--faint)' }}>Outbound network log · session</span>
        <span className="mono faint" style={{ fontSize: 11 }}>{netlog.length} requests · {teleCount} telemetry</span>
      </div>
      <div className="netlog">
        <div className="nlhead"><span>Time</span><span>Method</span><span>URL</span><span>Purpose</span><span>Bytes</span></div>
        {netlog.length === 0 && <div className="nlrow"><span className="faint" style={{ gridColumn: '1 / -1' }}>No outbound requests yet.</span></div>}
        {netlog.slice(0, 40).map(n => (
          <div key={n.id} className="nlrow">
            <span className="faint">{new Date(n.ts).toLocaleTimeString()}</span>
            <span>{n.method}</span>
            <span style={{ color: 'var(--ink)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{n.url}</span>
            <span className="purpose">{n.purpose}</span>
            <span className="muted">{n.bytesIn > 0 ? `${(n.bytesIn / 1024).toFixed(0)}k` : '—'}</span>
          </div>
        ))}
      </div>
      <div className="mono faint" style={{ fontSize: 11, marginTop: 14, lineHeight: 1.6 }}>Local IPC to <span style={{ color: 'var(--muted)' }}>127.0.0.1</span> (the llama-server sidecar) is not egress and never appears here — it never leaves the machine.</div>
    </div>
  )
}
