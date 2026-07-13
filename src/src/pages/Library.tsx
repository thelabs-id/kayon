import { useEffect, useState } from 'react'
import { api, type InstalledModel, type DownloadState, type FitVerdict, type OllamaModel } from '../lib/api'
import { VerdictChip } from '../components/icons'

const fmt = (b: number) => b < 1024 ** 3 ? `${(b / 1024 ** 2).toFixed(0)} MB` : `${(b / 1024 ** 3).toFixed(1)} GB`

export default function Library({ goBrowser, goPrivacy, onChange, goChat }: {
  goBrowser: () => void; goPrivacy: () => void; onChange: () => void; goChat: () => void
}) {
  const [models, setModels] = useState<InstalledModel[]>([])
  const [downloads, setDownloads] = useState<DownloadState[]>([])
  const [verdicts, setVerdicts] = useState<Record<string, FitVerdict>>({})
  const [ollama, setOllama] = useState<OllamaModel[]>([])
  const [confirmDel, setConfirmDel] = useState<string | null>(null)

  const load = async () => {
    const [m, d, o] = await Promise.all([api.library(), api.downloads(), api.ollamaModels()])
    if (m.ok && m.data) {
      setModels(m.data)
      for (const model of m.data) api.localVerdict(model.id, 4096, 2).then(r => { if (r.ok && r.data) setVerdicts(v => ({ ...v, [model.id]: r.data! })) })
    }
    if (d.ok && d.data) setDownloads(d.data)
    if (o.ok && o.data) setOllama(o.data)
  }
  useEffect(() => { load(); const iv = setInterval(load, 1500); return () => clearInterval(iv) }, [])

  const active = downloads.filter(d => d.status === 'active' || d.status === 'queued')
  const totalBytes = models.reduce((s, m) => s + m.bytes, 0)
  const adoptable = ollama.filter(o => o.adoptable)

  const loadModel = async (m: InstalledModel) => {
    const r = await api.runtimeLoad(m.id, 4096, 2)
    if (!r.ok) alert('Failed: ' + r.error)
    else goChat()
  }
  const del = async (id: string) => {
    if (confirmDel !== id) { setConfirmDel(id); setTimeout(() => setConfirmDel(c => c === id ? null : c), 3000); return }
    await api.deleteModel(id); setConfirmDel(null); onChange(); load()
  }
  const adopt = async (m: OllamaModel) => {
    let mode: 'copy' | undefined
    if (!m.sameVolumeAsLibrary) { if (!window.confirm(`${m.name}:${m.tag} is on a different drive. Copy it into the library (${fmt(m.sizeBytes)})?`)) return; mode = 'copy' }
    const r = await api.ollamaAdopt({ name: m.name, tag: m.tag, mode })
    if (!r.ok) alert('Adopt failed: ' + r.error)
    onChange(); load()
  }

  return (
    <div className="cinner">
      <div className="pagehead">
        <div>
          <p className="eyebrow">Library · downloaded + adopted</p>
          <h1 className="ptitle">Your models</h1>
          <p className="psub">One place for everything installed — app downloads and Ollama models adopted in place by hard link. Zero bytes re-downloaded.</p>
        </div>
        <button className="btn btn-line btn-sm" onClick={goBrowser}>+ Browse catalog</button>
      </div>

      {active.map(d => (
        <div key={d.id} className="dlcard">
          <div className="fx ac jb">
            <div className="fx ac gap12">
              <span className="livedot" style={{ background: 'var(--iris)' }} />
              <div>
                <div style={{ fontWeight: 600, fontSize: 14.5 }}>{d.modelId} · {d.quantLabel}</div>
                <div className="mono faint" style={{ fontSize: 11.5, marginTop: 2 }}>downloading · resumable · SHA-256 verified on completion</div>
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div className="mono" style={{ fontSize: 13, color: 'var(--iris)' }}>{d.totalBytes > 0 ? ((d.receivedBytes / d.totalBytes) * 100).toFixed(0) : 0}%</div>
              <div className="mono faint" style={{ fontSize: 11 }}>{(d.throughputBps / 1024 ** 2).toFixed(1)} MB/s{d.etaSeconds != null ? ` · ${d.etaSeconds}s` : ''}</div>
            </div>
          </div>
          <div className="dlbar"><div className="dlfill" style={{ width: `${d.totalBytes > 0 ? (d.receivedBytes / d.totalBytes) * 100 : 0}%` }} /></div>
        </div>
      ))}

      <div style={{ marginBottom: 10 }} className="fx ac jb">
        <span className="mono" style={{ fontSize: 11, letterSpacing: '.1em', textTransform: 'uppercase', color: 'var(--faint)' }}>Installed · {models.length} models · {fmt(totalBytes)}</span>
      </div>

      {models.length === 0 && <div className="panel" style={{ textAlign: 'center', color: 'var(--muted)' }}>No models yet. Browse the catalog or adopt from Ollama below.</div>}

      {models.map(m => {
        const v = verdicts[m.id]
        return (
          <div key={m.id} className="libr">
            <div>
              <div style={{ fontWeight: 600, fontSize: 15, display: 'flex', alignItems: 'center', gap: 10 }}>
                {m.modelId}
                <span className={`srcbadge ${m.source === 'adopted' ? 'adopt' : ''}`}>{m.source}</span>
              </div>
              <div className="mono faint" style={{ fontSize: 11.5, marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{m.path}</div>
            </div>
            <span className="mono" style={{ fontSize: 12.5, color: 'var(--muted)' }}>{m.quantLabel} · {fmt(m.bytes)}</span>
            {v ? <VerdictChip v={v.verdict} /> : <span className="mono faint" style={{ fontSize: 12 }}>…</span>}
            <div className="fx gap8">
              {m.needsNewerRuntime
                ? <button className="btn btn-sm btn-line" disabled style={{ opacity: .5 }}>Needs newer runtime</button>
                : <button className="btn btn-sm btn-solid" onClick={() => loadModel(m)}>Load &amp; chat</button>}
              <button className={`btn btn-sm ${confirmDel === m.id ? 'btn-danger' : 'btn-line'}`} onClick={() => del(m.id)}>{confirmDel === m.id ? 'Confirm delete' : 'Delete'}</button>
            </div>
          </div>
        )
      })}

      <div className="panel" style={{ marginTop: 26, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 20, flexWrap: 'wrap' }}>
        <div className="fx ac gap12">
          <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="var(--amber)" strokeWidth="1.8"><circle cx="12" cy="12" r="9" /><path d="M8 12h8M12 8v8" /></svg>
          <div>
            <div style={{ fontWeight: 600 }}>{ollama.length > 0 ? `Ollama detected · ${ollama.length} models, ${adoptable.length} adoptable` : 'Ollama not detected'}</div>
            <div className="mono faint" style={{ fontSize: 12, marginTop: 2 }}>%USERPROFILE%\.ollama\models · hard-link ready</div>
          </div>
        </div>
        <button className="btn btn-line btn-sm" onClick={goPrivacy}>Manage adoption →</button>
      </div>

      {ollama.length > 0 && (
        <div style={{ marginTop: 16 }}>
          {ollama.map(o => (
            <div key={`${o.name}:${o.tag}`} className="libr" style={{ marginTop: 12 }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: 15, display: 'flex', alignItems: 'center', gap: 10 }}>{o.name}:{o.tag}<span className="srcbadge adopt">ollama</span></div>
                <div className="mono faint" style={{ fontSize: 11.5, marginTop: 4 }}>{o.architecture ?? 'gguf'} · {o.sameVolumeAsLibrary ? 'same volume ✓' : 'cross-volume'}</div>
              </div>
              <span className="mono" style={{ fontSize: 12.5, color: 'var(--muted)' }}>{fmt(o.sizeBytes)}</span>
              <span className="mono faint" style={{ fontSize: 11 }}>{o.needsNewerRuntime ? 'needs newer runtime' : o.adoptable ? 'ready' : (o.adoptReason ?? '')}</span>
              <div className="fx gap8"><button className="btn btn-sm btn-line" disabled={!o.adoptable} onClick={() => adopt(o)}>Adopt (hard link)</button></div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
