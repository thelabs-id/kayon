import { useEffect, useState } from 'react'
import { api, type Catalog, type UpdateStatus } from '../lib/api'

export default function Settings() {
  const [libDir, setLibDir] = useState('')
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [upd, setUpd] = useState<UpdateStatus | null>(null)

  useEffect(() => {
    api.libraryDir().then(r => { if (r.ok && r.data) setLibDir(r.data) })
    api.catalog().then(r => { if (r.ok && r.data) setCatalog(r.data) })
    api.updateStatus().then(r => { if (r.ok && r.data) setUpd(r.data) })
  }, [])

  // UPD-1: asking for a check is itself consent, so this runs even with the launch check off.
  const checkNow = async () => {
    setUpd(u => (u ? { ...u, checking: true } : u))
    const r = await api.updateCheck()
    if (r.ok && r.data) setUpd(r.data)
    else { const s = await api.updateStatus(); if (s.ok && s.data) setUpd(s.data) }
  }

  const toggleAuto = async () => {
    const r = await api.updateSetAuto(!upd?.autoCheck)
    if (r.ok && r.data) setUpd(r.data)
  }

  const relocate = async () => {
    const dest = window.prompt('Move the Kayon library to this folder (e.g. on your Ollama drive for zero-copy adoption):', libDir)
    if (!dest || dest === libDir) return
    const r = await api.relocateLibrary(dest)
    if (!r.ok) { alert('Relocate failed: ' + r.error); return }
    alert(`Library moved to ${r.data!.libraryDir} (${r.data!.movedFiles} file(s) migrated).`)
    setLibDir(r.data!.libraryDir)
  }

  return (
    <div className="cinner">
      <div className="pagehead">
        <div>
          <p className="eyebrow">Preferences</p>
          <h1 className="ptitle">Settings</h1>
          <p className="psub">Local, transactional, and yours — stored in a single SQLite file under your profile.</p>
        </div>
      </div>
      <div className="g2">
        <div className="panel">
          <span className="mkey">Library directory</span>
          <div className="mono" style={{ fontSize: 13, margin: '12px 0', color: 'var(--ink)', overflow: 'hidden', textOverflow: 'ellipsis' }}>{libDir || '…'}</div>
          <div className="fx gap8"><button className="btn btn-line btn-sm" onClick={relocate}>Relocate…</button><span className="mono faint" style={{ fontSize: 11, alignSelf: 'center' }}>move-in-place migration</span></div>
        </div>
        <div className="panel">
          <span className="mkey">Runtime</span>
          <div className="speclist" style={{ marginTop: 8 }}>
            <div className="specr" style={{ borderTop: 0, paddingTop: 0 }}><span className="speck">llama-server</span><span className="specv" title="Vulkan keeps the download at ~20 MB; the CUDA build would add roughly 600 MB. Slower than CUDA on NVIDIA, mostly on prompt processing.">Vulkan sidecar</span></div>
            <div className="specr"><span className="speck">KV cache</span><span className="specv">f16 (default)</span></div>
            <div className="specr"><span className="speck">Flash attention</span><span className="specv">auto</span></div>
          </div>
        </div>
        <div className="panel">
          <span className="mkey">Catalog</span>
          <div className="speclist" style={{ marginTop: 8 }}>
            <div className="specr" style={{ borderTop: 0, paddingTop: 0 }}><span className="speck">Revision</span><span className="specv">r{catalog?.revision ?? '?'}</span></div>
            <div className="specr"><span className="speck">Source</span><span className="specv" title={catalog?.source === 'huggingface' ? 'Discovered live from Hugging Face; every quant checksum-pinned to HF’s published SHA-256 (CAT-7).' : 'Bundled catalog, ed25519-signed and verified against the baked key (CAT-5).'}>{catalog?.source === 'huggingface' ? 'Hugging Face · checksum-pinned' : catalog?.verifiedSignature === 'verified' ? 'bundled · ed25519 ✓' : 'bundled'}</span></div>
            <div className="specr"><span className="speck">Entries</span><span className="specv">{catalog?.entries.length ?? 0} models</span></div>
            <div className="specr"><span className="speck">Auto-update</span><span className="specv">live from Hugging Face</span></div>
          </div>
        </div>
        <div className="panel">
          <span className="mkey">Updates</span>
          <div className="speclist" style={{ marginTop: 8 }}>
            <div className="specr" style={{ borderTop: 0, paddingTop: 0 }}><span className="speck">This version</span><span className="specv">{upd?.current ?? '…'}</span></div>
            <div className="specr"><span className="speck">Status</span><span className="specv">
              {!upd?.supported ? 'n/a in this build'
                : upd.checking ? 'checking'
                : upd.ready ? `${upd.available} ready to install`
                : upd.available ? `${upd.available} available`
                : upd.error ? 'check failed'
                : 'up to date'}</span></div>
            <div className="specr"><span className="speck">Check on launch</span><span className="specv">{upd?.autoCheck ? 'on' : 'off'}</span></div>
          </div>
          <p className="mono faint" style={{ fontSize: 11, marginTop: 10, lineHeight: 1.6 }}>
            A check asks GitHub for a version manifest and nothing else — no identifiers. It is recorded
            in the network log like every other request. Turn it off and Kayon makes no update call at all.
            Updates are never downloaded until you ask.
          </p>
          <div className="fx gap8" style={{ marginTop: 10 }}>
            <button className="btn btn-line btn-sm" disabled={!upd?.supported || upd?.checking} onClick={checkNow}>
              {upd?.checking ? 'Checking…' : 'Check for updates'}
            </button>
            <button className="btn btn-line btn-sm" disabled={!upd?.supported} onClick={toggleAuto}>
              {upd?.autoCheck ? 'Turn off launch check' : 'Turn on launch check'}
            </button>
          </div>
        </div>
        <div className="panel">
          <span className="mkey">Account</span>
          <div style={{ margin: '12px 0' }}>
            <div className="serif ital" style={{ fontSize: 24, color: 'var(--muted)' }}>No account. Ever.</div>
            <p className="mono faint" style={{ fontSize: 12, marginTop: 8, lineHeight: 1.6 }}>There is no login surface in Kayon. Nothing to sign into, nothing stored elsewhere.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
