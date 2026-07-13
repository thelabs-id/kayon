import { useEffect, useState } from 'react'
import { api, type MachineProfile, type FitVerdict, type CatalogEntry } from '../lib/api'
import VerdictBadge from '../components/VerdictBadge'

function fmt(b: number): string { return b < 1024**3 ? `${(b/1024**2).toFixed(0)} MB` : `${(b/1024**3).toFixed(2)} GB` }

export default function Onboarding({ onDone }: { onDone: () => void }) {
  const [step, setStep] = useState(0)
  const [machine, setMachine] = useState<MachineProfile|null>(null)
  const [bestPick, setBestPick] = useState<{entry: CatalogEntry; verdict: FitVerdict}|null>(null)
  const [installing, setInstalling] = useState(false)
  const [installed, setInstalled] = useState(false)

  useEffect(() => {
    api.hardware().then(r => { if (r.ok && r.data) setMachine(r.data) })
    Promise.all([api.catalog(), api.verdicts()]).then(([c,v]) => {
      if (!c.ok || !c.data || !v.ok || !v.data) return
      const vm = new Map<string, FitVerdict>()
      for (const x of v.data) vm.set(`${x.modelId}|${x.quantLabel}`, x)
      let best: {entry: CatalogEntry; verdict: FitVerdict}|null = null
      for (const entry of c.data.entries) {
        for (const q of entry.quants) {
          const v = vm.get(`${entry.id}|${q.label}`)
          if (v && (v.verdict === 'FITS_FULLY' || v.verdict === 'FITS_TIGHT') && !best) best = { entry, verdict: v }
        }
      }
      setBestPick(best)
    })
  }, [])

  const gpu = machine?.gpus?.[0]

  // FR-2/FR-3: the computed best pick is offered as an explicit one-click confirm.
  // Nothing downloads until the user clicks — no auto-download.
  const installBestPick = async () => {
    if (!bestPick) return
    const q = bestPick.entry.quants.find(x => x.label === bestPick.verdict.quantLabel) || bestPick.entry.quants[0]
    if (!q) return
    setInstalling(true)
    const r = await api.startDownload({ modelId: bestPick.entry.id, quantLabel: q.label })
    setInstalling(false)
    if (r.ok) setInstalled(true)
    else alert('Could not start install: ' + (r.error || 'unknown'))
  }

  return (
    <div style={{minHeight:'100vh',display:'flex',alignItems:'center',justifyContent:'center',background:'var(--bg)',padding:24}}>
      <div style={{maxWidth:600,width:'100%'}}>
        <div style={{textAlign:'center',marginBottom:32}}>
          <h1 style={{fontSize:28,fontWeight:600}}>Welcome to Kayon</h1>
          <div className="text-muted" style={{marginTop:4}}>Honest, private, local LLM workstation</div>
        </div>
        <div className="card animate-in" key={step}>
          {step === 0 && <div>
            <div className="card-title" style={{marginBottom:8}}>What Kayon is</div>
            <p className="text-muted" style={{lineHeight:1.7}}>Kayon detects your GPU, tells you which models actually fit via a real memory model, manages downloads, adopts Ollama models without re-downloading, and runs everything locally with no cloud dependency.</p>
            <div style={{marginTop:12,padding:10,background:'var(--bg)',borderRadius:6,fontSize:13}}>
              <strong>You can skip any step.</strong> Dashboard, browser, library, and Ollama adoption all work with zero models.
            </div>
          </div>}
          {step === 1 && <div>
            <div className="card-title" style={{marginBottom:8}}>Hardware detected</div>
            {gpu ? <div style={{padding:12,background:'var(--bg)',borderRadius:6}}>
              <div className="font-medium">{gpu.name}</div>
              <div className="text-sm text-muted" style={{lineHeight:1.8,marginTop:4}}>
                Architecture: {gpu.architecture||'unknown'} | Compute: {gpu.computeCapability||'?'}<br/>
                Driver: {gpu.driverVersion||'?'} | CUDA: {gpu.cudaVersion||'?'}<br/>
                VRAM: {fmt(gpu.totalVramBytes)}
              </div>
              {machine && <div className="text-xs text-muted" style={{marginTop:8}}>{machine.cpu.brand} | {machine.cpu.threadCount} threads | {fmt(machine.ram.totalBytes)} RAM</div>}
            </div> : <div className="text-muted">No GPU detected. Verdicts will be RAM-based.</div>}
          </div>}
          {step === 2 && <div>
            <div className="card-title" style={{marginBottom:8}}>Your best fit</div>
            {bestPick ? <div>
              <p className="text-muted" style={{lineHeight:1.7,marginBottom:12}}>Based on your hardware, this model fits best:</p>
              <div style={{padding:12,background:'var(--bg)',borderRadius:6}}>
                <div className="font-medium">{bestPick.entry.id}</div>
                <div className="text-sm text-muted">{bestPick.entry.family} | {bestPick.entry.params} | {bestPick.verdict.quantLabel}</div>
                <div style={{marginTop:8}}><VerdictBadge verdict={bestPick.verdict.verdict}/></div>
                <div className="text-xs text-muted" style={{marginTop:8,lineHeight:1.5}}>{bestPick.verdict.explainability}</div>
              </div>
              <div style={{marginTop:12,padding:10,background:'rgba(224,145,107,0.08)',borderRadius:6,fontSize:13}}>
                Nothing downloads until you confirm. This is an explicit one-click install — no auto-download.
              </div>
              <div style={{marginTop:12}}>
                {installed ? (
                  <div className="badge badge-success">Install started — track it in Library</div>
                ) : (
                  <button className="btn btn-primary btn-sm" disabled={installing} onClick={installBestPick}>
                    {installing ? 'Starting…' : `Install ${bestPick.entry.id} (${bestPick.verdict.quantLabel})`}
                  </button>
                )}
              </div>
            </div> : <p className="text-muted">No catalog model currently fits. You can explore the browser or adopt Ollama models.</p>}
          </div>}
          <div style={{display:'flex',justifyContent:'space-between',marginTop:24,alignItems:'center'}}>
            <div style={{display:'flex',gap:6}}>
              {[0,1,2].map(i=><span key={i} style={{width:8,height:8,borderRadius:'50%',background:i===step?'var(--accent)':'var(--border)'}}/>)}
            </div>
            <div style={{display:'flex',gap:8}}>
              {step>0 && <button className="btn btn-ghost btn-sm" onClick={()=>setStep(step-1)}>Back</button>}
              <button className="btn btn-secondary btn-sm" onClick={onDone}>Skip</button>
              {step<2 ? <button className="btn btn-primary btn-sm" onClick={()=>setStep(step+1)}>Next</button> : <button className="btn btn-primary btn-sm" onClick={onDone}>Open Kayon</button>}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
