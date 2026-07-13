import { useEffect, useState } from 'react'
import { api, type CatalogEntry, type FitVerdict, type MachineProfile } from '../lib/api'
import { KMark, Check } from '../components/icons'

const g = (b: number) => (b / 1024 ** 3).toFixed(1)
const fmt = (b: number) => b < 1024 ** 3 ? `${(b / 1024 ** 2).toFixed(0)} MB` : `${(b / 1024 ** 3).toFixed(1)} GB`
const isPinned = (s: string) => /^[0-9a-f]{64}$/i.test((s || '').trim())
const tier: Record<string, number> = { FITS_FULLY: 4, FITS_TIGHT: 3, GPU_CPU_SPLIT: 2, CPU_ONLY: 1 }

export default function Onboarding({ machine, onFinish, goBrowser }: { machine: MachineProfile | null; onFinish: () => void; goBrowser: () => void }) {
  const [best, setBest] = useState<{ entry: CatalogEntry; verdict: FitVerdict; q: any } | null>(null)
  const [installing, setInstalling] = useState(false)
  const [installed, setInstalled] = useState(false)

  useEffect(() => {
    Promise.all([api.catalog(), api.verdicts(4096, 2)]).then(([c, v]) => {
      if (!c.ok || !c.data || !v.ok || !v.data) return
      const vm = new Map<string, FitVerdict>(); for (const x of v.data) vm.set(`${x.modelId}|${x.quantLabel}`, x)
      let winner: any = null, bestScore = -1
      for (const entry of c.data.entries) for (const q of entry.quants) {
        const vd = vm.get(`${entry.id}|${q.label}`)
        if (!vd || !(vd.verdict in tier) || !isPinned(q.sha256)) continue
        const s = tier[vd.verdict] * 1e15 + q.bytes
        if (s > bestScore) { bestScore = s; winner = { entry, verdict: vd, q } }
      }
      setBest(winner)
    })
  }, [])

  const gpu = machine?.gpus?.[0]

  const install = async () => {
    if (!best) return
    setInstalling(true)
    const r = await api.startDownload({ modelId: best.entry.id, quantLabel: best.q.label })
    setInstalling(false)
    if (r.ok) { setInstalled(true); setTimeout(() => goBrowser(), 700) }
    else alert('Could not start: ' + (r.error || 'unknown'))
  }

  return (
    <div className="onb">
      <div className="onbinner">
        <div className="fx ac gap12" style={{ marginBottom: 26 }}>
          <KMark size={34} />
          <span className="kword" style={{ fontSize: 26 }}>Kayon<span className="kdot">.</span></span>
        </div>
        <div className="stepdots"><span className="stepdot on" /><span className="stepdot on" /><span className="stepdot" /></div>
        <h1 className="ptitle" style={{ fontSize: 44 }}>Your machine, measured.</h1>
        <p className="psub" style={{ fontSize: 16, marginBottom: 6 }}>Kayon probed your hardware directly. Here's the honest picture — and the one model that fits it best.</p>
        <div className="probe">
          <div className="prober"><Check />{gpu ? `${gpu.name} — ${gpu.architecture ?? ''}, ${g(gpu.totalVramBytes)} GB VRAM` : 'No supported NVIDIA GPU — verdicts fall back to RAM'}</div>
          <div className="prober"><Check />{machine ? `${machine.cpu.brand} · ${machine.cpu.threadCount} threads · ${g(machine.ram.totalBytes)} GB RAM` : 'Probing CPU…'}</div>
          <div className="prober"><Check />{gpu ? `Driver ${gpu.driverVersion ?? '?'} · CUDA ${gpu.cudaVersion ?? '?'} · llama-server ready` : 'Runtime ready'}</div>
        </div>
        <div className="bestpick">
          <div className="fx ac jb" style={{ marginBottom: 10 }}>
            <span className="tag" style={{ color: 'var(--iris)', borderColor: 'color-mix(in oklab, var(--iris) 45%, var(--line2))' }}><span className="dotc" />Computed best pick — not a hardcoded default</span>
          </div>
          {best ? (
            <div className="fx ac jb" style={{ gap: 16, flexWrap: 'wrap' }}>
              <div>
                <div className="serif" style={{ fontSize: 28 }}>{best.entry.id} · {best.q.label}</div>
                <div className="mono muted" style={{ fontSize: 12.5, marginTop: 4 }}>{fmt(best.q.bytes)} · {best.verdict.verdict.replace(/_/g, ' ')} · {best.verdict.nGpuLayers} GPU layers</div>
              </div>
              <span className="verdict" style={{ color: 'var(--v-full)', background: 'color-mix(in oklab, var(--v-full) 16%, transparent)' }}><span className="vsw" style={{ background: 'var(--v-full)' }} />{best.verdict.verdict.replace(/_/g, ' ')}</span>
            </div>
          ) : <div className="mono muted" style={{ fontSize: 13 }}>No downloadable catalog model fits yet — you can still explore with zero models.</div>}
        </div>
        <div className="fx gap12" style={{ flexWrap: 'wrap' }}>
          {best && <button className="btn btn-iris" disabled={installing || installed} onClick={install}>{installed ? 'Install started ✓' : installing ? 'Starting…' : `Install this model · ${fmt(best.q.bytes)}`}</button>}
          <button className="btn btn-line" onClick={onFinish}>Skip — explore with zero models</button>
        </div>
        <p className="mono faint" style={{ fontSize: 11, marginTop: 18, lineHeight: 1.6 }}>Nothing downloads until you confirm. The dashboard, browser, and Ollama adoption all work with no models installed.</p>
      </div>
    </div>
  )
}
