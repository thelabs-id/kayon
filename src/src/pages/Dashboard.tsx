import { useState } from 'react'
import { api, type MachineProfile, type RuntimeStatus, type BenchmarkResult } from '../lib/api'

const gb = (b: number) => (b / 1024 ** 3).toFixed(1)
const tb = (b: number) => b / 1024 ** 4

function DiskRow({ mount, kind, free, total, color }: { mount: string; kind: string; free: number; total: number; color: string }) {
  const usedPct = total > 0 ? ((total - free) / total) * 100 : 0
  const unit = total >= 1024 ** 4 ? `${tb(free).toFixed(1)} / ${tb(total).toFixed(1)} TB free` : `${gb(free)} / ${gb(total)} GB free`
  return (
    <>
      <div className="specr" style={{ borderTop: 0, paddingTop: 0 }}><span className="speck">{mount} · {kind}</span><span className="specv">{unit}</span></div>
      <div className="bar" style={{ marginTop: 0, marginBottom: 12 }}><div className="barfill" style={{ width: `${usedPct}%`, background: color }} /></div>
    </>
  )
}

export default function Dashboard({ machine, runtime, onReonboard }: { machine: MachineProfile | null; runtime: RuntimeStatus | null; onReonboard: () => void }) {
  const [bench, setBench] = useState<BenchmarkResult | null>(null)
  const [benching, setBenching] = useState(false)
  const gpu = machine?.gpus?.[0]
  const t = gpu?.telemetry

  const totalV = gpu?.totalVramBytes ?? 0
  const usedV = t?.vramUsedBytes ?? 0
  const freeV = t?.vramFreeBytes ?? 0
  const circ = 2 * Math.PI * 66
  const usedFrac = totalV > 0 ? usedV / totalV : 0
  const rsvGB = Math.max(1, (totalV / 1024 ** 3) * 0.1)
  const rsvFrac = totalV > 0 ? (rsvGB * 1024 ** 3) / totalV : 0
  const usedLen = circ * usedFrac
  const rsvLen = circ * rsvFrac

  const cpuUtil = machine ? Math.round(machine.cpu.usagePercent) : 0
  const ramUsed = machine ? gb(machine.ram.usedBytes) : '0'
  const ramTotal = machine ? gb(machine.ram.totalBytes) : '0'
  const ramPct = machine && machine.ram.totalBytes > 0 ? (machine.ram.usedBytes / machine.ram.totalBytes) * 100 : 0
  const diskColors = ['var(--v-cpu)', 'var(--iris)', 'var(--amber)', 'var(--v-full)']

  const runBench = async () => {
    if (!runtime || runtime.kind !== 'running') { alert('Load a model first (Library → Load & Chat).'); return }
    setBenching(true)
    const r = await api.benchmark({ modelId: runtime.modelId ?? '', quantLabel: runtime.quantLabel ?? '', contextLength: 4096 })
    setBenching(false)
    if (r.ok && r.data) setBench(r.data)
  }

  return (
    <div className="cinner">
      <div className="pagehead">
        <div>
          <p className="eyebrow">Hardware · live at 1&nbsp;Hz</p>
          <h1 className="ptitle">Instrument cluster</h1>
          <p className="psub">Everything your machine is, and everything it can run — measured directly from NVML, not guessed.</p>
        </div>
        <button className="btn btn-line btn-sm" onClick={onReonboard}><span className="mono" style={{ fontSize: 12 }}>↻</span> Re-run first-run</button>
      </div>

      <div className="grid-dash" style={{ marginBottom: 20 }}>
        <div className="panel gpucard">
          {gpu ? <>
            <div className="fx ac jb">
              <span className="tag"><span className="dotc" />NVIDIA · detected</span>
              <span className="mono" style={{ fontSize: 11, color: 'var(--faint)' }}>{gpu.pciId ? `PCI ${gpu.pciId.split(':').slice(1).join(':').replace('.0', '.0')}` : ''}</span>
            </div>
            <div className="gpuname">{gpu.name.replace(/NVIDIA /i, '')}</div>
            <div className="speclist">
              <div className="specr"><span className="speck">Architecture</span><span className="specv">{gpu.architecture ?? '—'} · CC {gpu.computeCapability ?? '?'}</span></div>
              <div className="specr"><span className="speck">Driver / CUDA</span><span className="specv">{gpu.driverVersion ?? '?'} · {gpu.cudaVersion ?? '?'}</span></div>
              <div className="specr"><span className="speck">Total VRAM</span><span className="specv">{gb(totalV)} GB</span></div>
              <div className="specr"><span className="speck">Temperature</span><span className="specv">{t?.temperatureC.toFixed(0) ?? '—'} °C</span></div>
              <div className="specr"><span className="speck">Power draw</span><span className="specv">{t?.powerWatts.toFixed(0) ?? '—'} W</span></div>
              <div className="specr"><span className="speck">Core / Mem clock</span><span className="specv">{t?.coreClockMhz ?? '—'} · {t?.memClockMhz ?? '—'} MHz</span></div>
            </div>
          </> : <div style={{ padding: '20px 0' }}><div className="gpuname">No supported GPU</div><p className="psub" style={{ marginTop: 10 }}>NVML found no NVIDIA GPU (or compute capability &lt; 5.0). Verdicts fall back to system RAM.</p></div>}
        </div>

        <div className="panel">
          <div className="fx ac jb" style={{ marginBottom: 6 }}><span className="mkey">VRAM · dedicated</span><span className="mono" style={{ fontSize: 11, color: 'var(--faint)' }}>NVML memory_info</span></div>
          <div className="ringwrap">
            <div className="posrel" style={{ width: 160, height: 160, flex: 'none' }}>
              <svg className="ring" width="160" height="160" viewBox="0 0 160 160">
                <circle className="ringtrack" cx="80" cy="80" r="66" strokeWidth="14" />
                <circle className="ringrsv" cx="80" cy="80" r="66" strokeWidth="14" strokeDasharray={`${rsvLen} ${circ}`} strokeDashoffset={-usedLen} />
                <circle className="ringused" cx="80" cy="80" r="66" strokeWidth="14" strokeDasharray={`${usedLen} ${circ}`} />
              </svg>
              <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <div className="mono" style={{ fontSize: 11, color: 'var(--faint)' }}>USED</div>
                <div style={{ fontFamily: "'Instrument Serif',Georgia,serif", fontSize: 38, lineHeight: 1 }}>{gb(usedV)}</div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>of {gb(totalV)} GB</div>
              </div>
            </div>
            <div className="legend">
              <div className="legrow"><span className="legsw" style={{ background: 'var(--iris)' }} /><div><div style={{ fontWeight: 600 }}>{gb(usedV)} GB used</div><div className="mono faint" style={{ fontSize: 11 }}>model weights + KV</div></div></div>
              <div className="legrow"><span className="legsw" style={{ background: 'var(--amber)' }} /><div><div style={{ fontWeight: 600 }}>{rsvGB.toFixed(1)} GB reserved</div><div className="mono faint" style={{ fontSize: 11 }}>display headroom</div></div></div>
              <div className="legrow"><span className="legsw" style={{ background: 'var(--line2)' }} /><div><div style={{ fontWeight: 600 }}>{gb(freeV)} GB free</div><div className="mono faint" style={{ fontSize: 11 }}>available to allocate</div></div></div>
            </div>
          </div>
        </div>
      </div>

      <div className="g3" style={{ marginBottom: 20 }}>
        <div className="metric"><div className="mtop"><span className="mkey">GPU util</span><span className="mono faint" style={{ fontSize: 11 }}>core</span></div><div><span className="mbig">{t?.utilizationPercent.toFixed(0) ?? 0}</span><span className="munit">%</span></div><div className="bar"><div className="barfill" style={{ width: `${t?.utilizationPercent ?? 0}%`, background: 'var(--iris)' }} /></div></div>
        <div className="metric"><div className="mtop"><span className="mkey">CPU</span><span className="mono faint" style={{ fontSize: 11 }}>{machine ? `${machine.cpu.coreCount}c · ${machine.cpu.threadCount}T` : ''}</span></div><div><span className="mbig">{cpuUtil}</span><span className="munit">%</span></div><div className="bar"><div className="barfill" style={{ width: `${cpuUtil}%`, background: 'var(--v-cpu)' }} /></div></div>
        <div className="metric"><div className="mtop"><span className="mkey">System RAM</span><span className="mono faint" style={{ fontSize: 11 }}>of {ramTotal} GB</span></div><div><span className="mbig">{ramUsed}</span><span className="munit">GB</span></div><div className="bar"><div className="barfill" style={{ width: `${ramPct}%`, background: 'var(--amber)' }} /></div></div>
      </div>

      <div className="g2">
        <div className="panel">
          <div className="fx ac jb" style={{ marginBottom: 16 }}><span className="mkey">Speed benchmark · warm run</span>{runtime?.kind === 'running' && <span className="tag"><span className="dotc" style={{ background: 'var(--ok)' }} />{runtime.quantLabel}</span>}</div>
          <div className="fx ac gap24" style={{ alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <div><div className="mono faint" style={{ fontSize: 11 }}>GENERATION</div><div><span className="mbig iris">{bench ? bench.genTokPerS.toFixed(0) : '—'}</span><span className="munit">tok/s</span></div></div>
            <div><div className="mono faint" style={{ fontSize: 11 }}>PROMPT EVAL</div><div><span className="mbig">{bench ? bench.promptEvalTokPerS.toFixed(0) : '—'}</span><span className="munit">tok/s</span></div></div>
            <div style={{ flex: 1, minWidth: 140 }}><div className="mono faint" style={{ fontSize: 11, marginBottom: 6 }}>@ 4096 ctx · fixed prompt</div><button className="btn btn-line btn-sm" disabled={benching} onClick={runBench}>{benching ? 'Running…' : '↻ Re-run benchmark'}</button></div>
          </div>
        </div>
        <div className="panel">
          <div className="fx ac jb" style={{ marginBottom: 14 }}><span className="mkey">Per-drive disk</span></div>
          <div className="speclist">
            {(machine?.disks ?? []).slice(0, 3).map((d, i) => (
              <DiskRow key={d.mount} mount={d.mount} kind={i === 0 ? 'system' : 'storage'} free={d.freeBytes} total={d.totalBytes} color={diskColors[i % diskColors.length]} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
