import { useEffect, useRef, useState } from 'react'
import { type MachineProfile, type RuntimeStatus } from '../lib/api'

interface Msg { role: 'user' | 'assistant'; content: string; reasoning?: string }

const g = (b: number) => (b / 1024 ** 3).toFixed(1)

export default function Chat({ machine, runtime }: { machine: MachineProfile | null; runtime: RuntimeStatus | null }) {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [sideOpen, setSideOpen] = useState(true)
  const [sys, setSys] = useState('You are a precise coding assistant. Prefer minimal diffs and explain tradeoffs in one sentence.')
  const [temp, setTemp] = useState(0.7)
  const [topP, setTopP] = useState(0.95)
  const [maxTok, setMaxTok] = useState(2048)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const running = runtime?.kind === 'running'
  const gpu = machine?.gpus?.[0]
  const stats = { gen: 0, eval: 0 } // live tok/s comes from the shared telemetry / benchmark

  const send = async () => {
    if (!input.trim() || !running || busy) return
    const text = input; setInput(''); setBusy(true)
    const history = msgs.map(m => ({ role: m.role, content: m.content }))
    setMsgs(m => [...m, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    try {
      const port = runtime!.port
      const resp = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: [{ role: 'system', content: sys }, ...history, { role: 'user', content: text }], temperature: temp, top_p: topP, max_tokens: maxTok, stream: true }),
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const reader = resp.body?.getReader(); const dec = new TextDecoder(); let acc = '', buffer = ''
      while (reader) {
        const { done, value } = await reader.read()
        if (value) buffer += dec.decode(value, { stream: true })
        const lines = buffer.split('\n'); buffer = done ? '' : (lines.pop() ?? '')
        for (const line of lines) {
          const t = line.trim(); if (!t.startsWith('data:')) continue
          const d = t.slice(5).trim(); if (d === '' || d === '[DONE]') continue
          try { const j = JSON.parse(d); const c = j.choices?.[0]?.delta?.content; if (c) { acc += c; setMsgs(m => { const cp = [...m]; cp[cp.length - 1] = { role: 'assistant', content: acc }; return cp }) } } catch { /* partial */ }
        }
        if (done) break
      }
    } catch (e: any) {
      setMsgs(m => { const cp = [...m]; cp[cp.length - 1] = { role: 'assistant', content: 'Error: ' + (e?.message || 'unknown') }; return cp })
    } finally { setBusy(false) }
  }

  const vramUsed = gpu ? g(gpu.telemetry.vramUsedBytes) : '0'
  const gpuUtil = gpu ? gpu.telemetry.utilizationPercent.toFixed(0) : '0'

  if (!running) {
    return (
      <div className="cinner">
        <div className="pagehead"><div><p className="eyebrow">Runtime · assistant-ui</p><h1 className="ptitle">Chat</h1><p className="psub">No model is loaded. Open the Library and press <span className="iris">Load &amp; chat</span> on any installed model — it launches the llama-server sidecar and streams entirely on your GPU.</p></div></div>
        <div className="panel" style={{ textAlign: 'center', padding: 48, color: 'var(--muted)' }}>{runtime?.kind === 'starting' ? 'Starting llama-server…' : 'No model loaded.'}</div>
      </div>
    )
  }

  return (
    <div className="chat">
      <div className="chatmain">
        <div className="chathead">
          <div className="fx ac gap12">
            <span className="livedot" />
            <div>
              <div style={{ fontWeight: 600, fontSize: 14 }}>{runtime?.modelId}</div>
              <div className="mono faint" style={{ fontSize: 11 }}>{runtime?.quantLabel} · loaded · {runtime?.contextLength} ctx · llama-server :{runtime?.port}</div>
            </div>
          </div>
          <div className="fx ac gap8">
            <span className="tag">Streaming</span><span className="tag">Reasoning</span><span className="tag">Tools</span>
            <button className={`sidetgl ${sideOpen ? 'on' : ''}`} onClick={() => setSideOpen(o => !o)} title="Parameters panel">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="8" x2="20" y2="8" /><circle cx="15" cy="8" r="2.4" fill="var(--paper)" /><line x1="4" y1="16" x2="20" y2="16" /><circle cx="9" cy="16" r="2.4" fill="var(--paper)" /></svg>
            </button>
          </div>
        </div>

        <div className="chatscroll softscroll">
          {msgs.length === 0 && <div className="msg"><div className="answer faint" style={{ textAlign: 'center' }}>Message the model to begin. Nothing leaves your machine.</div></div>}
          {msgs.map((m, i) => m.role === 'user'
            ? <div key={i} className="msg user"><div className="msgrole">You</div><div className="bubble">{m.content}</div></div>
            : <div key={i} className="msg"><div className="msgrole"><span className="iris">◆</span> Assistant · local</div>
                {m.reasoning && <div className="reasoning"><div className="rlabel"><span>◇</span> Reasoning</div>{m.reasoning}</div>}
                <div className="answer">{m.content || (busy && i === msgs.length - 1 ? <span className="faint mono" style={{ fontSize: 13 }}>generating<span style={{ animation: 'pulse 1.2s infinite' }}>▋</span></span> : '')}</div>
              </div>)}
          <div ref={end} />
        </div>

        <div className="composer">
          <div className="cbox">
            <textarea className="cinput" rows={1} placeholder="Message the model — runs entirely on your GPU, nothing leaves the machine" value={input} disabled={busy}
              onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }} />
            <div className="cfoot">
              <div className="statline">
                <span>gen <b>{stats.gen}</b> tok/s</span>
                <span>eval <b>{stats.eval}</b> tok/s</span>
                <span>VRAM <b>{vramUsed}</b> GB</span>
                <span>GPU <b>{gpuUtil}%</b></span>
              </div>
              <button className="btn btn-iris btn-sm" onClick={send} disabled={busy || !input.trim()}>Send ↵</button>
            </div>
          </div>
        </div>
      </div>

      {sideOpen && (
        <div className="chatside softscroll">
          <div className="field"><div className="flabel">System prompt</div><textarea className="ftext" value={sys} onChange={e => setSys(e.target.value)} /></div>
          <div className="field"><div className="flabel"><span>Temperature</span><span className="lsval">{temp.toFixed(1)}</span></div><input className="range" type="range" min={0} max={2} step={0.1} value={temp} onChange={e => setTemp(+e.target.value)} /></div>
          <div className="field"><div className="flabel"><span>Top-p</span><span className="lsval">{topP.toFixed(2)}</span></div><input className="range" type="range" min={0} max={1} step={0.05} value={topP} onChange={e => setTopP(+e.target.value)} /></div>
          <div className="field"><div className="flabel"><span>Max tokens</span><span className="lsval">{maxTok}</span></div><input className="range" type="range" min={256} max={8192} step={256} value={maxTok} onChange={e => setMaxTok(+e.target.value)} /></div>
          <div className="flabel" style={{ marginTop: 22 }}>Live inference</div>
          <div className="livestat">
            <div className="lsrow"><span className="muted">Generation</span><span className="lsval iris">{stats.gen} tok/s</span></div>
            <div className="lsrow"><span className="muted">Prompt eval</span><span className="lsval">{stats.eval} tok/s</span></div>
            <div className="lsrow"><span className="muted">VRAM</span><span className="lsval">{vramUsed} GB</span></div>
            <div className="lsrow"><span className="muted">GPU util</span><span className="lsval">{gpuUtil}%</span></div>
          </div>
          <div className="mono faint" style={{ fontSize: 10.5, marginTop: 14, lineHeight: 1.5 }}>Same telemetry source as the dashboard. assistant-ui · Cloud disabled · history local only.</div>
        </div>
      )}
    </div>
  )
}
