import { useEffect, useRef, useState } from 'react'
import { api, type MachineProfile, type RuntimeStatus, type ChatSessionSummary } from '../lib/api'

interface Msg { role: 'user' | 'assistant'; content: string; reasoning?: string }

const g = (b: number) => (b / 1024 ** 3).toFixed(1)

// Compact relative time for the session list.
function rel(iso: string): string {
  const then = new Date(iso).getTime()
  const s = Math.max(0, (Date.now() - then) / 1000)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  if (s < 604800) return `${Math.floor(s / 86400)}d ago`
  return new Date(iso).toLocaleDateString()
}

const DEFAULT_SYS = 'You are a precise coding assistant. Prefer minimal diffs and explain tradeoffs in one sentence.'

export default function Chat({ machine, runtime }: { machine: MachineProfile | null; runtime: RuntimeStatus | null }) {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [sideOpen, setSideOpen] = useState(false)
  const [railOpen, setRailOpen] = useState(true)
  const [sys, setSys] = useState(DEFAULT_SYS)
  const [temp, setTemp] = useState(0.7)
  const [topP, setTopP] = useState(0.95)
  const [maxTok, setMaxTok] = useState(2048)

  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [activeTitle, setActiveTitle] = useState('New chat')
  const [editingTitle, setEditingTitle] = useState(false)
  const [confirmDel, setConfirmDel] = useState<string | null>(null)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])
  useEffect(() => { loadSessions() }, [])

  const loadSessions = async () => {
    const r = await api.chatSessions()
    if (r.ok && r.data) setSessions(r.data)
  }

  const running = runtime?.kind === 'running'
  const gpu = machine?.gpus?.[0]
  const stats = { gen: 0, eval: 0 } // live tok/s comes from the shared telemetry / benchmark

  const newChat = () => {
    if (busy) return // don't reset the view out from under an in-flight stream
    setActiveId(null); setActiveTitle('New chat'); setMsgs([]); setInput('')
    setSys(DEFAULT_SYS); setEditingTitle(false)
  }

  const openSession = async (id: string) => {
    // Switching sessions mid-stream would let the still-running loop write chunks into the newly
    // shown transcript while the assistant reply persists to the old session. Block it while busy.
    if (busy || id === activeId) return
    const r = await api.chatSession(id)
    if (!r.ok || !r.data) return
    const d = r.data
    setActiveId(d.id); setActiveTitle(d.title)
    setSys(d.systemPrompt || DEFAULT_SYS); setTemp(d.temperature); setTopP(d.topP); setMaxTok(d.maxTokens)
    setMsgs(d.messages.map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content, reasoning: m.reasoning })))
    setEditingTitle(false); setInput('')
  }

  const removeSession = async (id: string) => {
    if (busy) return // deleting the streaming session mid-flight would strand the in-flight append
    await api.deleteChatSession(id)
    setConfirmDel(null)
    if (id === activeId) newChat()
    loadSessions()
  }

  const renameActive = async (title: string) => {
    const t = title.trim(); setEditingTitle(false)
    if (!activeId || !t || t === activeTitle) return
    setActiveTitle(t)
    await api.renameChatSession(activeId, t)
    loadSessions()
  }

  const settingsBody = () => ({ systemPrompt: sys, temperature: temp, topP, maxTokens: maxTok, modelId: runtime?.modelId })

  // Ensure a persisted session exists before the first message lands; auto-title from that message.
  const ensureSession = async (firstText: string): Promise<string | null> => {
    if (activeId) { await api.updateChatSettings(activeId, settingsBody()); return activeId }
    const title = firstText.trim().slice(0, 48) || 'New chat'
    const r = await api.createChatSession({ title, ...settingsBody() })
    if (!r.ok || !r.data) return null
    setActiveId(r.data.id); setActiveTitle(title)
    return r.data.id
  }

  const send = async () => {
    if (!input.trim() || !running || busy) return
    const text = input; setInput(''); setBusy(true)
    const history = msgs.map(m => ({ role: m.role, content: m.content }))
    setMsgs(m => [...m, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    const sid = await ensureSession(text)
    if (sid) await api.appendChatMessage(sid, { role: 'user', content: text })
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
      if (sid) await api.appendChatMessage(sid, { role: 'assistant', content: acc })
    } catch (e: any) {
      setMsgs(m => { const cp = [...m]; cp[cp.length - 1] = { role: 'assistant', content: 'Error: ' + (e?.message || 'unknown') }; return cp })
    } finally { setBusy(false); loadSessions() }
  }

  const vramUsed = gpu ? g(gpu.telemetry.vramUsedBytes) : '0'
  const gpuUtil = gpu ? gpu.telemetry.utilizationPercent.toFixed(0) : '0'

  const rail = (
    <div className={`chatsessions softscroll ${railOpen ? '' : 'hidden'} ${busy ? 'locked' : ''}`} title={busy ? 'Locked while generating' : ''}>
      <button className="btn btn-sm newchat" onClick={newChat} disabled={busy}>＋ New chat</button>
      <div className="slabel" style={{ marginTop: 14 }}>History</div>
      {sessions.length === 0 && <div className="mono faint" style={{ fontSize: 11, padding: '8px 6px' }}>No saved chats yet.</div>}
      {sessions.map(s => (
        <div key={s.id} className={`sessitem ${s.id === activeId ? 'on' : ''}`} onClick={() => openSession(s.id)}>
          <div className="sesstitle">{s.title}</div>
          <div className="sessmeta mono">{s.messageCount} msg · {rel(s.updatedAt)}</div>
          {confirmDel === s.id
            ? <button className="sessdel danger" onClick={e => { e.stopPropagation(); removeSession(s.id) }} title="Confirm delete">confirm</button>
            : <button className="sessdel" onClick={e => { e.stopPropagation(); setConfirmDel(s.id) }} title="Delete chat">×</button>}
        </div>
      ))}
    </div>
  )

  return (
    <div className="chat">
      {rail}
      <div className="chatmain">
        <div className="chathead">
          <div className="fx ac gap12">
            <button className={`sidetgl ${railOpen ? 'on' : ''}`} onClick={() => setRailOpen(o => !o)} title="Chat history">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="6" x2="20" y2="6" /><line x1="4" y1="12" x2="20" y2="12" /><line x1="4" y1="18" x2="14" y2="18" /></svg>
            </button>
            <span className={`livedot ${running ? '' : 'off'}`} />
            <div>
              {editingTitle
                ? <input className="titleedit" autoFocus defaultValue={activeTitle}
                    onBlur={e => renameActive(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur(); if (e.key === 'Escape') setEditingTitle(false) }} />
                : <div style={{ fontWeight: 600, fontSize: 14, cursor: activeId ? 'text' : 'default' }} title={activeId ? 'Click to rename' : ''} onClick={() => activeId && setEditingTitle(true)}>{activeTitle}</div>}
              <div className="mono faint" style={{ fontSize: 11 }}>{running ? `${runtime?.modelId} · ${runtime?.quantLabel} · :${runtime?.port}` : 'no model loaded'}</div>
            </div>
          </div>
          <div className="fx ac gap8">
            <span className="tag">Streaming</span><span className="tag">Local history</span>
            <button className={`sidetgl ${sideOpen ? 'on' : ''}`} onClick={() => setSideOpen(o => !o)} title="Parameters panel">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><line x1="4" y1="8" x2="20" y2="8" /><circle cx="15" cy="8" r="2.4" fill="var(--paper)" /><line x1="4" y1="16" x2="20" y2="16" /><circle cx="9" cy="16" r="2.4" fill="var(--paper)" /></svg>
            </button>
          </div>
        </div>

        <div className="chatscroll softscroll">
          {msgs.length === 0 && <div className="msg"><div className="answer faint" style={{ textAlign: 'center' }}>{running ? 'Message the model to begin. Nothing leaves your machine.' : 'Open a past chat from the left, or load a model to start a new one.'}</div></div>}
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
            <textarea className="cinput" rows={1} placeholder={running ? 'Message the model — runs entirely on your GPU, nothing leaves the machine' : 'Load a model from the Library to send messages'} value={input} disabled={busy || !running}
              onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }} />
            <div className="cfoot">
              <div className="statline">
                <span>gen <b>{stats.gen}</b> tok/s</span>
                <span>eval <b>{stats.eval}</b> tok/s</span>
                <span>VRAM <b>{vramUsed}</b> GB</span>
                <span>GPU <b>{gpuUtil}%</b></span>
              </div>
              <button className="btn btn-iris btn-sm" onClick={send} disabled={busy || !input.trim() || !running}>Send ↵</button>
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
          <div className="mono faint" style={{ fontSize: 10.5, marginTop: 14, lineHeight: 1.5 }}>Settings save with this chat. History is stored locally · Cloud disabled · nothing leaves your machine.</div>
        </div>
      )}
    </div>
  )
}
