import { useEffect, useRef, useState } from 'react'
import { api, type MachineProfile, type RuntimeStatus, type ChatSessionSummary } from '../lib/api'
import { Globe, Folder, Bolt, Alert, Paperclip, FileIcon } from '../components/icons'

// A tool call shown inline in the transcript (TOOL-7). `confirm` means it's paused awaiting the
// user's Approve/Deny for a side-effectful tool (TOOL-6).
interface ToolCall {
  callId: string
  name: string
  args: unknown
  status: 'running' | 'confirm' | 'ok' | 'error'
  result?: string
}
interface Msg { role: 'user' | 'assistant'; content: string; reasoning?: string; tools?: ToolCall[] }

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

// Parse a persisted tool-call trace (TOOL-7) back into cards; tolerate absent/garbled JSON.
function parseTools(raw?: string): ToolCall[] | undefined {
  if (!raw) return undefined
  try { const v = JSON.parse(raw); return Array.isArray(v) && v.length ? v : undefined } catch { return undefined }
}

const DEFAULT_SYS = 'You are a precise coding assistant. Prefer minimal diffs and explain tradeoffs in one sentence.'
const DEFAULT_TEMP = 0.7
const DEFAULT_TOP_P = 0.95
const DEFAULT_MAX_TOK = 2048

export default function Chat({ machine, runtime }: { machine: MachineProfile | null; runtime: RuntimeStatus | null }) {
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [sideOpen, setSideOpen] = useState(false)
  const [railOpen, setRailOpen] = useState(true)
  const [sys, setSys] = useState(DEFAULT_SYS)
  const [temp, setTemp] = useState(DEFAULT_TEMP)
  const [topP, setTopP] = useState(DEFAULT_TOP_P)
  const [maxTok, setMaxTok] = useState(DEFAULT_MAX_TOK)

  // TOOL family: per-session workspace folder + Web toggle + side-effect auto-approve.
  const [workspace, setWorkspace] = useState('')
  const [webEnabled, setWebEnabled] = useState(false)
  const [autoApprove, setAutoApprove] = useState(false)
  const [wsEdit, setWsEdit] = useState(false)
  const [staged, setStaged] = useState<{ name: string }[]>([]) // files attached but not yet sent
  const fileInput = useRef<HTMLInputElement>(null)

  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [activeTitle, setActiveTitle] = useState('New chat')
  const [editingTitle, setEditingTitle] = useState(false)
  const [confirmDel, setConfirmDel] = useState<string | null>(null)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])
  useEffect(() => { loadSessions() }, [])

  const running = runtime?.kind === 'running'
  const supportsTools = !!runtime?.supportsTools

  // RUN-5 + TOOL: per-session settings (prompt, sampling, workspace/web/auto-approve) are session
  // state and must persist across switches / new chat / close even without a send. A latest-value ref
  // lets us flush on those transitions without a stale closure.
  const settingsRef = useRef({ activeId, sys, temp, topP, maxTok, modelId: runtime?.modelId, workspace, webEnabled, autoApprove })
  useEffect(() => { settingsRef.current = { activeId, sys, temp, topP, maxTok, modelId: runtime?.modelId, workspace, webEnabled, autoApprove } })
  const flushSettings = () => {
    const s = settingsRef.current
    if (s.activeId) api.updateChatSettings(s.activeId, { systemPrompt: s.sys, temperature: s.temp, topP: s.topP, maxTokens: s.maxTok, modelId: s.modelId, workspace: s.workspace || undefined, webEnabled: s.webEnabled, autoApprove: s.autoApprove })
  }
  useEffect(() => {
    if (!activeId) return
    const t = setTimeout(flushSettings, 400)
    return () => clearTimeout(t)
  }, [sys, temp, topP, maxTok, workspace, webEnabled, autoApprove, activeId])
  useEffect(() => () => flushSettings(), []) // final backstop on unmount (navigating away / closing)

  const loadSessions = async () => {
    const r = await api.chatSessions()
    if (r.ok && r.data) setSessions(r.data)
  }

  const gpu = machine?.gpus?.[0]
  const stats = { gen: 0, eval: 0 } // live tok/s comes from the shared telemetry / benchmark

  const newChat = () => {
    if (busy) return // don't reset the view out from under an in-flight stream
    flushSettings() // save the outgoing session's settings before clearing them
    setActiveId(null); setActiveTitle('New chat'); setMsgs([]); setInput('')
    // Reset ALL per-session settings so a fresh chat never inherits the last session's params.
    setSys(DEFAULT_SYS); setTemp(DEFAULT_TEMP); setTopP(DEFAULT_TOP_P); setMaxTok(DEFAULT_MAX_TOK)
    setWorkspace(''); setWebEnabled(false); setAutoApprove(false); setStaged([])
    setEditingTitle(false)
  }

  const openSession = async (id: string) => {
    if (busy || id === activeId) return
    flushSettings()
    const r = await api.chatSession(id)
    if (!r.ok || !r.data) return
    const d = r.data
    setActiveId(d.id); setActiveTitle(d.title)
    setSys(d.systemPrompt ?? DEFAULT_SYS); setTemp(d.temperature); setTopP(d.topP); setMaxTok(d.maxTokens)
    setWorkspace(d.workspace ?? ''); setWebEnabled(!!d.webEnabled); setAutoApprove(!!d.autoApprove); setStaged([])
    setMsgs(d.messages.map(m => ({ role: m.role === 'assistant' ? 'assistant' : 'user', content: m.content, reasoning: m.reasoning, tools: parseTools(m.tools) })))
    setEditingTitle(false); setInput('')
  }

  const removeSession = async (id: string) => {
    if (busy) return
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

  const settingsBody = () => ({ systemPrompt: sys, temperature: temp, topP, maxTokens: maxTok, modelId: runtime?.modelId, workspace: workspace || undefined, webEnabled, autoApprove })

  const ensureSession = async (firstText: string): Promise<string | null> => {
    if (activeId) { await api.updateChatSettings(activeId, settingsBody()); return activeId }
    const title = firstText.trim().slice(0, 48) || 'New chat'
    const r = await api.createChatSession({ title, ...settingsBody() })
    if (!r.ok || !r.data) return null
    setActiveId(r.data.id); setActiveTitle(title)
    return r.data.id
  }

  // A session must exist before a file can be attached (files land in its workspace). Create a bare
  // one if the chat hasn't started yet.
  const ensureSessionForAttach = async (): Promise<string | null> => {
    if (activeId) return activeId
    const r = await api.createChatSession({ title: 'New chat', ...settingsBody() })
    if (!r.ok || !r.data) return null
    setActiveId(r.data.id); setActiveTitle('New chat'); loadSessions()
    return r.data.id
  }

  const fileToBase64 = (f: File) => new Promise<string>((resolve, reject) => {
    const rd = new FileReader()
    rd.onerror = reject
    rd.onload = () => { const s = String(rd.result); resolve(s.slice(s.indexOf(',') + 1)) }
    rd.readAsDataURL(f)
  })

  const onFilesPicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    e.target.value = '' // allow re-picking the same file
    if (!files.length || busy) return
    const sid = await ensureSessionForAttach()
    if (!sid) return
    for (const f of files) {
      try {
        const r = await api.attachFile(sid, f.name, await fileToBase64(f))
        if (r.ok && r.data) setStaged(s => [...s, { name: r.data!.name }])
        // On failure the chip simply doesn't appear (visible signal); nothing is half-attached.
      } catch { /* skip a file that failed to read */ }
    }
  }

  // Update the last (assistant) message immutably.
  const patchLast = (fn: (m: Msg) => Msg) =>
    setMsgs(m => { const cp = [...m]; cp[cp.length - 1] = fn(cp[cp.length - 1]); return cp })

  // Insert or update a tool card on the current assistant message.
  const upsertTool = (callId: string, patch: Partial<ToolCall> & { name?: string; args?: unknown }) =>
    patchLast(m => {
      const tools = [...(m.tools ?? [])]
      const i = tools.findIndex(t => t.callId === callId)
      if (i >= 0) tools[i] = { ...tools[i], ...patch }
      else tools.push({ callId, name: patch.name ?? '', args: patch.args, status: patch.status ?? 'running', result: patch.result })
      return { ...m, tools }
    })

  // Changing or detaching the workspace must drop auto-approve: trust is per-folder, so a new folder
  // must never inherit a previous folder's auto-approval (it would run write_file/code unconfirmed).
  const changeWorkspace = (v: string) => { setWorkspace(v); setAutoApprove(false) }

  const decide = async (callId: string, approved: boolean) => {
    upsertTool(callId, { status: approved ? 'running' : 'error', result: approved ? undefined : 'declined' })
    await api.toolDecision(callId, approved)
  }

  const send = async () => {
    if (!input.trim() || !running || busy) return
    // Note any newly-attached files so the model knows to read them from the workspace.
    const note = staged.length ? `[Attached files in the workspace: ${staged.map(f => f.name).join(', ')}]\n\n` : ''
    const text = note + input; setInput(''); setBusy(true)
    const history = msgs.map(m => ({ role: m.role, content: m.content }))
    setMsgs(m => [...m, { role: 'user', content: text }, { role: 'assistant', content: '' }])
    const sid = await ensureSession(input.trim() || staged.map(f => f.name).join(', '))
    if (sid) await api.appendChatMessage(sid, { role: 'user', content: text })
    setStaged([])
    try {
      const { text: answer, tools } = supportsTools ? await streamAgent(text, history, sid) : await streamPlain(text, history)
      // Persist the assistant answer AND the tool-call trace (TOOL-7), so reopening the chat still
      // shows what was approved / executed.
      if (sid) await api.appendChatMessage(sid, { role: 'assistant', content: answer, tools: tools.length ? JSON.stringify(tools) : undefined })
    } catch (e) {
      patchLast(m => ({ ...m, content: (m.content || '') + '\nError: ' + ((e as Error)?.message || 'unknown') }))
    } finally {
      setBusy(false)
      loadSessions()
    }
  }

  // TOOL-1: stream the agentic loop (tokens + inline tool calls) from the server. Returns the final
  // visible assistant text plus the tool-call trace (for persistence / audit).
  const streamAgent = async (text: string, history: { role: string; content: string }[], sid: string | null): Promise<{ text: string; tools: ToolCall[] }> => {
    // Local mirror of the tool cards, so we can persist the trace without racing React state.
    const trace: ToolCall[] = []
    const upTrace = (callId: string, patch: Partial<ToolCall> & { name?: string; args?: unknown }) => {
      const i = trace.findIndex(t => t.callId === callId)
      if (i >= 0) trace[i] = { ...trace[i], ...patch }
      else trace.push({ callId, name: patch.name ?? '', args: patch.args, status: patch.status ?? 'running', result: patch.result })
    }
    const resp = await fetch(api.agentUrl(), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: [...history, { role: 'user', content: text }],
        systemPrompt: sys, temperature: temp, topP, maxTokens: maxTok,
        workspace: workspace || undefined, sessionId: sid || undefined, webEnabled, autoApprove,
      }),
    })
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const reader = resp.body?.getReader(); const dec = new TextDecoder(); let acc = '', buffer = ''
    while (reader) {
      const { done, value } = await reader.read()
      if (value) buffer += dec.decode(value, { stream: true })
      const lines = buffer.split('\n'); buffer = done ? '' : (lines.pop() ?? '')
      for (const line of lines) {
        const t = line.trim(); if (!t.startsWith('data:')) continue
        const d = t.slice(5).trim(); if (!d) continue
        let e: any; try { e = JSON.parse(d) } catch { continue }
        switch (e.type) {
          case 'token': acc += e.text || ''; patchLast(m => ({ ...m, content: acc })); break
          case 'tool_call': upsertTool(e.callId, { name: e.name, args: e.args, status: 'running' }); upTrace(e.callId, { name: e.name, args: e.args, status: 'running' }); break
          case 'confirm': upsertTool(e.callId, { name: e.name, args: e.args, status: 'confirm' }); upTrace(e.callId, { name: e.name, args: e.args, status: 'confirm' }); break
          case 'tool_result': upsertTool(e.callId, { name: e.name, status: e.ok ? 'ok' : 'error', result: e.result }); upTrace(e.callId, { name: e.name, status: e.ok ? 'ok' : 'error', result: e.result }); break
          case 'error': acc += (acc ? '\n' : '') + 'Error: ' + (e.message || 'unknown'); patchLast(m => ({ ...m, content: acc })); break
          case 'done': break
        }
      }
      if (done) break
    }
    return { text: acc, tools: trace }
  }

  // Plain streaming for models without tool support — talk straight to llama-server (RUN-3).
  const streamPlain = async (text: string, history: { role: string; content: string }[]): Promise<{ text: string; tools: ToolCall[] }> => {
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
        try { const j = JSON.parse(d); const c = j.choices?.[0]?.delta?.content; if (c) { acc += c; patchLast(m => ({ ...m, content: acc })) } } catch { /* partial */ }
      }
      if (done) break
    }
    return { text: acc, tools: [] }
  }

  const vramUsed = gpu ? g(gpu.telemetry.vramUsedBytes) : '0'
  const gpuUtil = gpu ? gpu.telemetry.utilizationPercent.toFixed(0) : '0'
  const wsName = workspace ? workspace.replace(/[\\/]+$/, '').split(/[\\/]/).pop() : ''

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
                    onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); renameActive(e.currentTarget.value) } else if (e.key === 'Escape') setEditingTitle(false) }} />
                : <div style={{ fontWeight: 600, fontSize: 14, cursor: activeId ? 'text' : 'default' }} title={activeId ? 'Click to rename' : ''} onClick={() => activeId && setEditingTitle(true)}>{activeTitle}</div>}
              <div className="mono faint" style={{ fontSize: 11 }}>{running ? `${runtime?.modelId} · ${runtime?.quantLabel} · :${runtime?.port}` : 'no model loaded'}</div>
            </div>
          </div>
          <div className="fx ac gap8">
            {running && (supportsTools ? <span className="tag" title="This model's chat template supports tool calling">Tools</span> : <span className="tag" style={{ opacity: 0.5 }} title="This model's template has no tool-call support">No tools</span>)}
            {wsName && <span className="tag fx ac gap6" title={`Filesystem access scoped to ${workspace}`}><Folder size={12} /> {wsName}</span>}
            {webEnabled && <span className="tag fx ac gap6" title="Web tools enabled for this chat"><Globe size={12} /> Web</span>}
            <span className="tag">Local history</span>
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
                {m.tools?.map(t => <ToolCard key={t.callId} t={t} onDecide={decide} />)}
                <div className="answer">{m.content || (busy && i === msgs.length - 1 && !(m.tools?.length) ? <span className="faint mono" style={{ fontSize: 13 }}>generating<span style={{ animation: 'pulse 1.2s infinite' }}>▋</span></span> : '')}</div>
              </div>)}
          <div ref={end} />
        </div>

        {running && supportsTools && (
          <>
            <div className="toolbar">
              <button className="toolchip" onClick={() => fileInput.current?.click()} disabled={busy} title={workspace ? 'Attach files (copied into the attached folder)' : 'Attach files (kept in this chat’s workspace)'}><Paperclip /> Attach files</button>
              <input ref={fileInput} type="file" multiple hidden onChange={onFilesPicked} />
              <button className={`toolchip ${webEnabled ? 'on' : ''}`} onClick={() => setWebEnabled(v => !v)} disabled={busy} title="Allow web search / fetch for this chat (logged; off by default)"><Globe /> Web {webEnabled ? 'on' : 'off'}</button>
              {wsEdit || workspace
                ? <span className="fx ac gap6">
                    <input className="wsinput mono" placeholder="C:\path\to\folder" value={workspace} disabled={busy}
                      onChange={e => changeWorkspace(e.target.value)} onBlur={() => setWsEdit(false)}
                      onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); setWsEdit(false) } }} autoFocus={wsEdit && !workspace} />
                    {workspace && <button className="toolchip" onClick={() => changeWorkspace('')} disabled={busy} title="Detach folder — files go back to this chat’s workspace">✕</button>}
                  </span>
                : <button className="toolchip" onClick={() => setWsEdit(true)} disabled={busy} title="Scope file + code tools to a real folder (else a per-chat workspace is used)"><Folder /> Attach folder</button>}
              <button className={`toolchip ${autoApprove ? 'on' : ''}`} onClick={() => setAutoApprove(v => !v)} disabled={busy} title="Auto-approve write_file / code for this chat (off = confirm each). Auto-workspace writes never prompt regardless."><Bolt /> Auto-approve {autoApprove ? 'on' : 'off'}</button>
            </div>
            {staged.length > 0 && (
              <div className="stagedrow">
                {staged.map((f, i) => (
                  <span key={i} className="stagedchip mono"><FileIcon size={12} /> {f.name}
                    <button onClick={() => setStaged(s => s.filter((_, j) => j !== i))} title="Remove from this message">✕</button>
                  </span>
                ))}
              </div>
            )}
          </>
        )}

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

// Inline tool-call card (TOOL-7): name, arguments, and result — never hidden. Side-effect calls in
// the `confirm` state show Approve / Deny (TOOL-6).
function ToolCard({ t, onDecide }: { t: ToolCall; onDecide: (id: string, ok: boolean) => void }) {
  const dot = t.status === 'ok' ? '#3fb950' : t.status === 'error' ? '#e5484d' : t.status === 'confirm' ? 'var(--iris)' : 'var(--muted)'
  const argStr = (() => { try { return JSON.stringify(t.args) } catch { return '' } })()
  return (
    <div className="toolcard">
      <div className="fx ac gap8" style={{ justifyContent: 'space-between' }}>
        <div className="fx ac gap6">
          <span className="tcdot" style={{ background: dot }} />
          <span className="mono" style={{ fontSize: 12, fontWeight: 600 }}>{t.name || 'tool'}</span>
          <span className="mono faint" style={{ fontSize: 11 }}>{t.status}</span>
        </div>
        {t.status === 'confirm' && (
          <span className="fx ac gap6">
            <button className="btn btn-sm btn-iris" onClick={() => onDecide(t.callId, true)}>Approve</button>
            <button className="btn btn-sm" onClick={() => onDecide(t.callId, false)}>Deny</button>
          </span>
        )}
      </div>
      {argStr && argStr !== '{}' && <div className="tcargs mono">{argStr}</div>}
      {t.status === 'confirm' && <div className="tcwarn mono fx gap6"><Alert size={13} /><span>{t.name === 'code' ? 'Runs real code on your machine with your account’s permissions — this is not a sandbox.' : 'Writes to your attached folder.'} Approve only if you trust it.</span></div>}
      {t.result != null && t.result !== '' && <div className={`tcresult mono ${t.status === 'error' ? 'err' : ''}`}>{t.result.length > 1200 ? t.result.slice(0, 1200) + ' …' : t.result}</div>}
    </div>
  )
}
