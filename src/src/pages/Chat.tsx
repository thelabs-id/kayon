import { useEffect, useRef, useState } from 'react'
import { api, type RuntimeStatus } from '../lib/api'

interface Msg { role: 'user'|'assistant'|'system'; content: string }

export default function Chat() {
  const [status, setStatus] = useState<RuntimeStatus|null>(null)
  const [msgs, setMsgs] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [sysPrompt, setSysPrompt] = useState('You are a helpful assistant.')
  const [temp, setTemp] = useState(0.7)
  const [busy, setBusy] = useState(false)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const iv = setInterval(async () => { const r = await api.runtimeStatus(); if (r.ok && r.data) setStatus(r.data) }, 1500)
    return () => clearInterval(iv)
  }, [])
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [msgs])

  const ready = status?.kind === 'running'

  const send = async () => {
    if (!input.trim() || !ready || busy) return
    const um: Msg = { role: 'user', content: input }
    const text = input; setInput(''); setBusy(true)
    setMsgs(m => [...m, um, { role: 'assistant', content: '' }])
    try {
      const port = status.port || 8080
      const resp = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({
          messages: [{role:'system',content:sysPrompt}, ...msgs.filter(m=>m.role!=='system').map(m=>({role:m.role,content:m.content})), {role:'user',content:text}],
          temperature: temp, max_tokens: 512, stream: true,
        }),
      })
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const reader = resp.body?.getReader(); const dec = new TextDecoder(); let acc = ''
      while (reader) {
        const { done, value } = await reader.read(); if (done) break
        for (const line of dec.decode(value, {stream:true}).split('\n')) {
          if (!line.startsWith('data: ')) continue; const d = line.slice(6).trim(); if (d === '[DONE]') continue
          try { const j = JSON.parse(d); const c = j.choices?.[0]?.delta?.content; if (c) { acc += c; setMsgs(m => { const cp=[...m]; cp[cp.length-1]={role:'assistant',content:acc}; return cp }) } } catch {}
        }
      }
    } catch (e: any) { setMsgs(m => [...m, {role:'system',content:'Error: '+(e?.message||'unknown')}]) }
    finally { setBusy(false) }
  }

  return (
    <div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:16}}>
        <h1 className="page-title" style={{margin:0}}>Chat</h1>
        <span className={`badge ${ready?'badge-success':'badge-neutral'}`}>{status?.kind || 'stopped'}</span>
      </div>
      {!ready ? (
        <div className="card"><div className="empty-state">
          <div className="empty-state-title">No model loaded</div>
          <div className="empty-state-text">{status?.kind==='starting'?'Starting...':'Open Library and click Load & Chat.'}</div>
        </div></div>
      ) : (
        <div style={{display:'flex',flexDirection:'column',height:'calc(100vh - 140px)'}}>
          <div className="card" style={{padding:12,marginBottom:12}}>
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:12}}>
              <div><label className="label">System prompt</label><input className="input" value={sysPrompt} onChange={e=>setSysPrompt(e.target.value)} style={{fontSize:13}}/></div>
              <div><label className="label">Temperature: {temp.toFixed(2)}</label><input type="range" min={0} max={2} step={0.05} value={temp} onChange={e=>setTemp(+e.target.value)} style={{width:'100%'}}/></div>
              <div><button className="btn btn-secondary btn-sm" style={{marginTop:24}} onClick={()=>api.runtimeStop()}>Stop model</button></div>
            </div>
          </div>
          <div className="card" style={{flex:1,overflow:'hidden',display:'flex',flexDirection:'column'}}>
            <div style={{flex:1,overflowY:'auto',padding:16,display:'flex',flexDirection:'column',gap:12}}>
              {msgs.length===0 && <div className="empty-state"><div className="empty-state-text">Start chatting...</div></div>}
              {msgs.map((m,i) => (
                <div key={i} className={`chat-message ${m.role}`} style={{alignSelf:m.role==='user'?'flex-end':'flex-start',maxWidth:'80%'}}>
                  {m.content || (m.role==='assistant'&&busy ? <span style={{opacity:0.5}}>...</span> : null)}
                </div>
              ))}
              <div ref={end}/>
            </div>
            <div style={{padding:12,borderTop:'1px solid var(--border)',display:'flex',gap:8}}>
              <input className="input" placeholder="Send a message..." value={input} disabled={busy}
                onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}}}/>
              <button className="btn btn-primary" onClick={send} disabled={busy||!input.trim()}>Send</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
