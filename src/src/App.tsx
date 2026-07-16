import { useEffect, useState, useCallback, type ReactNode } from 'react'
import { api, type MachineProfile, type RuntimeStatus } from './lib/api'
import Dashboard from './pages/Dashboard'
import Browser from './pages/Browser'
import Library from './pages/Library'
import Chat from './pages/Chat'
import Privacy from './pages/Privacy'
import Settings from './pages/Settings'
import Onboarding from './pages/Onboarding'
import { KMark } from './components/icons'

export type View = 'dashboard' | 'browser' | 'library' | 'chat' | 'privacy' | 'settings'

const gb = (b: number) => (b / 1024 ** 3).toFixed(1)

const NAV: { id: View; label: string; icon: ReactNode; badge?: boolean }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="5" rx="1"/><rect x="13" y="10" width="8" height="11" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/></svg> },
  { id: 'browser', label: 'Model browser', icon: <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4"/></svg> },
  { id: 'library', label: 'Library', icon: <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 4h5v16H4zM10 4h4v16h-4zM16 5l4 1-3 15-4-1z"/></svg>, badge: true },
  { id: 'chat', label: 'Chat', icon: <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a8 8 0 0 1-8 8H4l2-3a8 8 0 1 1 15-5z"/></svg> },
]
const NAV_SYS: { id: View; label: string; icon: ReactNode }[] = [
  { id: 'privacy', label: 'Privacy & network', icon: <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 3l7 3v5c0 4.5-3 8.5-7 10-4-1.5-7-5.5-7-10V6z"/></svg> },
  { id: 'settings', label: 'Settings', icon: <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.1-1.2l2-1.5-2-3.5-2.4 1a7 7 0 0 0-2-1.2L14 1h-4l-.5 2.6a7 7 0 0 0-2 1.2l-2.4-1-2 3.5 2 1.5A7 7 0 0 0 5 12a7 7 0 0 0 .1 1.2l-2 1.5 2 3.5 2.4-1a7 7 0 0 0 2 1.2L10 23h4l.5-2.6a7 7 0 0 0 2-1.2l2.4 1 2-3.5-2-1.5A7 7 0 0 0 19 12z"/></svg> },
]

export default function App() {
  const [view, setView] = useState<View>('dashboard')
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [sideMini, setSideMini] = useState(false)
  const [machine, setMachine] = useState<MachineProfile | null>(null)
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null)
  const [libCount, setLibCount] = useState(0)
  const [showOnboarding, setShowOnboarding] = useState(() => !localStorage.getItem('kayon_onboarded'))

  // shared 1 Hz telemetry (dashboard + chat + sidecard all read this)
  useEffect(() => {
    const tick = async () => {
      const [h, r] = await Promise.all([api.hardware(), api.runtimeStatus()])
      if (h.ok && h.data) setMachine(h.data)
      if (r.ok && r.data) setRuntime(r.data)
    }
    tick()
    const iv = setInterval(tick, 1000)
    return () => clearInterval(iv)
  }, [])

  useEffect(() => {
    api.library().then(r => { if (r.ok && r.data) setLibCount(r.data.length) })
  }, [view])

  const finishOnboarding = useCallback(() => {
    localStorage.setItem('kayon_onboarded', 'true')
    setShowOnboarding(false)
  }, [])

  const gpu = machine?.gpus?.[0]
  const gpuShort = gpu ? gpu.name.replace(/NVIDIA GeForce /i, '').replace(/ Laptop GPU/i, '') : 'No GPU'
  const vramUsed = gpu ? gb(gpu.telemetry.vramUsedBytes) : '0'
  const vramTotal = gpu ? gb(gpu.totalVramBytes) : '0'
  const vramPct = gpu && gpu.totalVramBytes > 0 ? (gpu.telemetry.vramUsedBytes / gpu.totalVramBytes) * 100 : 0
  const running = runtime?.kind === 'running'

  const page = () => {
    switch (view) {
      case 'dashboard': return <Dashboard machine={machine} runtime={runtime} onReonboard={() => setShowOnboarding(true)} />
      case 'browser': return <Browser machine={machine} goLibrary={() => setView('library')} />
      case 'library': return <Library goBrowser={() => setView('browser')} goPrivacy={() => setView('privacy')} onChange={() => api.library().then(r => r.ok && r.data && setLibCount(r.data.length))} goChat={() => setView('chat')} />
      case 'chat': return <Chat machine={machine} runtime={runtime} />
      case 'privacy': return <Privacy />
      case 'settings': return <Settings />
    }
  }

  return (
    <div className={`desk ${theme === 'light' ? 'light' : ''}`}>
      <div className={`kyn ${theme === 'light' ? 'light' : ''}`}>
        <div className="win">
          {/* title bar */}
          <div className="tbar">
            <div className="fx ac gap12">
              <div className="traffic"><button className="tdot tclose"/><button className="tdot tmin"/><button className="tdot tmax"/></div>
              <button className="tmenu" onClick={() => setSideMini(m => !m)} title="Collapse menu" style={{ marginLeft: 6 }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><line x1="9" y1="4" x2="9" y2="20"/></svg>
              </button>
            </div>
            <div className="tcenter">
              <KMark size={20} />
              <span className="kword">Kayon<span className="kdot">.</span></span>
            </div>
            <div className="tright">
              <span className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>{gpuShort}</span>
              <button className="ticon" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')} title="Toggle theme">
                {theme === 'dark'
                  ? <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/></svg>
                  : <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/></svg>}
              </button>
            </div>
          </div>

          {/* body */}
          <div className="shell">
            <aside className={`side ${sideMini ? 'mini' : ''}`}>
              <div className="slabel">Workstation</div>
              {NAV.map(n => (
                <div key={n.id} className={`navitem ${view === n.id ? 'on' : ''}`} onClick={() => setView(n.id)} title={n.label}>
                  <span className="nico">{n.icon}</span>
                  <span className="nlabel">{n.label}</span>
                  {n.badge && libCount > 0 && <span className="nbadge">{libCount}</span>}
                </div>
              ))}
              <div className="slabel">System</div>
              {NAV_SYS.map(n => (
                <div key={n.id} className={`navitem ${view === n.id ? 'on' : ''}`} onClick={() => setView(n.id)} title={n.label}>
                  <span className="nico">{n.icon}</span>
                  <span className="nlabel">{n.label}</span>
                </div>
              ))}
              <div className="sspacer" />
              <div className="sidecard">
                <div className="sc-top">
                  <span className="livedot" style={running ? {} : { background: 'var(--faint)', animation: 'none' }} />
                  <span className="mono" style={{ fontSize: 11, color: 'var(--muted)' }}>ACTIVE MODEL</span>
                </div>
                <div style={{ fontWeight: 600, fontSize: 13.5 }}>{running ? runtime?.modelId : 'None loaded'}</div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--faint)', marginTop: 2 }}>
                  {running ? `${runtime?.quantLabel ?? ''} · :${runtime?.port ?? ''}` : 'load one from Library'}
                </div>
                <div className="mono" style={{ fontSize: 10, color: 'var(--muted)', display: 'flex', justifyContent: 'space-between', marginTop: 12 }}>
                  <span>VRAM</span><span>{vramUsed} / {vramTotal} GB</span>
                </div>
                <div className="mini-meter"><div className="mini-fill" style={{ width: `${vramPct}%` }} /></div>
                <div className="mono sfoot" style={{ fontSize: 10, color: 'var(--faint)', textAlign: 'center', marginTop: 12 }}>Kayon 1.4.1 · private by construction</div>
              </div>
            </aside>

            <main className="content softscroll">
              {page()}
            </main>
          </div>

          {showOnboarding && <Onboarding machine={machine} onFinish={finishOnboarding} goBrowser={() => { finishOnboarding(); setView('browser') }} />}
        </div>
      </div>
    </div>
  )
}
