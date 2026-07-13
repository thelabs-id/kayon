import { useState, useEffect } from 'react'
import Dashboard from './pages/Dashboard'
import Browser from './pages/Browser'
import Library from './pages/Library'
import Chat from './pages/Chat'
import Settings from './pages/Settings'
import Onboarding from './pages/Onboarding'

type Page = 'dashboard' | 'browser' | 'library' | 'chat' | 'settings' | 'onboarding'

const navItems: { id: Page; icon: string; label: string }[] = [
  { id: 'dashboard', icon: '⬡', label: 'Dashboard' },
  { id: 'browser', icon: '⊞', label: 'Browser' },
  { id: 'library', icon: '📁', label: 'Library' },
  { id: 'chat', icon: '💬', label: 'Chat' },
  { id: 'settings', icon: '⚙', label: 'Settings' },
]

function KayonLogo() {
  return (
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M32 4C23 18 18 26 18 36C18 48 24 54 32 60C40 54 46 48 46 36C46 26 41 18 32 4Z"
        fill="none" stroke="#F3EEE3" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
      <line x1="32" y1="16" x2="32" y2="52" stroke="#F3EEE3" strokeWidth="3" strokeLinecap="round"/>
      <circle cx="32" cy="32" r="5" fill="#E0916B"/>
    </svg>
  )
}

export default function App() {
  const [page, setPage] = useState<Page>('dashboard')
  const [gpuName, setGpuName] = useState<string>('')
  const [gpuFree, setGpuFree] = useState<number>(0)
  const [gpuTotal, setGpuTotal] = useState<number>(0)
  const [showOnboarding, setShowOnboarding] = useState(false)

  useEffect(() => {
    fetch('/api/hardware')
      .then(r => r.json())
      .then(d => {
        if (d.ok && d.data) {
          const gpus = d.data.gpus || []
          if (gpus.length > 0) {
            setGpuName(gpus[0].name)
            setGpuFree(gpus[0].telemetry.vramFreeBytes)
            setGpuTotal(gpus[0].totalVramBytes)
          }
        }
      })
      .catch(() => {})

    const prefs = localStorage.getItem('kayon_onboarded')
    if (!prefs) setShowOnboarding(true)
  }, [])

  const handleOnboardingDone = () => {
    localStorage.setItem('kayon_onboarded', 'true')
    setShowOnboarding(false)
  }

  if (showOnboarding) {
    return <Onboarding onDone={handleOnboardingDone} />
  }

  const renderPage = () => {
    switch (page) {
      case 'dashboard': return <Dashboard />
      case 'browser': return <Browser />
      case 'library': return <Library />
      case 'chat': return <Chat />
      case 'settings': return <Settings />
      default: return <Dashboard />
    }
  }

  const gb = (bytes: number) => (bytes / (1024**3)).toFixed(1)
  const gpuFreePct = gpuTotal > 0 ? (gpuFree / gpuTotal * 100).toFixed(0) : '0'

  return (
    <div className="app-layout">
      <nav className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <KayonLogo />
            <span className="sidebar-logo-text">Kayon</span>
          </div>
        </div>
        <div className="sidebar-nav">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`nav-item ${page === item.id ? 'active' : ''}`}
              onClick={() => setPage(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </div>
        <div className="sidebar-footer">
          {gpuName ? (
            <div className="gpu-indicator">
              <span className="gpu-dot"></span>
              <span style={{flex:1, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap'}}>{gpuName}</span>
            </div>
          ) : (
            <div className="gpu-indicator">
              <span className="gpu-dot" style={{background:'#888'}}></span>
              <span>No GPU</span>
            </div>
          )}
          {gpuTotal > 0 && (
            <div style={{marginTop:8}}>
              <div className="meter-bar">
                <div className="meter-fill" style={{
                  width: `${100 - Number(gpuFreePct)}%`,
                  background: Number(gpuFreePct) > 20 ? 'var(--gpu-green)' : Number(gpuFreePct) > 5 ? 'var(--warning)' : 'var(--danger)'
                }}></div>
              </div>
              <div style={{fontSize:11, marginTop:4, color:'var(--text-sidebar-muted)', fontFamily:'var(--font-mono)'}}>
                {gb(gpuTotal - gpuFree)} / {gb(gpuTotal)} GB
              </div>
            </div>
          )}
        </div>
      </nav>
      <main className="main-content animate-in">
        {renderPage()}
      </main>
    </div>
  )
}
