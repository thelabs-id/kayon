import { useEffect, useRef } from 'react'

interface Props {
  open: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
  actions?: React.ReactNode
}

export default function Modal({ open, onClose, title, children, actions }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (open) ref.current?.focus()
  }, [open])

  if (!open) return null

  return (
    <div
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div ref={ref} tabIndex={-1} style={{
        background: 'var(--bg-card)', borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow-lg)',
        padding: 24, maxWidth: 440, width: '90%', outline: 'none',
      }}>
        <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>{title}</h3>
        <div style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 20, lineHeight: 1.6 }}>{children}</div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
          <button className="btn btn-secondary btn-sm" onClick={onClose}>Cancel</button>
          {actions}
        </div>
      </div>
    </div>
  )
}
