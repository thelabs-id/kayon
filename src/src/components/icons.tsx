import type { VerdictKind } from '../lib/api'

export function KMark({ size = 20, iris }: { size?: number; iris?: boolean }) {
  return (
    <svg className="kmk" viewBox="0 0 64 64" width={size} height={size}>
      <path className="ko" d="M32 7 C23 18 18 24 18 34 C18 45 24 51 32 57 C40 51 46 45 46 34 C46 24 41 18 32 7 Z" style={iris ? { stroke: 'var(--iris)' } : undefined} />
      <line className="ko" x1="32" y1="15" x2="32" y2="49" />
      <circle className="khub" cx="32" cy="31" r="4" />
    </svg>
  )
}

const VMAP: Record<VerdictKind, { label: string; c: string }> = {
  FITS_FULLY: { label: 'FITS FULLY', c: 'var(--v-full)' },
  FITS_TIGHT: { label: 'FITS TIGHT', c: 'var(--v-tight)' },
  GPU_CPU_SPLIT: { label: 'GPU + CPU SPLIT', c: 'var(--v-split)' },
  CPU_ONLY: { label: 'CPU ONLY', c: 'var(--v-cpu)' },
  EXCEEDS_MACHINE: { label: 'EXCEEDS MACHINE', c: 'var(--v-exceed)' },
  UNVERIFIED_ARCH: { label: 'UNVERIFIED ARCH', c: 'var(--v-unv)' },
}

export function verdictColor(v: VerdictKind) { return VMAP[v].c }

export function VerdictChip({ v }: { v: VerdictKind }) {
  const { label, c } = VMAP[v]
  return (
    <span className="verdict" style={{ color: c, background: `color-mix(in oklab, ${c} 15%, transparent)` }}>
      <span className="vsw" style={{ background: c }} />
      {label}
    </span>
  )
}

export function Check() {
  return <svg className="checkmk" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M20 6L9 17l-5-5" /></svg>
}

// Line-art glyphs for the Chat tool controls — monochrome, stroke=currentColor, matching the nav
// icon language (never full-colour emoji, which clash with the app's look and feel).
function Glyph({ size = 14, children }: { size?: number; children: React.ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flex: 'none' }}>
      {children}
    </svg>
  )
}

export const Globe = ({ size }: { size?: number }) => (
  <Glyph size={size}><circle cx="12" cy="12" r="9" /><path d="M3 12h18" /><path d="M12 3c2.5 2.7 3.8 5.7 3.8 9s-1.3 6.3-3.8 9c-2.5-2.7-3.8-5.7-3.8-9S9.5 5.7 12 3z" /></Glyph>
)
export const Folder = ({ size }: { size?: number }) => (
  <Glyph size={size}><path d="M3 7a2 2 0 0 1 2-2h3.5l2 2H19a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" /></Glyph>
)
export const Bolt = ({ size }: { size?: number }) => (
  <Glyph size={size}><path d="M13 2L4.5 13.5H11l-1 8.5L19.5 10H13z" /></Glyph>
)
export const Alert = ({ size }: { size?: number }) => (
  <Glyph size={size}><path d="M12 3.5l9 15.5H3z" /><line x1="12" y1="10" x2="12" y2="14" /><line x1="12" y1="16.7" x2="12" y2="16.8" /></Glyph>
)
/** Disclosure caret. Points right when collapsed, down when open. */
export const Caret = ({ size = 12, open }: { size?: number; open?: boolean }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="2.5"
    strokeLinecap="round" strokeLinejoin="round"
    style={{ flex: 'none', transform: open ? 'rotate(90deg)' : 'none', transition: 'transform .12s ease' }}>
    <path d="M9 6l6 6-6 6" />
  </svg>
)
export const Paperclip = ({ size }: { size?: number }) => (
  <Glyph size={size}><path d="M21.4 11l-9.2 9.2a5 5 0 0 1-7-7l9.2-9.2a3.3 3.3 0 0 1 4.7 4.7l-9.2 9.2a1.7 1.7 0 0 1-2.3-2.3l8.5-8.5" /></Glyph>
)
export const FileIcon = ({ size }: { size?: number }) => (
  <Glyph size={size}><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></Glyph>
)
