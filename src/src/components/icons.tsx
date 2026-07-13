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
