import type { VerdictKind } from '../lib/api'

const config: Record<VerdictKind, { label: string; className: string; color: string }> = {
  FITS_FULLY: { label: 'Fits Fully', className: 'badge-success', color: '#22c55e' },
  FITS_TIGHT: { label: 'Fits Tight', className: 'badge-warning', color: '#eab308' },
  GPU_CPU_SPLIT: { label: 'GPU+CPU Split', className: 'badge-accent', color: '#E0916B' },
  CPU_ONLY: { label: 'CPU Only', className: 'badge-info', color: '#3b82f6' },
  EXCEEDS_MACHINE: { label: 'Exceeds', className: 'badge-danger', color: '#ef4444' },
  UNVERIFIED_ARCH: { label: 'Unverified Arch', className: 'badge-neutral', color: '#888' },
}

interface Props {
  verdict: VerdictKind
  showTooltip?: boolean
  explainability?: string
  nGpuLayers?: number
}

export default function VerdictBadge({ verdict, showTooltip, explainability, nGpuLayers }: Props) {
  const cfg = config[verdict] || config.UNVERIFIED_ARCH
  const tooltip = [
    explainability,
    nGpuLayers !== undefined ? `n_gpu_layers: ${nGpuLayers}` : null,
  ].filter(Boolean).join('\n')

  return (
    <span
      className={`badge ${cfg.className}`}
      style={{ cursor: showTooltip ? 'help' : 'default' }}
      title={showTooltip ? tooltip : undefined}
    >
      {cfg.label}
    </span>
  )
}
