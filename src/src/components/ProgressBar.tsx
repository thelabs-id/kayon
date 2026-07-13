interface Props {
  percent: number
  throughputBps?: number
  etaSeconds?: number
  status?: string
}

function formatBytes(bps: number): string {
  if (bps < 1024) return `${bps} B/s`
  if (bps < 1024 * 1024) return `${(bps / 1024).toFixed(1)} KB/s`
  return `${(bps / (1024 * 1024)).toFixed(1)} MB/s`
}

function formatEta(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`
  if (sec < 3600) return `${Math.round(sec / 60)}m`
  return `${(sec / 3600).toFixed(1)}h`
}

export default function ProgressBar({ percent, throughputBps, etaSeconds, status }: Props) {
  return (
    <div style={{ width: '100%' }}>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: 12, color: 'var(--text-muted)' }}>
        <span className="mono">{percent.toFixed(1)}%</span>
        <span>
          {throughputBps ? formatBytes(throughputBps) : ''}
          {etaSeconds != null ? ` \u00b7 ${formatEta(etaSeconds)} left` : ''}
          {status && status !== 'active' ? ` \u00b7 ${status}` : ''}
        </span>
      </div>
    </div>
  )
}
