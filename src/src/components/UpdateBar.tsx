import { useEffect, useState } from 'react'
import { api, type UpdateStatus } from '../lib/api'
import { Alert, Bolt } from './icons'

const mb = (b: number) => (b / 1048576).toFixed(1)

/**
 * UPD: the update affordance.
 *
 * Announce, then wait. A newer version is never fetched until the user clicks Download (UPD-2, the
 * same rule models live under), and it is never installed until they click Relaunch to update. The
 * bar renders nothing at all when there is nothing to say, so the app is not decorated with a
 * permanent nag.
 */
export default function UpdateBar() {
  const [st, setSt] = useState<UpdateStatus | null>(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    let dead = false
    const poll = async () => {
      const r = await api.updateStatus()
      if (!dead && r.ok && r.data) setSt(r.data)
    }
    poll()
    // Slow poll: this only has to notice the launch check finishing, or a download progressing.
    const t = setInterval(poll, 2000)
    return () => { dead = true; clearInterval(t) }
  }, [])

  if (!st || !st.supported || dismissed) return null
  if (!st.available && !st.error) return null

  const download = async () => { setSt(await api.updateDownload().then(r => r.data ?? st)) }
  const relaunch = async () => { await api.updateInstall() }

  return (
    <div className="updbar">
      {st.error ? <Alert size={13} /> : <Bolt size={13} />}
      <div className="updtext">
        {st.error ? (
          <>Update failed: <span className="mono">{st.error}</span></>
        ) : st.ready ? (
          <><b>Kayon {st.available} is ready.</b> Your chats and library are untouched by updating.</>
        ) : st.downloading ? (
          <>Downloading {st.available}
            {st.totalBytes ? ` — ${mb(st.downloadedBytes)} of ${mb(st.totalBytes)} MB` : ` — ${mb(st.downloadedBytes)} MB`}</>
        ) : (
          <><b>Kayon {st.available} is available.</b> You're on {st.current}. Nothing is downloaded until you say so.</>
        )}
      </div>
      {st.ready
        ? <button className="btn btn-iris btn-sm" onClick={relaunch}>Relaunch to update</button>
        : st.downloading
          ? null
          : !st.error && <button className="btn btn-iris btn-sm" onClick={download}>Download</button>}
      <button className="updclose" onClick={() => setDismissed(true)} title="Not now">×</button>
    </div>
  )
}
