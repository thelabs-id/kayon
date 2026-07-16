import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { marked } from 'marked'
import * as pdfjs from 'pdfjs-dist'
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url'
import { api } from '../lib/api'
import { Alert, FileIcon } from './icons'

// The engine and every asset it fetches are local (TOOL-8): viewing a document never touches the
// network. pdf.js would otherwise default to a CDN for these, which is exactly the silent egress
// PRIV-1 forbids.
pdfjs.GlobalWorkerOptions.workerSrc = workerUrl
const PDF_ASSETS = {
  cMapUrl: '/pdfjs/cmaps/',
  cMapPacked: true,
  standardFontDataUrl: '/pdfjs/standard_fonts/',
  wasmUrl: '/pdfjs/wasm/',
}

export type ViewKind = 'md' | 'text' | 'image' | 'pdf' | 'html'

export function kindOf(name: string): ViewKind {
  const ext = name.toLowerCase().split('.').pop() ?? ''
  if (['md', 'markdown'].includes(ext)) return 'md'
  // SVG sits with HTML on purpose: it is a document that can carry script, not an inert picture.
  if (['html', 'htm', 'svg'].includes(ext)) return 'html'
  if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'ico'].includes(ext)) return 'image'
  if (ext === 'pdf') return 'pdf'
  return 'text'
}

/** Something in the artifact that will not work offline. Surfaced rather than dropped (OD-12). */
interface Block { uri: string; directive: string }

const MAX_BLOCKS = 12

/**
 * Remote references in an artifact, found by reading it rather than by watching it fail.
 *
 * The earlier design let the frame report its own CSP violations. That needed scripts to run in the
 * frame, and scripts are exactly what must not run: a CSP blocks `fetch`, but nothing in it blocks
 * `location.href = 'https://…'`, so a script could still navigate the frame out and take the page's
 * data with it in the query string. Reading the markup in the parent gets the same honest answer
 * with nothing executing.
 */
function remoteRefs(html: string): Block[] {
  const out = new Map<string, string>()
  // Only things the page would *fetch*. An <a href> to a remote site is not a load — nothing
  // requests it, and in this frame it cannot even be followed — so flagging it as blocked would be
  // a lie about what the viewer withheld.
  const patterns: [RegExp, string][] = [
    [/\bsrc\s*=\s*["']?((?:https?:)?\/\/[^"'\s>]+)/gi, 'src'],
    [/<link\b[^>]*\bhref\s*=\s*["']?((?:https?:)?\/\/[^"'\s>]+)/gi, 'stylesheet'],
    [/url\(\s*["']?((?:https?:)?\/\/[^"')\s]+)/gi, 'css url()'],
  ]
  for (const [re, kind] of patterns) {
    for (let m = re.exec(html); m && out.size < MAX_BLOCKS; m = re.exec(html)) {
      const uri = m[1].slice(0, 300)
      if (!out.has(uri)) out.set(uri, kind)
    }
  }
  return [...out].map(([uri, directive]) => ({ uri, directive }))
}

/** Whether the artifact carries script that this viewer deliberately will not run. */
const hasScript = (html: string) => /<script[\s>]|\son[a-z]+\s*=\s*["']/i.test(html)

// Belt to the sandbox attribute's braces. `default-src 'none'` denies every fetch; no script-src
// entry at all means no script runs even if the sandbox were widened by accident.
const FRAME_CSP = [
  "default-src 'none'",
  "style-src 'unsafe-inline'",
  "img-src data:",
  "font-src data:",
  "media-src data:",
  "base-uri 'none'",
  "form-action 'none'",
].join('; ')

interface Theme { dark: boolean; ink: string; muted: string; line: string; code: string; link: string; paper: string }

const themeKey = (t: Theme) => `${t.dark}|${t.ink}|${t.muted}|${t.line}|${t.code}|${t.link}|${t.paper}`

/**
 * The app's live palette, read from the CSS variables on `.kyn` rather than restated here. The
 * frame has an opaque origin, so it inherits none of the app's styling and has to be handed
 * concrete values — but hardcoding a second palette would drift the moment a theme changes, and a
 * wrong guess renders text the same colour as its background.
 */
function readTheme(): Theme {
  const root = typeof document !== 'undefined' ? document.querySelector('.kyn') : null
  const cs = root ? getComputedStyle(root) : null
  const v = (name: string, fallback: string) => cs?.getPropertyValue(name).trim() || fallback
  return {
    dark: !root?.classList.contains('light'),
    ink: v('--ink', '#1a1a1a'),
    muted: v('--muted', '#6b6862'),
    line: v('--line2', '#e0ded9'),
    code: v('--panel2', '#f0efec'),
    link: v('--iris', '#c0553a'),
    paper: v('--paper', '#faf9f5'),
  }
}

const frameCss = (t: Theme) => `
  :root{color-scheme:${t.dark ? 'dark' : 'light'}}
  body{margin:0;padding:18px 20px;font:14px/1.65 system-ui,-apple-system,Segoe UI,sans-serif;
    color:${t.ink};background:${t.paper}}
  a{color:${t.link}}
  h1,h2,h3{line-height:1.25;margin:1.3em 0 .5em;color:${t.ink}}
  h1{font-size:1.55em}h2{font-size:1.3em}h3{font-size:1.1em}
  code,pre{font-family:ui-monospace,Consolas,monospace;font-size:.9em}
  code{background:${t.code};padding:.15em .35em;border-radius:4px}
  pre{background:${t.code};padding:12px 14px;border-radius:8px;overflow-x:auto}
  pre code{background:none;padding:0}
  table{border-collapse:collapse;margin:1em 0}
  th,td{border:1px solid ${t.line};padding:6px 10px;text-align:left}
  blockquote{margin:1em 0;padding-left:14px;border-left:3px solid ${t.line};color:${t.muted}}
  hr{border:0;border-top:1px solid ${t.line}}
  img{max-width:100%}
`

/**
 * Renders untrusted markup (a model-written HTML artifact, or markdown we converted) in a frame with
 * an opaque origin, no network, and — deliberately — no script execution.
 *
 * `sandbox=""` denies everything the sandbox can deny, scripts included. That last part is the point
 * and is not conservatism for its own sake: a CSP stops an artifact from *fetching*, but nothing
 * stops a script from doing `location.href = 'https://…?data=' + secrets`. Navigation is not a
 * fetch, no CSP directive governs it (`navigate-to` was never shipped), and the request leaves the
 * machine — which is precisely the silent egress PRIV-1 forbids. Verified: with scripts enabled this
 * frame really did navigate to a live external site. No script, no navigation.
 *
 * The cost is honest and stated in the UI: an artifact's JavaScript does not run here.
 */
function SandboxFrame({ body, theme }: { body: string; theme: Theme }) {
  const css = frameCss(theme)
  const doc = useMemo(
    () =>
      `<!doctype html><html><head><meta charset="utf-8">` +
      `<meta http-equiv="Content-Security-Policy" content="${FRAME_CSP}">` +
      `<style>${css}</style>` +
      `</head><body>${body}</body></html>`,
    [body, css],
  )

  return <iframe className="vframe" sandbox="" srcDoc={doc} title="artifact" />
}

function PdfView({ url }: { url: string }) {
  const [pages, setPages] = useState(0)
  const [page, setPage] = useState(1)
  const [scale, setScale] = useState(1.2)
  const [err, setErr] = useState('')
  const canvas = useRef<HTMLCanvasElement>(null)
  const doc = useRef<pdfjs.PDFDocumentProxy | null>(null)

  useEffect(() => {
    let dead = false
    ;(async () => {
      try {
        const r = await fetch(url)
        if (!r.ok) throw new Error(await r.text())
        const data = new Uint8Array(await r.arrayBuffer())
        const d = await pdfjs.getDocument({ data, ...PDF_ASSETS }).promise
        // Teardown runs through the loading task, which owns the worker; dropping the proxy alone
        // would leak it.
        if (dead) { d.loadingTask.destroy(); return }
        doc.current = d
        setPages(d.numPages)
        setPage(1)
      } catch (e) {
        if (!dead) setErr((e as Error)?.message || 'could not read this PDF')
      }
    })()
    return () => { dead = true; doc.current?.loadingTask.destroy(); doc.current = null }
  }, [url])

  useEffect(() => {
    let task: pdfjs.RenderTask | null = null
    ;(async () => {
      const d = doc.current
      const cv = canvas.current
      if (!d || !cv || page < 1 || page > d.numPages) return
      try {
        const pg = await d.getPage(page)
        const viewport = pg.getViewport({ scale })
        cv.width = Math.floor(viewport.width)
        cv.height = Math.floor(viewport.height)
        task = pg.render({ canvas: cv, viewport })
        await task.promise
      } catch (e) {
        // A cancelled render is the expected outcome of paging quickly; only real faults surface.
        if ((e as Error)?.name !== 'RenderingCancelledException') setErr((e as Error)?.message || 'render failed')
      }
    })()
    return () => { task?.cancel() }
  }, [page, scale, pages])

  if (err) return <div className="vmsg"><Alert size={13} /> {err}</div>

  return (
    <div className="vpdf">
      <div className="vbar">
        <button className="btn btn-line btn-sm" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>Prev</button>
        <span className="mono faint" style={{ fontSize: 12 }}>{pages ? `${page} / ${pages}` : 'loading'}</span>
        <button className="btn btn-line btn-sm" disabled={page >= pages} onClick={() => setPage(p => p + 1)}>Next</button>
        <span className="vgap" />
        <button className="btn btn-line btn-sm" onClick={() => setScale(s => Math.max(0.5, +(s - 0.2).toFixed(1)))}>−</button>
        <span className="mono faint" style={{ fontSize: 12 }}>{Math.round(scale * 100)}%</span>
        <button className="btn btn-line btn-sm" onClick={() => setScale(s => Math.min(3, +(s + 0.2).toFixed(1)))}>+</button>
      </div>
      <div className="vscroll"><canvas ref={canvas} className="vcanvas" /></div>
    </div>
  )
}

/** TOOL-8: read-only view of one workspace file — an attached document or a model-made artifact. */
export default function Viewer({ sessionId, name, onClose }: { sessionId: string; name: string; onClose: () => void }) {
  const kind = kindOf(name)
  const url = api.workspaceFileUrl(sessionId, name)
  const [text, setText] = useState<string | null>(null)
  const [err, setErr] = useState('')
  const [blocks, setBlocks] = useState<Block[]>([])
  // After commit, not during render: a theme toggle re-renders this component, but the new class is
  // not on the DOM until React commits, so reading computed styles mid-render returns the OLD theme
  // and the frame would keep last theme's colours. The equality guard stops the re-render loop.
  const [theme, setTheme] = useState<Theme>(readTheme)
  useLayoutEffect(() => {
    const next = readTheme()
    setTheme(prev => (themeKey(prev) === themeKey(next) ? prev : next))
  })

  useEffect(() => {
    setBlocks([]); setErr(''); setText(null)
    if (kind === 'image' || kind === 'pdf') return
    let dead = false
    ;(async () => {
      try {
        const r = await fetch(url)
        const body = await r.text()
        if (dead) return
        if (!r.ok) setErr(body || `HTTP ${r.status}`)
        else setText(body)
      } catch (e) {
        if (!dead) setErr((e as Error)?.message || 'could not read this file')
      }
    })()
    return () => { dead = true }
  }, [url, kind])

  // marked is given the file's text only; the result is still rendered inside the sandbox, so a
  // markdown file carrying raw HTML is contained exactly like an HTML artifact is.
  const body = useMemo(() => {
    if (text === null) return ''
    if (kind === 'md') return marked.parse(text, { async: false }) as string
    return text
  }, [text, kind])

  // What this artifact wants but will not get, read off the markup before anything renders.
  const scripted = (kind === 'md' || kind === 'html') && text !== null && hasScript(body)
  useEffect(() => { setBlocks(body ? remoteRefs(body) : []) }, [body])

  return (
    <div className="viewer">
      <div className="vhead">
        <FileIcon size={13} />
        <span className="vname" title={name}>{name}</span>
        <span className="vkind mono">{kind}</span>
        <span className="vgap" />
        <a className="btn btn-line btn-sm" href={url} download={name}>Save a copy</a>
        <button className="btn btn-line btn-sm" onClick={onClose}>Close</button>
      </div>

      {(scripted || blocks.length > 0) && (
        <div className="vblocked">
          <Alert size={13} />
          <div>
            {scripted && (
              <div><b>This artifact’s JavaScript does not run.</b> Kayon renders artifacts offline, and a script could
                send your data out by navigating the page, which no policy can prevent once it runs. Anything drawn by
                script will be missing. Save a copy to run it in a browser you trust.</div>
            )}
            {blocks.length > 0 && (
              <div style={{ marginTop: scripted ? 6 : 0 }}>
                <b>{blocks.length}{blocks.length === MAX_BLOCKS ? '+' : ''} remote reference{blocks.length > 1 ? 's' : ''} will not load.</b>{' '}
                Nothing is fetched from the network to show you a file.
                <ul className="vblocklist mono">{blocks.map(b => <li key={b.uri}>{b.uri}</li>)}</ul>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="vbody">
        {err && <div className="vmsg"><Alert size={13} /> {err}</div>}
        {!err && kind === 'image' && <div className="vscroll"><img className="vimg" src={url} alt={name} /></div>}
        {!err && kind === 'pdf' && <PdfView url={url} />}
        {!err && (kind === 'md' || kind === 'html') && text !== null && (
          <SandboxFrame body={body} theme={theme} />
        )}
        {!err && kind === 'text' && text !== null && <div className="vscroll"><pre className="vtext mono">{text}</pre></div>}
        {!err && text === null && kind !== 'image' && kind !== 'pdf' && <div className="vmsg faint">loading</div>}
      </div>
    </div>
  )
}
