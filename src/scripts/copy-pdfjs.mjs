// TOOL-8: vendor the PDF engine's runtime assets into public/ so pages render fully offline.
//
// pdf.js fetches these at runtime rather than bundling them. Without them a PDF using a
// non-embedded standard font (very common), a CJK encoding, or a JPEG2000 image renders wrong or
// not at all — and the natural "fix" is pdf.js's default CDN URL, which would be silent network
// (PRIV-1). Copying them locally is what makes the offline guarantee real.
//
// Copied from node_modules at build time and gitignored: vendored bytes don't belong in the repo.
import { cp, mkdir, rm } from 'node:fs/promises'

const SRC = 'node_modules/pdfjs-dist'
const OUT = 'public/pdfjs'

await rm(OUT, { recursive: true, force: true })
await mkdir(OUT, { recursive: true })
for (const dir of ['cmaps', 'standard_fonts', 'wasm']) {
  await cp(`${SRC}/${dir}`, `${OUT}/${dir}`, { recursive: true })
}
console.log(`pdfjs assets -> ${OUT}`)
