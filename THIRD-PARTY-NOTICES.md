# Third-Party Notices

Kayon (MIT-licensed — see [`LICENSE`](LICENSE)) is built on and redistributes third-party software.
This file lists those components and their licenses. All are permissive (Apache-2.0, MIT, BSD, ISC,
MPL-2.0, SIL OFL, Unicode) — none impose copyleft obligations on Kayon's own source.

This is a summary. The **authoritative, per-package** license list is derived from the lockfiles:

```bash
# Rust crates (full transitive tree)
cd src-tauri && cargo license            # or: cargo about generate
# npm packages
cd src && npx license-checker --summary
```

---

## Bundled / redistributed binaries

### llama.cpp — `llama-server.exe` (Vulkan)
The installer bundles the prebuilt llama.cpp server (and its `ggml` libraries) as the inference
runtime. Upstream: <https://github.com/ggml-org/llama.cpp>.
**License: MIT** — © 2023 The ggml authors. The full MIT text applies (see the upstream `LICENSE`);
it must ship alongside the binary, which this notice satisfies.

### Microsoft Edge WebView2 Runtime
The desktop shell renders its UI in the system-provided **Microsoft Edge WebView2 Runtime**. It is a
Microsoft component governed by the Microsoft Software License Terms for the WebView2 Runtime, not by
Kayon; Kayon does not modify or redistribute it (it is installed/serviced by Windows/Microsoft).

---

## Application framework

- **Tauri** (tauri, tauri-build, tauri-runtime, wry, tao, and the `tauri-plugin-single-instance`) —
  **MIT OR Apache-2.0**. <https://github.com/tauri-apps/tauri>

---

## Rust dependencies

Kayon's Rust core depends on several hundred crates (direct + transitive). Direct dependencies:

> tauri, tauri-plugin-single-instance, axum, tokio, tower-http, serde, serde_json, rusqlite,
> nvml-wrapper, sysinfo, sha2, ed25519-dalek, rand, reqwest, url, pdf-extract, futures-util,
> tokio-util, chrono, uuid, log, env_logger, anyhow, thiserror, base64, hex, dirs, notify,
> tokio-stream

License distribution across the **full** crate tree (from `cargo license`):

| License | Crates | Notes |
|---|---:|---|
| Apache-2.0 OR MIT | ~365 | serde, tokio, reqwest, axum, uuid, base64, url, hyper-stack, windows-*, etc. |
| MIT | ~133 | rusqlite, tower, tokio-macros, lopdf, pdf-extract, sysinfo, hyper, webview2-com, etc. |
| Apache-2.0 OR MIT OR Zlib | ~20 | bytemuck, objc2-* (macOS), raw-window-handle, tinyvec |
| Unicode-3.0 | ~18 | ICU crates (icu_*, zerovec, tinystr) |
| MIT OR Unlicense | ~8 | memchr, aho-corasick, walkdir, byteorder, jiff |
| BSD-3-Clause | ~6 | curve25519-dalek, ed25519-dalek, subtle, instant |
| ISC | ~6 | rustls-webpki, untrusted, libloading, inotify |
| **MPL-2.0** | 5 | cssparser, cssparser-macros, dtoa-short, option-ext, selectors — weak (file-level) copyleft; used unmodified, sources at crates.io |
| Apache-2.0 | 3 | openssl, tao, sync_wrapper |
| Apache-2.0 AND ISC | 1 | ring |
| CC0-1.0 | 1 | notify |
| Zlib | 1 | foldhash |
| other permissive combos | a few | BSD-1/2/3-Clause, BSL-1.0, CC0/MIT-0 alternatives |

No GPL, LGPL, or AGPL crates are present. Each crate's exact terms are in its published source on
[crates.io](https://crates.io); regenerate the exact list with `cargo license` as shown above.

---

## Frontend dependencies (npm)

- **React**, **React DOM** — MIT. <https://github.com/facebook/react>
- **pdf.js** (`pdfjs-dist`, Mozilla) — **Apache-2.0**. <https://github.com/mozilla/pdf.js>
  Renders PDF pages in the document viewer. Redistributed in the installer: the library, its worker,
  and its runtime assets (`cmaps`, `standard_fonts`, `wasm`), which are copied from the package at
  build time into `pdfjs/`. They are bundled rather than fetched so a PDF is never sent anywhere to
  be viewed; pdf.js would otherwise default to a CDN for them.
- **marked** — MIT. <https://github.com/markedjs/marked> Converts markdown artifacts to HTML, which
  is then rendered inside the viewer's script-free sandbox.
- **Vite** — MIT. <https://github.com/vitejs/vite>
- **TypeScript** — Apache-2.0. <https://github.com/microsoft/TypeScript>
- **@vitejs/plugin-react**, **oxlint**, **pureimage**, **@fontsource*** wrappers, `@tauri-apps/cli`,
  `@types/*` — MIT (dev/build tooling).

## Fonts (bundled as WOFF2)

- **Geist** and **Geist Mono** (Vercel) — **SIL Open Font License 1.1**.
  <https://github.com/vercel/geist-font>
- **Instrument Serif** — **SIL Open Font License 1.1**.
  <https://fonts.google.com/specimen/Instrument+Serif>

The SIL OFL permits bundling and redistribution of the font files; the license text ships with each
`@fontsource` package under `src/node_modules/`.

---

*Model weights are **not** bundled.* Kayon downloads models on the user's explicit request and
verifies them against a pinned checksum; each model carries its own license from its source (e.g.
Hugging Face) and is the user's responsibility to accept.
