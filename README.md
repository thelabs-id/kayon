<div align="center">

<img src="docs/assets/logo.svg" width="84" alt="Kayon">

# Kayon

**An honest, private, local-LLM workstation for Windows + NVIDIA.**

[![Download](https://img.shields.io/github/v/release/thelabs-id/kayon?label=Download&color=C0553A&style=flat-square)](https://github.com/thelabs-id/kayon/releases/latest)
[![License](https://img.shields.io/badge/license-MIT-4FA97C?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Windows%2010%2F11-x64-6E665A?style=flat-square)](https://github.com/thelabs-id/kayon/releases/latest)
[![Built with Rust](https://img.shields.io/badge/Rust-C0553A?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![Tauri 2](https://img.shields.io/badge/Tauri%202-E0916B?style=flat-square&logo=tauri&logoColor=white)](https://tauri.app)

Most local-LLM apps decide what "fits" your GPU by comparing a file size to your VRAM. That's a lie —
it ignores the KV cache, compute buffers, and the memory your display already took. **Kayon does the
real arithmetic**, tells you the truth per quant, and runs everything on your machine: no account, no
cloud, no telemetry unless you turn it on.

**[⬇ Download for Windows](https://github.com/thelabs-id/kayon/releases/latest)**

<img src="docs/assets/model-browser.png" width="100%" alt="Kayon's model browser showing honest per-quant fit verdicts computed from real free VRAM">

</div>

---

## Why Kayon

### 1. Honest fit — not `file_size < VRAM`

Every quant of every model gets a verdict computed for **your** GPU at **your** chosen context length,
from a real memory model — **weights + KV cache + compute buffers + display headroom** vs. actual free
VRAM:

| Verdict | Meaning |
|---|---|
| `FITS_FULLY` | Runs entirely on the GPU, comfortably |
| `FITS_TIGHT` | Fits, but with little headroom |
| `GPU_CPU_SPLIT` | Partially offloaded — runs, slower |
| `CPU_ONLY` | Won't fit on the GPU; fits in RAM |
| `EXCEEDS_MACHINE` | Won't run here |
| `UNVERIFIED_ARCH` | Non-standard attention (SSM/linear/hybrid) — **honestly unverifiable** rather than a fabricated number |

Expand any quant for the full breakdown. When **nothing** fits, Kayon says so plainly instead of
crowning a model that can't run.

### 2. Adopt your Ollama models in place — zero bytes re-downloaded

Kayon finds your Ollama store and **hard-links** the blobs into its library: no copy, no re-download.
It records Ollama's blob digest as the checksum for free, and flags models whose architecture needs a
newer runtime. Deleting Kayon's link never touches Ollama's blob.

### 3. Private by construction

No account. No cloud. Telemetry is **off by default** — and when you do enable it, you see the
**literal payload** before anything is sent. Every outbound request in the whole app funnels through
one instrumented choke point and lands in a **network log** you can read.

---

## Download

**[⬇ Get the latest release](https://github.com/thelabs-id/kayon/releases/latest)** →
`Kayon_<version>_x64-setup.exe`

Run the installer and launch **Kayon** from the Start menu. The llama.cpp CUDA runtime is **bundled** —
chat and the benchmark work out of the box. No separate download, no env vars, no Python.

**Requirements**

- Windows 10/11 x64
- An NVIDIA GPU + driver for GPU inference — *optional*: without one, Kayon still runs and gives you
  honest RAM-based verdicts instead of pretending
- Optional: [Ollama](https://ollama.com), to adopt models you already have

> [!NOTE]
> The installer isn't code-signed yet, so Windows SmartScreen will warn *"Windows protected your PC."*
> Click **More info → Run anyway**. Code signing is on the roadmap.

**Upgrading?** Run the newer installer over your existing install — library, chat history, and settings
are preserved.

---

## Screenshots

<table>
<tr>
<td width="50%">

**Instrument cluster** — your hardware, measured directly from NVML at 1 Hz. Not guessed.

<img src="docs/assets/dashboard.png" alt="Kayon dashboard showing GPU, VRAM, CPU and RAM telemetry">

</td>
<td width="50%">

**Chat with tools** — local chat with an agentic tool loop. Every tool call is shown inline.

<img src="docs/assets/chat-tools.png" alt="Kayon chat showing an inline calculator tool call and its result">

</td>
</tr>
</table>

---

## Features

- **Fit engine** — per-quant verdicts with a full explainability breakdown, at any context length,
  with an f16 / q8_0 KV-cache toggle.
- **Live catalog** — discovered from Hugging Face at launch, every quant's real SHA-256 and byte size
  pinned straight from Git-LFS metadata (no multi-GB download to learn a hash). Stays current with zero
  hand-editing.
- **Checksum-gated downloads** — resumable, with **pause / resume / cancel**. Nothing enters your
  library without matching its pinned hash.
- **Ollama adoption** — hard-linked, zero-copy, with a cross-volume copy fallback.
- **Managed runtime** — llama.cpp `llama-server` (CUDA) supervised as a sidecar, launched with the
  exact `n_gpu_layers` / context the verdict promised.
- **Local chat with sessions** — conversations saved in local SQLite; reopen and continue any of them.
  Each session keeps its own system prompt and sampling params.
- **Agentic tools** — see [Tools](#tools) below.
- **Privacy surface** — a network log accounting for every outbound request, and telemetry that shows
  you its payload before sending.

---

## Tools

When a loaded model's GGUF chat template actually supports tool calling (**detected at load, never
guessed**), Chat offers a built-in, vetted tool set through a server-side **agent loop**: the model's
tool calls run locally, results feed back, and the loop continues until a final answer. Every call —
name, arguments, result — is rendered **inline** and **persisted** with the message, so saved history
stays auditable.

- **Built-in tools** — `calculator` (a deterministic evaluator, no `eval`), `read_file` / `list_dir` /
  `write_file`, `read_selection`, a Python `code` interpreter, and web `search` / `fetch_url`.
  `read_file` **extracts text from PDFs**, refuses other binaries with a clear message instead of
  feeding the model garbage, and forgives a truncated/approximate filename.
- **Session workspace + artifacts** — every chat has a workspace: a folder you attach, or an
  auto-created `~/.kayon/workspace/<session>/`. **Attach files** and they're copied in; model-created
  files land there as artifacts. Filesystem/code tools operate **only** within it — `..`, absolute
  paths, and symlink escapes (including a symlinked write target) are refused.
- **Web is opt-in, per session** — a **Web toggle**, off by default, gates `search` / `fetch_url`.
  DuckDuckGo by default (no key, no account, straight from your machine), and every query lands in the
  network log. `fetch_url` is **SSRF-guarded**: the host is parsed with the same parser the HTTP client
  uses, resolved, and refused if loopback/private/link-local; the vetted IP is pinned (no DNS-rebind)
  and every redirect hop is re-checked.
- **Side effects are confirmation-gated** — `code` **always** asks per call; `write_file` asks only
  when writing into a folder *you* attached (auto-workspace artifacts flow through). An off-by-default
  auto-approve overrides it.

> [!IMPORTANT]
> **The confirmation is the security boundary.** The `code` tool runs an isolated-mode,
> cwd-in-workspace, output-capped, killed-on-timeout Python subprocess — but it is honestly **not a
> sandbox**: approved code runs with your OS permissions. A real WASM/OS-jail sandbox is a post-v1
> goal. Approve only what you trust.

MCP / user-defined tool servers are a planned extension (built-in set first).

---

## Architecture

| Layer | Tech |
|---|---|
| **Core** | Rust — NVML probe, GGUF reader, the fit engine, resumable/checksummed downloads, SQLite (`rusqlite`), ed25519 catalog verification, llama-server supervisor |
| **Shell** | Tauri 2 — a native WebView2 window over the Rust core |
| **UI** | React + TypeScript (Vite) — dashboard, model browser, library, chat, privacy, settings |
| **Runtime** | Prebuilt llama.cpp `llama-server` (CUDA) as a sidecar, driven over its OpenAI-compatible HTTP API |
| **Catalog** | Live from Hugging Face, checksum-pinned; an ed25519-signed catalog is bundled as the offline anchor |

The app launches its local API on `127.0.0.1:9518` on a background thread and renders the UI in the
window — no browser required. The same API also serves the UI standalone, which is how the app is
driven in automated end-to-end tests.

<details>
<summary><b>Repository layout</b></summary>

```
src-tauri/            Rust core
  src/
    probe/            NVML + system telemetry
    gguf/             GGUF header reader
    fit/              the fit engine
    catalog/          bundled anchor · verify · parse
    discovery/        live Hugging Face catalog discovery
    download/         resumable, checksummed
    library/          index · deterministic paths
    ollama/           discover · adopt (hard link)
    runtime/          llama-server sidecar supervisor
    tools/            built-in tool set + executor
    agent/            server-side agentic tool loop
    telemetry/        opt-in gate + outbound network log
    db/               SQLite (rusqlite)
    ipc/              typed command/response contract
    bin/catgen.rs     catalog generator (auto-discovery)
    bin/catsign.rs    sign the bundled catalog
  catalog/            bundled catalog.json + .sig
src/                  React + TypeScript UI (Vite)
```

</details>

---

## Building from source

> Just want to *use* Kayon? Grab the [installer](#download) — this section is for building it yourself.

**Prerequisites:** Windows 10/11 x64 · Rust (stable) · Node.js 18+ · a llama.cpp `llama-server.exe`
(CUDA build recommended) · optionally an NVIDIA GPU + driver, and Ollama.

```bash
# Build the installer
cd src && npm install && npm run build     # build the UI once
cd ..
src/node_modules/.bin/tauri build          # → src-tauri/target/release/bundle/nsis/Kayon_<version>_x64-setup.exe
```

```bash
# Or run it in development
cd src && npm install && npm run build
cd ../src-tauri && cargo run --bin kayon   # the desktop window
# cargo run --bin server                   # just the API + UI on http://127.0.0.1:9518
```

For hot-reload UI work, run `npm run dev` in `src/` (Vite on :3000, proxying `/api` to :9518).

> [!NOTE]
> **llama-server sidecar.** The *installer* bundles `llama-server.exe` (CUDA) as a Tauri resource, so
> it works out of the box. When *building from source* the CUDA binary isn't committed (large
> artifact) — place it under `src-tauri/binaries/llama/` before `tauri build`, or point
> `KAYON_LLAMA_SERVER` at it. The resolver checks: env var → bundled resource → dev path → `PATH`.

**Tests**

```bash
cd src-tauri && cargo test    # fit golden cases, catalog signature/tamper, tool scoping, SSRF guard
```

<details>
<summary><b>Catalog tooling</b></summary>

The catalog is **discovered live from Hugging Face at runtime** — not hand-curated, and not fetched
from any Kayon-hosted file. On launch (in the background) Kayon queries the most-downloaded GGUF models
from a trusted allow-list of quantizers (default: `bartowski`), pins each real checksum + byte size
straight from HF's Git-LFS metadata (the LFS `oid` *is* the SHA-256), and derives the architecture from
a range-fetched header. Results cache to `~/.kayon/catalog/discovered.json`.

A signed catalog is still **bundled** as the offline anchor, generated by the same code path:

```bash
cargo run --bin catgen -- auto [per_author] [author,...]   # regenerate the bundled anchor from HF
cargo run --bin catsign -- pubkey                          # print the baked-in verifying key
cargo run --bin catsign -- sign                            # sign → catalog/catalog.json.sig
```

Discovery is a normal, logged network call and is independently controllable — set the
`catalog_auto_refresh` preference to `off` to stay on the bundled/cached catalog.

**Trust note:** runtime-discovered entries are pinned to Hugging Face's published hash and enforced by
the download checksum gate, but are *not* Kayon-signed like the bundled anchor. Since HF is already the
download origin, this keeps trust to a single party — a deliberate, documented tradeoff.

</details>

---

## Design notes & honest tradeoffs

Documented decisions, not silent divergences:

- **Code execution is not sandboxed (v1).** It's confirmation-gated, isolated-mode, cwd-scoped, and
  killed on timeout — but approved code has your OS permissions. Said plainly in the UI rather than
  dressed up as isolation. A WASM/OS-jail sandbox is the post-v1 hardening.
- **Discovered catalog entries aren't Kayon-signed** — they're pinned to Hugging Face's hash and
  enforced by the checksum gate. The bundled anchor stays signed.
- **The catalog signing key is not in source** — it comes from `KAYON_CATALOG_SEED` or a gitignored
  key file; the verifying key is baked into the binary. In production it belongs in a secret store.
- **Fit constants** (CUDA overhead, compute buffer, headroom) ship at conservative defaults;
  on-device calibration via the benchmark is a follow-up.
- **Chat is a hand-rolled streaming client** over the OpenAI-compatible endpoint, not a chat library.
- **Tool-call traces are persisted per message**; summarized long-term memory and cross-chat recall
  are not in this build.
- **Windows + NVIDIA only** in v1 — no macOS/AMD, no multi-GPU offload, no multimodal, no serving.

**Changelog:** see the [Releases](https://github.com/thelabs-id/kayon/releases) page.

---

## License

Kayon is released under the **[MIT License](LICENSE)**.

It bundles third-party components under their own permissive licenses — llama.cpp (MIT), Tauri
(MIT/Apache-2.0), the Rust crates, and the fonts. See **[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)**.

Model weights are **not** bundled: Kayon downloads models at your request and verifies them against a
pinned checksum. Each model carries its own license from its source.

<div align="center">
<sub><i>Kayon</i> — the Tree-of-Life figure a dalang plants center-screen to frame the world of the play.</sub>
</div>
