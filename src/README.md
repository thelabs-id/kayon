# Kayon UI (React + TypeScript + Vite)

The Kayon desktop app's frontend. It renders inside the Tauri WebView2 window and talks to the Rust
core's local API on `127.0.0.1:9518`.

- **Pages** (`src/pages/`): Dashboard, Model browser, Library, Chat, Privacy & network, Settings, Onboarding.
  Chat drives the agentic **tool loop** for tool-capable models — a per-session Web toggle, an
  attach-folder control (workspace scope), inline tool-call cards with Approve/Deny confirmation for
  side effects, and a persisted, auditable tool trace (TOOL family).
- **API client** (`src/lib/api.ts`): typed wrappers over the core's HTTP endpoints (hardware, catalog +
  live discovery status, fit verdicts, downloads + pause/resume/cancel, library, Ollama, runtime, chat
  sessions, privacy).
- **Design system** (`src/design.css`): the app's visual language (colors, typography, components).

## Develop

```bash
npm install
npm run build     # type-check (tsc -b) + production bundle → dist/ (served by the Rust core)
npm run dev       # Vite dev server (used by `tauri dev`)
npm run lint      # oxlint
```

The app is drivable in a plain browser for testing: run `cargo run --bin server` in `../src-tauri`
(serves this UI + the API on `127.0.0.1:9518`), then open that URL.

See the repository [`README.md`](../README.md) for the full picture.
