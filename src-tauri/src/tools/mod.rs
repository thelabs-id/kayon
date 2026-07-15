//! TOOL family — built-in, Kayon-vetted tool set for model-native tool calling (TOOL-3).
//!
//! The set is: `calculator` (deterministic math), `read_file` / `list_dir` / `write_file`,
//! `read_selection`, a `code` interpreter, and web `search` / `fetch_url`. Filesystem tools and the
//! code interpreter are **scoped to an attached session folder** (TOOL-4): every resolved real path
//! must stay inside the folder — `..`, absolute paths, and symlink escapes are refused. Web tools are
//! **per-session opt-in** (TOOL-5) and every request flows through the single instrumented client so
//! it lands in the network log (PRIV-5). `write_file` and `code` are **side-effectful** and the agent
//! loop confirmation-gates them (TOOL-6); everything else is read-only.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::db::Database;

/// Per-turn tool context: what the session has enabled. The available tool set is the intersection
/// of (model supports tools) × (folder attached) × (Web toggle) — TOOL-1.
pub struct ToolContext {
    pub workspace: Option<PathBuf>,
    /// True when `workspace` is the Kayon-owned auto-workspace (`~/.kayon/workspace/<session>/`)
    /// rather than a folder the user explicitly attached. Drives the write-confirmation policy.
    pub is_auto_workspace: bool,
    pub web_enabled: bool,
    pub selection: Option<String>,
    pub db: Arc<Database>,
}

/// Side-effectful tools (may need confirmation depending on target — see `needs_confirmation`).
pub fn is_side_effect(name: &str) -> bool {
    matches!(name, "write_file" | "code")
}

/// Whether a tool call must be user-confirmed before running (TOOL-6). `auto_approve` is the
/// per-session master override. Otherwise: **code execution always confirms** (it runs on the
/// machine with the user's OS permissions, not a sandbox, regardless of folder); **write_file**
/// confirms only when writing into an explicitly *attached* folder — writes into the Kayon-owned
/// auto-workspace flow freely, so model-generated artifacts appear without a click.
pub fn needs_confirmation(name: &str, ctx: &ToolContext, auto_approve: bool) -> bool {
    if auto_approve {
        return false;
    }
    match name {
        "code" => true,
        "write_file" => !ctx.is_auto_workspace,
        _ => false,
    }
}

/// The OpenAI-format `tools` array to advertise for this turn. Empty when the model can't use tools.
/// Filesystem/code tools appear only with a workspace; web tools only with the Web toggle on.
pub fn tool_specs(supports_tools: bool, ctx: &ToolContext) -> Vec<Value> {
    if !supports_tools {
        return vec![];
    }
    let mut specs = vec![spec(
        "calculator",
        "Evaluate a deterministic arithmetic expression and return the numeric result. Supports + - * / % ^, parentheses, and unary minus.",
        json!({
            "type": "object",
            "properties": { "expression": { "type": "string", "description": "e.g. \"(2+3)*4/2\"" } },
            "required": ["expression"]
        }),
    )];
    if ctx.selection.is_some() {
        specs.push(spec(
            "read_selection",
            "Return the text the user attached as the current selection for this chat.",
            json!({ "type": "object", "properties": {} }),
        ));
    }
    if ctx.workspace.is_some() {
        specs.push(spec(
            "list_dir",
            "List the entries (name, kind, size) of a directory inside the attached workspace folder.",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Path relative to the workspace root. Use \".\" for the root." } },
                "required": ["path"]
            }),
        ));
        specs.push(spec(
            "read_file",
            "Read a UTF-8 text file inside the attached workspace folder and return its contents (truncated if very large).",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Path relative to the workspace root." } },
                "required": ["path"]
            }),
        ));
        specs.push(spec(
            "write_file",
            "Create or overwrite a UTF-8 text file inside the attached workspace folder. Requires user confirmation.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path relative to the workspace root." },
                    "content": { "type": "string", "description": "Full file contents to write." }
                },
                "required": ["path", "content"]
            }),
        ));
        specs.push(spec(
            "code",
            "Run a short Python 3 program with its working directory set to the attached workspace folder, and return stdout+stderr. Requires user confirmation.",
            json!({
                "type": "object",
                "properties": {
                    "language": { "type": "string", "description": "Only \"python\" is supported in v1.", "enum": ["python"] },
                    "source": { "type": "string", "description": "The program source." }
                },
                "required": ["source"]
            }),
        ));
    }
    if ctx.web_enabled {
        specs.push(spec(
            "search",
            "Search the web (DuckDuckGo) and return the top results (title, URL, snippet).",
            json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        ));
        specs.push(spec(
            "fetch_url",
            "Fetch a URL over HTTP(S) and return its text content (HTML stripped, truncated if large).",
            json!({
                "type": "object",
                "properties": { "url": { "type": "string" } },
                "required": ["url"]
            }),
        ));
    }
    specs
}

fn spec(name: &str, description: &str, parameters: Value) -> Value {
    json!({
        "type": "function",
        "function": { "name": name, "description": description, "parameters": parameters }
    })
}

/// Execute a tool call. Returns the textual result the model will see, or an `Err` describing the
/// failure (surfaced to the model AND the UI — never swallowed, TOOL-7). Side-effect confirmation is
/// handled by the caller (the agent loop) before this is invoked.
pub async fn execute(name: &str, args: &Value, ctx: &ToolContext) -> Result<String, String> {
    match name {
        "calculator" => {
            let expr = str_arg(args, "expression")?;
            calc::eval(&expr).map(|n| format_num(n)).map_err(|e| format!("calculator: {e}"))
        }
        "read_selection" => ctx
            .selection
            .clone()
            .ok_or_else(|| "no selection is attached to this chat".to_string()),
        "list_dir" => list_dir(ctx, &str_arg(args, "path")?),
        "read_file" => read_file(ctx, &str_arg(args, "path")?),
        "write_file" => write_file(ctx, &str_arg(args, "path")?, &str_arg(args, "content")?),
        "code" => {
            let lang = args.get("language").and_then(|v| v.as_str()).unwrap_or("python");
            run_code(ctx, lang, &str_arg(args, "source")?).await
        }
        "search" => web_search(ctx, &str_arg(args, "query")?).await,
        "fetch_url" => fetch_url(ctx, &str_arg(args, "url")?).await,
        other => Err(format!("unknown tool: {other}")),
    }
}

fn str_arg(args: &Value, key: &str) -> Result<String, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("missing required string argument '{key}'"))
}

fn format_num(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

// ---- filesystem scoping (TOOL-4) ------------------------------------------------------------

const MAX_READ_BYTES: usize = 256 * 1024;
/// Cap on the text a single `fetch_url` returns to the model — a whole web page routinely strips to
/// tens of thousands of tokens, which would overflow a local model's context in one call.
const WEB_FETCH_MAX_CHARS: usize = 8000;

/// Resolve `rel` against the workspace root and guarantee the real path stays inside it. For reads we
/// canonicalize the target (which resolves symlinks, so an escaping symlink is caught). For writes the
/// file may not exist yet, so we canonicalize its parent and re-join the file name.
fn resolve_in_workspace(ctx: &ToolContext, rel: &str, for_write: bool) -> Result<PathBuf, String> {
    let ws = ctx.workspace.as_ref().ok_or("no workspace folder is attached to this chat")?;
    let root = ws
        .canonicalize()
        .map_err(|e| format!("workspace folder is unavailable: {e}"))?;
    if Path::new(rel).is_absolute() {
        return Err("absolute paths are not allowed; use a path relative to the workspace".into());
    }
    let joined = root.join(rel);
    if for_write {
        if joined.exists() {
            // An existing target: refuse a symlink outright (writing through it could clobber a file
            // outside the workspace — parent-only canonicalization would miss that), then require the
            // real, symlink-resolved path to stay inside the root.
            let meta = std::fs::symlink_metadata(&joined).map_err(|e| e.to_string())?;
            if meta.file_type().is_symlink() {
                return Err("refusing to write through a symlink".into());
            }
            let real = joined.canonicalize().map_err(|e| e.to_string())?;
            if !real.starts_with(&root) {
                return Err("path escapes the attached workspace folder".into());
            }
            Ok(real)
        } else {
            // A new file: canonicalize the (must-exist) parent — which resolves any symlink in the
            // directory chain — and require it inside the root.
            let parent = joined
                .parent()
                .ok_or("invalid path")?
                .canonicalize()
                .map_err(|e| format!("parent directory does not exist: {e}"))?;
            if !parent.starts_with(&root) {
                return Err("path escapes the attached workspace folder".into());
            }
            let name = joined.file_name().ok_or("a file name is required")?;
            Ok(parent.join(name))
        }
    } else {
        // Reads: canonicalize (resolves symlinks) and require the result inside the root.
        let real = joined
            .canonicalize()
            .map_err(|e| format!("path does not exist: {e}"))?;
        if !real.starts_with(&root) {
            return Err("path escapes the attached workspace folder".into());
        }
        Ok(real)
    }
}

/// Resolve a path for *reading*, forgivingly. Models often mangle an attached file's long name
/// (dropping a prefix, an en-dash, or spaces — e.g. asking for `Timeline.pdf` when the file is
/// `Avrist Agent App – Timeline.pdf`). So if the exact path isn't found, fall back to a UNIQUE
/// case-insensitive match among the workspace's files by basename (exact → suffix → substring). If
/// nothing unambiguously matches, return an error that lists the real file names so the model can
/// retry with the right one instead of hallucinating.
fn resolve_read_path(ctx: &ToolContext, rel: &str) -> Result<PathBuf, String> {
    if let Ok(p) = resolve_in_workspace(ctx, rel, false) {
        return Ok(p);
    }
    let root = ctx
        .workspace
        .as_ref()
        .ok_or("no workspace folder is attached to this chat")?
        .canonicalize()
        .map_err(|e| format!("workspace folder is unavailable: {e}"))?;
    let want = std::path::Path::new(rel)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(rel)
        .to_lowercase();
    // Names of files (not subdirs) that are direct children of the workspace root.
    let names: Vec<String> = std::fs::read_dir(&root)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .filter(|e| e.metadata().map(|m| m.is_file()).unwrap_or(false))
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .collect();
    let unique = |pred: &dyn Fn(&str) -> bool| -> Option<String> {
        let mut it = names.iter().filter(|n| pred(&n.to_lowercase()));
        match (it.next(), it.next()) {
            (Some(n), None) => Some(n.clone()),
            _ => None,
        }
    };
    if let Some(name) = unique(&|n| n == want)
        .or_else(|| unique(&|n| n.ends_with(&want)))
        .or_else(|| unique(&|n| n.contains(&want)))
    {
        // Re-validate the matched name through the scoped resolver so the confinement guarantee holds
        // even for the fuzzy path: it canonicalizes (resolving symlinks) and rejects any escape.
        return resolve_in_workspace(ctx, &name, false);
    }
    Err(format!(
        "no file matching '{rel}' in the workspace. Available files: {}",
        if names.is_empty() { "(none)".to_string() } else { names.join(", ") }
    ))
}

fn list_dir(ctx: &ToolContext, rel: &str) -> Result<String, String> {
    let dir = resolve_in_workspace(ctx, rel, false)?;
    let md = std::fs::metadata(&dir).map_err(|e| e.to_string())?;
    if !md.is_dir() {
        return Err("not a directory".into());
    }
    // Cap the number of entries we collect so an enormous directory can't blow up memory or the
    // persisted tool result — note the truncation rather than silently dropping.
    const MAX_ENTRIES: usize = 1000;
    let mut out = String::new();
    let mut entries: Vec<_> = Vec::new();
    let mut more = false;
    for e in std::fs::read_dir(&dir).map_err(|e| e.to_string())?.filter_map(|e| e.ok()) {
        if entries.len() >= MAX_ENTRIES {
            more = true;
            break;
        }
        entries.push(e);
    }
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().to_string_lossy().to_string();
        let m = e.metadata().ok();
        let (kind, size) = match &m {
            Some(m) if m.is_dir() => ("dir", 0),
            Some(m) => ("file", m.len()),
            None => ("?", 0),
        };
        if kind == "dir" {
            out.push_str(&format!("{name}/\t<dir>\n"));
        } else {
            out.push_str(&format!("{name}\t{size} bytes\n"));
        }
    }
    if out.is_empty() {
        out.push_str("(empty directory)");
    }
    if more {
        out.push_str(&format!("\n[... more than {MAX_ENTRIES} entries; listing truncated ...]"));
    }
    Ok(out)
}

fn read_file(ctx: &ToolContext, rel: &str) -> Result<String, String> {
    use std::io::Read;
    let path = resolve_read_path(ctx, rel)?;

    // A PDF is binary — reading it as UTF-8 gives the model garbage — so extract its text instead.
    // Detect by magic bytes (authoritative) with the extension as a fallback.
    let mut head = [0u8; 5];
    let head_n = std::fs::File::open(&path).and_then(|mut f| f.read(&mut head)).unwrap_or(0);
    let is_pdf = (head_n >= 4 && &head[..4] == b"%PDF")
        || path.extension().map(|e| e.eq_ignore_ascii_case("pdf")).unwrap_or(false);
    if is_pdf {
        return read_pdf(&path);
    }

    // Read at most MAX_READ_BYTES+1 (the +1 only tells us whether truncation happened) rather than
    // loading a multi-GB file fully into memory before slicing.
    let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take((MAX_READ_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let truncated = bytes.len() > MAX_READ_BYTES;
    let slice = &bytes[..bytes.len().min(MAX_READ_BYTES)];
    // Binary guard: a NUL byte almost never appears in real text, and returning from_utf8_lossy of a
    // binary blob would feed the model replacement-character garbage. Say so honestly instead.
    if slice.contains(&0) {
        return Err(format!(
            "'{}' looks like a binary file ({} bytes) — read_file only returns text. PDFs are \
             extracted; other binary formats (images, office docs, archives) are not supported.",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
            bytes.len()
        ));
    }
    let mut text = String::from_utf8_lossy(slice).to_string();
    if truncated {
        text.push_str(&format!("\n\n[... truncated at {MAX_READ_BYTES} bytes ...]"));
    }
    Ok(text)
}

/// Extract the text of a PDF so the model reads the document's words (TOOL-3, attach-files flow).
/// PDFs must be parsed whole, so we read the entire file (bounded) rather than the byte-capped slice
/// `read_file` uses for text. Scanned/image-only PDFs have no text layer — we say so honestly rather
/// than returning nothing (OCR is out of scope for v1).
fn read_pdf(path: &Path) -> Result<String, String> {
    const MAX_PDF_BYTES: u64 = 25 * 1024 * 1024;
    // A PDF must be parsed whole (its xref/trailer live at the end), so reject an oversized file up
    // front rather than truncating it — a truncated PDF just fails as "malformed".
    let len = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if len > MAX_PDF_BYTES {
        return Err(format!(
            "PDF is too large ({} MB) — the limit is {} MB",
            len / (1024 * 1024),
            MAX_PDF_BYTES / (1024 * 1024)
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    // pdf-extract can panic on some malformed PDFs; isolate it so a bad file can't take down the loop.
    let extracted = std::panic::catch_unwind(|| pdf_extract::extract_text_from_mem(&bytes))
        .map_err(|_| "could not parse the PDF (it may be malformed or encrypted)".to_string())?
        .map_err(|e| format!("PDF text extraction failed: {e}"))?;
    let text = extracted.trim();
    if text.is_empty() {
        return Err("the PDF has no extractable text — it is likely a scanned image (OCR is not supported in v1)".into());
    }
    let truncated = text.chars().count() > MAX_READ_BYTES;
    let mut out: String = text.chars().take(MAX_READ_BYTES).collect();
    if truncated {
        out.push_str("\n\n[... PDF text truncated for length ...]");
    }
    Ok(out)
}

fn write_file(ctx: &ToolContext, rel: &str, content: &str) -> Result<String, String> {
    let path = resolve_in_workspace(ctx, rel, true)?;
    std::fs::write(&path, content).map_err(|e| e.to_string())?;
    Ok(format!("wrote {} bytes to {}", content.len(), rel))
}

// ---- code interpreter (TOOL-6) --------------------------------------------------------------

async fn run_code(ctx: &ToolContext, language: &str, source: &str) -> Result<String, String> {
    if !language.eq_ignore_ascii_case("python") && !language.eq_ignore_ascii_case("python3") {
        return Err(format!("unsupported language '{language}'; only python is supported in v1"));
    }
    let ws = ctx
        .workspace
        .as_ref()
        .ok_or("no workspace folder is attached; code execution is workspace-scoped")?
        .canonicalize()
        .map_err(|e| format!("workspace folder is unavailable: {e}"))?;

    // The script itself lives in the OS temp dir (not the user's workspace); only its *working
    // directory* is the workspace, so relative file access resolves there.
    let script = std::env::temp_dir().join(format!("kayon-code-{}.py", uuid::Uuid::new_v4()));
    std::fs::write(&script, source).map_err(|e| format!("failed to stage script: {e}"))?;

    // NOTE: this is a confirmation-gated subprocess with its working directory set to the workspace
    // (TOOL-6) — it is NOT a security sandbox. Python here can still read/write absolute paths and
    // open sockets; the confirmation gate (and off-by-default auto-approve) is the boundary. `-I`
    // (isolated mode) trims some footguns (ignores user site-packages and PYTHON* env). A real WASM
    // sandbox is the documented post-v1 hardening (OD-11).
    let py = python_binary();
    let mut cmd = tokio::process::Command::new(&py);
    cmd.arg("-I")
        .arg(&script)
        .current_dir(&ws)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // On the timeout path we drop the `output()` future; kill_on_drop ensures the Python child
        // is actually terminated rather than left running (e.g. an infinite loop) after we return.
        .kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW (tokio's Command has an inherent method)

    // Stream stdout/stderr with a per-stream byte cap rather than buffering the whole output: a
    // runaway script (an infinite print loop) could otherwise produce hundreds of MB before the
    // timeout. We keep draining past the cap (discarding the excess) so the child never blocks on a
    // full pipe, and rely on kill_on_drop + the timeout to stop a hung process.
    const OUT_CAP: usize = 64 * 1024;
    let run = async {
        let mut child = cmd.spawn().map_err(|e| format!("failed to launch {py}: {e}"))?;
        let stdout = child.stdout.take().ok_or("no stdout pipe")?;
        let stderr = child.stderr.take().ok_or("no stderr pipe")?;
        let (o, e) = tokio::join!(read_capped_pipe(stdout, OUT_CAP), read_capped_pipe(stderr, OUT_CAP));
        let status = child.wait().await.map_err(|e| e.to_string())?;
        Ok::<_, String>((o, e, status))
    };
    let result = tokio::time::timeout(std::time::Duration::from_secs(30), run).await;
    let _ = std::fs::remove_file(&script);
    let (out_b, err_b, status) = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return Err(e),
        Err(_) => return Err("code execution timed out after 30s".into()),
    };

    let mut text = String::new();
    let stdout = String::from_utf8_lossy(&out_b);
    let stderr = String::from_utf8_lossy(&err_b);
    if !stdout.trim().is_empty() {
        text.push_str(stdout.trim_end());
        if out_b.len() >= OUT_CAP { text.push_str("\n[stdout truncated]"); }
        text.push('\n');
    }
    if !stderr.trim().is_empty() {
        text.push_str("[stderr]\n");
        text.push_str(stderr.trim_end());
        if err_b.len() >= OUT_CAP { text.push_str("\n[stderr truncated]"); }
        text.push('\n');
    }
    text.push_str(&format!("[exit code {}]", status.code().unwrap_or(-1)));
    Ok(text)
}

/// Read an async pipe to EOF, keeping at most `cap` bytes but continuing to drain (and discard) the
/// rest so the child never blocks writing to a full pipe.
async fn read_capped_pipe<R: tokio::io::AsyncRead + Unpin>(mut r: R, cap: usize) -> Vec<u8> {
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        match r.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => {
                if buf.len() < cap {
                    let take = (cap - buf.len()).min(n);
                    buf.extend_from_slice(&tmp[..take]);
                }
            }
            Err(_) => break,
        }
    }
    buf
}

fn python_binary() -> String {
    // Prefer python3 (what the user put on PATH), fall back to python.
    for c in ["python3", "python"] {
        if std::process::Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return c.to_string();
        }
    }
    "python3".to_string()
}

// ---- web tools (TOOL-5) — logged at the single egress choke point (PRIV-5) -------------------

/// Hard cap on raw bytes read from a web response before we stop. Web tools can be pointed at huge
/// or binary URLs; we reject an oversized `Content-Length` up front and otherwise stream, aborting
/// once we have enough — so a single tool call can't buffer unbounded memory/bandwidth.
const MAX_FETCH_RAW_BYTES: usize = 1024 * 1024;

async fn read_body_capped(resp: reqwest::Response, max_bytes: usize) -> Result<(String, u64), String> {
    if let Some(len) = resp.content_length() {
        if len > (max_bytes as u64) * 16 {
            return Err(format!("response too large ({len} bytes)"));
        }
    }
    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        let b = chunk.map_err(|e| e.to_string())?;
        if buf.len() < max_bytes {
            let take = (max_bytes - buf.len()).min(b.len());
            buf.extend_from_slice(&b[..take]);
        }
        if buf.len() >= max_bytes {
            break; // enough — abort the download rather than pull the whole (possibly huge) body
        }
    }
    let n = buf.len() as u64;
    Ok((String::from_utf8_lossy(&buf).to_string(), n))
}

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) Kayon/1.2")
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .unwrap_or_default()
}

/// Resolve a `Location` header against the current URL using the URL parser's RFC-3986 `join`, so a
/// path-relative redirect like `Location: next` from `/dir/page` correctly becomes `/dir/next`
/// (browser/HTTP semantics), and protocol-relative / absolute forms work too. Returns `None` for a
/// non-http(s) result so a redirect can't escape into `file:`/`ftp:` etc.
fn resolve_redirect(base: &str, loc: &str) -> Option<String> {
    let joined = url::Url::parse(base).ok()?.join(loc.trim()).ok()?;
    match joined.scheme() {
        "http" | "https" => Some(joined.to_string()),
        _ => None,
    }
}

async fn web_search(ctx: &ToolContext, query: &str) -> Result<String, String> {
    if !ctx.web_enabled {
        return Err("web tools are off for this chat".into());
    }
    let url = "https://html.duckduckgo.com/html/";
    let client = http_client();
    let resp = client.get(url).query(&[("q", query)]).send().await;
    let (status, body, err) = match resp {
        Ok(r) => {
            let st = r.status().as_u16();
            match read_body_capped(r, MAX_FETCH_RAW_BYTES).await {
                Ok((t, _)) => (Some(st), t, None),
                Err(e) => (Some(st), String::new(), Some(e)),
            }
        }
        Err(e) => (None, String::new(), Some(e.to_string())),
    };
    crate::telemetry::log_network_request_full(
        &ctx.db,
        "GET",
        &format!("{url}?q={query}"),
        "tool:search",
        query.len() as u64,
        body.len() as u64,
        status,
        err.clone(),
    );
    if let Some(e) = err {
        return Err(format!("search failed: {e}"));
    }
    let results = parse_ddg(&body);
    if results.is_empty() {
        return Ok("(no results)".into());
    }
    let mut out = String::new();
    for (i, (title, link, snippet)) in results.iter().take(5).enumerate() {
        out.push_str(&format!("{}. {}\n{}\n{}\n\n", i + 1, title, link, snippet));
    }
    Ok(out.trim_end().to_string())
}

/// Extract (title, url, snippet) triples from DuckDuckGo's HTML results page without an HTML-parsing
/// dependency. Tolerant string scanning: if the markup shifts, we return whatever we could recover.
fn parse_ddg(html: &str) -> Vec<(String, String, String)> {
    let mut out = Vec::new();
    for block in html.split("result__body").skip(1) {
        let title = between(block, "result__a", "</a>")
            .and_then(|seg| seg.split_once('>').map(|(_, t)| strip_tags(t)));
        let href = between(block, "result__a\"", ">")
            .and_then(|seg| attr(seg, "href"))
            .map(decode_ddg_href);
        let snippet = between(block, "result__snippet", "</a>")
            .and_then(|seg| seg.split_once('>').map(|(_, t)| strip_tags(t)));
        if let (Some(t), Some(h)) = (title, href) {
            if !t.is_empty() && !h.is_empty() {
                out.push((t, h, snippet.unwrap_or_default()));
            }
        }
    }
    out
}

fn between<'a>(hay: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let i = hay.find(start)? + start.len();
    let rest = &hay[i..];
    let j = rest.find(end)?;
    Some(&rest[..j])
}

fn attr(seg: &str, name: &str) -> Option<String> {
    let key = format!("{name}=\"");
    let i = seg.find(&key)? + key.len();
    let rest = &seg[i..];
    let j = rest.find('"')?;
    Some(rest[..j].to_string())
}

/// DuckDuckGo HTML wraps result links in a redirector like `//duckduckgo.com/l/?uddg=<encoded>`.
/// Recover the real URL when present.
fn decode_ddg_href(href: String) -> String {
    if let Some(i) = href.find("uddg=") {
        let enc = &href[i + 5..];
        let enc = enc.split('&').next().unwrap_or(enc);
        return percent_decode(enc);
    }
    if let Some(stripped) = href.strip_prefix("//") {
        return format!("https://{stripped}");
    }
    href
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(if bytes[i] == b'+' { b' ' } else { bytes[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// The host of an http(s) URL, parsed with the SAME parser reqwest uses (`url::Url`) so the SSRF
/// check can't disagree with what actually gets connected (e.g. a backslash trick like
/// `http://127.0.0.1\@example.com/` that a hand-rolled parser reads as `example.com` but WHATWG
/// normalizes to host `127.0.0.1`). An IP literal needs no DNS; a domain is resolved.
enum UrlHost {
    Ip(std::net::IpAddr, u16),
    Domain(String, u16),
}

fn parse_url_host(raw: &str) -> Option<UrlHost> {
    let u = url::Url::parse(raw).ok()?;
    if !matches!(u.scheme(), "http" | "https") {
        return None;
    }
    let port = u.port_or_known_default()?;
    match u.host()? {
        url::Host::Ipv4(ip) => Some(UrlHost::Ip(ip.into(), port)),
        url::Host::Ipv6(ip) => Some(UrlHost::Ip(ip.into(), port)),
        url::Host::Domain(d) => Some(UrlHost::Domain(d.to_string(), port)),
    }
}

fn ip_is_private(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v) => {
            v.is_loopback() || v.is_private() || v.is_link_local() || v.is_unspecified()
                || v.is_broadcast() || v.octets()[0] == 0
                // 100.64.0.0/10 (CGNAT) and 169.254 covered by link_local; 192.0.0.0/24 etc left as-is
        }
        IpAddr::V6(v) => {
            let s = v.segments();
            v.is_loopback() || v.is_unspecified()
                || (s[0] & 0xfe00) == 0xfc00 // fc00::/7 unique-local
                || (s[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
                || v.to_ipv4_mapped().map(IpAddr::V4).map(ip_is_private).unwrap_or(false)
        }
    }
}

/// Require the URL's target to be public, returning the domain to pin (if any) plus the vetted
/// addresses. The caller PINS these for the actual request so `reqwest` cannot re-resolve to a
/// private IP between the check and the connection (DNS-rebinding TOCTOU). An IP literal is checked
/// directly and needs no pinning (reqwest connects to it as-is).
async fn reject_private_target(url: &str) -> Result<(Option<String>, Vec<std::net::SocketAddr>), String> {
    match parse_url_host(url).ok_or("could not parse the URL host")? {
        UrlHost::Ip(ip, port) => {
            if ip_is_private(ip) {
                return Err("refusing to fetch a loopback / private / link-local address".into());
            }
            Ok((None, vec![std::net::SocketAddr::new(ip, port)]))
        }
        UrlHost::Domain(host, port) => {
            let hl = host.to_lowercase();
            if hl == "localhost" || hl.ends_with(".localhost") {
                return Err("refusing to fetch a local address".into());
            }
            let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host.as_str(), port))
                .await
                .map_err(|e| format!("could not resolve host: {e}"))?
                .collect();
            if addrs.is_empty() {
                return Err("host did not resolve".into());
            }
            for addr in &addrs {
                if ip_is_private(addr.ip()) {
                    return Err("refusing to fetch a loopback / private / link-local address".into());
                }
            }
            Ok((Some(host), addrs))
        }
    }
}

/// A no-redirect client whose DNS for `host` (when a domain) is pinned to the already-vetted `addrs`,
/// so the actual connection goes to the IP we checked — closing the DNS-rebinding gap between check
/// and connect. For an IP-literal URL there is no host to pin (reqwest connects to the literal).
fn http_client_pinned(host: Option<&str>, addrs: &[std::net::SocketAddr]) -> reqwest::Client {
    let mut b = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) Kayon/1.2")
        .timeout(std::time::Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none());
    if let Some(h) = host {
        if !addrs.is_empty() {
            b = b.resolve_to_addrs(h, addrs);
        }
    }
    b.build().unwrap_or_default()
}

async fn fetch_url(ctx: &ToolContext, url: &str) -> Result<String, String> {
    if !ctx.web_enabled {
        return Err("web tools are off for this chat".into());
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("only http(s) URLs are supported".into());
    }
    // SSRF guard: refuse loopback / private / link-local targets — otherwise a tool-capable model
    // could read Kayon's own unauthenticated loopback API (or other local services) via fetch_url and
    // exfiltrate it through `search`. Redirects are followed MANUALLY, re-checking AND re-pinning the
    // resolved IP on every hop, so neither reqwest's auto-redirect nor DNS-rebinding between check and
    // connect can bounce us to a private address. Every hop that actually opens a socket is logged
    // (PRIV-5); a target refused before any socket opens is not (we log real traffic, not intent).
    let mut current = url.to_string();
    let mut hops = 0u8;
    let body = loop {
        let (host, addrs) = reject_private_target(&current).await.map_err(|e| format!("fetch failed: {e}"))?;
        let client = http_client_pinned(host.as_deref(), &addrs);
        let r = match client.get(&current).send().await {
            Ok(r) => r,
            Err(e) => {
                crate::telemetry::log_network_request_full(&ctx.db, "GET", &current, "tool:fetch_url", 0, 0, None, Some(e.to_string()));
                return Err(format!("fetch failed: {e}"));
            }
        };
        let st = r.status().as_u16();
        if r.status().is_redirection() {
            crate::telemetry::log_network_request_full(&ctx.db, "GET", &current, "tool:fetch_url", 0, 0, Some(st), None);
            if hops >= 5 {
                return Err("fetch failed: too many redirects".into());
            }
            let loc = r.headers().get(reqwest::header::LOCATION).and_then(|v| v.to_str().ok());
            match loc.and_then(|l| resolve_redirect(&current, l)) {
                Some(next) => { current = next; hops += 1; continue; }
                None => return Err("fetch failed: unsupported redirect target".into()),
            }
        }
        match read_body_capped(r, MAX_FETCH_RAW_BYTES).await {
            Ok((t, n)) => {
                crate::telemetry::log_network_request_full(&ctx.db, "GET", &current, "tool:fetch_url", 0, n, Some(st), None);
                break t;
            }
            Err(e) => {
                crate::telemetry::log_network_request_full(&ctx.db, "GET", &current, "tool:fetch_url", 0, 0, Some(st), Some(e.clone()));
                return Err(format!("fetch failed: {e}"));
            }
        }
    };
    // Web pages routinely strip to tens of thousands of tokens — far more than a local model's
    // context. Cap what we hand back so a single fetch can't blow the loop's next request (a raw
    // page can be 40k+ tokens against a 4k context). read_file has its own, larger cap.
    let text = strip_tags(&body);
    let truncated = text.chars().count() > WEB_FETCH_MAX_CHARS;
    let mut text = text.chars().take(WEB_FETCH_MAX_CHARS).collect::<String>();
    if truncated {
        text.push_str("\n\n[... truncated for length ...]");
    }
    Ok(text)
}

/// Minimal HTML → text: drop script/style bodies and tags, collapse whitespace. Good enough to feed a
/// model; not a full renderer.
fn strip_tags(html: &str) -> String {
    let mut s = html.to_string();
    for (open, close) in [("<script", "</script>"), ("<style", "</style>")] {
        while let (Some(a), Some(b)) = (s.find(open), s.find(close)) {
            if b > a {
                s.replace_range(a..b + close.len(), " ");
            } else {
                break;
            }
        }
    }
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    // decode a few common entities, collapse whitespace
    let out = out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ");
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ---- calculator (deterministic, no eval) ----------------------------------------------------

mod calc {
    //! A tiny recursive-descent evaluator for + - * / % ^, parentheses, and unary minus. No `eval`,
    //! no external process — deterministic arithmetic only (TOOL-3).
    pub fn eval(input: &str) -> Result<f64, String> {
        let mut p = Parser { chars: input.chars().collect(), pos: 0 };
        let v = p.expr()?;
        p.skip_ws();
        if p.pos != p.chars.len() {
            return Err(format!("unexpected trailing input at position {}", p.pos));
        }
        Ok(v)
    }

    struct Parser {
        chars: Vec<char>,
        pos: usize,
    }

    impl Parser {
        fn skip_ws(&mut self) {
            while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
                self.pos += 1;
            }
        }
        fn peek(&mut self) -> Option<char> {
            self.skip_ws();
            self.chars.get(self.pos).copied()
        }
        fn expr(&mut self) -> Result<f64, String> {
            let mut v = self.term()?;
            while let Some(op) = self.peek() {
                if op == '+' || op == '-' {
                    self.pos += 1;
                    let rhs = self.term()?;
                    v = if op == '+' { v + rhs } else { v - rhs };
                } else {
                    break;
                }
            }
            Ok(v)
        }
        fn term(&mut self) -> Result<f64, String> {
            let mut v = self.power()?;
            while let Some(op) = self.peek() {
                if op == '*' || op == '/' || op == '%' {
                    self.pos += 1;
                    let rhs = self.power()?;
                    v = match op {
                        '*' => v * rhs,
                        '/' => {
                            if rhs == 0.0 {
                                return Err("division by zero".into());
                            }
                            v / rhs
                        }
                        _ => {
                            if rhs == 0.0 {
                                return Err("modulo by zero".into());
                            }
                            v % rhs
                        }
                    };
                } else {
                    break;
                }
            }
            Ok(v)
        }
        fn power(&mut self) -> Result<f64, String> {
            let base = self.unary()?;
            if let Some('^') = self.peek() {
                self.pos += 1;
                let exp = self.power()?; // right-associative
                return Ok(base.powf(exp));
            }
            Ok(base)
        }
        fn unary(&mut self) -> Result<f64, String> {
            match self.peek() {
                Some('-') => {
                    self.pos += 1;
                    Ok(-self.unary()?)
                }
                Some('+') => {
                    self.pos += 1;
                    self.unary()
                }
                _ => self.atom(),
            }
        }
        fn atom(&mut self) -> Result<f64, String> {
            match self.peek() {
                Some('(') => {
                    self.pos += 1;
                    let v = self.expr()?;
                    match self.peek() {
                        Some(')') => {
                            self.pos += 1;
                            Ok(v)
                        }
                        _ => Err("expected ')'".into()),
                    }
                }
                Some(c) if c.is_ascii_digit() || c == '.' => self.number(),
                Some(c) => Err(format!("unexpected character '{c}'")),
                None => Err("unexpected end of expression".into()),
            }
        }
        fn number(&mut self) -> Result<f64, String> {
            self.skip_ws();
            let start = self.pos;
            while self.pos < self.chars.len()
                && (self.chars[self.pos].is_ascii_digit() || self.chars[self.pos] == '.')
            {
                self.pos += 1;
            }
            let s: String = self.chars[start..self.pos].iter().collect();
            s.parse::<f64>().map_err(|_| format!("invalid number '{s}'"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculator_evaluates() {
        assert_eq!(calc::eval("(2+3)*4/2").unwrap(), 10.0);
        assert_eq!(calc::eval("2^10").unwrap(), 1024.0);
        assert_eq!(calc::eval("-3 + 5").unwrap(), 2.0);
        assert_eq!(calc::eval("10 % 3").unwrap(), 1.0);
        assert!(calc::eval("1/0").is_err());
        assert!(calc::eval("2 +").is_err());
        assert!(calc::eval("abc").is_err());
    }

    fn ctx_with_ws(ws: PathBuf) -> ToolContext {
        ToolContext {
            workspace: Some(ws),
            is_auto_workspace: false,
            web_enabled: false,
            selection: None,
            db: Arc::new(Database::open_in_memory().unwrap()),
        }
    }

    #[test]
    fn confirmation_policy() {
        let ctx = |auto: bool| ToolContext {
            workspace: Some(PathBuf::from(".")), is_auto_workspace: auto, web_enabled: false,
            selection: None, db: Arc::new(Database::open_in_memory().unwrap()),
        };
        // code always confirms unless the session auto-approves.
        assert!(needs_confirmation("code", &ctx(true), false));
        assert!(needs_confirmation("code", &ctx(false), false));
        assert!(!needs_confirmation("code", &ctx(true), true)); // auto_approve overrides
        // write_file: gated in an attached folder, free in the auto-workspace.
        assert!(needs_confirmation("write_file", &ctx(false), false)); // attached folder
        assert!(!needs_confirmation("write_file", &ctx(true), false)); // auto-workspace
        // read-only never confirms.
        assert!(!needs_confirmation("read_file", &ctx(false), false));
    }

    #[test]
    fn workspace_escapes_are_refused() {
        let tmp = std::env::temp_dir().join(format!("kayon-tools-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("inside.txt"), "hello").unwrap();
        let ctx = ctx_with_ws(tmp.clone());

        assert!(read_file(&ctx, "inside.txt").is_ok());
        assert!(read_file(&ctx, "../secret.txt").is_err()); // .. escape
        assert!(read_file(&ctx, "/etc/passwd").is_err()); // absolute
        assert!(resolve_in_workspace(&ctx, "..", false).is_err());

        // Writes are scoped too: a new file inside is fine; escaping via `..` or absolute is refused.
        assert!(write_file(&ctx, "new.txt", "hi").is_ok());
        assert!(write_file(&ctx, "../escape.txt", "x").is_err()); // .. escape on write
        assert!(write_file(&ctx, "/tmp/abs.txt", "x").is_err()); // absolute on write
        assert!(resolve_in_workspace(&ctx, "../escape.txt", true).is_err());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn ssrf_host_parsing_and_private_ranges() {
        use std::net::IpAddr;
        // Domains vs IP literals, parsed by the WHATWG url crate (same as reqwest).
        match parse_url_host("http://user@host.tld:81/").unwrap() {
            UrlHost::Domain(h, p) => { assert_eq!(h, "host.tld"); assert_eq!(p, 81); }
            _ => panic!("expected domain"),
        }
        match parse_url_host("http://1.2.3.4:8080/a").unwrap() {
            UrlHost::Ip(ip, p) => { assert_eq!(ip.to_string(), "1.2.3.4"); assert_eq!(p, 8080); }
            _ => panic!("expected ip"),
        }
        // The parser-differential attack: a hand-rolled parser reads example.com, but WHATWG (and
        // reqwest) normalize the backslash so the real host is 127.0.0.1 — which we must catch.
        match parse_url_host(r"http://127.0.0.1\@example.com/").unwrap() {
            UrlHost::Ip(ip, _) => assert!(ip.is_loopback(), "backslash trick must resolve to 127.0.0.1"),
            UrlHost::Domain(h, _) => panic!("must not read {h} as the host"),
        }
        assert!(parse_url_host("ftp://example.com/").is_none()); // non-http scheme refused

        // Relative redirects resolve against the full path (RFC-3986 join), not just the origin.
        assert_eq!(resolve_redirect("https://example.com/dir/page", "next").unwrap(), "https://example.com/dir/next");
        assert_eq!(resolve_redirect("https://example.com/dir/page", "/root").unwrap(), "https://example.com/root");
        assert_eq!(resolve_redirect("https://example.com/x", "https://other.com/y").unwrap(), "https://other.com/y");
        assert!(resolve_redirect("https://example.com/x", "ftp://evil/").is_none());

        for p in ["127.0.0.1", "10.1.2.3", "192.168.0.5", "169.254.1.1", "0.0.0.0", "::1"] {
            assert!(ip_is_private(p.parse::<IpAddr>().unwrap()), "{p} should be private");
        }
        for p in ["8.8.8.8", "1.1.1.1", "93.184.216.34"] {
            assert!(!ip_is_private(p.parse::<IpAddr>().unwrap()), "{p} should be public");
        }
    }

    #[test]
    fn read_file_text_vs_binary() {
        let tmp = std::env::temp_dir().join(format!("kayon-read-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("notes.txt"), "plain readable text").unwrap();
        std::fs::write(tmp.join("blob.bin"), [0x89u8, 0x00, 0x01, b'x']).unwrap(); // has a NUL
        let ctx = ctx_with_ws(tmp.clone());
        assert!(read_file(&ctx, "notes.txt").unwrap().contains("plain readable text"));
        // A binary file is refused with a message, not returned as garbage.
        let err = read_file(&ctx, "blob.bin").unwrap_err();
        assert!(err.contains("binary"), "got: {err}");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn read_file_forgives_truncated_names() {
        let tmp = std::env::temp_dir().join(format!("kayon-fuzzy-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("Avrist Agent App – Timeline.txt"), "the real content").unwrap();
        let ctx = ctx_with_ws(tmp.clone());
        // The model asks for the truncated tail; we resolve it uniquely by suffix.
        assert!(read_file(&ctx, "Timeline.txt").unwrap().contains("the real content"));
        // Case-insensitive substring also resolves.
        assert!(read_file(&ctx, "timeline").unwrap().contains("the real content"));
        // A miss lists the available files rather than erroring blankly.
        let err = read_file(&ctx, "nope.txt").unwrap_err();
        assert!(err.contains("Available files") && err.contains("Timeline"), "got: {err}");
        // Ambiguous: two files sharing a suffix -> no silent guess.
        std::fs::write(tmp.join("Another Timeline.txt"), "other").unwrap();
        assert!(read_file(&ctx, "Timeline.txt").is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn side_effect_classification() {
        assert!(is_side_effect("write_file"));
        assert!(is_side_effect("code"));
        assert!(!is_side_effect("read_file"));
        assert!(!is_side_effect("search"));
        assert!(!is_side_effect("calculator"));
    }
}
