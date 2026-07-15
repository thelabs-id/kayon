//! The agentic tool loop (TOOL-1). Runs server-side because tools execute locally (filesystem
//! scoping, code execution, logged web egress). It calls the loaded model over `llama-server`'s
//! OpenAI-compatible `/v1/chat/completions` with the advertised `tools`, and on `tool_calls`
//! dispatches to the local executor, appends `tool` results, and re-calls — bounded by a
//! max-iteration cap — until the model returns a final answer.
//!
//! Events are streamed to the UI as JSON lines over SSE so each tool call (name, args, result) is
//! shown inline and never hidden (TOOL-7). Side-effectful tools (`write_file`, `code`) pause for a
//! per-call confirmation carried over a oneshot channel, unless the session auto-approves (TOOL-6).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

use crate::db::Database;
use crate::tools::{self, ToolContext};

/// Pending-confirmation registry: callId → the sender the decision endpoint resolves.
pub type Decisions = Arc<Mutex<HashMap<String, oneshot::Sender<bool>>>>;

const MAX_ITERS: usize = 8;
const CONFIRM_TIMEOUT_SECS: u64 = 300;
/// Safety cap on the tool-result text appended to the conversation before the next model call. Any
/// single tool (a large file, a web page) is truncated here so it can never overflow the model's
/// context in one step — the tool's own caps are tighter, this is the last-line guard.
const MAX_TOOL_RESULT_CHARS: usize = 12000;

/// Truncate on a char boundary with a visible marker, so appended tool output stays context-safe.
fn cap_result(s: String) -> String {
    if s.chars().count() <= MAX_TOOL_RESULT_CHARS {
        return s;
    }
    let mut out: String = s.chars().take(MAX_TOOL_RESULT_CHARS).collect();
    out.push_str("\n[... tool result truncated for length ...]");
    out
}

pub struct AgentRequest {
    pub messages: Vec<(String, String)>, // (role, content) — history + latest user turn
    pub system_prompt: String,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: i64,
    pub workspace: Option<String>,
    pub session_id: Option<String>,
    pub web_enabled: bool,
    pub auto_approve: bool,
    pub selection: Option<String>,
}

/// Drive the loop, emitting JSON event strings into `tx`. Event shapes (all with a `type` field):
/// `token{text}`, `tool_call{callId,name,args}`, `confirm{callId,name,args}`,
/// `tool_result{callId,name,ok,result}`, `error{message}`, `done`.
pub async fn run(
    db: Arc<Database>,
    port: u16,
    supports_tools: bool,
    req: AgentRequest,
    decisions: Decisions,
    tx: mpsc::Sender<String>,
) {
    // Effective workspace: an explicitly attached folder, else the Kayon-owned per-session
    // auto-workspace (~/.kayon/workspace/<session>/), created lazily. So the filesystem/code tools
    // always have a home — the model can produce artifacts without the user attaching anything first.
    let (workspace, is_auto_workspace) = match req.workspace.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
        Some(w) => (Some(std::path::PathBuf::from(w)), false),
        None => match req.session_id.as_ref() {
            Some(sid) => {
                let dir = crate::kayon_home().join("workspace").join(sid);
                let _ = std::fs::create_dir_all(&dir);
                (Some(dir), true)
            }
            None => (None, false),
        },
    };
    let ctx = ToolContext {
        workspace,
        is_auto_workspace,
        web_enabled: req.web_enabled,
        selection: req.selection.clone(),
        db: db.clone(),
    };
    let specs = tools::tool_specs(supports_tools, &ctx);

    // Build the running OpenAI message array, starting with the system prompt.
    let mut messages: Vec<Value> = Vec::new();
    if !req.system_prompt.trim().is_empty() {
        messages.push(json!({ "role": "system", "content": req.system_prompt }));
    }
    for (role, content) in &req.messages {
        messages.push(json!({ "role": role, "content": content }));
    }

    let client = reqwest::Client::new();
    let url = format!("http://127.0.0.1:{port}/v1/chat/completions");

    for _iter in 0..MAX_ITERS {
        // Cancellation: if the SSE receiver is gone (user navigated away / closed the chat), stop
        // before making another model call or running any tool — otherwise auto-approved side
        // effects could keep executing invisibly, violating TOOL-7 transparency.
        if tx.is_closed() {
            return;
        }
        let mut body = json!({
            "model": "kayon",
            "messages": messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "stream": true,
        });
        if !specs.is_empty() {
            body["tools"] = json!(specs);
            body["tool_choice"] = json!("auto");
        }

        let resp = match client.post(&url).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                emit(&tx, json!({ "type": "error", "message": format!("model request failed: {e}") })).await;
                return;
            }
        };
        if !resp.status().is_success() {
            let code = resp.status().as_u16();
            let detail = resp.text().await.unwrap_or_default();
            emit(&tx, json!({ "type": "error", "message": format!("model returned HTTP {code}: {}", detail.chars().take(300).collect::<String>()) })).await;
            return;
        }

        // ---- consume the streamed completion, splitting content from tool_calls -------------
        let mut content = String::new();
        let mut calls: Vec<AccCall> = Vec::new();
        let mut finish = String::new();
        // Buffer raw BYTES, not a lossy per-chunk string: a multi-byte UTF-8 character (or an emoji
        // in tool arguments) can straddle two `bytes_stream()` chunks, and decoding each chunk
        // separately would corrupt it with replacement characters. We only decode a line once it is
        // complete (terminated by '\n'), at which point its bytes are whole.
        let mut buf: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(b) => b,
                Err(e) => {
                    emit(&tx, json!({ "type": "error", "message": format!("stream error: {e}") })).await;
                    return;
                }
            };
            buf.extend_from_slice(&bytes);
            // SSE frames are newline-delimited; keep the trailing partial line (and any partial
            // multi-byte char at its end) in `buf`.
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
                let line = String::from_utf8_lossy(&line_bytes).trim().to_string();
                let Some(data) = line.strip_prefix("data:") else { continue };
                let data = data.trim();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let Ok(j) = serde_json::from_str::<Value>(data) else { continue };
                let Some(choice) = j.get("choices").and_then(|c| c.get(0)) else { continue };
                if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    finish = fr.to_string();
                }
                let delta = choice.get("delta").cloned().unwrap_or(Value::Null);
                if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
                    if !c.is_empty() {
                        content.push_str(c);
                        emit(&tx, json!({ "type": "token", "text": c })).await;
                    }
                }
                if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                    for tc in tcs {
                        let idx = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        while calls.len() <= idx {
                            calls.push(AccCall::default());
                        }
                        let slot = &mut calls[idx];
                        if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                            if !id.is_empty() {
                                slot.id = id.to_string();
                            }
                        }
                        if let Some(f) = tc.get("function") {
                            if let Some(n) = f.get("name").and_then(|v| v.as_str()) {
                                if !n.is_empty() {
                                    slot.name = n.to_string();
                                }
                            }
                            if let Some(a) = f.get("arguments").and_then(|v| v.as_str()) {
                                slot.args.push_str(a);
                            }
                        }
                    }
                }
            }
        }

        // No tool calls -> this streamed message was the final answer (finish_reason "stop").
        let _ = finish;
        if calls.is_empty() {
            emit(&tx, json!({ "type": "done" })).await;
            return;
        }

        // Record the assistant turn (with its tool_calls) so the follow-up `tool` messages attach.
        let assistant_tool_calls: Vec<Value> = calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                json!({
                    "id": call_id(c, i),
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.args }
                })
            })
            .collect();
        messages.push(json!({
            "role": "assistant",
            "content": content,
            "tool_calls": assistant_tool_calls
        }));

        // ---- execute each tool call ---------------------------------------------------------
        for (i, c) in calls.iter().enumerate() {
            // Abort mid-turn too: don't run (or confirm) further tools once the client is gone.
            if tx.is_closed() {
                return;
            }
            // Two distinct ids: `cid` is the model's tool_call_id (correlates the assistant call with
            // its `tool` reply in the OpenAI messages); `ui_id` is a globally-unique id for the UI
            // cards and the confirmation registry, so concurrent agent streams that both synthesize
            // e.g. `call_0` can never collide and mis-route one stream's approval to another.
            let cid = call_id(c, i);
            let ui_id = uuid::Uuid::new_v4().to_string();
            let args: Value = serde_json::from_str(&c.args).unwrap_or(json!({}));
            emit(&tx, json!({ "type": "tool_call", "callId": ui_id, "name": c.name, "args": args })).await;

            let result: Result<String, String> = if c.name.is_empty() {
                Err("tool call had no name".into())
            } else if tools::needs_confirmation(&c.name, &ctx, req.auto_approve) {
                // TOOL-6: pause for user confirmation (code always; write_file only to an attached
                // folder — auto-workspace writes flow through for the artifact UX).
                let (dtx, drx) = oneshot::channel();
                decisions.lock().unwrap().insert(ui_id.clone(), dtx);
                emit(&tx, json!({ "type": "confirm", "callId": ui_id, "name": c.name, "args": args })).await;
                let approved = match tokio::time::timeout(
                    std::time::Duration::from_secs(CONFIRM_TIMEOUT_SECS),
                    drx,
                )
                .await
                {
                    Ok(Ok(v)) => v,
                    _ => false,
                };
                decisions.lock().unwrap().remove(&ui_id);
                if approved {
                    tools::execute(&c.name, &args, &ctx).await
                } else {
                    Err("user declined to run this tool".into())
                }
            } else {
                tools::execute(&c.name, &args, &ctx).await
            };

            let (ok, text) = match &result {
                Ok(s) => (true, s.clone()),
                Err(e) => (false, e.clone()),
            };
            emit(&tx, json!({ "type": "tool_result", "callId": ui_id, "name": c.name, "ok": ok, "result": text })).await;
            // The model sees the result (or the error) as a `tool` message — errors are not swallowed.
            messages.push(json!({
                "role": "tool",
                "tool_call_id": cid,
                "content": cap_result(if ok { text } else { format!("ERROR: {text}") })
            }));
        }
        // loop: re-call the model with the tool results appended.
    }

    emit(&tx, json!({ "type": "error", "message": format!("stopped after {MAX_ITERS} tool iterations") })).await;
}

#[derive(Default)]
struct AccCall {
    id: String,
    name: String,
    args: String,
}

/// A stable id for a tool call: the model-provided id when present, else a synthesized one so the
/// UI card and the `tool` reply still correlate.
fn call_id(c: &AccCall, i: usize) -> String {
    if c.id.is_empty() {
        format!("call_{i}")
    } else {
        c.id.clone()
    }
}

async fn emit(tx: &mpsc::Sender<String>, v: Value) {
    let _ = tx.send(v.to_string()).await;
}
