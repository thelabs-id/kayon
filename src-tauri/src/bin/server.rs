//! Standalone API server (no Tauri window) — serves the same API + built UI on
//! http://127.0.0.1:9518 for browser-based development and automated testing.
//! The desktop app (`kayon`) runs this exact server behind a native WebView2 window.

fn main() {
    let _ = env_logger::try_init();
    kayon::start_api_server();
    // Park the main thread forever; the server runs on its own thread.
    loop {
        std::thread::park();
    }
}
