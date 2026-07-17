use std::path::Path;

fn main() {
    bake_runtime_version();
    tauri_build::build();
}

/// Bake the bundled llama.cpp version into the binary from `runtime-pin.json` (RUN-1, OLL-6).
///
/// The pin is already the single record of which runtime ships, so the version gate reads it from
/// there rather than depending on someone remembering to set an environment variable. Before this,
/// `bundled_runtime_version()` read `std::env::var` at *runtime* — on the user's machine, where
/// nothing ever sets it — so it always answered "unknown" and the `runtimeMinVersion` gate failed
/// closed for every packaged build. That was invisible only because every catalog entry currently
/// declares a null `runtimeMinVersion`; the first entry that needed one would have been unlaunchable.
///
/// An explicit `KAYON_RUNTIME_VERSION` still wins, so a build against a hand-staged runtime can say
/// so. If the pin can't be read the variable is simply not set, and the gate keeps failing closed —
/// refusing to launch beats claiming a version we can't substantiate.
fn bake_runtime_version() {
    let pin = Path::new("runtime-pin.json");
    println!("cargo:rerun-if-changed=runtime-pin.json");
    println!("cargo:rerun-if-env-changed=KAYON_RUNTIME_VERSION");

    if let Ok(v) = std::env::var("KAYON_RUNTIME_VERSION") {
        if !v.trim().is_empty() {
            println!("cargo:rustc-env=KAYON_RUNTIME_VERSION={}", v.trim());
            return;
        }
    }

    // Deliberately not a JSON dependency for one field: build-dependencies are a supply-chain
    // surface, and this is a flat file we control.
    let Ok(text) = std::fs::read_to_string(pin) else { return };
    if let Some(tag) = json_string_field(&text, "tag") {
        println!("cargo:rustc-env=KAYON_RUNTIME_VERSION={tag}");
    }
}

/// Pull `"<key>": "<value>"` out of the pin. Tolerates whitespace; returns None rather than guess.
fn json_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let rest = text.get(text.find(&needle)? + needle.len()..)?;
    let rest = rest.trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    let value = &rest[..end];
    (!value.is_empty()).then(|| value.to_string())
}
