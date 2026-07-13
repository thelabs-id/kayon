use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

use crate::db::Database;
use crate::ipc::*;

pub fn library_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".kayon")
        .join("models")
}

pub fn deterministic_path(model_id: &str, quant_label: &str) -> String {
    let dir = library_dir();
    let filename = format!("{}-{}.gguf", model_id.replace('/', "_"), quant_label);
    dir.join(filename).to_string_lossy().to_string()
}

pub fn init_library_dir() -> Result<()> {
    let dir = library_dir();
    std::fs::create_dir_all(&dir)?;
    Ok(())
}

pub fn list_installed(db: &Database) -> Result<Vec<InstalledModel>> {
    db.list_installed_models()
}

pub fn delete_model(db: &Database, id: &str, confirm: bool) -> Result<bool> {
    if !confirm {
        return Err(anyhow!("two-step delete required: call with confirm=true"));
    }
    let model = db.get_installed_model(id)?
        .ok_or_else(|| anyhow!("model not found: {}", id))?;
    let path = Path::new(&model.path);
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    db.remove_installed_model(id)?;
    Ok(true)
}

pub fn scan_library_dir(db: &Database) -> Result<Vec<InstalledModel>> {
    let dir = library_dir();
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut found = vec![];
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "gguf").unwrap_or(false) {
            let path_str = path.to_string_lossy().to_string();
            if let Ok(Some(existing)) = db.find_installed_by_path(&path_str) {
                found.push(existing);
            }
        }
    }
    Ok(found)
}
