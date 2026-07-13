use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};

use crate::db::Database;
use crate::ipc::*;

fn kayon_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".kayon")
}

fn library_dir_config() -> PathBuf {
    kayon_dir().join("library_dir.txt")
}

/// The managed library directory (LIB-1). Relocatable: if a location has been set (move-in-place
/// migration), it's read from a small config file; otherwise the default under the user profile.
pub fn library_dir() -> PathBuf {
    if let Ok(s) = std::fs::read_to_string(library_dir_config()) {
        let t = s.trim();
        if !t.is_empty() {
            return PathBuf::from(t);
        }
    }
    kayon_dir().join("models")
}

fn set_library_dir(new_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(kayon_dir())?;
    std::fs::write(library_dir_config(), new_dir.to_string_lossy().as_bytes())?;
    Ok(())
}

/// Relocate the managed library to `new_dir` (LIB-1 move-in-place; enables zero-copy Ollama
/// adoption on the target volume, OLL-4). Moves each managed file (rename on same volume, else
/// copy+remove), updates the DB paths, then records the new location.
pub fn relocate_library(db: &Database, new_dir: &str) -> Result<usize> {
    let new_dir = PathBuf::from(new_dir);
    if new_dir.as_os_str().is_empty() {
        return Err(anyhow!("new library path is empty"));
    }
    std::fs::create_dir_all(&new_dir)?;
    let old_dir = library_dir();
    let mut moved = 0usize;
    for m in db.list_installed_models()? {
        let old_path = PathBuf::from(&m.path);
        if !old_path.exists() {
            continue;
        }
        let fname = match old_path.file_name() {
            Some(f) => f,
            None => continue,
        };
        let new_path = new_dir.join(fname);
        if std::fs::rename(&old_path, &new_path).is_err() {
            // Cross-volume: copy then remove the original.
            std::fs::copy(&old_path, &new_path)?;
            let _ = std::fs::remove_file(&old_path);
        }
        db.update_installed_path(&m.id, &new_path.to_string_lossy())?;
        moved += 1;
    }
    set_library_dir(&new_dir)?;
    let _ = old_dir; // old dir left in place if non-empty; only managed files are moved
    Ok(moved)
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

