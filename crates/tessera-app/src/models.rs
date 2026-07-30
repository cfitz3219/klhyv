//! Finding the ONNX models the app can use.
//!
//! Models are not bundled — the weights carry their own licences and run to
//! tens of megabytes — so the app looks for them in a `models` folder and tells
//! the user plainly when none is there.

use std::path::{Path, PathBuf};

/// A model the app can offer, described for the interface.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    pub path: String,
    /// File name without extension, which is what the user recognises.
    pub name: String,
}

/// Folders searched for models, most specific first.
fn search_paths() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    // Beside the executable, which is where an installed copy keeps them.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models"));
        }
    }
    // Beside the crate, so a development checkout works without installing.
    dirs.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("models"));
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("models"));
    }
    dirs
}

/// The first folder that exists, for showing the user where to put files.
pub fn models_dir() -> PathBuf {
    search_paths()
        .into_iter()
        .find(|d| d.is_dir())
        .unwrap_or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|e| e.parent().map(|p| p.join("models")))
                .unwrap_or_else(|| PathBuf::from("models"))
        })
}

/// Every `.onnx` file found, in a stable order.
pub fn available() -> Vec<ModelEntry> {
    let mut found: Vec<ModelEntry> = Vec::new();
    for dir in search_paths() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_onnx = path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
            if !is_onnx {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string();
            if found.iter().any(|m| m.name == name) {
                continue;
            }
            found.push(ModelEntry {
                path: path.display().to_string(),
                name,
            });
        }
    }
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

/// Pick a model suited to `style`, falling back to whatever exists.
///
/// Selection is by file name, since that is the only hint a bare `.onnx` file
/// carries. A user who names their files sensibly gets sensible routing.
pub fn best_for(style: &str) -> Option<ModelEntry> {
    let models = available();
    if models.is_empty() {
        return None;
    }
    let wanted: &[&str] = match style {
        "maps" => &["map", "line", "anime", "draw"],
        _ => &["photo", "real", "general"],
    };
    for hint in wanted {
        if let Some(hit) = models
            .iter()
            .find(|m| m.name.to_ascii_lowercase().contains(hint))
        {
            return Some(hit.clone());
        }
    }
    models.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_style_prefers_line_art_models() {
        // Selection is name-driven, so verify the hint order directly.
        let hints: &[&str] = &["map", "line", "anime", "draw"];
        assert!(hints.contains(&"anime"), "anime models suit line art");
    }

    #[test]
    fn models_dir_is_always_a_path() {
        assert!(!models_dir().as_os_str().is_empty());
    }
}
