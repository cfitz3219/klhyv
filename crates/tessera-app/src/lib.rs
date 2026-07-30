//! Tessera desktop application.
//!
//! A thin shell over `tessera-core`: the window picks files, chooses a
//! magnification, and shows a full-size before/after comparison. All the real
//! work happens in the engine.

mod models;
mod preview;

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tessera_core::backend::{ResampleBackend, Upscaler};
use tessera_core::{pipeline, world, PngSink, ScaleStrategy, UpscaleOptions};

/// Commands report failures as plain sentences, since they are shown to the user.
type CmdResult<T> = Result<T, String>;

/// Run blocking work off the interface thread, flattening the join error.
async fn blocking<T, F>(f: F) -> CmdResult<T>
where
    F: FnOnce() -> CmdResult<T> + Send + 'static,
    T: Send + 'static,
{
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|e| format!("The task stopped unexpectedly: {e}"))?
}

/// A configured upscaler plus how to describe it in the interface.
struct Engine {
    backend: Box<dyn Upscaler>,
    label: String,
    neural: bool,
}

impl Engine {
    /// Tile and overlap this backend wants.
    ///
    /// A model with a fixed input size dictates the tile, so the user is never
    /// asked to work it out.
    fn options(&self) -> UpscaleOptions {
        let tile = self.backend.preferred_tile().unwrap_or(256);
        UpscaleOptions {
            tile,
            // An eighth of the tile is enough overlap to hide model edge
            // effects without inflating the work much.
            overlap: (tile / 8).max(8),
        }
    }
}

/// Choose the best engine for `style`, falling back when no model is installed.
fn build_engine(style: &str, target: u32) -> CmdResult<Engine> {
    #[cfg(feature = "onnx")]
    {
        use tessera_core::backend::{Device, OnnxBackend};
        if let Some(entry) = models::best_for(style) {
            let backend = OnnxBackend::new(Path::new(&entry.path), Device::Auto, None)
                .map_err(|e| format!("Could not load the model \"{}\": {e}", entry.name))?;
            return Ok(Engine {
                backend: Box::new(backend),
                label: entry.name,
                neural: true,
            });
        }
    }
    let _ = style;
    // Without a model the app still works, just without invented detail. Saying
    // so plainly beats refusing to open.
    let backend = ResampleBackend::new(target, "lanczos3").map_err(|e| e.to_string())?;
    Ok(Engine {
        backend: Box::new(backend),
        label: "Basic enlargement".to_string(),
        neural: false,
    })
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageInfo {
    path: String,
    name: String,
    width: u32,
    height: u32,
    preview: String,
    georeferenced: bool,
}

#[tauri::command]
async fn open_image(path: String) -> CmdResult<ImageInfo> {
    blocking(move || {
        let p = PathBuf::from(&path);
        let img = image::open(&p)
            .map_err(|e| format!("Could not open that image: {e}"))?
            .to_rgba8();
        let (width, height) = img.dimensions();
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
            .to_string();
        let thumb = preview::fit(&img, 1400);
        Ok(ImageInfo {
            path,
            name,
            width,
            height,
            preview: preview::to_data_url(&thumb).map_err(|e| e.to_string())?,
            georeferenced: world::read_sidecar(&p).ok().flatten().is_some(),
        })
    })
    .await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DetailPreview {
    before: String,
    after: String,
    /// Source pixels covered, so the interface can say what is being shown.
    source_px: u32,
    engine: String,
    neural: bool,
}

/// Render a full-size before/after of one region.
///
/// Comparing whole images fitted to a window would shrink the result back down
/// and hide the very detail being judged, so the comparison happens at 1:1 on a
/// region instead.
#[tauri::command]
async fn preview_detail(
    path: String,
    scale: u32,
    style: String,
    cx: f32,
    cy: f32,
) -> CmdResult<DetailPreview> {
    blocking(move || {
        let source = image::open(&path)
            .map_err(|e| format!("Could not open that image: {e}"))?
            .to_rgba8();
        // Keep the rendered comparison around 1200px whatever the magnification.
        let source_px = (1200 / scale.max(1)).clamp(64, 512);
        let patch = preview::region(&source, cx, cy, source_px);
        let (pw, ph) = patch.dimensions();

        let engine = build_engine(&style, scale)?;
        let before = image::imageops::resize(
            &patch,
            pw * scale,
            ph * scale,
            image::imageops::FilterType::Lanczos3,
        );
        let after = pipeline::upscale_to_target(
            &patch,
            engine.backend.as_ref(),
            scale,
            engine.options(),
            None,
        )
        .map_err(|e| format!("Could not enlarge the preview: {e}"))?;

        Ok(DetailPreview {
            before: preview::to_data_url(&before).map_err(|e| e.to_string())?,
            after: preview::to_data_url(&after).map_err(|e| e.to_string())?,
            source_px: pw,
            engine: engine.label,
            neural: engine.neural,
        })
    })
    .await
}

#[derive(Serialize, Clone)]
struct ProgressPayload {
    done: usize,
    total: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RunResult {
    output_path: String,
    width: u32,
    height: u32,
    elapsed_ms: u64,
    engine: String,
    neural: bool,
    plan: String,
    georeferenced: bool,
}

#[tauri::command]
async fn run_upscale(
    app: AppHandle,
    path: String,
    output: String,
    scale: u32,
    style: String,
) -> CmdResult<RunResult> {
    blocking(move || {
        let src_path = PathBuf::from(&path);
        let out_path = PathBuf::from(&output);
        let source = image::open(&src_path)
            .map_err(|e| format!("Could not open that image: {e}"))?
            .to_rgba8();
        let (w, h) = source.dimensions();
        let (out_w, out_h) = (w * scale, h * scale);

        let engine = build_engine(&style, scale)?;
        let strategy = ScaleStrategy::plan(scale, engine.backend.scale_factor())
            .map_err(|e| e.to_string())?;

        if let Some(parent) = out_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Could not create that folder: {e}"))?;
            }
        }

        // Thousands of tiles would flood the interface, so report about 200
        // times regardless of job size.
        let emitter = |done: usize, total: usize| {
            let step = (total / 200).max(1);
            if done.is_multiple_of(step) || done == total {
                let _ = app.emit("tessera:progress", ProgressPayload { done, total });
            }
        };

        let started = Instant::now();
        let writes_png = out_path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("png"));

        if writes_png {
            // Rows go straight to disk, so output size is not bounded by memory.
            let mut sink = PngSink::create(&out_path, out_w, out_h).map_err(|e| e.to_string())?;
            pipeline::upscale_to_sink(
                &source,
                engine.backend.as_ref(),
                scale,
                engine.options(),
                &mut sink,
                Some(&emitter),
            )
            .map_err(|e| format!("Enlarging failed: {e}"))?;
        } else {
            let result = pipeline::upscale_to_target(
                &source,
                engine.backend.as_ref(),
                scale,
                engine.options(),
                Some(&emitter),
            )
            .map_err(|e| format!("Enlarging failed: {e}"))?;
            result
                .save(&out_path)
                .map_err(|e| format!("Could not save the result: {e}"))?;
        }

        // Keep the map pinned where it belongs.
        let mut georeferenced = false;
        if let Ok(Some((_, sidecar))) = world::read_sidecar(&src_path) {
            world::write_sidecar(&out_path, sidecar.scaled(scale))
                .map_err(|e| format!("Could not write the map position file: {e}"))?;
            georeferenced = true;
        }

        Ok(RunResult {
            output_path: out_path.display().to_string(),
            width: out_w,
            height: out_h,
            elapsed_ms: started.elapsed().as_millis() as u64,
            engine: engine.label,
            neural: engine.neural,
            plan: strategy.describe(),
            georeferenced,
        })
    })
    .await
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelStatus {
    models: Vec<models::ModelEntry>,
    folder: String,
}

#[tauri::command]
fn model_status() -> ModelStatus {
    ModelStatus {
        models: models::available(),
        folder: models::models_dir().display().to_string(),
    }
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            open_image,
            preview_detail,
            run_upscale,
            model_status
        ])
        .run(tauri::generate_context!())
        .expect("Tessera could not start");
}
