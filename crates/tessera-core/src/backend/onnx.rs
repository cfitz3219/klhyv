//! ONNX Runtime backend: runs Real-ESRGAN-style super-resolution models.
//!
//! ONNX Runtime was chosen over ncnn because its execution providers cover the
//! desktop GPU landscape from one codebase (CUDA and TensorRT for NVIDIA,
//! DirectML for any vendor on Windows, CoreML on Apple), and because its CPU
//! provider means the whole path is testable on machines without a GPU.
//!
//! The model is supplied by the user rather than bundled: super-resolution
//! weights carry their own licences, and which model suits a scanned map is a
//! different answer than for a photograph.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, bail, ensure, Context, Result};
use image::{GrayImage, Luma, Rgba, RgbaImage};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Tensor, ValueType};

use super::Upscaler;

/// Which execution providers to offer the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Register every provider compiled in, best first, falling back to CPU.
    #[default]
    Auto,
    /// Force CPU even when a GPU provider is available.
    Cpu,
}

/// Spatial dimensions a model accepts, when it fixes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedInput {
    width: u32,
    height: u32,
}

pub struct OnnxBackend {
    /// ONNX Runtime needs `&mut` to run, and the pipeline calls backends from
    /// several threads, so inference is serialised here. Little is lost: the
    /// runtime parallelises each model internally across its own thread pool.
    session: Mutex<Session>,
    input_name: String,
    output_name: String,
    fixed_input: Option<FixedInput>,
    scale: u32,
    name: String,
}

impl OnnxBackend {
    /// Load a model from disk.
    ///
    /// `scale_override` is only needed for models whose input and output shapes
    /// are both dynamic, where the factor cannot be read off the graph.
    pub fn new(model: &Path, device: Device, scale_override: Option<u32>) -> Result<Self> {
        ensure!(
            model.is_file(),
            "no model file at {} (pass --model with a path to a .onnx file)",
            model.display()
        );

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // ort's builder errors borrow the builder, so they are not Send and
        // cannot cross into anyhow directly; flatten them to strings first.
        let session = (|| -> Result<Session, String> {
            let mut builder = Session::builder().map_err(|e| e.to_string())?;
            if device == Device::Auto {
                builder = register_gpu_providers(builder)?;
            }
            builder
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| e.to_string())?
                .with_intra_threads(threads)
                .map_err(|e| e.to_string())?
                .commit_from_file(model)
                .map_err(|e| e.to_string())
        })()
        .map_err(|e| anyhow!(e))
        .with_context(|| format!("loading {}", model.display()))?;

        let input = session
            .inputs()
            .first()
            .context("model declares no inputs")?;
        let output = session
            .outputs()
            .first()
            .context("model declares no outputs")?;
        let input_name = input.name().to_string();
        let output_name = output.name().to_string();

        let in_shape = tensor_shape(input.dtype())
            .with_context(|| format!("input {input_name:?} is not a tensor"))?;
        let out_shape = tensor_shape(output.dtype())
            .with_context(|| format!("output {output_name:?} is not a tensor"))?;

        ensure!(
            in_shape.len() == 4 && out_shape.len() == 4,
            "expected 4-D NCHW tensors, got input {in_shape:?} and output {out_shape:?}"
        );
        // -1 marks a dynamic dimension.
        ensure!(
            in_shape[1] == 3 || in_shape[1] == -1,
            "expected a 3-channel RGB input, got {} channels",
            in_shape[1]
        );

        let fixed_input = match (in_shape[3], in_shape[2]) {
            (w, h) if w > 0 && h > 0 => Some(FixedInput {
                width: w as u32,
                height: h as u32,
            }),
            _ => None,
        };

        let scale = match (scale_override, infer_scale(&in_shape, &out_shape)) {
            (Some(given), Some(found)) => {
                ensure!(
                    given == found,
                    "model scales by {found}x but {given}x was requested"
                );
                given
            }
            (Some(given), None) => given,
            (None, Some(found)) => found,
            (None, None) => bail!(
                "model has dynamic input and output shapes, so its scale factor \
                 cannot be read from the graph; pass --scale explicitly"
            ),
        };
        ensure!(scale >= 1, "model scale factor must be >= 1, got {scale}");

        let label = model
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");

        Ok(Self {
            session: Mutex::new(session),
            input_name,
            output_name,
            fixed_input,
            scale,
            name: format!("onnx:{label}"),
        })
    }

    /// Spatial size this tile must be presented at.
    fn model_input_size(&self, tile_w: u32, tile_h: u32) -> Result<(u32, u32)> {
        match self.fixed_input {
            None => Ok((tile_w, tile_h)),
            Some(fixed) => {
                ensure!(
                    tile_w <= fixed.width && tile_h <= fixed.height,
                    "model accepts at most {}x{} pixels but the tile is {}x{}; \
                     rerun with --tile {}",
                    fixed.width,
                    fixed.height,
                    tile_w,
                    tile_h,
                    fixed.width.min(fixed.height)
                );
                Ok((fixed.width, fixed.height))
            }
        }
    }
}

impl Upscaler for OnnxBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn scale_factor(&self) -> u32 {
        self.scale
    }

    fn preferred_tile(&self) -> Option<u32> {
        // Square tiles only, so the smaller side governs.
        self.fixed_input.map(|f| f.width.min(f.height))
    }

    fn upscale(&self, tile: &RgbaImage) -> Result<RgbaImage> {
        let (tw, th) = tile.dimensions();
        ensure!(tw > 0 && th > 0, "cannot upscale a zero-sized tile");

        let (mw, mh) = self.model_input_size(tw, th)?;
        // Replicate rather than zero-pad: a black border is real content to the
        // model, and the artifacts it provokes bleed back into the valid area.
        let padded = if (mw, mh) == (tw, th) {
            tile.clone()
        } else {
            pad_replicate(tile, mw, mh)
        };

        let planes = to_nchw(&padded);
        let input = Tensor::from_array(([1usize, 3, mh as usize, mw as usize], planes))
            .map_err(|e| anyhow!("building input tensor: {e}"))?;

        // Scoped so the borrowed outputs, and then the lock, are released before
        // the comparatively slow pixel work below.
        let rgb = {
            let mut session = self
                .session
                .lock()
                .map_err(|_| anyhow!("model session poisoned by a previous panic"))?;
            let outputs = session
                .run(ort::inputs![self.input_name.as_str() => input])
                .map_err(|e| anyhow!("inference failed: {e}"))?;
            let (shape, data) = outputs[self.output_name.as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("reading model output: {e}"))?;

            let dims: Vec<i64> = shape.iter().copied().collect();
            ensure!(
                dims.len() == 4,
                "expected a 4-D output tensor, got shape {dims:?}"
            );
            let (out_h, out_w) = (dims[2] as u32, dims[3] as u32);
            ensure!(
                out_w >= tw * self.scale && out_h >= th * self.scale,
                "model returned {out_w}x{out_h}, too small for a {}x{} result",
                tw * self.scale,
                th * self.scale
            );
            from_nchw(data, out_w, out_h)
        };

        // Discard the padding, then restore alpha, which the model never saw.
        let cropped =
            image::imageops::crop_imm(&rgb, 0, 0, tw * self.scale, th * self.scale).to_image();
        Ok(reattach_alpha(cropped, tile, self.scale))
    }
}

/// Register GPU execution providers that were compiled in.
///
/// Providers are advisory: ONNX Runtime silently falls back to CPU when one is
/// unavailable at runtime, so a binary built with CUDA still works without it.
#[allow(unused_mut, clippy::let_and_return)]
fn register_gpu_providers(
    mut builder: ort::session::builder::SessionBuilder,
) -> Result<ort::session::builder::SessionBuilder, String> {
    #[cfg(feature = "cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;
        builder = builder
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .map_err(|e| e.to_string())?;
    }
    #[cfg(feature = "directml")]
    {
        use ort::execution_providers::DirectMLExecutionProvider;
        builder = builder
            .with_execution_providers([DirectMLExecutionProvider::default().build()])
            .map_err(|e| e.to_string())?;
    }
    #[cfg(feature = "coreml")]
    {
        use ort::execution_providers::CoreMLExecutionProvider;
        builder = builder
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(|e| e.to_string())?;
    }
    Ok(builder)
}

fn tensor_shape(ty: &ValueType) -> Option<Vec<i64>> {
    match ty {
        ValueType::Tensor { shape, .. } => Some(shape.iter().copied().collect()),
        _ => None,
    }
}

/// Read the magnification off the graph, when both shapes are static.
fn infer_scale(input: &[i64], output: &[i64]) -> Option<u32> {
    let (iw, ih, ow, oh) = (input[3], input[2], output[3], output[2]);
    if iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0 {
        return None;
    }
    if ow % iw != 0 || oh % ih != 0 {
        return None;
    }
    let (sx, sy) = (ow / iw, oh / ih);
    // A model that scales the axes differently is not something this pipeline
    // can place correctly, so decline rather than guess.
    if sx != sy {
        return None;
    }
    u32::try_from(sx).ok()
}

fn pad_replicate(src: &RgbaImage, width: u32, height: u32) -> RgbaImage {
    let (sw, sh) = src.dimensions();
    RgbaImage::from_fn(width, height, |x, y| {
        *src.get_pixel(x.min(sw - 1), y.min(sh - 1))
    })
}

/// RGBA image to planar NCHW float, scaled to 0..1. Alpha is dropped.
fn to_nchw(img: &RgbaImage) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let plane = (w * h) as usize;
    let mut out = vec![0.0f32; plane * 3];
    for (i, px) in img.pixels().enumerate() {
        out[i] = px.0[0] as f32 / 255.0;
        out[plane + i] = px.0[1] as f32 / 255.0;
        out[2 * plane + i] = px.0[2] as f32 / 255.0;
    }
    out
}

/// Planar NCHW float back to an opaque RGBA image.
fn from_nchw(data: &[f32], width: u32, height: u32) -> RgbaImage {
    let plane = (width * height) as usize;
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    RgbaImage::from_fn(width, height, |x, y| {
        let i = (y * width + x) as usize;
        Rgba([
            to_u8(data[i]),
            to_u8(data[plane + i]),
            to_u8(data[2 * plane + i]),
            255,
        ])
    })
}

/// Restore the alpha channel the model discarded.
///
/// Super-resolution models are RGB, so alpha is resampled classically and
/// reattached. Fully opaque tiles, which is most map content, skip the work.
fn reattach_alpha(mut rgb: RgbaImage, original: &RgbaImage, scale: u32) -> RgbaImage {
    if original.pixels().all(|p| p.0[3] == 255) {
        return rgb;
    }
    let (w, h) = original.dimensions();
    let alpha = GrayImage::from_fn(w, h, |x, y| Luma([original.get_pixel(x, y).0[3]]));
    let scaled = image::imageops::resize(
        &alpha,
        w * scale,
        h * scale,
        image::imageops::FilterType::Lanczos3,
    );
    for (x, y, px) in rgb.enumerate_pixels_mut() {
        px.0[3] = scaled.get_pixel(x, y).0[0];
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_is_read_from_static_shapes() {
        assert_eq!(infer_scale(&[1, 3, 128, 128], &[1, 3, 512, 512]), Some(4));
        assert_eq!(infer_scale(&[1, 3, 64, 64], &[1, 3, 128, 128]), Some(2));
    }

    #[test]
    fn dynamic_shapes_yield_no_scale() {
        assert_eq!(infer_scale(&[1, 3, -1, -1], &[1, 3, -1, -1]), None);
    }

    #[test]
    fn anisotropic_and_fractional_scales_are_declined() {
        assert_eq!(infer_scale(&[1, 3, 100, 100], &[1, 3, 200, 400]), None);
        assert_eq!(infer_scale(&[1, 3, 100, 100], &[1, 3, 150, 150]), None);
    }

    #[test]
    fn replicate_padding_extends_edges() {
        let src = RgbaImage::from_fn(2, 2, |x, y| {
            Rgba([(x * 10) as u8, (y * 20) as u8, 0, 255])
        });
        let padded = pad_replicate(&src, 4, 4);
        assert_eq!(padded.get_pixel(3, 3).0, [10, 20, 0, 255]);
        assert_eq!(padded.get_pixel(0, 3).0, [0, 20, 0, 255]);
        // No black border: padding must never introduce a colour the tile lacks.
        assert!(padded.pixels().all(|p| p.0[3] == 255));
    }

    #[test]
    fn nchw_round_trips() {
        let src = RgbaImage::from_fn(5, 3, |x, y| {
            Rgba([(x * 40) as u8, (y * 60) as u8, ((x + y) * 20) as u8, 255])
        });
        let restored = from_nchw(&to_nchw(&src), 5, 3);
        assert_eq!(restored, src);
    }

    #[test]
    fn opaque_tiles_keep_full_alpha() {
        let original = RgbaImage::from_pixel(4, 4, Rgba([1, 2, 3, 255]));
        let rgb = RgbaImage::from_pixel(8, 8, Rgba([9, 9, 9, 255]));
        let out = reattach_alpha(rgb, &original, 2);
        assert!(out.pixels().all(|p| p.0[3] == 255));
    }

    #[test]
    fn transparency_is_carried_across() {
        let original = RgbaImage::from_fn(8, 8, |x, _| {
            Rgba([0, 0, 0, if x < 4 { 255 } else { 0 }])
        });
        let rgb = RgbaImage::from_pixel(16, 16, Rgba([9, 9, 9, 255]));
        let out = reattach_alpha(rgb, &original, 2);
        assert_eq!(out.get_pixel(1, 1).0[3], 255);
        assert_eq!(out.get_pixel(14, 1).0[3], 0);
    }

    #[test]
    fn missing_model_file_is_reported_clearly() {
        let err = match OnnxBackend::new(Path::new("/nonexistent/model.onnx"), Device::Cpu, None) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("loading a missing model should fail"),
        };
        assert!(err.contains("no model file"), "unhelpful error: {err}");
    }
}
