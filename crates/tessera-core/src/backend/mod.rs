//! Upscaling backends.
//!
//! The pipeline only knows about this trait, so a neural backend (ncnn/Vulkan,
//! ONNX Runtime, or a vendor SDK) can be dropped in later without the tiling,
//! blending, or I/O code changing.

use anyhow::Result;
use image::RgbaImage;

mod resample;

#[cfg(feature = "onnx")]
mod onnx;

pub use resample::ResampleBackend;

#[cfg(feature = "onnx")]
pub use onnx::{Device, OnnxBackend};

/// Turns one tile into a `scale_factor()`-times larger tile.
///
/// Implementations must be deterministic for a given input: the blend assumes
/// two tiles covering the same pixel broadly agree about it.
pub trait Upscaler: Send + Sync {
    /// Short identifier, surfaced in CLI output.
    fn name(&self) -> &str;

    /// Fixed integer magnification this backend produces.
    fn scale_factor(&self) -> u32;

    /// Tile size this backend requires, if it constrains one.
    ///
    /// Models with a fixed input shape must be fed exactly that many pixels.
    /// Reporting it here means callers configure themselves instead of leaving
    /// the user to discover the right tile size from an error message.
    fn preferred_tile(&self) -> Option<u32> {
        None
    }

    /// Upscale a single tile. Output must be exactly `scale_factor()` times the
    /// input in both dimensions.
    fn upscale(&self, tile: &RgbaImage) -> Result<RgbaImage>;
}
