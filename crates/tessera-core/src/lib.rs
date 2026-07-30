//! Tessera: seam-free tiled upscaling for large images.
//!
//! The engine splits an image into overlapping tiles, runs each through a
//! pluggable [`backend::Upscaler`], and cross-fades the results back together.
//! Tiling is what makes gigapixel scans possible at all, and the cross-fade is
//! what keeps the tile grid from showing in the result.
//!
//! ```no_run
//! use tessera_core::{backend::ResampleBackend, pipeline, UpscaleOptions};
//!
//! let source = image::open("map.tif")?.to_rgba8();
//! let backend = ResampleBackend::new(4, "lanczos3")?;
//! let out = pipeline::upscale_tiled(&source, &backend, UpscaleOptions::default(), None)?;
//! out.save("map-4x.png")?;
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod backend;
pub mod pipeline;
pub mod scale;
pub mod tiling;
pub mod world;

pub use backend::Upscaler;
pub use pipeline::{accumulator_bytes, upscale_tiled, upscale_to_target, UpscaleOptions};
pub use scale::ScaleStrategy;
pub use tiling::TilePlan;
pub use world::WorldFile;
