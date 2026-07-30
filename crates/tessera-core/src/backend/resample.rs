//! Classical resampling backend.
//!
//! This is the reference implementation. It adds no detail, so it is not the
//! point of the project, but it exercises the whole pipeline on any machine and
//! gives the tiling tests a deterministic backend to check seams against. It is
//! also a sensible fallback when no GPU is present.

use anyhow::{ensure, Result};
use image::imageops::FilterType;
use image::RgbaImage;

use super::Upscaler;

pub struct ResampleBackend {
    scale: u32,
    filter: FilterType,
    name: String,
}

impl ResampleBackend {
    /// `filter` is matched by name: `lanczos3`, `catmullrom`, `gaussian`,
    /// `triangle`, or `nearest`.
    pub fn new(scale: u32, filter: &str) -> Result<Self> {
        ensure!(scale >= 1, "scale must be >= 1, got {scale}");
        let filter_ty = match filter.to_ascii_lowercase().as_str() {
            "lanczos3" => FilterType::Lanczos3,
            "catmullrom" => FilterType::CatmullRom,
            "gaussian" => FilterType::Gaussian,
            "triangle" => FilterType::Triangle,
            "nearest" => FilterType::Nearest,
            other => anyhow::bail!(
                "unknown filter {other:?}; expected one of \
                 lanczos3, catmullrom, gaussian, triangle, nearest"
            ),
        };
        Ok(Self {
            scale,
            filter: filter_ty,
            name: format!("resample:{}", filter.to_ascii_lowercase()),
        })
    }
}

impl Upscaler for ResampleBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn scale_factor(&self) -> u32 {
        self.scale
    }

    fn upscale(&self, tile: &RgbaImage) -> Result<RgbaImage> {
        let (w, h) = tile.dimensions();
        ensure!(w > 0 && h > 0, "cannot upscale a zero-sized tile");
        Ok(image::imageops::resize(
            tile,
            w * self.scale,
            h * self.scale,
            self.filter,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_matches_the_declared_scale_factor() {
        let backend = ResampleBackend::new(4, "lanczos3").unwrap();
        let out = backend.upscale(&RgbaImage::new(16, 9)).unwrap();
        assert_eq!(out.dimensions(), (64, 36));
    }

    #[test]
    fn rejects_unknown_filters() {
        assert!(ResampleBackend::new(2, "bicubish").is_err());
    }
}
