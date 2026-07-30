//! Tiled upscaling: cut, upscale each piece, cross-fade back together.

use std::sync::Mutex;

use anyhow::{ensure, Result};
use image::{Rgba, RgbaImage};
use rayon::prelude::*;

use crate::backend::Upscaler;
use crate::tiling::{Tile, TilePlan};

/// Knobs for a single run.
#[derive(Debug, Clone, Copy)]
pub struct UpscaleOptions {
    /// Tile edge length in source pixels.
    pub tile: u32,
    /// Requested overlap between neighbouring tiles, in source pixels.
    pub overlap: u32,
}

impl Default for UpscaleOptions {
    fn default() -> Self {
        // 256px tiles fit comfortably in modest VRAM; 32px of overlap is enough
        // for a cross-fade to hide the edge effects of typical models.
        Self {
            tile: 256,
            overlap: 32,
        }
    }
}

/// Bytes of scratch the blend accumulator needs for a given output size.
///
/// Four f32 colour channels plus one f32 of weight, per output pixel.
pub fn accumulator_bytes(out_width: u32, out_height: u32) -> u64 {
    out_width as u64 * out_height as u64 * 20
}

/// Accumulates premultiplied colour and coverage, then resolves to 8-bit RGBA.
///
/// Premultiplying matters where tiles meet across a partially transparent
/// region: blending straight alpha there drags colour towards whatever the
/// transparent pixels happen to hold.
struct Accumulator {
    width: u32,
    height: u32,
    /// Premultiplied RGB and alpha, interleaved as 4 floats per pixel.
    color: Vec<f32>,
    weight: Vec<f32>,
}

impl Accumulator {
    fn new(width: u32, height: u32) -> Self {
        let px = width as usize * height as usize;
        Self {
            width,
            height,
            color: vec![0.0; px * 4],
            weight: vec![0.0; px],
        }
    }

    fn add_tile(&mut self, tile: &Tile, patch: &RgbaImage) {
        for v in 0..tile.dst.h {
            let row = (tile.dst.y + v) as usize * self.width as usize;
            for u in 0..tile.dst.w {
                let w = tile.weight_at(u, v);
                let px = patch.get_pixel(u, v).0;
                let a = px[3] as f32 / 255.0;
                let idx = row + (tile.dst.x + u) as usize;

                let c = &mut self.color[idx * 4..idx * 4 + 4];
                c[0] += px[0] as f32 * a * w;
                c[1] += px[1] as f32 * a * w;
                c[2] += px[2] as f32 * a * w;
                c[3] += a * w;
                self.weight[idx] += w;
            }
        }
    }

    fn resolve(self) -> RgbaImage {
        let mut out = RgbaImage::new(self.width, self.height);
        for (idx, px) in out.pixels_mut().enumerate() {
            let total = self.weight[idx];
            let c = &self.color[idx * 4..idx * 4 + 4];
            // `axis_weight` never returns zero, so every covered pixel has
            // weight; this guard only covers a pixel no tile reached at all.
            if total <= 0.0 {
                *px = Rgba([0, 0, 0, 0]);
                continue;
            }
            let alpha = c[3] / total;
            if alpha <= 0.0 {
                *px = Rgba([0, 0, 0, 0]);
                continue;
            }
            // Undo the premultiply: colour was accumulated scaled by alpha.
            let unmul = |v: f32| (v / total / alpha).round().clamp(0.0, 255.0) as u8;
            *px = Rgba([
                unmul(c[0]),
                unmul(c[1]),
                unmul(c[2]),
                (alpha * 255.0).round().clamp(0.0, 255.0) as u8,
            ]);
        }
        out
    }
}

/// Upscale `source` with `backend`, tiling as needed.
///
/// `progress` is called with (completed, total) after each tile.
pub fn upscale_tiled(
    source: &RgbaImage,
    backend: &dyn Upscaler,
    opts: UpscaleOptions,
    progress: Option<&(dyn Fn(usize, usize) + Sync)>,
) -> Result<RgbaImage> {
    let (width, height) = source.dimensions();
    ensure!(width > 0 && height > 0, "source image is empty");

    let scale = backend.scale_factor();
    let plan = TilePlan::new(width, height, scale, opts.tile, opts.overlap);
    ensure!(!plan.tiles.is_empty(), "tile plan produced no work");

    let total = plan.tiles.len();
    let acc = Mutex::new(Accumulator::new(plan.out_width(), plan.out_height()));
    let done = Mutex::new(0usize);

    // Inference dominates, so upscaling runs in parallel and the lock is held
    // only for the comparatively cheap blend.
    plan.tiles.par_iter().try_for_each(|tile| -> Result<()> {
        let patch = image::imageops::crop_imm(source, tile.src.x, tile.src.y, tile.src.w, tile.src.h)
            .to_image();
        let scaled = backend.upscale(&patch)?;

        ensure!(
            scaled.dimensions() == (tile.dst.w, tile.dst.h),
            "backend {} returned {:?} for a {}x{} tile, expected {}x{}",
            backend.name(),
            scaled.dimensions(),
            tile.src.w,
            tile.src.h,
            tile.dst.w,
            tile.dst.h
        );

        acc.lock().unwrap().add_tile(tile, &scaled);

        if let Some(cb) = progress {
            let mut n = done.lock().unwrap();
            *n += 1;
            cb(*n, total);
        }
        Ok(())
    })?;

    Ok(acc.into_inner().unwrap().resolve())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ResampleBackend;

    /// Deterministic, detailed test pattern. Flat colour would hide seams.
    fn test_pattern(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| {
            Rgba([
                ((x * 7 + y * 3) % 256) as u8,
                ((x * x + y) % 256) as u8,
                ((x ^ (y * 5)) % 256) as u8,
                255,
            ])
        })
    }

    fn max_channel_diff(a: &RgbaImage, b: &RgbaImage) -> u8 {
        assert_eq!(a.dimensions(), b.dimensions());
        a.pixels()
            .zip(b.pixels())
            .flat_map(|(p, q)| (0..4).map(move |i| p.0[i].abs_diff(q.0[i])))
            .max()
            .unwrap_or(0)
    }

    /// The headline property. A single-tile run is the un-tiled reference; a
    /// heavily tiled run of the same image must reproduce it. Any seam, weight
    /// bug, or off-by-one in the blend shows up here as a large difference.
    #[test]
    fn tiling_reproduces_the_untiled_result() {
        let src = test_pattern(240, 176);
        let backend = ResampleBackend::new(2, "lanczos3").unwrap();

        let reference = upscale_tiled(
            &src,
            &backend,
            UpscaleOptions { tile: 4096, overlap: 0 },
            None,
        )
        .unwrap();

        for (tile, overlap) in [(64, 32), (96, 24), (48, 16), (128, 48)] {
            let tiled =
                upscale_tiled(&src, &backend, UpscaleOptions { tile, overlap }, None).unwrap();
            assert_eq!(tiled.dimensions(), reference.dimensions());
            // Measured worst case is 1-4/255 across these configurations; 6
            // leaves headroom for filter changes without letting a real seam by.
            let diff = max_channel_diff(&tiled, &reference);
            assert!(
                diff <= 6,
                "tile={tile} overlap={overlap}: max channel difference {diff} \
                 against the untiled reference suggests a visible seam"
            );
        }
    }

    /// A seam shows up as a spike in column-to-column difference at tile
    /// boundaries. Compare the worst boundary against the typical column.
    #[test]
    fn no_discontinuity_at_tile_boundaries() {
        let src = test_pattern(200, 64);
        let backend = ResampleBackend::new(2, "lanczos3").unwrap();
        let out = upscale_tiled(
            &src,
            &backend,
            UpscaleOptions { tile: 50, overlap: 12 },
            None,
        )
        .unwrap();

        let (w, h) = out.dimensions();
        let column_delta: Vec<f32> = (1..w)
            .map(|x| {
                (0..h)
                    .map(|y| {
                        let a = out.get_pixel(x, y).0;
                        let b = out.get_pixel(x - 1, y).0;
                        (0..3).map(|i| a[i].abs_diff(b[i]) as f32).sum::<f32>()
                    })
                    .sum::<f32>()
                    / h as f32
            })
            .collect();

        let mean = column_delta.iter().sum::<f32>() / column_delta.len() as f32;
        let worst = column_delta.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            worst < mean * 4.0,
            "worst column step {worst:.2} vs mean {mean:.2}: looks like a seam"
        );
    }

    #[test]
    fn output_dimensions_follow_the_scale_factor() {
        let src = test_pattern(100, 70);
        let backend = ResampleBackend::new(3, "catmullrom").unwrap();
        let out = upscale_tiled(&src, &backend, UpscaleOptions::default(), None).unwrap();
        assert_eq!(out.dimensions(), (300, 210));
    }

    #[test]
    fn transparency_survives_the_blend() {
        let src = RgbaImage::from_fn(96, 96, |x, _| {
            if x < 48 {
                Rgba([200, 40, 40, 255])
            } else {
                Rgba([0, 0, 0, 0])
            }
        });
        let backend = ResampleBackend::new(2, "nearest").unwrap();
        let out = upscale_tiled(&src, &backend, UpscaleOptions { tile: 32, overlap: 8 }, None)
            .unwrap();

        assert_eq!(out.get_pixel(10, 10).0[3], 255, "opaque side went transparent");
        assert_eq!(out.get_pixel(180, 10).0[3], 0, "transparent side gained alpha");
        // Premultiplied blending must not drag the opaque colour toward black.
        let px = out.get_pixel(10, 10).0;
        assert!(px[0] > 190 && px[1] < 60, "colour shifted: {px:?}");
    }

    #[test]
    fn progress_reports_every_tile() {
        let src = test_pattern(200, 200);
        let backend = ResampleBackend::new(2, "triangle").unwrap();
        let seen = Mutex::new(Vec::new());
        let out = upscale_tiled(
            &src,
            &backend,
            UpscaleOptions { tile: 64, overlap: 16 },
            Some(&|n, total| seen.lock().unwrap().push((n, total))),
        );
        assert!(out.is_ok());

        let seen = seen.into_inner().unwrap();
        let total = seen[0].1;
        assert_eq!(seen.len(), total);
        let mut counts: Vec<usize> = seen.iter().map(|&(n, _)| n).collect();
        counts.sort_unstable();
        assert_eq!(counts, (1..=total).collect::<Vec<_>>());
    }

    #[test]
    fn empty_source_is_an_error() {
        let backend = ResampleBackend::new(2, "nearest").unwrap();
        assert!(
            upscale_tiled(&RgbaImage::new(0, 0), &backend, UpscaleOptions::default(), None)
                .is_err()
        );
    }
}
