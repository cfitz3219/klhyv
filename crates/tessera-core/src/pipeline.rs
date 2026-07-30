//! Tiled upscaling: cut, upscale each piece, cross-fade back together.
//!
//! Output is produced one row of tiles at a time and handed straight to a
//! [`RowSink`]. Only the rows still being blended are held, so memory depends on
//! the tile size rather than the size of the result.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use anyhow::{ensure, Result};
use image::imageops::FilterType;
use image::RgbaImage;
use rayon::prelude::*;

use crate::backend::Upscaler;
use crate::scale::ScaleStrategy;
use crate::sink::{MemorySink, RowSink};
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

/// Peak scratch memory for a run, in bytes.
///
/// Only the tile rows being blended are resident, so this depends on the tile
/// size and output width, not on output height.
pub fn accumulator_bytes(out_width: u32, tile: u32, scale: u32) -> u64 {
    // Four f32 colour channels plus one f32 of weight, per pixel, over the rows
    // of one tile band plus the band carried into the next.
    let band_rows = (tile as u64 * scale as u64) * 2;
    out_width as u64 * band_rows * 20
}

/// Accumulates premultiplied colour and coverage for a horizontal band.
///
/// Premultiplying matters where tiles meet across a partially transparent
/// region: blending straight alpha there drags colour towards whatever the
/// transparent pixels happen to hold.
struct Accumulator {
    width: u32,
    /// Absolute output row held at local row 0.
    y_offset: u32,
    height: u32,
    /// Premultiplied RGB and alpha, interleaved as 4 floats per pixel.
    color: Vec<f32>,
    weight: Vec<f32>,
}

impl Accumulator {
    fn new(width: u32, y_offset: u32, height: u32) -> Self {
        let px = width as usize * height as usize;
        Self {
            width,
            y_offset,
            height,
            color: vec![0.0; px * 4],
            weight: vec![0.0; px],
        }
    }

    /// Adopt partial sums carried over from the previous band.
    fn seed(&mut self, color: &[f32], weight: &[f32]) {
        self.color[..color.len()].copy_from_slice(color);
        self.weight[..weight.len()].copy_from_slice(weight);
    }

    fn add_tile(&mut self, tile: &Tile, patch: &RgbaImage) {
        for v in 0..tile.dst.h {
            let abs_y = tile.dst.y + v;
            debug_assert!(abs_y >= self.y_offset && abs_y < self.y_offset + self.height);
            let row = (abs_y - self.y_offset) as usize * self.width as usize;
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

    /// Resolve absolute rows `from..to` to RGBA8 bytes.
    fn resolve_rows(&self, from: u32, to: u32) -> Vec<u8> {
        let start = (from - self.y_offset) as usize * self.width as usize;
        let end = (to - self.y_offset) as usize * self.width as usize;
        let mut out = Vec::with_capacity((end - start) * 4);
        for idx in start..end {
            let total = self.weight[idx];
            let c = &self.color[idx * 4..idx * 4 + 4];
            // `axis_weight` never returns zero, so every covered pixel has
            // weight; this guard only covers a pixel no tile reached at all.
            if total <= 0.0 {
                out.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let alpha = c[3] / total;
            if alpha <= 0.0 {
                out.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            // Undo the premultiply: colour was accumulated scaled by alpha.
            let unmul = |v: f32| (v / total / alpha).round().clamp(0.0, 255.0) as u8;
            out.extend_from_slice(&[
                unmul(c[0]),
                unmul(c[1]),
                unmul(c[2]),
                (alpha * 255.0).round().clamp(0.0, 255.0) as u8,
            ]);
        }
        out
    }

    /// Partial sums for absolute rows `from..` , to seed the next band.
    fn tail(&self, from: u32) -> (Vec<f32>, Vec<f32>) {
        let start = (from - self.y_offset) as usize * self.width as usize;
        (self.color[start * 4..].to_vec(), self.weight[start..].to_vec())
    }
}

/// Run one pass of the model over `source`, streaming rows into `sink`.
fn run_pass(
    source: &RgbaImage,
    backend: &dyn Upscaler,
    opts: UpscaleOptions,
    sink: &mut dyn RowSink,
    on_tile: Option<&(dyn Fn() + Sync)>,
) -> Result<()> {
    let (width, height) = source.dimensions();
    ensure!(width > 0 && height > 0, "source image is empty");

    let scale = backend.scale_factor();
    let plan = TilePlan::new(width, height, scale, opts.tile, opts.overlap);
    ensure!(!plan.tiles.is_empty(), "tile plan produced no work");
    let (out_w, out_h) = (plan.out_width(), plan.out_height());

    // Tiles in a row share a dst.y and height by construction.
    let mut bands: BTreeMap<u32, Vec<&Tile>> = BTreeMap::new();
    for tile in &plan.tiles {
        bands.entry(tile.dst.y).or_default().push(tile);
    }
    let bands: Vec<(u32, Vec<&Tile>)> = bands.into_iter().collect();

    let mut carry: Option<(Vec<f32>, Vec<f32>)> = None;
    let mut flushed = 0u32;

    for (i, (band_y, tiles)) in bands.iter().enumerate() {
        let band_end = tiles[0].dst.bottom();
        let acc = Mutex::new(Accumulator::new(out_w, *band_y, band_end - band_y));
        if let Some((color, weight)) = carry.take() {
            acc.lock().unwrap().seed(&color, &weight);
        }

        // Inference dominates, so tiles upscale in parallel and the lock is held
        // only for the comparatively cheap blend.
        tiles.par_iter().try_for_each(|tile| -> Result<()> {
            let patch =
                image::imageops::crop_imm(source, tile.src.x, tile.src.y, tile.src.w, tile.src.h)
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
            if let Some(cb) = on_tile {
                cb();
            }
            Ok(())
        })?;

        let acc = acc.into_inner().unwrap();
        // Rows above the next band's start can no longer receive contributions.
        let next_start = bands.get(i + 1).map(|(y, _)| *y).unwrap_or(out_h);
        let flush_to = next_start.min(band_end);
        sink.write_rows(&acc.resolve_rows(flushed, flush_to))?;
        flushed = flush_to;

        if flush_to < band_end {
            let (color, weight) = acc.tail(flush_to);
            carry = Some((color, weight));
        }
    }

    ensure!(
        flushed == out_h,
        "wrote {flushed} of {out_h} output rows; tile bands did not cover the image"
    );
    sink.finish()
}

/// Upscale `source` by the backend's native factor, returning the result.
///
/// `progress` is called with (completed, total) after each tile.
pub fn upscale_tiled(
    source: &RgbaImage,
    backend: &dyn Upscaler,
    opts: UpscaleOptions,
    progress: Option<&(dyn Fn(usize, usize) + Sync)>,
) -> Result<RgbaImage> {
    let (w, h) = source.dimensions();
    ensure!(w > 0 && h > 0, "source image is empty");
    let scale = backend.scale_factor();
    let total = TilePlan::new(w, h, scale, opts.tile, opts.overlap)
        .tiles
        .len();

    let done = AtomicUsize::new(0);
    let relay = || {
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(cb) = progress {
            cb(n, total);
        }
    };

    let mut sink = MemorySink::new(w * scale, h * scale);
    run_pass(
        source,
        backend,
        opts,
        &mut sink,
        progress.is_some().then_some(&relay),
    )?;
    sink.into_image()
}

/// Total tiles across every pass needed to reach `target`.
fn total_tiles(dims: (u32, u32), native: u32, passes: u32, opts: UpscaleOptions) -> usize {
    let mut dims = dims;
    let mut total = 0;
    for _ in 0..passes {
        total += TilePlan::new(dims.0, dims.1, native, opts.tile, opts.overlap)
            .tiles
            .len();
        dims = (dims.0 * native, dims.1 * native);
    }
    total
}

/// Upscale `source` to exactly `target` times its size, into `sink`.
///
/// The final pass streams straight into `sink` when `target` is a whole number
/// of model passes. Otherwise the last pass must be held in memory so it can be
/// resampled down to the exact size, which is the one case where output size is
/// still bounded by available memory.
pub fn upscale_to_sink(
    source: &RgbaImage,
    backend: &dyn Upscaler,
    target: u32,
    opts: UpscaleOptions,
    sink: &mut dyn RowSink,
    progress: Option<&(dyn Fn(usize, usize) + Sync)>,
) -> Result<()> {
    let (src_w, src_h) = source.dimensions();
    ensure!(src_w > 0 && src_h > 0, "source image is empty");

    let native = backend.scale_factor();
    let strategy = ScaleStrategy::plan(target, native)?;
    let total = total_tiles((src_w, src_h), native, strategy.passes, opts);

    let done = AtomicUsize::new(0);
    let relay = || {
        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(cb) = progress {
            cb(n, total);
        }
    };
    let relay_ref = progress.is_some().then_some(&relay as &(dyn Fn() + Sync));

    // Every pass but the last produces input for the next, so it must be
    // materialised regardless.
    let mut current = source.clone();
    for _ in 0..strategy.passes.saturating_sub(1) {
        let (w, h) = current.dimensions();
        let mut mem = MemorySink::new(w * native, h * native);
        run_pass(&current, backend, opts, &mut mem, relay_ref)?;
        current = mem.into_image()?;
    }

    if strategy.needs_resample() {
        let (w, h) = current.dimensions();
        let mut mem = MemorySink::new(w * native, h * native);
        run_pass(&current, backend, opts, &mut mem, relay_ref)?;
        let resized = image::imageops::resize(
            &mem.into_image()?,
            src_w * target,
            src_h * target,
            FilterType::Lanczos3,
        );
        sink.write_rows(resized.as_raw())?;
        sink.finish()?;
    } else {
        run_pass(&current, backend, opts, sink, relay_ref)?;
    }

    Ok(())
}

/// Upscale `source` to exactly `target` times its size, returning the result.
pub fn upscale_to_target(
    source: &RgbaImage,
    backend: &dyn Upscaler,
    target: u32,
    opts: UpscaleOptions,
    progress: Option<&(dyn Fn(usize, usize) + Sync)>,
) -> Result<RgbaImage> {
    let (w, h) = source.dimensions();
    ensure!(w > 0 && h > 0, "source image is empty");
    let mut sink = MemorySink::new(w * target, h * target);
    upscale_to_sink(source, backend, target, opts, &mut sink, progress)?;
    sink.into_image()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ResampleBackend;
    use crate::sink::PngSink;
    use image::Rgba;

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

    /// Banding must not change a single pixel. If the carry-over between tile
    /// rows were wrong, horizontal seams would appear here and nowhere else.
    #[test]
    fn banded_output_is_identical_regardless_of_band_count() {
        let src = test_pattern(200, 300);
        let backend = ResampleBackend::new(2, "lanczos3").unwrap();

        // One tall tile: a single band, nothing carried.
        let single = upscale_tiled(
            &src,
            &backend,
            UpscaleOptions { tile: 4096, overlap: 0 },
            None,
        )
        .unwrap();

        // Small tiles force many bands and many carries.
        for tile in [32, 48, 64, 100] {
            let many = upscale_tiled(
                &src,
                &backend,
                UpscaleOptions { tile, overlap: 16 },
                None,
            )
            .unwrap();
            assert_eq!(many.dimensions(), single.dimensions());
            assert!(
                max_channel_diff(&many, &single) <= 6,
                "tile={tile}: banding changed the result"
            );
        }
    }

    /// Smooth, wrap-free content. `test_pattern` uses modulo arithmetic whose
    /// wraparound produces genuine 255-level row jumps, which would swamp any
    /// seam this test is looking for.
    fn smooth_pattern(w: u32, h: u32) -> RgbaImage {
        RgbaImage::from_fn(w, h, |x, y| {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let ch = |v: f32| (128.0 + 100.0 * v).clamp(0.0, 255.0) as u8;
            Rgba([
                ch((fx * 6.0).sin() * (fy * 5.0).cos()),
                ch(((fx + fy) * 7.0).sin()),
                ch((fy * 9.0).cos()),
                255,
            ])
        })
    }

    /// A horizontal seam is a row-to-row jump at a band boundary. Rather than
    /// judge that against an absolute threshold, which depends on the content,
    /// compare against the same image produced in a single band.
    #[test]
    fn banding_adds_no_horizontal_seam() {
        let src = smooth_pattern(64, 200);
        let backend = ResampleBackend::new(2, "lanczos3").unwrap();

        let worst_row_step = |opts| {
            let out = upscale_tiled(&src, &backend, opts, None).unwrap();
            let (w, h) = out.dimensions();
            (1..h)
                .map(|y| {
                    (0..w)
                        .map(|x| {
                            let a = out.get_pixel(x, y).0;
                            let b = out.get_pixel(x, y - 1).0;
                            (0..3).map(|i| a[i].abs_diff(b[i]) as f32).sum::<f32>()
                        })
                        .sum::<f32>()
                        / w as f32
                })
                .fold(0.0_f32, f32::max)
        };

        let un_banded = worst_row_step(UpscaleOptions { tile: 4096, overlap: 0 });
        let banded = worst_row_step(UpscaleOptions { tile: 40, overlap: 12 });
        assert!(
            banded <= un_banded * 1.25 + 1.0,
            "worst row step {banded:.2} banded vs {un_banded:.2} un-banded: \
             banding introduced a seam"
        );
    }

    /// Streaming to disk must match building the image in memory exactly.
    #[test]
    fn streamed_png_matches_the_in_memory_result() {
        let src = test_pattern(150, 190);
        let backend = ResampleBackend::new(2, "lanczos3").unwrap();
        let opts = UpscaleOptions { tile: 48, overlap: 16 };

        let in_memory = upscale_to_target(&src, &backend, 2, opts, None).unwrap();

        let dir = std::env::temp_dir().join("tessera-stream-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("streamed.png");
        let mut sink = PngSink::create(&path, 300, 380).unwrap();
        upscale_to_sink(&src, &backend, 2, opts, &mut sink, None).unwrap();

        let streamed = image::open(&path).unwrap().to_rgba8();
        assert_eq!(streamed.dimensions(), in_memory.dimensions());
        assert_eq!(
            max_channel_diff(&streamed, &in_memory),
            0,
            "streaming changed pixels"
        );
        let _ = std::fs::remove_file(&path);
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

    /// The slider's promise: whatever stop the user picks, the output is
    /// exactly that many times the input, including factors the model has no
    /// native pass for.
    #[test]
    fn every_target_scale_lands_exactly() {
        let src = test_pattern(40, 28);
        // A 2x model must still deliver 3x, 5x, 7x and so on.
        let backend = ResampleBackend::new(2, "lanczos3").unwrap();
        for target in 1..=10u32 {
            let out = upscale_to_target(
                &src,
                &backend,
                target,
                UpscaleOptions { tile: 32, overlap: 8 },
                None,
            )
            .unwrap();
            assert_eq!(
                out.dimensions(),
                (40 * target, 28 * target),
                "target {target}x produced the wrong size"
            );
        }
    }

    /// Progress must climb once to its total across all passes, not restart.
    #[test]
    fn progress_is_continuous_across_passes() {
        let src = test_pattern(64, 64);
        let backend = ResampleBackend::new(2, "triangle").unwrap();
        let seen = Mutex::new(Vec::new());
        // 2x model, 7x target: three passes to 8x, then a resample down.
        upscale_to_target(
            &src,
            &backend,
            7,
            UpscaleOptions { tile: 32, overlap: 8 },
            Some(&|n, total| seen.lock().unwrap().push((n, total))),
        )
        .unwrap();

        let seen = seen.into_inner().unwrap();
        let total = seen[0].1;
        assert_eq!(seen.len(), total, "reported count must match the total");
        let mut counts: Vec<usize> = seen.iter().map(|&(n, _)| n).collect();
        counts.sort_unstable();
        assert_eq!(counts, (1..=total).collect::<Vec<_>>());
    }

    #[test]
    fn unit_target_returns_the_original_size() {
        let src = test_pattern(50, 30);
        let backend = ResampleBackend::new(4, "lanczos3").unwrap();
        let out =
            upscale_to_target(&src, &backend, 1, UpscaleOptions::default(), None).unwrap();
        assert_eq!(out.dimensions(), (50, 30));
    }

    /// Peak memory must not grow with output height, or nothing is gained.
    #[test]
    fn scratch_memory_is_independent_of_output_height() {
        let short = accumulator_bytes(80_000, 256, 4);
        let tall = accumulator_bytes(80_000, 256, 4);
        assert_eq!(short, tall);
        // A 20k-wide map at 4x should need a bounded band, not 128 GiB.
        assert!(
            accumulator_bytes(80_000, 256, 4) < 4 << 30,
            "band buffer should stay under 4 GiB"
        );
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
