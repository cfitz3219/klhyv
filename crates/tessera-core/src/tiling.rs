//! Tile planning and blend-weight computation.
//!
//! Large scans cannot be pushed through a neural upscaler in one piece, so the
//! source is cut into overlapping tiles, each upscaled independently, then
//! recombined. Independent upscales disagree slightly near tile edges, which
//! shows up as a visible grid unless the shared region is cross-faded.
//!
//! Two details keep the grid from showing. Tile origins are distributed evenly
//! rather than stepped at a fixed stride with a short final step, so no pair of
//! tiles overlaps far more than the rest. And each cross-fade is sized from the
//! *actual* overlap with that specific neighbour, not the requested overlap, so
//! the ramps always span exactly the shared region.

/// A rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub fn right(&self) -> u32 {
        self.x + self.w
    }

    pub fn bottom(&self) -> u32 {
        self.y + self.h
    }
}

/// Per-side cross-fade widths, in output pixels.
///
/// A side is feathered only where a neighbouring tile actually covers it. The
/// outer border of the image stays at full weight, otherwise the image would
/// fade out at its own edges.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Feather {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

/// One unit of work: read `src`, upscale it, blend into `dst`.
#[derive(Debug, Clone, Copy)]
pub struct Tile {
    pub src: Rect,
    pub dst: Rect,
    pub feather: Feather,
}

impl Tile {
    /// Blend weight for a pixel at (`u`, `v`) relative to this tile's `dst`.
    ///
    /// Sampling at pixel centres (the `+ 0.5`) is what makes a pair of opposing
    /// ramps sum to 1 rather than to `1 ± 1/(2 * feather)`.
    pub fn weight_at(&self, u: u32, v: u32) -> f32 {
        axis_weight(u, self.dst.w, self.feather.left, self.feather.right)
            * axis_weight(v, self.dst.h, self.feather.top, self.feather.bottom)
    }
}

fn axis_weight(pos: u32, extent: u32, lead: u32, trail: u32) -> f32 {
    let mut w = 1.0_f32;
    if lead > 0 && pos < lead {
        w *= (pos as f32 + 0.5) / lead as f32;
    }
    if trail > 0 && pos >= extent.saturating_sub(trail) {
        let from_end = extent - pos - 1;
        w *= (from_end as f32 + 0.5) / trail as f32;
    }
    // Never return exactly zero: every output pixel must have some contributor,
    // or normalisation divides by zero.
    w.clamp(f32::MIN_POSITIVE, 1.0)
}

/// One tile position along a single axis.
struct Span {
    origin: u32,
    extent: u32,
    lead_overlap: u32,
    trail_overlap: u32,
}

/// How the source image is cut up, and where each piece lands in the output.
#[derive(Debug, Clone)]
pub struct TilePlan {
    pub src_width: u32,
    pub src_height: u32,
    pub scale: u32,
    pub tiles: Vec<Tile>,
}

impl TilePlan {
    pub fn out_width(&self) -> u32 {
        self.src_width * self.scale
    }

    pub fn out_height(&self) -> u32 {
        self.src_height * self.scale
    }

    /// Plan a tiling of a `width` x `height` source.
    ///
    /// `tile` and `overlap` are in source pixels. `overlap` is clamped below
    /// `tile` so the grid always advances; a zero-size image yields no tiles.
    pub fn new(width: u32, height: u32, scale: u32, tile: u32, overlap: u32) -> Self {
        assert!(scale >= 1, "scale must be >= 1");
        assert!(tile >= 1, "tile must be >= 1");

        let overlap = overlap.min(tile.saturating_sub(1));
        let cols = axis_spans(width, tile, overlap);
        let rows = axis_spans(height, tile, overlap);

        let mut tiles = Vec::with_capacity(cols.len() * rows.len());
        for row in &rows {
            for col in &cols {
                tiles.push(Tile {
                    src: Rect {
                        x: col.origin,
                        y: row.origin,
                        w: col.extent,
                        h: row.extent,
                    },
                    dst: Rect {
                        x: col.origin * scale,
                        y: row.origin * scale,
                        w: col.extent * scale,
                        h: row.extent * scale,
                    },
                    feather: Feather {
                        left: col.lead_overlap * scale,
                        right: col.trail_overlap * scale,
                        top: row.lead_overlap * scale,
                        bottom: row.trail_overlap * scale,
                    },
                });
            }
        }

        TilePlan {
            src_width: width,
            src_height: height,
            scale,
            tiles,
        }
    }
}

/// Tile positions along one axis, with each neighbour overlap measured.
fn axis_spans(extent: u32, tile: u32, overlap: u32) -> Vec<Span> {
    if extent == 0 {
        return Vec::new();
    }
    if extent <= tile {
        return vec![Span {
            origin: 0,
            extent,
            lead_overlap: 0,
            trail_overlap: 0,
        }];
    }

    // Distance the tile window must travel, and how many positions it needs.
    let step = tile - overlap;
    let span = extent - tile;
    let count = span.div_ceil(step) + 1;

    // Spread positions evenly across `span`. Compared with a fixed stride plus a
    // clamped final tile, this avoids one huge overlap at the right/bottom edge.
    let origins: Vec<u32> = (0..count)
        .map(|i| (span as u64 * i as u64 / (count as u64 - 1)) as u32)
        .collect();

    origins
        .iter()
        .enumerate()
        .map(|(i, &origin)| {
            // Every tile is full width here: origin <= extent - tile by construction.
            let lead_overlap = if i > 0 {
                (origins[i - 1] + tile).saturating_sub(origin)
            } else {
                0
            };
            let trail_overlap = if i + 1 < origins.len() {
                (origin + tile).saturating_sub(origins[i + 1])
            } else {
                0
            };
            Span {
                origin,
                extent: tile,
                lead_overlap,
                trail_overlap,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CASES: [(u32, u32, u32, u32); 5] = [
        (500, 300, 128, 16),
        (77, 91, 32, 8),
        (1024, 1024, 256, 64),
        (1000, 40, 64, 32),
        (313, 257, 100, 37),
    ];

    #[test]
    fn single_tile_when_image_fits() {
        let plan = TilePlan::new(100, 100, 2, 256, 32);
        assert_eq!(plan.tiles.len(), 1);
        assert_eq!(plan.tiles[0].feather, Feather::default());
        assert_eq!(plan.tiles[0].dst, Rect { x: 0, y: 0, w: 200, h: 200 });
    }

    #[test]
    fn tiles_cover_every_source_pixel() {
        for (w, h, tile, overlap) in CASES {
            let plan = TilePlan::new(w, h, 1, tile, overlap);
            let mut covered = vec![false; (w * h) as usize];
            for t in &plan.tiles {
                for y in t.src.y..t.src.bottom() {
                    for x in t.src.x..t.src.right() {
                        covered[(y * w + x) as usize] = true;
                    }
                }
            }
            assert!(covered.iter().all(|&c| c), "gap in coverage for {w}x{h}");
        }
    }

    #[test]
    fn tiles_stay_inside_the_source() {
        for (w, h, tile, overlap) in CASES {
            let plan = TilePlan::new(w, h, 3, tile, overlap);
            for t in &plan.tiles {
                assert!(t.src.right() <= w && t.src.bottom() <= h);
                assert!(t.dst.right() <= plan.out_width() && t.dst.bottom() <= plan.out_height());
            }
        }
    }

    /// Normalisation divides by the accumulated weight, so a zero total anywhere
    /// would punch a hole in the output.
    #[test]
    fn every_output_pixel_has_weight() {
        for (w, h, tile, overlap) in CASES {
            let plan = TilePlan::new(w, h, 2, tile, overlap);
            let mut acc = vec![0.0_f32; (plan.out_width() * plan.out_height()) as usize];
            for t in &plan.tiles {
                for v in 0..t.dst.h {
                    for u in 0..t.dst.w {
                        let idx = ((t.dst.y + v) * plan.out_width() + t.dst.x + u) as usize;
                        acc[idx] += t.weight_at(u, v);
                    }
                }
            }
            assert!(
                acc.iter().all(|&x| x > 0.0),
                "zero total weight somewhere in {w}x{h}"
            );
        }
    }

    #[test]
    fn feathers_agree_between_neighbours() {
        let spans = axis_spans(1000, 128, 24);
        for pair in spans.windows(2) {
            assert_eq!(
                pair[0].trail_overlap, pair[1].lead_overlap,
                "cross-fade must be symmetric or the pair will not sum to 1"
            );
        }
    }

    #[test]
    fn actual_overlap_is_never_less_than_requested() {
        let spans = axis_spans(1000, 128, 24);
        for pair in spans.windows(2) {
            assert!(pair[0].trail_overlap >= 24, "overlap shrank below the request");
        }
    }

    #[test]
    fn overlap_is_clamped_below_tile_size() {
        // Would divide by a zero step if the clamp were missing.
        let plan = TilePlan::new(300, 40, 1, 64, 999);
        assert!(plan.tiles.len() > 1);
    }

    #[test]
    fn zero_sized_image_plans_nothing() {
        assert!(TilePlan::new(0, 0, 2, 64, 8).tiles.is_empty());
    }
}
