//! World files: the georeferencing sidecar that upscalers normally throw away.
//!
//! A world file pins an image to map coordinates with six affine coefficients.
//! Upscaling changes the pixel size, so carrying the sidecar across unchanged
//! would misplace the map by the full width of the image. Rewriting it is
//! cheap, and it is the difference between output a GIS can open and output it
//! cannot.
//!
//! Full GeoTIFF tags live inside the TIFF itself and are a separate job; this
//! covers the sidecar convention (`.tfw`, `.jgw`, `.pgw`, `.wld`) that most
//! scanned-map workflows use.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};

/// The six affine coefficients, in world-file line order.
///
/// The mapping is `x = a*col + b*row + c` and `y = d*col + e*row + f`, where
/// `col`/`row` index pixel *centres* from the upper-left pixel at (0, 0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldFile {
    /// Line 1: pixel width in x.
    pub a: f64,
    /// Line 2: row rotation (y-component of column step).
    pub d: f64,
    /// Line 3: column rotation (x-component of row step).
    pub b: f64,
    /// Line 4: pixel height in y, conventionally negative.
    pub e: f64,
    /// Line 5: x of the upper-left pixel centre.
    pub c: f64,
    /// Line 6: y of the upper-left pixel centre.
    pub f: f64,
}

impl WorldFile {
    pub fn parse(text: &str) -> Result<Self> {
        let nums: Vec<f64> = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(|l| l.parse::<f64>().with_context(|| format!("bad number {l:?}")))
            .collect::<Result<_>>()?;
        ensure!(
            nums.len() >= 6,
            "world file needs 6 coefficients, found {}",
            nums.len()
        );
        Ok(Self {
            a: nums[0],
            d: nums[1],
            b: nums[2],
            e: nums[3],
            c: nums[4],
            f: nums[5],
        })
    }

    pub fn to_text(self) -> String {
        [self.a, self.d, self.b, self.e, self.c, self.f]
            .iter()
            .map(|v| format!("{v:.12}"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    /// Adjust for an image magnified by `scale`.
    ///
    /// Pixels shrink by `scale`, and the upper-left pixel *centre* moves inward
    /// by the shrinkage, because the outer corner of the image is what stays
    /// fixed on the map.
    pub fn scaled(self, scale: u32) -> Self {
        let s = scale as f64;
        let (a, b, d, e) = (self.a / s, self.b / s, self.d / s, self.e / s);
        Self {
            a,
            b,
            d,
            e,
            c: self.c - (self.a + self.b) / 2.0 + (a + b) / 2.0,
            f: self.f - (self.d + self.e) / 2.0 + (d + e) / 2.0,
        }
    }
}

/// Sidecar paths to try for an image, most conventional first.
///
/// The usual rule takes the first and last letter of the image extension and
/// appends `w` (`tif` -> `tfw`, `png` -> `pgw`); `.wld` is accepted for anything.
pub fn sidecar_candidates(image: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(ext) = image.extension().and_then(|e| e.to_str()) {
        let ext = ext.to_ascii_lowercase();
        let chars: Vec<char> = ext.chars().collect();
        if chars.len() >= 2 {
            let short = format!("{}{}w", chars[0], chars[chars.len() - 1]);
            out.push(image.with_extension(&short));
            out.push(image.with_extension(short.to_ascii_uppercase()));
        }
        out.push(image.with_extension(format!("{ext}w")));
    }
    out.push(image.with_extension("wld"));
    out
}

/// Read the world file beside `image`, if one exists.
pub fn read_sidecar(image: &Path) -> Result<Option<(PathBuf, WorldFile)>> {
    for candidate in sidecar_candidates(image) {
        if candidate.is_file() {
            let text = fs::read_to_string(&candidate)
                .with_context(|| format!("reading {}", candidate.display()))?;
            let parsed = WorldFile::parse(&text)
                .with_context(|| format!("parsing {}", candidate.display()))?;
            return Ok(Some((candidate, parsed)));
        }
    }
    Ok(None)
}

/// Write `world` to the conventional sidecar path for `image`.
pub fn write_sidecar(image: &Path, world: WorldFile) -> Result<PathBuf> {
    let path = sidecar_candidates(image)
        .into_iter()
        .next()
        .context("image path has no usable extension for a sidecar")?;
    fs::write(&path, world.to_text()).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "2.0\n0.0\n0.0\n-2.0\n500000.0\n4649000.0\n";

    #[test]
    fn parses_line_order() {
        let w = WorldFile::parse(SAMPLE).unwrap();
        assert_eq!(w.a, 2.0);
        assert_eq!(w.e, -2.0);
        assert_eq!(w.c, 500000.0);
        assert_eq!(w.f, 4649000.0);
    }

    #[test]
    fn round_trips_through_text() {
        let w = WorldFile::parse(SAMPLE).unwrap();
        assert_eq!(WorldFile::parse(&w.to_text()).unwrap(), w);
    }

    #[test]
    fn rejects_short_files() {
        assert!(WorldFile::parse("1.0\n2.0\n").is_err());
    }

    #[test]
    fn scaling_shrinks_pixels() {
        let w = WorldFile::parse(SAMPLE).unwrap().scaled(4);
        assert_eq!(w.a, 0.5);
        assert_eq!(w.e, -0.5);
    }

    /// The invariant that matters: the outer corner of the image must not move,
    /// or the upscaled map lands offset from the original.
    #[test]
    fn scaling_keeps_the_image_corner_fixed() {
        let original = WorldFile::parse("2.0\n0.0\n0.0\n-2.0\n500000.0\n4649000.0\n").unwrap();
        let corner = |w: &WorldFile| {
            (
                w.c - (w.a + w.b) / 2.0,
                w.f - (w.d + w.e) / 2.0,
            )
        };
        for scale in [2, 3, 4, 8] {
            let (x0, y0) = corner(&original);
            let (x1, y1) = corner(&original.scaled(scale));
            assert!(
                (x0 - x1).abs() < 1e-9 && (y0 - y1).abs() < 1e-9,
                "corner moved at scale {scale}: ({x0}, {y0}) -> ({x1}, {y1})"
            );
        }
    }

    /// Rotated/skewed world files are rare but legal; scaling must not drop the
    /// rotation terms.
    #[test]
    fn scaling_preserves_rotation_terms() {
        let w = WorldFile::parse("2.0\n0.5\n0.25\n-2.0\n1000.0\n2000.0\n")
            .unwrap()
            .scaled(2);
        assert_eq!(w.d, 0.25);
        assert_eq!(w.b, 0.125);
    }

    #[test]
    fn sidecar_naming_follows_convention() {
        let names = |p: &str| {
            sidecar_candidates(Path::new(p))
                .iter()
                .filter_map(|c| c.extension().and_then(|e| e.to_str()).map(str::to_owned))
                .collect::<Vec<_>>()
        };
        assert_eq!(names("/m/scan.tif")[0], "tfw");
        assert_eq!(names("/m/scan.png")[0], "pgw");
        assert_eq!(names("/m/scan.jpg")[0], "jgw");
        assert!(names("/m/scan.tif").contains(&"wld".to_string()));
    }
}
