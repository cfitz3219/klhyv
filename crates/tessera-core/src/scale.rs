//! Reaching an arbitrary magnification with a fixed-factor model.
//!
//! Super-resolution models bake their factor into the weights: a ×4 model only
//! ever produces ×4. Offering a free choice of magnification therefore means
//! running the model enough times to *overshoot* the request, then resampling
//! down to the exact size.
//!
//! Overshooting and shrinking beats undershooting and stretching. Reaching ×7
//! by running a ×4 model twice to ×16 and shrinking keeps detail the model
//! synthesised; running it once to ×4 and stretching to ×7 just blurs.

use anyhow::{ensure, Result};

/// Guards against a pathological model reporting a tiny factor and provoking a
/// very long chain. Four passes of a ×2 model already reach ×16.
const MAX_PASSES: u32 = 6;

/// How to reach `target` using a model with a fixed `native` factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleStrategy {
    /// Requested final magnification.
    pub target: u32,
    /// The model's own factor.
    pub native: u32,
    /// How many times to run the model.
    pub passes: u32,
    /// Magnification after the last pass, before the final resample.
    pub intermediate: u32,
}

impl ScaleStrategy {
    /// Plan the passes needed to reach `target`.
    ///
    /// `target` of 1 still runs the model once and shrinks back. That is not a
    /// no-op: it is how these models remove compression artifacts and scanning
    /// noise without changing the image's size.
    pub fn plan(target: u32, native: u32) -> Result<Self> {
        ensure!(target >= 1, "target scale must be at least 1");
        ensure!(native >= 1, "model scale factor must be at least 1");

        // A ×1 model never grows, so one pass is all that can be asked of it;
        // any enlargement then falls to the resampler.
        if native == 1 {
            return Ok(Self {
                target,
                native,
                passes: 1,
                intermediate: 1,
            });
        }

        let mut passes = 1;
        let mut intermediate = native;
        while intermediate < target && passes < MAX_PASSES {
            intermediate *= native;
            passes += 1;
        }
        ensure!(
            intermediate >= target,
            "cannot reach {target}x with a {native}x model in {MAX_PASSES} passes"
        );

        Ok(Self {
            target,
            native,
            passes,
            intermediate,
        })
    }

    /// Whether a resampling step is needed after the final model pass.
    pub fn needs_resample(&self) -> bool {
        self.intermediate != self.target
    }

    /// Human-readable summary for the CLI and, later, the interface.
    pub fn describe(&self) -> String {
        match (self.passes, self.needs_resample()) {
            (1, false) => format!("1 model pass at {}x", self.native),
            (1, true) => format!(
                "1 model pass at {}x, then resample to {}x",
                self.native, self.target
            ),
            (n, false) => format!("{n} model passes at {}x", self.native),
            (n, true) => format!(
                "{n} model passes to {}x, then resample to {}x",
                self.intermediate, self.target
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_multiples_need_no_resample() {
        let s = ScaleStrategy::plan(4, 4).unwrap();
        assert_eq!((s.passes, s.intermediate), (1, 4));
        assert!(!s.needs_resample());

        let s = ScaleStrategy::plan(16, 4).unwrap();
        assert_eq!((s.passes, s.intermediate), (2, 16));
        assert!(!s.needs_resample());
    }

    /// The core promise of the slider: every stop from 1 to 10 is reachable,
    /// and always by overshooting rather than stretching.
    #[test]
    fn every_slider_stop_is_reachable_by_overshoot() {
        for native in [2, 3, 4] {
            for target in 1..=10 {
                let s = ScaleStrategy::plan(target, native).unwrap();
                assert!(
                    s.intermediate >= target,
                    "{native}x model undershot {target}x (reached {})",
                    s.intermediate
                );
                assert!(s.passes >= 1);
            }
        }
    }

    #[test]
    fn uses_the_fewest_passes_that_clear_the_target() {
        // 4^1 = 4 < 7 <= 16 = 4^2, so two passes and no more.
        let s = ScaleStrategy::plan(7, 4).unwrap();
        assert_eq!(s.passes, 2);
        assert_eq!(s.intermediate, 16);

        // 4 already clears 3, so one pass is enough.
        assert_eq!(ScaleStrategy::plan(3, 4).unwrap().passes, 1);
    }

    /// 1x means restore without resizing, which still costs a model pass.
    #[test]
    fn unit_scale_still_runs_the_model() {
        let s = ScaleStrategy::plan(1, 4).unwrap();
        assert_eq!(s.passes, 1);
        assert_eq!(s.intermediate, 4);
        assert!(s.needs_resample(), "must shrink back to the original size");
    }

    #[test]
    fn a_unit_model_falls_back_to_resampling() {
        let s = ScaleStrategy::plan(4, 1).unwrap();
        assert_eq!((s.passes, s.intermediate), (1, 1));
        assert!(s.needs_resample());
    }

    #[test]
    fn rejects_nonsense_input() {
        assert!(ScaleStrategy::plan(0, 4).is_err());
        assert!(ScaleStrategy::plan(4, 0).is_err());
    }

    #[test]
    fn descriptions_read_sensibly() {
        assert_eq!(ScaleStrategy::plan(4, 4).unwrap().describe(), "1 model pass at 4x");
        assert!(ScaleStrategy::plan(7, 4).unwrap().describe().contains("2 model passes to 16x"));
    }
}
