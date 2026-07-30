//! Command-line front end for the Tessera tiling engine.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use tessera_core::backend::{ResampleBackend, Upscaler};
use tessera_core::{accumulator_bytes, pipeline, world, UpscaleOptions};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    /// Classical resampling on the CPU. Runs anywhere; adds no detail.
    Resample,
}

#[derive(Parser, Debug)]
#[command(
    name = "tessera",
    about = "Seam-free tiled upscaling for large images and scanned maps",
    version
)]
struct Args {
    /// Image to upscale.
    input: PathBuf,

    /// Where to write the result. Format follows the extension.
    output: PathBuf,

    /// Magnification factor.
    #[arg(short, long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=16))]
    scale: u32,

    /// Tile edge length, in source pixels. Lower this if you run out of memory.
    #[arg(short, long, default_value_t = 256)]
    tile: u32,

    /// Overlap between neighbouring tiles, in source pixels.
    #[arg(short = 'v', long, default_value_t = 32)]
    overlap: u32,

    /// Upscaling backend.
    #[arg(short, long, value_enum, default_value_t = BackendKind::Resample)]
    backend: BackendKind,

    /// Resampling filter, for the resample backend.
    #[arg(
        short,
        long,
        default_value = "lanczos3",
        value_parser = ["lanczos3", "catmullrom", "gaussian", "triangle", "nearest"]
    )]
    filter: String,

    /// Do not read or write world-file sidecars.
    #[arg(long)]
    no_world: bool,

    /// Suppress progress output.
    #[arg(short, long)]
    quiet: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let source = image::open(&args.input)
        .with_context(|| format!("opening {}", args.input.display()))?
        .to_rgba8();
    let (w, h) = source.dimensions();

    let backend: Box<dyn Upscaler> = match args.backend {
        BackendKind::Resample => Box::new(ResampleBackend::new(args.scale, &args.filter)?),
    };

    let opts = UpscaleOptions {
        tile: args.tile,
        overlap: args.overlap,
    };

    let (out_w, out_h) = (w * args.scale, h * args.scale);
    if !args.quiet {
        eprintln!(
            "{}x{} -> {}x{}  scale {}x  backend {}  tile {} overlap {}",
            w,
            h,
            out_w,
            out_h,
            args.scale,
            backend.name(),
            args.tile,
            args.overlap
        );
        eprintln!(
            "blend buffer needs ~{:.1} GiB",
            accumulator_bytes(out_w, out_h) as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    let started = Instant::now();
    let progress = |done: usize, total: usize| {
        // Carriage return keeps this to one line; stderr so stdout stays clean.
        eprint!("\rtile {done}/{total}");
        let _ = io::stderr().flush();
    };
    let result = pipeline::upscale_tiled(
        &source,
        backend.as_ref(),
        opts,
        if args.quiet { None } else { Some(&progress) },
    )?;
    if !args.quiet {
        eprintln!("\rdone in {:.1}s{:12}", started.elapsed().as_secs_f64(), "");
    }

    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    result
        .save(&args.output)
        .with_context(|| format!("writing {}", args.output.display()))?;

    if !args.no_world {
        carry_world_file(&args, !args.quiet)?;
    }

    Ok(())
}

/// Rewrite the georeferencing sidecar for the new pixel size, if there is one.
fn carry_world_file(args: &Args, verbose: bool) -> Result<()> {
    let Some((src_path, world_in)) = world::read_sidecar(&args.input)? else {
        return Ok(());
    };
    let written = world::write_sidecar(&args.output, world_in.scaled(args.scale))?;
    if verbose {
        eprintln!(
            "georeference {} -> {}",
            src_path.display(),
            written.display()
        );
    }
    Ok(())
}
