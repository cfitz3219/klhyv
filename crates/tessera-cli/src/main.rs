//! Command-line front end for the Tessera tiling engine.

use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use tessera_core::backend::{ResampleBackend, Upscaler};
use tessera_core::{accumulator_bytes, pipeline, world, ScaleStrategy, UpscaleOptions};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum BackendKind {
    /// Classical resampling on the CPU. Runs anywhere; adds no detail.
    Resample,
    /// Neural upscaling through an ONNX model. Needs --model.
    #[cfg(feature = "onnx")]
    Onnx,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum DeviceKind {
    /// Use a GPU provider when one is compiled in and present.
    Auto,
    /// Force CPU inference.
    Cpu,
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

    /// Magnification, 1 to 10. With a fixed-factor model the run overshoots
    /// this and resamples down, so any value is reachable. 1 restores without
    /// resizing.
    #[arg(short, long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=10))]
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

    /// Path to a .onnx super-resolution model, for --backend onnx.
    #[cfg(feature = "onnx")]
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// The model's own magnification. Only needed for models whose shapes are
    /// dynamic, where it cannot be read from the file.
    #[cfg(feature = "onnx")]
    #[arg(long)]
    model_scale: Option<u32>,

    /// Where to run inference, for --backend onnx.
    #[cfg(feature = "onnx")]
    #[arg(long, value_enum, default_value_t = DeviceKind::Auto)]
    device: DeviceKind,

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

fn build_backend(args: &Args) -> Result<Box<dyn Upscaler>> {
    match args.backend {
        // Resampling reaches any factor directly, so it needs no pass chaining.
        BackendKind::Resample => Ok(Box::new(ResampleBackend::new(args.scale, &args.filter)?)),
        #[cfg(feature = "onnx")]
        BackendKind::Onnx => {
            use tessera_core::backend::{Device, OnnxBackend};
            let model = args
                .model
                .as_ref()
                .context("--backend onnx needs --model pointing at a .onnx file")?;
            let device = match args.device {
                DeviceKind::Auto => Device::Auto,
                DeviceKind::Cpu => Device::Cpu,
            };
            Ok(Box::new(OnnxBackend::new(model, device, args.model_scale)?))
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.tile >= 1, "--tile must be at least 1");

    let source = image::open(&args.input)
        .with_context(|| format!("opening {}", args.input.display()))?
        .to_rgba8();
    let (w, h) = source.dimensions();

    let backend = build_backend(&args)?;
    let scale = args.scale;
    let strategy = ScaleStrategy::plan(scale, backend.scale_factor())?;

    let opts = UpscaleOptions {
        tile: args.tile,
        overlap: args.overlap,
    };

    let (out_w, out_h) = (w * scale, h * scale);
    if !args.quiet {
        eprintln!(
            "{}x{} -> {}x{}  backend {}  tile {} overlap {}",
            w,
            h,
            out_w,
            out_h,
            backend.name(),
            args.tile,
            args.overlap
        );
        eprintln!("plan: {}", strategy.describe());
        // Peak memory is set by the largest intermediate, not the final size.
        eprintln!(
            "blend buffer needs ~{:.1} GiB at peak",
            accumulator_bytes(w * strategy.intermediate, h * strategy.intermediate) as f64
                / (1024.0 * 1024.0 * 1024.0)
        );
    }

    let started = Instant::now();
    let progress = |done: usize, total: usize| {
        // Carriage return keeps this to one line; stderr so stdout stays clean.
        eprint!("\rtile {done}/{total}");
        let _ = io::stderr().flush();
    };
    let result = pipeline::upscale_to_target(
        &source,
        backend.as_ref(),
        scale,
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
        carry_world_file(&args, scale, !args.quiet)?;
    }

    Ok(())
}

/// Rewrite the georeferencing sidecar for the new pixel size, if there is one.
fn carry_world_file(args: &Args, scale: u32, verbose: bool) -> Result<()> {
    let Some((src_path, world_in)) = world::read_sidecar(&args.input)? else {
        return Ok(());
    };
    let written = world::write_sidecar(&args.output, world_in.scaled(scale))?;
    if verbose {
        eprintln!(
            "georeference {} -> {}",
            src_path.display(),
            written.display()
        );
    }
    Ok(())
}
