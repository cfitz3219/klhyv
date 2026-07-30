# Tessera

Seam-free tiled upscaling for large images and scanned maps.

Big scans can't be pushed through a neural upscaler in one piece — a 20,000 ×
20,000 map doesn't fit in VRAM, and most models fix their input size anyway. The
standard answer is to cut the image into tiles and upscale each one, but a
neural model is not shift-invariant: two tiles covering the same pixel disagree
about it, and that disagreement shows up as a grid across the output.

Tessera is the part that makes tiling invisible: overlapping tiles, cross-faded
with weights that reconstruct the un-tiled result.

It also carries georeferencing across, which general-purpose upscalers drop.

## Does the blending actually work?

Measured, not asserted. Upscaling a texture-rich image 4× with Real-ESRGAN at
128px tiles, then measuring the column-to-column step at the tile edge against
the local median step of nearby columns:

| Setting | Step at tile edge vs. local baseline |
|---|---|
| `--overlap 0` (butted) | **3.83×** — a clear seam |
| `--overlap 32` (blended) | **0.99–1.05×** — indistinguishable from content |

Against the classical backend, where an un-tiled reference exists to compare
with, the worst channel error across tile configurations is 1–4 out of 255.

## Status

Early, but the core works end to end. Engine, CLI, neural backend, and
georeferencing are built and tested. No GUI yet — see [Roadmap](#roadmap).

## Build

```sh
cargo build --release
```

This fetches a prebuilt ONNX Runtime. To build without it:

```sh
cargo build --release --no-default-features
```

### GPU

The CPU provider works everywhere and needs no flags. GPU providers are opt-in
at build time, because each links a vendor runtime:

```sh
cargo build --release --features cuda        # NVIDIA
cargo build --release --features tensorrt    # NVIDIA, faster, longer warm-up
cargo build --release --features directml    # any vendor, Windows
cargo build --release --features coreml      # Apple
```

ONNX Runtime falls back to CPU when a provider is absent at runtime, so a binary
built with `cuda` still works on a machine without it. Use `--device cpu` to
force CPU.

## Use

```sh
# classical resampling — no model needed, adds no detail
tessera scan.png out.png --scale 4

# neural upscaling
tessera map.tif map-4x.png --backend onnx --model realesrgan-x4.onnx --tile 128
```

The scale factor is read from the model, so `--scale` is only needed for models
with fully dynamic shapes. Run `tessera --help` for all options.

Tiles are processed in parallel. Lower `--tile` if memory is tight; raise
`--overlap` if a model produces strong edge artifacts. If a model fixes its
input size, `--tile` must match it — Tessera says so if it doesn't.

### Models

Models are not bundled: super-resolution weights carry their own licences, and
the right model for a scanned map is not the right one for a photograph. Any
Real-ESRGAN-style ONNX model works — RGB `NCHW` in, RGB `NCHW` out, values in
0..1. Tested against `real-esrgan-x4plus-128`.

Alpha is handled separately, since these models are RGB-only: it is resampled
classically and reattached. Fully opaque images skip that work.

### Georeferencing

If a world file sits beside the input (`.tfw`, `.jgw`, `.pgw`, `.wld`), Tessera
rewrites it for the new pixel size and writes it beside the output. Pixel size
is divided by the scale factor and the origin shifts so the *outer corner* of
the image stays put — carrying the sidecar across unchanged would offset the map
by half a pixel of the original grid. Rotation and skew terms are preserved.

GeoTIFF tags stored inside the TIFF are not handled yet.

## Layout

| Crate | Contents |
|---|---|
| `tessera-core` | Tiling, blending, backends, world files |
| `tessera-cli` | `tessera` command-line binary |

Inside `tessera-core`:

- `tiling.rs` — tile placement and blend weights
- `pipeline.rs` — parallel tile processing and premultiplied accumulation
- `backend/resample.rs` — classical reference backend
- `backend/onnx.rs` — neural backend via ONNX Runtime
- `world.rs` — world-file parsing, scaling, and sidecar naming

## How the blend works

Tile origins are spread evenly rather than stepped at a fixed stride with a
short final step, so no pair of tiles overlaps far more than the rest. Each
cross-fade is then sized from the *actual* overlap with that specific
neighbour, so opposing ramps always span exactly the shared region and sum to
one. Contributions accumulate in premultiplied alpha and are normalised by total
weight, which keeps the blend correct where tiles meet across transparency.

Adding a backend means implementing `Upscaler` — one method that takes a tile
and returns it magnified. Nothing in the tiling, blending, or I/O path changes.

## Roadmap

- **Streaming I/O.** The blend accumulator is in memory at ~20 bytes per output
  pixel, which caps practical output size. Streaming tiles to and from disk
  removes the cap. The CLI prints the buffer estimate before starting.
- **GeoTIFF tags.** Read and rewrite the coordinate system embedded in the TIFF,
  not just the sidecar.
- **Content-aware model routing.** Maps are text, line art, and halftone, not
  photographs; photo-trained models hallucinate texture into linework. Route by
  content type, with halftone descreening before upscaling.
- **GUI.** A Tauri shell over this engine — chosen over Electron because the
  engine competes with the UI for memory, and a tiled canvas viewer is needed
  for gigapixel output regardless of framework.

## License

MIT. Models and the ONNX Runtime carry their own terms.
