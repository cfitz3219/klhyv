# Tessera

Seam-free tiled upscaling for large images and scanned maps.

Big scans can't be pushed through a neural upscaler in one piece — a 20,000 ×
20,000 map doesn't fit in VRAM, and most models fix their input size anyway. The
standard answer is to cut the image into tiles and upscale each one, but a
neural model is not shift-invariant: two tiles covering the same pixel disagree
about it, and that disagreement shows up as a grid across the output.

Tessera is the part that makes tiling invisible: overlapping tiles, cross-faded
with weights that reconstruct the un-tiled result. It streams the result to disk
a band at a time, so output size is not limited by memory. And it carries
georeferencing across, which general-purpose upscalers drop.

## Does it actually work?

Measured, not asserted.

**Seams.** Upscaling a texture-rich image 4× with Real-ESRGAN at 128px tiles,
then measuring the column-to-column step at the tile edge against the local
median step of nearby columns:

| Setting | Step at tile edge vs. local baseline |
|---|---|
| `--overlap 0` (butted) | **3.83×** — a clear seam |
| `--overlap 32` (blended) | **0.99–1.05×** — indistinguishable from content |

Against the classical backend, where an un-tiled reference exists to compare
with, the worst channel error across tile configurations is 1–4 out of 255.

**Memory.** A 3000 × 2000 source upscaled on a machine with no special
provisions:

| Run | Output | Peak memory |
|---|---|---|
| ×4, `--tile 256` | 12,000 × 8,000 | 405 MiB |
| ×4, `--tile 128` | 12,000 × 8,000 | 224 MiB |
| ×8, `--tile 256` | 24,000 × 16,000 (384 Mpx) | 1,306 MiB |

Holding that ×8 result in memory the naive way would need about 9 GiB.

## Status

Early, but the core works end to end. Engine, CLI, neural backend, streaming
output, and georeferencing are built and tested. No GUI yet — see
[Roadmap](#roadmap).

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
tessera map.tif map-7x.png --backend onnx --model realesrgan-x4.onnx --tile 128 --scale 7
```

Run `tessera --help` for all options.

### Choosing a magnification

`--scale` takes any whole number from 1 to 10, whatever factor the model was
trained at.

Models bake their factor into the weights — a ×4 model only ever produces ×4 —
so other factors are reached by running the model until the result *exceeds* the
request, then resampling down to the exact size. Asking for ×7 from a ×4 model
runs two passes to ×16 and shrinks. That beats one pass to ×4 stretched up to
×7, which only blurs what the model produced.

Tessera prints its plan before starting:

```
plan: 2 model passes to 16x, then resample to 7x
```

`--scale 1` is not a no-op: it runs one model pass and shrinks back, which is
how these models strip compression artifacts and scanning noise without
changing the image's size.

### Large images

Output is produced one row of tiles at a time. Finished rows go straight to the
file, and only the rows still being blended stay resident, so peak memory
depends on the tile size and image *width* — not on how tall the result is.

Two consequences worth knowing:

- **Lower `--tile` when memory is tight.** Halving it roughly halves peak usage.
- **Write PNG for the largest jobs.** PNG is encoded row by row as results
  arrive. Other formats have to be assembled in memory first.

Three limits remain, in rough order of when you would hit them:

- A magnification that is not a whole number of model passes (×3, ×5, ×6, ×7…)
  has to hold the final intermediate in memory so it can be resampled down.
  Exact multiples of the model's factor stream all the way through.
- The source image is held in memory. That is 4 bytes per pixel against 20 for
  the blend buffer, so it is rarely the binding constraint, but a 20,000 ×
  20,000 scan is still 1.6 GiB.
- Intermediate passes in a multi-pass run are held in memory.

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
| `tessera-core` | Tiling, blending, backends, streaming, world files |
| `tessera-cli` | `tessera` command-line binary |

Inside `tessera-core`:

- `tiling.rs` — tile placement and blend weights
- `pipeline.rs` — banded processing and premultiplied accumulation
- `sink.rs` — where finished rows go: memory, or streamed to a PNG
- `scale.rs` — reaching any magnification with a fixed-factor model
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

Because output is produced in bands, partial sums for the rows shared between
one row of tiles and the next are carried forward rather than resolved early.

Adding a backend means implementing `Upscaler` — one method that takes a tile
and returns it magnified. Nothing in the tiling, blending, or I/O path changes.

## Roadmap

- **GUI.** A Tauri shell over this engine — chosen over Electron because the
  engine competes with the UI for memory, and a tiled canvas viewer is needed
  for gigapixel output regardless of framework.
- **Streaming source and intermediates.** Removes the remaining memory limits
  listed under [Large images](#large-images).
- **GeoTIFF tags.** Read and rewrite the coordinate system embedded in the TIFF,
  not just the sidecar.
- **Content-aware model routing.** Maps are text, line art, and halftone, not
  photographs; photo-trained models hallucinate texture into linework. Route by
  content type, with halftone descreening before upscaling.

## License

MIT. Models and the ONNX Runtime carry their own terms.
