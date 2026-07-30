# Tessera

Seam-free tiled upscaling for large images and scanned maps.

Big scans can't be pushed through a neural upscaler in one piece — a 20,000 ×
20,000 map doesn't fit in VRAM. The standard answer is to cut the image into
tiles and upscale each one, but tiles upscaled independently disagree slightly
near their edges, and that disagreement shows up as a grid across the output.

Tessera is the part that makes tiling invisible: overlapping tiles, cross-faded
with weights that reconstruct the un-tiled result. On the reference backend the
worst channel error against an un-tiled upscale is 1–4 out of 255.

It also carries georeferencing across, which general-purpose upscalers drop.

## Status

Early. The engine, the CLI, and the world-file handling work and are tested.
The neural backend and the GUI are not built yet — see [Roadmap](#roadmap).

The only backend today is classical resampling. It adds no detail, so it is not
the point of the project; it exists so the pipeline runs on any machine and so
the seam tests have a deterministic backend to check against.

## Build

```sh
cargo build --release
```

## Use

```sh
# 4x upscale
tessera scan.png scan-4x.png --scale 4

# smaller tiles if memory is tight
tessera huge-map.tif out.png --scale 2 --tile 128 --overlap 24
```

Options: `--scale`, `--tile`, `--overlap`, `--filter`, `--backend`,
`--no-world`, `--quiet`. Run `tessera --help` for details.

Tiles are processed in parallel across cores. Lower `--tile` to cut memory use;
raise `--overlap` if a backend produces strong edge artifacts.

### Georeferencing

If a world file sits beside the input (`.tfw`, `.jgw`, `.pgw`, `.wld`), Tessera
rewrites it for the new pixel size and writes it beside the output. Pixel size
is divided by the scale factor and the origin is shifted so the *outer corner*
of the image stays put — carrying the sidecar across unchanged would offset the
map by half a pixel of the original grid. Rotation and skew terms are preserved.

GeoTIFF tags stored inside the TIFF are not handled yet.

## Layout

| Crate | Contents |
|---|---|
| `tessera-core` | Tiling, blending, backend trait, world files |
| `tessera-cli` | `tessera` command-line binary |

Inside `tessera-core`:

- `tiling.rs` — tile placement and blend weights
- `pipeline.rs` — parallel tile processing and premultiplied accumulation
- `backend/` — the `Upscaler` trait and the resampling reference backend
- `world.rs` — world-file parsing, scaling, and sidecar naming

## How the blend works

Tile origins are spread evenly rather than stepped at a fixed stride with a
short final step, so no pair of tiles overlaps far more than the rest. Each
cross-fade is then sized from the *actual* overlap with that specific
neighbour, so opposing ramps always span exactly the shared region and sum to
one. Contributions accumulate in premultiplied alpha and are normalised by
total weight, which keeps the blend correct where tiles meet across
transparency.

Adding a backend means implementing `Upscaler` — one method that takes a tile
and returns it magnified. Nothing in the tiling, blending, or I/O path needs to
change.

## Roadmap

- **Neural backend.** Real-ESRGAN via ncnn/Vulkan, behind the existing
  `Upscaler` trait. Vulkan rather than CUDA so AMD, Intel, and Apple GPUs work.
- **Streaming I/O.** The blend accumulator is currently in memory at ~20 bytes
  per output pixel, which caps practical output size. Streaming tiles to and
  from disk removes the cap. The CLI prints the buffer estimate before starting.
- **GeoTIFF tags.** Read and rewrite the coordinate system embedded in the TIFF,
  not just the sidecar.
- **Content-aware model routing.** Maps are text, line art, and halftone, not
  photographs; photo-trained models hallucinate texture into linework. Route by
  content type, with halftone descreening before upscaling.
- **GUI.** A Tauri shell over this engine — chosen over Electron because the
  engine competes with the UI for memory, and a tiled canvas viewer is needed
  for gigapixel output regardless of framework.

## License

MIT
