# CleanPlate

Open-source, fully local AI rotoscoping and cleanup for film. Click once, get a production matte.

No cloud, no API keys, no per-frame billing. Everything runs on your own machine —
Apple Silicon (MPS), CUDA, or CPU.

## Status

**Task 2 — from cutout to key: done.** The binary matte is now a soft key with despill,
corrective clicks kill flicker at its source, and every claim is measured.

| | |
|---|---|
| Device | Apple M3 Pro, MPS — **zero fallback ops in either stage** |
| Track (SAM 2.1 hiera-small) | 0.896 s/frame (1.12 fps) |
| Refine (MatAnyone v1.0.0) | 0.161 s/frame (6.21 fps) |
| Alpha | 256 levels, ~0.8–1.1 % of frame fractional (was 2 levels, 0 %) |
| Flicker on `walk` | mean −15.5 %, max −35.9 % |
| Pavement bleed | −93 % from one corrective negative click |
| Shots | `walk`, `dialogue`, `hair` — the RotoBench seed |

Hair is an honest partial negative: the silhouette is soft, but no interior strand
transparency is recovered (soft pixels sit a median of 2 px from the solid core). Read
[docs/TASK2_REPORT.md](docs/TASK2_REPORT.md) for the full assessment,
[docs/METRICS.md](docs/METRICS.md) for the tables, and
[docs/DECISIONS.md](docs/DECISIONS.md) for why MatAnyone.

RotoBench — a public benchmark of AI matte quality — comes in a later phase.

## Pipeline

```
movie ─► extract_shot ─► pick_point ─► track_matte ─► refine_matte ─► export_rgba ─► comp_preview
          (jpg frames)     (clicks)    (SAM 2 video,  (MatAnyone,     (RGBA +        (comp.mp4,
                                        binary mask)   soft alpha)     despill)       side_by_side,
                                                                                      v1_vs_v2)
                                                 └──────────► metrics ◄──────────┘
```

`contact_sheet.py` sits alongside these: scout a whole movie for shots, or review any
folder of numbered frames (source, masks, overlays, comps).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision opencv-python pillow numpy matplotlib
./scripts/download.sh          # footage + SAM 2 + checkpoints (nothing large is committed)

# cut a shot out of the movie
python src/extract_shot.py --name walk --start 270 --duration 4

# click the actor once (writes shots/walk/point.json)
python src/pick_point.py --shot walk

# propagate the matte through every frame; --at adds a corrective click on any frame
python src/track_matte.py --shot walk --point 360,200 \
    --at 67:352,250:+ --at 67:400,360:+ --at 67:415,60:+ --at 67:321,379:- \
    --out-name walk_v2

# turn the binary mask into a soft alpha  (optional stage, see the licence note below)
python src/refine_matte.py --shot walk --masks outputs/walk_v2/masks --out-name walk_v2

# merge frames + soft alpha into RGBA, suppressing green edge spill
python src/export_rgba.py --shot walk --out-name walk_v2 \
    --alpha outputs/walk_v2/alpha --despill green

# comp over a new background, and render a 2x2 against the previous run
python src/comp_preview.py --shot walk --out-name walk_v2 \
    --bg-crop 380,125,1680,600 --compare-with walk

# put numbers on it
python src/metrics.py --run "v1=outputs/walk/masks" --run "v2=outputs/walk_v2/alpha"
```

> **Licence note.** The refine stage uses **MatAnyone**, which is **S-Lab License 1.0 —
> non-commercial use only**. It is never bundled: `scripts/download.sh matanyone` clones it
> into gitignored `vendor/`, and CleanPlate runs fine without it, producing the binary
> SAM 2 matte. See [THIRD_PARTY.md](THIRD_PARTY.md).

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `track_matte.py` itself, so unimplemented MPS
ops degrade to CPU instead of crashing. Pass `--device cpu` to take MPS out of the picture
entirely.

## Layout

| Path | What lives there |
|---|---|
| `src/` | The pipeline scripts |
| `docs/` | Task reports, decisions, metrics |
| `vendor/` | SAM 2 and MatAnyone checkouts (gitignored, re-fetched) |
| `shots/` | Source movie and extracted JPEG frame folders (gitignored) |
| `outputs/` | Masks, RGBA sequences, comps (gitignored) |
| `scripts/` | `download.sh` — re-fetches footage, SAM 2, and weights |
| `checkpoints/` | SAM 2 weights (gitignored) |

Footage, model weights, the venv, and all outputs are gitignored by design.
`scripts/download.sh` re-fetches every one of them.

## License

MIT — see [LICENSE](LICENSE). Third-party components and footage credits are listed in
[THIRD_PARTY.md](THIRD_PARTY.md).
