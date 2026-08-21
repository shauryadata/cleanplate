# CleanPlate

Open-source, fully local AI rotoscoping and cleanup for film. Click once, get a production matte.

No cloud, no API keys, no per-frame billing. Everything runs on your own machine —
Apple Silicon (MPS), CUDA, or CPU.

## Status

**Task 1 — proof of life: done.** One Tears of Steel shot taken end to end from a single
click. Shot `walk` (t=270s, 96 frames, 960x400, 24fps): one point on the actor's chest in
frame 0, a matte propagated through all 96 frames by SAM 2.1 hiera-small, an RGBA
sequence, and a Mars background swap.

| | |
|---|---|
| Device | Apple M3 Pro, MPS (`torch 2.13.0`) |
| MPS fallback ops | **none** |
| Model | `sam2.1_hiera_s` + `sam2.1_hiera_small.pt` |
| Propagation | 81.48s for 96 frames = **0.8488s/frame** (1.18 fps) |
| Prompt | one positive point, `--point 360,200`, frame 0 |
| Coverage | 96/96 frames, no empty masks |

Known flaws in this first pass are catalogued in [docs/TASK1_REPORT.md](docs/TASK1_REPORT.md);
they are the input to Task 2.

RotoBench — a public benchmark of AI matte quality — comes in a later phase.

## Pipeline

```
movie ──► extract_shot.py ──► pick_point.py ──► track_matte.py ──► export_rgba.py ──► comp_preview.py
            (jpg frames)       (one click)      (SAM 2 video)      (RGBA png seq)      (comp.mp4)
                                                                                       (side_by_side.mp4)
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

# propagate the matte through every frame
python src/track_matte.py --shot walk --point 360,200

# merge frames + masks into an RGBA sequence
python src/export_rgba.py --shot walk

# comp over a new background and render the previews
python src/comp_preview.py --shot walk --bg-crop 380,125,1680,600
```

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `track_matte.py` itself, so unimplemented MPS
ops degrade to CPU instead of crashing. Pass `--device cpu` to take MPS out of the picture
entirely.

## Layout

| Path | What lives there |
|---|---|
| `src/` | The pipeline scripts |
| `docs/` | Task reports |
| `shots/` | Source movie and extracted JPEG frame folders (gitignored) |
| `outputs/` | Masks, RGBA sequences, comps (gitignored) |
| `scripts/` | `download.sh` — re-fetches footage, SAM 2, and weights |
| `checkpoints/` | SAM 2 weights (gitignored) |

Footage, model weights, the venv, and all outputs are gitignored by design.
`scripts/download.sh` re-fetches every one of them.

## License

MIT — see [LICENSE](LICENSE). Third-party components and footage credits are listed in
[THIRD_PARTY.md](THIRD_PARTY.md).
