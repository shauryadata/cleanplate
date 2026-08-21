# CleanPlate

Open-source, fully local AI rotoscoping and cleanup for film. Click once, get a production matte.

No cloud, no API keys, no per-frame billing. Everything runs on your own machine —
Apple Silicon (MPS), CUDA, or CPU.

## Status

**Task 1 — proof of life.** One Tears of Steel shot, end to end: click a point on the
actor in frame 1, propagate a matte through the whole shot with SAM 2, export an alpha
image sequence, and render a background-swap preview.

RotoBench — a public benchmark of AI matte quality — comes in a later phase.

## Pipeline

```
shot frames ──► pick_point.py ──► track_matte.py ──► export_rgba.py ──► comp_preview.py
   (jpg)          (one click)      (SAM 2 video)      (RGBA png seq)     (comp.mp4)
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision opencv-python pillow numpy matplotlib
./scripts/download.sh          # footage + SAM 2 + checkpoints (nothing large is committed)

python src/pick_point.py shots/<shot>/frames        # click the actor, note the pixel
python src/track_matte.py --shot <shot> --point X,Y
python src/export_rgba.py --shot <shot>
python src/comp_preview.py --shot <shot>
```

## Layout

| Path | What lives there |
|---|---|
| `src/` | The pipeline scripts |
| `shots/` | Source movie and extracted JPEG frame folders (gitignored) |
| `outputs/` | Masks, RGBA sequences, comps (gitignored) |
| `scripts/` | `download.sh` — re-fetches footage, SAM 2, and weights |
| `checkpoints/` | SAM 2 weights (gitignored) |

Footage, model weights, the venv, and all outputs are gitignored by design.
`scripts/download.sh` re-fetches every one of them.

## License

MIT — see [LICENSE](LICENSE). Third-party components and footage credits are listed in
[THIRD_PARTY.md](THIRD_PARTY.md).
