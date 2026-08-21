# Task 1 — Proof of Life

One Tears of Steel shot, end to end, fully local, from a single click.
Everything below was measured on the machine, not estimated.

## Shot

| | |
|---|---|
| Source | `tears_of_steel_1080p.webm`, 1920x800, VP8, 24fps, 734.167s |
| Origin | https://media.xiph.org/tearsofsteel/ (official Xiph mirror) |
| Credit | **(CC) Blender Foundation \| mango.blender.org** |
| Shot | `walk` — t=270.0s, 4.0s, 96 frames, 960x400, 24fps |
| Content | Thom walks laterally along an Amsterdam canal, camera tracking |

Two other candidates were extracted and rejected in favour of this one:
`dialogue` (t=30s, two figures on a bridge) and `hair` (t=177s, long grey hair in a
dim interior). Both remain on disk and are reproducible from `shots/*/shot.json`.

## Environment and performance

| | |
|---|---|
| Machine | MacBook M3 Pro, 18 GB unified memory, arm64 |
| Device used | **`mps`** — no CPU fallback was needed |
| torch / torchvision | 2.13.0 / 0.28.0 |
| Python | 3.12.7 (venv) |
| Model | SAM 2.1 `sam2.1_hiera_s.yaml` + `sam2.1_hiera_small.pt` (176 MB) |
| Precision | fp32. `--autocast` (bf16) exists but is off by default — bf16 autocast is not dependable on MPS |
| **MPS fallback ops** | **none.** `PYTORCH_ENABLE_MPS_FALLBACK=1` was set, and the run captured zero `aten::*` fallback warnings |

| Stage | Time |
|---|---|
| Model build | 0.9 s |
| Load 96 JPEG frames | 3.2 s |
| **Propagate 96 frames** | **81.5 s = 0.849 s/frame (1.18 fps)** |
| RGBA export (96 frames) | ~2 s |
| Comp + 3 video encodes | 8.9 s |

Prompt: **one** positive point, `--point 360,200` on frame 0 (centre of the chest).
No negative points, no box, no corrections on later frames.

Coverage: 96/96 frames, zero empty masks, mask area 9.34%–12.57% of frame.

## Outputs on disk

```
outputs/walk/
├── masks/00000.png … 00095.png      96 binary mattes, 960x400, 8-bit L
├── rgba/00000.png … 00095.png       96 RGBA PNGs, straight (unpremultiplied) alpha
├── comp_image/, comp_solid/         comp frames
├── comp.mp4                         actor over the Mars plate, 960x400  (533 KB)
├── comp_solid.mp4                   actor over flat #D4571E, 960x400    (356 KB)
├── side_by_side.mp4                 ORIGINAL | MATTE | COMP, 2880x400 (1.9 MB)
├── stills/                          the five report stills
├── backgrounds/                     plate + SOURCES.txt (licences)
├── track.json                       device, timings, fallback ops, coverage
└── rgba.json                        alpha coverage stats
```

Background plate: NASA/JPL Curiosity 360 panorama, **public domain**, verified through
the Wikimedia Commons API. Full provenance in `outputs/walk/backgrounds/SOURCES.txt`.

## Quality flaws

These are the real defects in this first pass. They are the input to Task 2.

### 1. Alpha is strictly binary — there is no soft edge anywhere
`numpy.unique(alpha)` over any mask returns exactly `[0, 255]`. Every edge in the
matte is a hard, aliased staircase. Motion-blurred limbs, hair, and any semi-transparent
pixel are all forced to fully-on or fully-off. This is the single biggest thing standing
between this and a usable matte — a production key needs sub-pixel coverage.
SAM 2 emits mask *logits*; thresholding at 0 throws that gradient away.

### 2. Background bleed at the lower back that pops on and off — the main flicker source
A wedge of pale grey pavement stays attached to the actor's lower back and hip, and
switches in and out between frames. Measured: frame-to-frame mask-area change averages
1.76% but spikes to **8.67% at f0070**, with the worst cluster at **f0064–f0073**
(consecutive-frame IoU bottoms out at 0.804 at f0067). Visible directly in
`comp_solid` frames 63–74: the wedge is present at f0063/67/68/71/74 and largely gone at
f0064/65/66/69/70/72/73. On the Mars comp this reads as a grey smear that blinks.

### 3. Boundary boil
Perimeter length swings between 766 and 1055 pixels across the shot (mean 895) with no
corresponding change in the subject's silhouette. The edge crawls frame to frame even
where the actor is barely moving. There is no temporal smoothing of the matte at all.

### 4. Chewed edges on the yellow lining
The yellow garment hanging at his hip is torn into ragged, comb-like strands at
f0071 and f0073, and partially dropped at f0070 (the shot's minimum mask area, 35,865 px
against a 48,279 px maximum). Thin, high-contrast, fast-moving detail is where the matte
breaks down first.

### 5. Green halo from the foliage
At 5x zoom on the head, a 1–2 px olive/green fringe survives along the right side of the
face and through the hair — background trees leaking into the matte. The same happens in
pale grey along the back of the jacket. No despill, no edge colour correction.

### 6. Hair is cut as a solid block
No individual strands anywhere. The fringe and the top of the head are a smooth blob
boundary. This shot has short dark hair; the `hair` candidate shot (long grey hair) would
almost certainly be much worse.

### 7. Hands and fingers are lumped
The trailing hand is a single mass — the gaps between fingers are never resolved, and the
edge is a visible staircase at 1:1.

### 8. Comp limitation, not a matte flaw: the background does not move
The source camera tracks the actor laterally, but the Mars plate is locked off. The comp
is therefore parallax-free and reads as a cutout on a still. Fixing this needs a camera
solve, which is out of scope for Task 1.

## What worked without a fight

- SAM 2 on MPS ran first try with zero fallback ops and no NaNs.
- A single click held the subject for all 96 frames with no loss of track, no ID swap,
  and no empty frames — through a full walk cycle with arms and legs crossing the body.
- The mask correctly touches the bottom frame edge in 96/96 frames as the legs leave
  frame, and never touches the top edge.

## Suggested Task 2

1. Keep SAM 2's mask logits instead of thresholding: derive a soft alpha, and validate it
   against the flat-colour comp.
2. Temporal stabilisation of the matte to kill the boil and the pavement pop.
3. Edge treatment: erode/feather controls exist in `export_rgba.py` but are untuned and
   off; add despill and edge colour correction.
4. Run the same pipeline on the `hair` and `dialogue` shots and see how much worse it gets
   — that comparison is the seed of RotoBench.
