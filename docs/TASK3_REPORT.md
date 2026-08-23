# Task 3 — The app

The CLI became a local web app: load a clip, click the subject, get a matte, correct it,
export. One command, no accounts, nothing leaves the machine.

![walkthrough](img/walkthrough.gif)

## What shipped

| | |
|---|---|
| `cleanplate/` | the pipeline as a package — paths, session, ingest, track, refine, compose, metrics, runs, viewer |
| `src/*.py` | the same CLIs, now thin wrappers over that package |
| `app.py` | a local Gradio 6.25 UI over the same functions |
| `scripts/regression_check.py` | proves a refactor did not change the numbers |
| `scripts/make_docs_assets.py` | drives the live app to produce every screenshot here |
| `scripts/capture_retroactive_demo.py` | reproduces the f067 incident on purpose |

## Checkpoint 0 — the refactor is numerically identical

Rebuilding `walk` from its committed `point.json` alone, through the new package:

```
[regression] shot=walk  reference=outputs/walk_v2
[regression] prompt from shots/walk/point.json: 5 clicks on frames [0, 67]
[regression] track: 0.8015 s/frame on mps, fallback ops none
  binary masks: IDENTICAL  pixel agreement 100.0000%  per-frame IoU min 1.00000
[regression] refine: 0.1261 s/frame on mps, fallback ops none
  soft alpha  : IDENTICAL  pixel agreement 100.0000%  per-frame IoU min 1.00000
  area_change_pct.mean       1.489  ->  1.489   yes
  area_change_pct.max        5.563  ->  5.563   yes
  iou_consecutive.min       0.8549  -> 0.8549   yes
  perimeter.norm_cv_pct      7.111  ->  7.111   yes
  softness...mean          0.00791 -> 0.00791   yes
[regression] PASS - refactor is numerically identical
```

Re-running `export_rgba` through the refactored CLI also reproduced the Task 2 RGBA
**byte for byte**, despill included.

### A bug that only exists once both models share a process

`sam2/__init__.py` registers its Hydra config module at import time; MatAnyone's
`get_matanyone_model()` calls `hydra.initialize()` unconditionally. As separate CLI
processes they never met. In the app they do, and the second one raised
*"GlobalHydra is already initialized"*. `refine.py` now scopes the swap and restores
SAM 2's config module afterwards — verified by running track → refine → track in one
process, with the second track bit-identical to the first.

## Timings, measured inside the app

From the app's own Export tab after a full pass on the 96-frame `walk` shot:

| Stage | Device | s/frame | total | MPS fallback ops |
|---|---|---|---|---|
| Track (SAM 2.1 hiera-small) | `mps` | 0.792 | 79.4 s | **none** |
| Refine (MatAnyone) | `mps` | 0.125 | 13.7 s | **none** |
| RGBA + despill | cpu | 0.007 | 0.7 s | — |

About **95 seconds** from click to exportable matte. Despill altered 482,731 px —
the Task 2 CLI figure was 482,728, the difference being a 3-pixel click offset.

## The Changes tab, and the incident reproduced on purpose

`scripts/capture_retroactive_demo.py` drives the app to: click the actor once on frame 0,
track, then add the *under-specified* correction from Task 2 on frame 67 — one keep on
the t-shirt, one exclude on the pavement — and track again.

![changes](img/09_changes.png)

> **Retroactive change: 63 frame(s) before your click on frame 67 moved.**
> Range f3–f66, worst f63 at IoU 0.691.

92 of 96 frames changed. The matte visibly loses the jacket. This is the Task 2 failure,
on demand, with the app naming it instead of leaving you to discover it three exports
later. Adding keep points that cover the jacket, jeans and head on frame 67 restores it.

The app also refuses to run a conditioning frame that carries only exclude points, which
was the other Task 2 trap: SAM 2 reads that as "object absent" and blanks the frame.

## A new finding: near-coincident clicks are destructive

While capturing screenshots, the automation added a keep point **one pixel** from the
existing one on frame 0. The matte collapsed to the t-shirt. Reproduced deliberately:

| prompt | mean mask area | min | max |
|---|---|---|---|
| committed 5-click prompt | 0.1071 | 0.0887 | 0.1232 |
| same + a second f0 keep at (360, 199) | 0.0702 | 0.0196 | 0.1223 |

Mean area **−34 %**, minimum frame area 8.87 % → 1.96 %, per-frame IoU between the two
runs as low as 0.19. Two near-coincident positives do not reinforce each other — they
change which mask hypothesis SAM 2 selects, here from "the whole person" to "the printed
t-shirt". The app now warns when a click lands within 12 px of an existing point on the
same frame.

## Checkpoint 6 — cold-clone ship test

Run in a fresh `mktemp -d`, nothing reused from the dev checkout.

```
$ df -h . | tail -1
/dev/disk3s5   460Gi   311Gi   120Gi    73%
$ python3 --version
Python 3.12.7
$ ffmpeg -version | head -1
ffmpeg version 8.1

$ git clone https://github.com/shauryadata/cleanplate.git
$ cd cleanplate && du -sh .
464K    .

$ python3 -m venv .venv
$ .venv/bin/pip install -r requirements.txt
Successfully installed ... gradio-6.25.0 ... torch-2.13.0 torchvision-0.28.0 ...

$ .venv/bin/python -c "import torch, gradio; ..."
torch 2.13.0 | mps True | gradio 6.25.0

$ ./scripts/download.sh all
==> footage -> shots/source/tears_of_steel_1080p.webm
==> cloning SAM 2 ... pip install -e vendor/sam2
==> checkpoint -> checkpoints/sam2.1_hiera_small.pt
==> MatAnyone is licensed S-Lab 1.0 - NON-COMMERCIAL USE ONLY. See THIRD_PARTY.md.
==> downloading matanyone.pth (~135 MB)
==> done.

$ du -sh shots/source checkpoints vendor .venv
560M    shots/source
321M    checkpoints
198M    vendor
1.2G    .venv

$ .venv/bin/python app.py
  CleanPlate v0.3.0 — local only, no uploads, no telemetry
  http://127.0.0.1:7860
```

The clone had **zero frames** on disk — footage is gitignored, only `shot.json` and
`point.json` are committed. Clicking **Load demo shot (walk)** in the browser cut the
shot out of the source movie with ffmpeg and produced 96 frames, then Run Track produced
a matte. **No CLI was touched.** Every screenshot in this report was captured from that
cold-clone install.

Total install: about 2.3 GB of downloads on top of a 464 KB clone.

## UX gaps found while using it, ranked

1. **Re-running Track is all-or-nothing, and it costs 80 seconds.** Adding one corrective
   click re-propagates the entire shot. Since conditioning frames are global this is
   *correct*, but it makes iteration slow and it is the single biggest drag on the
   click → look → fix loop. Caching the encoder pass over frames would help most.
2. **Progress renders once per output component.** A single Track click paints three
   progress bars because three components are being updated. Cosmetic but cheap-looking.
3. **The matte cannot be edited, only re-prompted.** There is no brush, no per-frame
   patch, no way to say "this frame only". Every fix is a global re-solve.
4. **No undo across runs.** Undo works on points; there is no way to get the *previous
   matte* back after a bad re-run other than removing points and re-running.
5. **State is in memory only.** Masks and alpha live in `gr.State`; a browser refresh
   loses the run, though `point.json` survives. Long jobs deserve a resume.
6. **The frame slider is the only navigation.** No playback, no stepping keys, no
   thumbnail strip. Finding the frame where a matte breaks means dragging.
7. **Six tabs barely fit.** They collapsed into a "…" overflow menu at first, hiding the
   Changes tab, which is the point of the app. Fixed by shortening labels and tightening
   tab padding, but it is fragile at smaller windows.
8. **Callout contrast was broken in light theme.** The dark-panel styling inherited the
   light theme's dark text, making the retroactive warning unreadable — the most
   important message in the app. Fixed with `prefers-color-scheme` variants.
9. **Export re-renders from scratch every time.** Choosing `comp.mp4` then
   `side_by_side.mp4` re-composites all 96 frames twice.
10. **The IoU chart is a static PNG.** No hover, no click-to-jump. Clicking a dip should
    take you to that frame.
11. **"Build RGBA" is a manual step.** Refining does not automatically rebuild it, so the
    Comp view can silently show pre-despill pixels until you press it.
12. **No cached-result labelling is needed yet — because nothing is cached.** If the
    encoder cache in (1) lands, every view showing reused output must say so.
