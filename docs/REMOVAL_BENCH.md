# Removal — scored on Tier R synthetic truth

Each clip is a clean background with a tracked foreground composited in, so
the background is the correct answer **by construction**. PSNR and SSIM are
measured inside the removal hole only: the rest of the frame is untouched and
would score infinity, which tells you nothing.

`warp_err_gt` is the same temporal metric computed on the true background —
the floor. A fill cannot be more temporally stable than the footage it
replaces, so read the two together.

| Metric | R1_actor_over_mars | R2_actor_over_city | R3_actor_over_canal | R4_big_over_harbour |
|---|---|---|---|---|
| PSNR in the hole (dB, higher better) | **22.185** | 21.589 | 15.955 | 18.356 |
| SSIM in the hole (higher better) | 0.50206 | **0.63452** | 0.43973 | 0.60215 |
| PSNR whole frame (dB, higher better) | **31.643** | 31.608 | 25.493 | 25.412 |
| Temporal warp error in the hole (x1e3, lower better) | 0.126 | **0.1067** | 2.0628 | 0.8894 |
|   same metric on the true background (reference floor) | 0 | 0 | 6.681 | 2.8253 |
| Hole area (fraction of frame) | 0.10954 | **0.09924** | 0.11083 | 0.19683 |

## Cost and memory

| Clip | s/frame | swap during stage | lowest available |
|---|---|---|---|
| R1_actor_over_mars | 2.15 | 9023 → 11631 MB | 1117 MB |
| R2_actor_over_city | 2.17 | 10885 → 11994 MB | 731 MB |
| R3_actor_over_canal | 2.21 | 10754 → 12131 MB | 904 MB |
| R4_big_over_harbour | 2.34 | 11575 → 14052 MB | 854 MB |

Settings: ProPainter, fp16, chunks of 8 frames, hole dilated 12 px. Those defaults were measured, not guessed — ProPainter's own defaults (fp32, chunks of 80) grew swap by 6 GB on 24 frames and the guard killed the run.

## What the numbers say

**Static backgrounds** (R1_actor_over_mars, R2_actor_over_city) have a warp floor of exactly zero: the true background never moves, so every bit of motion in the fill is spurious. Measured: 0.126, 0.107 (x1e3). That is low-amplitude flicker rather than drift - stable enough to cut with, but not frozen the way the plate under it is.

**Moving backgrounds** (R3_actor_over_canal, R4_big_over_harbour) fail the opposite way, and they fail by almost the same amount: the fill reproduces 31%, 31% of the true background motion. Under-movement, not over-movement. ProPainter's transformer regresses toward a temporally smooth answer, so moving water and passing architecture come back too calm.

This is the trap in reading the temporal metric alone. The worst clip by PSNR is R3_actor_over_canal at 16.0 dB, and its warp error still sits *below* the ground-truth floor (2.063 against 6.681). The fill is not noisy, it is too placid, and a stability score on its own would have called that a success. A fill that scores well on temporal stability and badly on PSNR is smoothing, not tracking; neither metric catches that by itself.

## Three real removals

The scored clips above are synthetic, because that is the only way to have a correct answer to compare against. These three are real shots with no ground truth, which is what the tool will actually meet.

| Demo | Frames | Hole | s/frame | swap during stage | lowest available |
|---|---|---|---|---|---|
| A_walk_lamppost — walk shot: the cast-iron lamppost at frame left | 24 | 6.04% | 2.17 | 9811 → 12061 MB | 1408 MB |
| B_greenscreen_markers — 08_3a plate: tracking markers on the green backing | 48 | 2.03% | 6.30 | 5028 → 10874 MB | 502 MB |
| C_dialogue_one_actor — dialogue: one of the two actors, leaving the other | 96 | 14.44% | 2.20 | 10381 → 14431 MB | 1069 MB |

**A, the lamppost.** The walk shot is a tracking shot, so a piece of street furniture at frame left is gone within a second: the lamppost is in shot for 24 of the 96 frames and the demo covers those. That camera move is the reason this is the best-looking of the three. Panning past an object is parallax, and parallax genuinely reveals what is behind it, so the fill has evidence instead of having to invent. The building, the doorway, the kerb and the railing all continue correctly; the railing is slightly soft where it crosses the hole.

**B, the tracking markers.** The mask here comes from a detector, not a click: markers are the small non-green blobs on the backing once the subject is excluded using the Tier B keyed alpha. The largest one is so far out of focus that green bleeds through it and it still passes a greenness test, so it is caught on saturation instead — measured at 0.287 below the local backing against 0.182 for clean backing at p99.9, which is what sets the 0.20 threshold. Fifteen blobs per frame, 0.75% of the frame, all of them gone, the actor untouched. Recall is not perfect: one small mark at frame right survives.

**C, one actor of two.** Selective removal — the man is taken out and the woman is left, which is the case that matters for dialogue coverage. SAM 2 kept them separate throughout despite them touching by the end of the shot. The bridge railing and the arch reconstruct well. The weakness is the vertical smudge where his head was, and it has a specific cause: he barely moves relative to the camera for 96 frames, so nothing in the clip ever reveals the building behind him. Compare demo A, where the pan does.

## Where it breaks

**Moving stochastic texture.** The worst case measured. Water, foliage and crowds come back too calm: the fill carries about a third of the true motion, so it reads as a still patch sliding over a moving plate. Worse the longer the shot stays on it.

**Large occlusions.** Doubling the hole from ~10% to ~20% of frame costs real accuracy even on an otherwise easy background, because there is less surrounding evidence per hole pixel. Expect degradation, not failure.

**No parallax.** An object that holds still relative to the camera never reveals what is behind it, and no amount of temporal context helps: the inpainter is inventing, not recovering. This is the single best predictor of a bad fill, and it is visible in demo C. A camera move is the friend of removal, which is the opposite of the intuition.

**Long holes.** The two failures above compound with shot length. Nothing here is scored past 96 frames, and the pipeline has not been tested on a shot where the object is occluded for hundreds of frames.

**The memory ceiling is real and it is the binding constraint.** On this 18 GB machine, 96 frames of 960x506 aborted the guard twice — once at 465 MB reclaimable, once on 6,034 MB of swap growth — while 96 frames of 960x400 ran fine. Demo B only completed at 48 frames, and then with 502 MB of headroom against a 500 MB limit. It also cost 6.30 s/frame against 2.17 for demo A, because the smaller chunk that makes it fit also makes it slow. Removal is roughly 10x the cost of matting per frame and it is the stage that will stop a bigger job.

The guard thresholds were not relaxed to make any of this pass.
