# Task 2 — From cutout to key

Task 1 produced a binary cutout. Task 2 turns it into a soft key, kills the flicker at
its source, and puts numbers on both. Every figure below is measured from files on disk.

## Pipeline as it now stands

```
movie ─► extract_shot ─► pick_point ─► track_matte ─► refine_matte ─► export_rgba ─► comp_preview
                          (clicks)     (SAM 2, binary)  (MatAnyone,     (RGBA +        (comp, comps,
                                                          soft alpha)    despill)       v1_vs_v2)
                                                    └─────────► metrics ◄─────────┘
```

## Performance — Apple M3 Pro, MPS, zero fallback ops

| Stage | Model | s/frame | fps | MPS fallback ops |
|---|---|---|---|---|
| Track | SAM 2.1 hiera-small | 0.896 | 1.12 | **none** |
| Refine | MatAnyone v1.0.0 | 0.161 | 6.21 | **none** |
| Refine (dialogue) | MatAnyone v1.0.0 | 0.127 | 7.85 | **none** |
| Refine (hair) | MatAnyone v1.0.0 | 0.128 | 7.81 | **none** |

The matting stage is ~5.6x *faster* than the segmentation stage. Neither run needed a CPU
fallback, and neither needed a CPU rerun. Full pipeline on a 96-frame shot: about 110 s.

## Checkpoint 1 — killing the pavement bleed at the source

Task 1's flaw 2 was a wedge of pavement that latched onto the actor's lower back from
frame 65 and flickered on and off. `track_matte.py` gained `--at FRAME:X,Y:+|-` for
corrective clicks on any frame. One negative click at `67:321,379` (paired with three
positives re-describing the subject) fixed it.

| Measured over f064–073 | v1 | v2 | |
|---|---|---|---|
| mean frame-to-frame area change | 4.79 % | 1.59 % | **−67 %** |
| max frame-to-frame area change | 8.67 % | 2.89 % | **−67 %** |
| min consecutive IoU | 0.804 | 0.859 | **+6.8 %** |
| pavement-wedge pixels, f060–095 (total) | 21 831 | 1 481 | **−93 %** |

### Two SAM 2 behaviours worth knowing

Both cost a run to discover and are now guarded or documented in the code.

1. **A conditioning frame with only negative points blanks that frame.** SAM 2 reads it as
   "the object is absent here" and emits an empty mask. `track_matte.py` now exits with
   instructions instead of silently producing a hole.
2. **Conditioning frames are global, not causal.** `select_closest_cond_frames` puts every
   conditioning frame into the memory bank for *every* timestep, so a corrective prompt at
   t=67 changes the mask at t=24. An under-specified correction — one positive point that
   happened to land on the t-shirt — redefined the object as "shirt, not jacket" and
   dropped the jacket from frame 24 onward, a 30 % area loss across two thirds of the shot.
   A corrective frame must describe the whole subject, not just the part you are fixing.

## Checkpoint 2 — the matting stage

MatAnyone was chosen over RVM and a trimap→ViTMatte pipeline. Full reasoning, including
the licence problem and how it is contained, is in [DECISIONS.md](DECISIONS.md).

The headline: it is the only candidate that is both **mask-guided** (so "click *this*
actor" survives) and **temporally modelled** (so it does not reintroduce boil). It also
ships genuine Apple Silicon support — `get_default_device()` returns `mps` and
`safe_autocast()` deliberately skips autocast there.

**Licence: S-Lab 1.0, non-commercial only.** CleanPlate stays MIT by never bundling it:
MatAnyone is cloned to gitignored `vendor/`, driven through a thin adapter, and the
binary SAM 2 matte remains the no-strings default. See THIRD_PARTY.md.

## Checkpoint 3–4 — metrics, and three shots

Full tables in [METRICS.md](METRICS.md). Headlines for `walk`:

| | v1 binary | v2 refined | |
|---|---|---|---|
| Mean flicker | 1.763 % | 1.489 % | **−15.5 %** |
| Max flicker | 8.673 % | 5.563 % | **−35.9 %** |
| Min consecutive IoU | 0.804 | 0.855 | **+6.3 %** |
| Shape-normalised perimeter CV (boil) | 7.580 % | 7.111 % | **−6.2 %** |
| Soft pixels (0 < α < 255) | **0 %** | **0.791 %** of frame | from nothing |
| Distinct alpha values | 2 | 256 | |

`dialogue` and `hair` are more honest reading: refinement adds softness everywhere but
**does not** improve flicker on those two shots, and slightly worsens it on `dialogue`
(mean +5.7 %, perimeter CV +17.2 %). The temporal win on `walk` came mostly from the
corrective click, not from the matting model.

### A metric that was wrong, and is now fixed

The first version of `metrics.py` counted raw connected components and reported `hair`
regressing from 1 to 4 — apparently catastrophic fragmentation. Inspection showed the
extra "components" were **1–6 pixel specks** sitting just over the 127 threshold, against
a 76 000-pixel body. Meanwhile `dialogue` scored 2.8 components on the *binary* mask, which
turned out to be the bridge railing legitimately occluding Celia into head, torso and legs
(36 % and 13 % of the main component) — correct behaviour, not a defect.

`metrics.py` now reports **significant components** (≥1 % of the largest) as the headline
and specks separately. Under the corrected metric `hair` is 1 component in both versions.
A benchmark that ships a misleading metric is worse than no benchmark.

## Hair — an honest partial negative

This is the result the brief asked to be assessed plainly, and it is half a failure.

**What improved.** The silhouette boundary is genuinely soft: a graded band replaces
v1's hard staircase, 1 678 soft pixels in the f084 hairline crop alone. The face profile,
which v1 rendered as a smooth blob, is now resolved.

**What did not.** No individual hair strands are recovered. The plate shows separated,
backlit clumps of white hair with visible gaps through to the wall. v2 renders that whole
region as **fully opaque** with a soft *outline* around it.

Measured, so it is not a matter of opinion — distance of each soft pixel from the nearest
fully-opaque pixel:

| shot | median | p90 | p99 | max |
|---|---|---|---|---|
| walk | 2.0 px | 4.0 px | 15.2 px | 25.5 px |
| dialogue | 2.0 px | 3.2 px | 6.4 px | 18.4 px |
| hair | 2.0 px | 5.1 px | 10.0 px | 31.0 px |

A median of 2 px means the softness is an **anti-aliased outline**, not interior
transparency. Real hair matting would place fractional alpha many pixels deep inside the
hair mass. It does not. Flaw 6 from Task 1 is **not fixed** — it is improved at the
silhouette and untouched in the interior. That is benchmark data, and it is the strongest
argument for RotoBench existing.

## Checkpoint 5 — comp v2 and despill

`export_rgba.py` gained `--alpha` (source a soft matte) and `--despill green`, which caps
green at the mean of red and blue inside the fractional-alpha band, grown by 2 px. Only
the excess is removed, so genuinely green subject pixels keep their colour. On `walk` it
altered ~5 000 px per frame — consistent with a ±2 px band on a ~900 px perimeter.

The flat-orange stress test at f048 shows both effects at once: v1 has a hard staircase
*and* a bright green foliage fringe running down the whole profile; v2 has neither.

## Task 1 flaw list — status

| # | Flaw | Status |
|---|---|---|
| 1 | Alpha strictly binary | **Fixed.** 256 levels, ~0.8–1.1 % of frame fractional |
| 2 | Pavement bleed / flicker | **Fixed.** wedge −93 %, window flicker −67 % |
| 3 | Boundary boil | **Partial.** −6.2 % on walk; +0.4 % and +8.1 % on hair and dialogue |
| 4 | Chewed thin detail (yellow lining) | **Partial.** softer, still ragged at f070 |
| 5 | Green halo | **Fixed.** despill; visibly gone on the orange stress test |
| 6 | Hair as a solid block | **Not fixed.** soft outline only, median 2 px deep |
| 7 | Fingers lumped | **Partial.** soft edge, gaps between fingers still unresolved |
| 8 | Background does not move | Out of scope — Task 3 |

One Task 1 observation turned out to be **wrong**: the pale band along the jacket's back
edge was called pavement bleed. Inspecting the plate at 7x shows it is a cream hoodie
collar, and the yellow beside it is the jacket lining. Both are correctly matted. Only
the lower-back wedge was ever a real bleed.

## Ranked for Task 3

1. **Hair interior transparency.** The headline failure. Options: higher-resolution
   inference (MatAnyone downsamples internally), a trimap→ViTMatte pass on the hair
   region only, or an alpha-detail model. Needs a ground-truth strategy to score against.
2. **Temporal stability of the alpha itself.** The matting stage does not reduce boil on
   two of three shots. Measure alpha L1 between frames, and consider explicit smoothing.
3. **Camera solve** (Task 1 flaw 8). The comp is parallax-free and reads as a cutout on a
   still. The single biggest gap between "a matte" and "a shot".
4. **Despill beyond green.** Achromatic pavement spill is untouched; a general
   "suppress the sampled background colour" pass would cover more.
5. **Correction ergonomics.** Corrective clicks are global and easy to get wrong. The tool
   should show which frames a correction changed, and warn on large retroactive shifts.
6. **Ground truth for RotoBench.** Everything here is a self-consistency metric. Without
   reference mattes there is no accuracy number, only stability numbers.
