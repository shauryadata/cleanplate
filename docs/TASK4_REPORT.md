# Task 4 — Ground truth, then hair

Task 2 ended with a confident claim: the matte's softness is "an anti-aliased outline,
median 2 px deep, not interior hair transparency", and that was called the headline
failure. Task 4 built reference alpha to check that claim against. **It does not
survive contact with truth**, and that is the most useful thing in this report.

## Checkpoint 0 — the Tears of Steel archive ships its plates, green screens included

Time-boxed investigation, findings in [DECISIONS.md](DECISIONS.md) D2.

`media.xiph.org/tearsofsteel/` carries the complete Mango VFX pipeline output, **all
Creative Commons Attribution 3.0**, free, no account, no paywall — so the $0 rule was
never actually tested. What is there: 2 shots of Sony F65 4K raw `.mxf` (21 GB, 13 GB),
**81 shots of VFX plates** as 4K (~51 MB/frame) and 1920×1012 (~6.2 MB/frame) OpenEXR,
76 shots of cleaned plates, 148 shots of final frames, plus the graded film by torrent.
Blender Studio's subscription library was not needed and was not used.

I pulled one frame from every one of the 81 plate shots
(`outputs/_scout/tos_plates_survey.jpg`). **About 25 are green-screen setups**, and
`08_3a` is the same actor as our `hair` shot — long grey backlit hair — against clean
green, 847 frames. That is the hardest case in the project with keyable reference alpha
available. Plates for our three graded shots exist too (`01_2a` the bridge two-shot,
`04_2d`/`04_3e` the warm interior, `05_1c`/`05_3*` the canal walk), but those are
ungraded source, not alpha.

## Truth, in two tiers that are never conflated

| | Tier A | Tier B |
|---|---|---|
| What | 5 VideoMatte240K test clips composited over varied static and moving backgrounds | Chroma key on ToS plate `08_3a`, then composited over a ToS plate |
| Resolution | 1920×1080, 96 frames | 1920×1012, 96 frames |
| Alpha status | **exact by construction** — it is the number used to make the pixel | **keyed reference, not gospel** — carries the keyer's own errors |
| Licence | VM240K: commercial and non-commercial | CC BY 3.0 |

Every clip regenerates bit-for-bit from a frozen recipe (`scripts/build_truth.py`,
`truth/*/recipe.json`). Tier B's key thresholds are **derived from data, not guessed**:
measured greenness on frame 48 gives subject p99 = 0.051 and backing p1 = 0.417, so
`alpha_lo`/`alpha_hi` sit in that gap. The keyer also drops the green screen's tracking
markers by keeping only what is connected to the subject, and a garbage box excludes the
grey flag at frame right.

## Checkpoint 3 — metrics, validated before use

`cleanplate/accuracy.py`: MAD, MSE, gradient error, dtSSD, boundary-F, each with a
hair-bounding-box variant. `tests/test_accuracy.py` checks all of them on cases with
known answers — 23 assertions covering identity, hand-computed MAD/MSE values, dtype
invariance, monotonicity under increasing displacement, boundary-F tolerance behaviour,
and that Grad reacts to edge *profile* where MAD does not. **One assertion I wrote was
itself wrong** — it compared Grad and MAD magnitudes, which live on different scales —
and now tests relative sensitivity instead.

Two implementation fixes came out of running it at 1920×1080: boundary-F used ~18
iterations of binary dilation per 2 MP mask and dominated the benchmark's runtime; it is
now a single Euclidean distance transform, which is both faster and the correct disk
tolerance rather than an octagon.

## Two benchmark bugs, fixed before any number was believed

Both are subject ambiguity, not matte quality, and both are recorded in the clip's
`recipe.json` with the reason:

- **A2** is a woman *and* a child, touching — one connected blob with one interior peak.
  The default single-click oracle returns one person against a two-person reference:
  MAD 58.9, boundary-F 0.18.
- **A4**'s interior peak landed on the **notepad she is holding**. SAM 2 correctly
  returned the notepad (1.3 % of frame) against a whole-person reference (17.2 %):
  MAD 157, boundary-F 0.000. The green-screen key counted the held object as foreground;
  a click cannot know that.

Rather than tune a heuristic until it happened to split them — fitting the benchmark to
the pipeline — the clicks are pinned, documented, and given identically to every method.

## Baseline and experiments

Full tables, per clip, in [BENCH.md](BENCH.md). Hair region, mean over all 6 clips:

| Metric | binary | **baseline** | guided (5c) | vitmatte (5b) | matanyone2 (5d) | hairzoom (5a) | **hairzoom2 (5a+5d)** |
|---|---|---|---|---|---|---|---|
| MAD ↓ | 15.46 | 10.14 | 21.36 | 11.32 | 8.665 | 7.516 | **6.33** |
| MSE ↓ | 10.66 | 4.248 | 6.385 | 3.845 | 3.979 | 4.024 | **3.307** |
| Grad ↓ | 0.5711 | 0.2184 | 0.3573 | 0.1714 | 0.1935 | 0.1821 | **0.1456** |
| dtSSD ↓ | 5.948 | 3.672 | 4.223 | 4.318 | 3.495 | 3.19 | **2.947** |
| Boundary-F ↑ | 0.9205 | 0.9559 | 0.957 | **0.9702** | 0.9581 | 0.9468 | 0.9655 |
| s/frame | 0.131 | 0.190 | 0.222 | 0.286 | 0.201 | 0.685 | 0.557 |

Combining the two things that worked — the crop-and-zoom of 5a with the newer model of
5d — beats either alone on four of five metrics **and costs less than 5a on its own**
(0.557 vs 0.685 s/frame), because MatAnyone 2 is the faster model.

Whole frame, `matanyone2_960` wins **every** metric (MAD 4.234, MSE 3.172, Grad 0.0486,
dtSSD 1.978, BF 0.9625) at 0.201 s/frame against the baseline's 0.190 — essentially free.

### 5c, the cheap shot, is a clear negative
The guided filter is **worse than doing nothing** on hair (MAD 21.36 vs baseline 10.14)
while still costing time. It was worth testing and it is now dead.

### 5a did not test what it looked like it tested
Running *both* stages at 1920 scored MAD 80.6 whole-frame and 210.7 in the hair region —
apparently catastrophic. It is not a matting failure. **SAM 2's multi-mask head picks a
different granularity at a different input resolution**: on A3 the same relative click
selected the woman's *skin* (face, neck, chest — hair excluded) at 1920 where it selected
the whole person at 960. The matting stage was then handed the wrong object. This is the
same sensitivity as the near-coincident-click finding in Task 3, triggered by resolution
instead. Full-frame numbers for `fullres_1920` are in BENCH.md and should be read as a
segmentation result, not a matting one.

The de-confounded version — SAM 2 held at 960, matting at 1920 — **was aborted**. On this
18 GB machine it drove swap to 13.83 GB of 14.34 GB with ~58 MB of free pages and made no
visible progress on one clip in forty minutes. The brief's constraint is explicit, so it
was killed rather than left to thrash; the record is in
`outputs/_bench/fullmatte_1920_ABORTED.json`. Corroborating evidence that this is memory
and not slowness: `fullres_1920`'s matting stage timed at 33.44, 10.06, 4.47, 4.04, 4.48
and 2.03 s/frame across successive clips as memory freed up, against 0.13–0.20 s/frame at
960. **Full-resolution video matting of a 96-frame 1920×1080 shot is not viable on 18 GB
with this implementation.** Buying resolution only where it matters — the crop-and-zoom —
is the practical form of the same hypothesis, and it is the one that won.

A caveat on the memory column in BENCH.md: `peak_rss_mb` counts resident pages, so a
thrashing process reports a *small* number while the system pages out. It understates
pressure and should not be read as a safety margin.

## The hair verdict, quantified — and it reverses Task 2

Soft-pixel depth inside the hair box (distance from the nearest fully-opaque pixel), at
frame 48:

| clip | TRUTH median | baseline | hairzoom |
|---|---|---|---|
| A1 | 2.0 px | 3.6 | 2.0 |
| A2 | 2.2 px | 4.0 | 2.0 |
| A3 | 2.2 px | 4.5 | 2.8 |
| A4 | 2.0 px | 4.0 | 2.2 |
| A5 | 2.0 px | 4.0 | 2.0 |
| B1 (real hair) | 2.8 px | 5.0 | 3.2 |

(`hairzoom` column; `hairzoom2` tracks it closely.)

**Task 2's diagnosis was wrong.** It concluded the matte was "too hard — a 2 px outline
with no interior transparency". Measured against reference alpha, 2 px *is what the truth
looks like*, and the baseline was running at 4–5 px: **not too hard, too soft, and
displaced**. The winning method improves by pulling the edge back towards the truth's
sharpness and position, not by discovering strands. Task 2 measured against an intuition;
this measures against an answer, and that is the whole point of building truth.

**How much interior transparency was actually recovered? Very little, and honestly the
truth barely contains any to recover.** Tier A's VM240K alphas were themselves keyed from
green screen by the dataset authors and are close to solid silhouettes with textured
edges. Tier B has genuine wispy strands — visible in
`outputs/_scout/hair_B1_tos_greenscreen_hair_f048.jpg` — and there the improvement is the
*smallest* of any clip: hair-region MAD 12.41 → 9.97 for the winner, **−19.7 %**,
against −44 % on the synthetic A3. On the one clip with real backlit hair and a real
keyed reference, the gain is roughly half what the synthetic clips suggest — which is
exactly why both tiers exist.

That is the honest answer, and it also indicts the truth: a keyer-derived reference
cannot contain much more strand transparency than a keyer can produce. Settling this
properly needs a reference that is not keyed at all.

## The ToS shots: self-consistency could not have told us any of this

`docs/TOS_SELFCHECK.md` runs baseline, new default and winner over the three graded ToS
shots, where no reference alpha exists. On the `hair` shot the Task 2 stability metrics
are **flat**: mean flicker 0.297 / 0.304 / 0.301 %, consecutive IoU 0.955 / 0.956 /
0.956, softness 0.881 / 0.945 / 0.990 % of frame. Three methods whose hair-region
accuracy differs by 37 % are indistinguishable to self-consistency.

That is the case for RotoBench in one table. Stability metrics tell you a matte is not
boiling; they cannot tell you it is right.

## Winner and integration

**`hairzoom2_960` — 2× crop-and-zoom over the head, matted with MatAnyone 2.** Hair-region
MAD 10.14 → **6.33, −37.6 %**, plus the best MSE, Grad and dtSSD, at 0.557 s/frame.

Integrated in the app as two changes:
- **New default refine: MatAnyone 2.** It beat v1 on every whole-frame metric at
  essentially identical cost (0.201 vs 0.190 s/frame), so there is no reason to keep v1
  as the default.
- **"High quality (hair)" toggle: the crop-and-zoom pass**, labelled with its measured
  gain and cost, with the real per-frame figure landing in the timings table after a run.
  Measured live on the `hair` shot: 0.128 s/frame default, 0.253 s/frame with the toggle.

Losers keep their rows in BENCH.md — including the guided filter, which is worse than
doing nothing.

## Ranked for Task 5

1. **The truth is the bottleneck now, not the method.** Both tiers are keyer-derived, so
   neither contains much strand transparency, and the hair question cannot be settled
   against them. What is needed is a reference that was never keyed: CG hair rendered
   with true alpha, or a captured plate with a known matte. Until then "recover hair
   transparency" is not a measurable objective.
2. **Prompt sensitivity is a first-class defect.** Three separate incidents now: a
   duplicate click 1 px away collapsing the matte (Task 3), a corrective click
   redefining the object globally (Task 2), and resolution changing which object SAM 2
   selects (Task 4). The pipeline needs a prompt that is robust to these, or a UI that
   detects them. The Task 3 warnings cover one of the three.
3. **Full-resolution matting needs chunking.** It is not viable in one pass on 18 GB.
   Tiling with overlap, or streaming the memory bank to disk, would make 5a testable
   properly rather than aborted.
4. **A2 and A4 show the reference itself is contestable.** A held notepad, a second
   person — a green-screen key says "foreground", a click says "this object". RotoBench
   needs a stated convention for what the subject *is*, not just an alpha.
5. **dtSSD rewards being static.** A method that never moves scores well on temporal
   accuracy when the reference barely moves. It needs pairing with a motion-magnitude
   normaliser before it can be trusted alone.
6. **Only six clips, one of them keyed by me.** The set is too small to rank methods
   that finish within a few percent of each other, and A4 alone swings the mean.
