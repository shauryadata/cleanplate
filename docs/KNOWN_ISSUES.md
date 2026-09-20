# Known issues

Things that are measured, reproducible, and currently unfixed. Each entry says how it
was measured so the next attempt can tell whether it moved.

## Face-edge dropout on a low-contrast shadowed jaw

**Symptom.** On the `hair` shot, stretches of frames lose the front of the beard and
jaw from the matte. Reported by the first user test as "part of the face drops out".

**Where.** Frames **73–87**, region roughly x 412–567, y 53–295 — the shadowed left
edge of the face as he turns away from the practical lamp.

**Measurement.** Skin-toned plate pixels that lie inside the subject's filled silhouette
but are excluded by the matte. Motion-invariant, so the camera push-in does not confound
it. Median across the shot is 0 px, so these frames are genuine outliers.

**Attribution: segmentation, not matting.** The pixels are already missing from the
SAM 2 binary mask before MatAnyone runs:

| frame | excluded by SAM 2 binary | excluded after matting |
|---|---|---|
| f73 | 173 | 213 |
| f77 | 219 | 235 |
| f79 | 152 | 292 |
| f81 | 159 | 282 |

Matting amplifies it on some frames but does not cause it.

**What was tried, and did not work.**

| Attempt | SAM 2 stage, mean excluded skin | Final, mean | Frames > 150 px |
|---|---|---|---|
| Their run (1 click, MatAnyone v1) | 52.9 | 69.6 | 26 / 96 |
| Well-formed clicks (hair/brow/cheek/torso) + HQ | 52.4 | 61.4 | 18 / 96 |
| Above **plus** a well-formed corrective click on the beard at f73 | **54.2 (worse)** | 61.5 | 18 / 96 |

Adding clicks on the face moved the SAM 2 stage by 0.5 px of 52.9 — nothing. A
corrective click aimed straight at the dropout made the segmentation *slightly worse*,
consistent with the Task 2 finding that conditioning frames are global and adding one
perturbs the whole shot. The −12 % that is recoverable comes from the matting stage
(MatAnyone 2 plus the hair zoom), not from prompting.

**Conclusion.** This is a genuine SAM 2 limitation on a low-contrast edge in shadow, not
a user error and not a prompting mistake. It is not fixable at the prompt layer.

**What would plausibly fix it** (untested, for Task 6): a matting stage that is allowed
to *add* coverage rather than only refine what segmentation gave it — the trimap route
with a wide unknown band scored best on boundary-F in Task 4 and is the obvious
candidate; or a segmentation model with better low-contrast edge behaviour.

## Full-resolution matting is not viable on 18 GB

Carried from Task 4. A 96-frame 1920×1080 matting pass drove swap to 13.8 GB of 14.3 GB
and made no progress in forty minutes. `cleanplate/memguard.py` now aborts this class of
run instead of thrashing. Buying resolution only where it matters — the hair crop-and-zoom
— is the practical workaround and is what ships.

## Hair strand transparency is better referenced now, but still keyer-derived

**Updated in Task 6.** Both earlier tiers were keyed by me, so neither held much strand
transparency to recover. RotoBench Tier P uses the compositing team's own keys, which on
the same frames carry **1.42% soft pixels against our keyer's 0.60%**, and edges 5.10 px
deep against 3.61. That is a materially better reference — and it ranks the seven methods
differently enough (Spearman 0.75 against our key) that the in-house key should no longer
be used to decide anything.

It is still a key. A reference that was never keyed — CG hair rendered with true alpha,
or a captured plate with a known matte — remains the thing that would settle interior
strand transparency. See docs/ROTOBENCH.md.

## The coverage bug cannot be fixed at the matting stage

**Measured in Task 6** against professional truth, over seven clips. Splitting the
pixels the default matte misses inside the reference's opaque core:

- **85.8%** are regions the segmentation never selected on any nearby frame — a whole
  jacket, a bank of monitors. A trimap solver has no confident foreground to anchor to
  there and answers background.
- **14.2%** are intermittent, the class the user reported ("part of the face drops out").

Two repairs were built and scored against a rule fixed before they ran: a wide unknown
band over dropout-prone regions merged with max() (`cover_960`), and a wide band
everywhere (`cover2_960`). They recover **0.4%** and **1.9%** of the missing coverage and
cost +7.0% and +29.8% of band MAD. **Neither was integrated**; both stay in the standings
as losers. See docs/ROTOBENCH_RESULTS.md.

The lever is on the other side. On P02, where the oracle prompt put both clicks on skin
and SAM 2 dropped the actor's jacket for 96 frames, **one** extra click took whole-frame
MAD from 129.8 to 4.8.

## The hair-zoom pass does not fit on wide subjects

The HQ pass re-mattes the head box at 2x, so its cost scales with that box. In RotoBench
it completed at 1.0–1.1 MP of zoomed area and was killed by the memory guard twice at
~1.9 MP (a two-person shot whose "head box" spans the frame). It could not run on five
of twelve clips. The app now refuses it above 1.2 MP and says so rather than thrashing;
tiling the pass would remove the limit.

## GPU numerics moved under the code

After the macOS 27.0 (26A428) update, HEAD itself — unmodified — no longer reproduces
the Task 2 regression reference bit for bit: binary masks differ at min per-frame IoU
0.99998, with every recomputed metric identical. `scripts/regression_check.py` gained an
EQUIVALENT verdict for exactly that signature. To certify a code change, A/B the working
tree against HEAD on the same machine and require bit-identity; Task 6 did that for the
multi-object tracker (0 differing pixels).
