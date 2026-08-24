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

## Hair strand transparency cannot currently be scored

Both ground-truth tiers are keyer-derived, so neither contains much interior strand
transparency to recover. Until there is a reference that was never keyed (CG hair
rendered with true alpha, or a captured plate with a known matte), "recover hair
transparency" is not a measurable objective. See docs/TASK4_REPORT.md.
