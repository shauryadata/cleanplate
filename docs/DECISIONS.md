# Decisions

Running log of the choices that shape CleanPlate, with the evidence behind them.

---

## D1 — Matting stage: MatAnyone (Task 2, 2026-08-21)

**Decision.** Use **MatAnyone** (CVPR 2025, `pq-yang/MatAnyone`) as the mask-guided video
matting stage, taking SAM 2's binary mask as the first-frame anchor and producing a soft
0–255 alpha. Wire it in as an *optional, separately-licensed* component under
`vendor/matanyone/`, never vendored into CleanPlate's own MIT source.

### Candidates considered

| | MatAnyone | RVM (RobustVideoMatting) | SAM2 mask → trimap → ViTMatte + smoothing |
|---|---|---|---|
| Mask-guided | **yes** — first-frame mask is the target assignment | no — auto human matting, no target selection | yes, via a synthesised trimap |
| Temporal model | **yes** — consistent memory propagation | yes — recurrent | **no** — per-frame, needs a smoothing hack |
| Built for hair | yes, explicitly | moderate | yes, but per frame |
| Multi-subject shots | picks the clicked subject | mattes *all* humans | picks the clicked subject |
| Extra moving parts | one model | one model | trimap heuristic + matting model + temporal filter |
| Licence | S-Lab 1.0, **non-commercial** | GPL-3.0 | ViTMatte Apache-2.0 (but a 3-stage pipeline) |

### Why MatAnyone

1. **It is the only candidate that is both mask-guided and temporally modelled.** CleanPlate's
   whole premise is "click once on *this* subject". RVM has no notion of a chosen subject —
   in the `dialogue` shot with two actors it would matte both, which breaks the product.
   The trimap→ViTMatte path is target-aware but per-frame, so it would reintroduce exactly
   the boundary boil Task 1 flagged as flaw 3, and then need a temporal filter bolted on to
   remove it. MatAnyone's memory propagation is the mechanism we want, not a workaround.
2. **Mac feasibility is first-class, not accidental.** `matanyone/utils/device.py` ships a
   `get_default_device()` that returns `mps` on Apple Silicon, and a `safe_autocast()` that
   deliberately *skips* autocast on MPS (autocast is only applied for cuda/cpu). Verified by
   reading the source: `get_default_device()` returns `device(type='mps')` on this machine.
   Neither of the alternatives has Apple-specific handling in-tree.
3. **Cost fits the budget.** One 135 MB checkpoint, no CUDA extension to compile.

### Costs accepted, and how they are handled

- **Licence: S-Lab License 1.0 — non-commercial use only.** This is a research-only licence
  and it is *not* compatible with CleanPlate being MIT. Handled by keeping MatAnyone out of
  the repo entirely: it is cloned to `vendor/` (gitignored) by `scripts/download.sh`, used
  through a thin adapter (`src/refine_matte.py`), and recorded in `THIRD_PARTY.md` with the
  restriction stated plainly. CleanPlate's own code stays MIT and runs without it — the
  binary SAM 2 matte remains the no-strings default. If a commercial path is ever needed,
  the ViTMatte route (Apache-2.0) is the fallback, which is why it stays documented above.
- **Dependency wall.** `pyproject.toml` demands `PySide6`, `gradio`, `tensorboard`,
  `pycocotools`, `netifaces`, `cchardet` (abandoned; fails to build on Python 3.11+) and a
  git dependency — none of which the inference path imports. Handled with
  `pip install --no-deps -e vendor/matanyone` plus the six packages inference actually
  needs (`einops safetensors huggingface_hub scipy imageio requests`). pip prints
  unsatisfied-requirement warnings for the training-only packages; they are expected.
- **Module-level device global.** `matanyone/model/matanyone.py` binds
  `device = get_default_device()` at import time and `encode_image()` pushes `pixel_mean`
  and `pixel_std` to *that* global, not to the model's device. So `--device cpu` on a Mac
  where MPS is available would mix devices and crash. `src/refine_matte.py` rebinds that
  module global to the requested device before inference, and says so when it does.
- **First-frame-only anchoring.** MatAnyone takes one mask and propagates on its own memory,
  so the Task 2 corrective click at frame 67 does not reach it directly. `refine_matte.py`
  therefore exposes `--reanchor FRAME`, which re-injects the SAM 2 mask mid-sequence
  (`InferenceCore.step` accepts a mask on any frame). Used only if measurement shows the
  bleed returning.

### Rejected, with reasons

- **RVM** — no target selection, so it cannot express "this actor". GPL-3.0 is also a
  stronger copyleft than an MIT project should take a dependency on.
- **SAM2 → trimap → ViTMatte + temporal smoothing** — three components to tune instead of
  one, per-frame by construction, and the trimap band width becomes a hand-tuned parameter
  per shot. Kept on the shelf as the permissively-licensed fallback.

---

## D2 — Ground truth: two tiers, both free (Task 4, 2026-08-23)

### Checkpoint 0 finding: Tears of Steel ships its VFX plates, and many are green screen

**The open production archive is real, complete, and free.** `media.xiph.org/tearsofsteel/`
carries the full Mango VFX pipeline output, all under **Creative Commons Attribution 3.0**
per the READMEs in each tree ("These are VFX plates from the mango open movie", "(CC)
Blender Foundation | mango.blender.org"). Nothing is paywalled and nothing needs an
account. What exists:

| Tree | Contents | Access |
|---|---|---|
| `raw/` | 2 shots as Sony F65 4K raw `.mxf` — 21 GB and 13 GB | free, huge |
| `linear-exr/` | 2 shots, 4K linear OpenEXR decoded from the raw | free |
| `tearsofsteel-footage-exr/` | **81 shots** of VFX plates, each with `linear/` (4K, ~51 MB/frame) and `linear_hd/` (1920x1012, ~6.2 MB/frame) | free |
| `tearsofsteel-cleaned-exr/` | 76 shots, plates with rig/marker cleanup | free |
| `tearsofsteel-frames-exr/` | 148 shots of final frames | free |
| `tearsofsteel-1080-png/`, `-4k-tiff/` | final graded frames, via torrent | free |

Blender Studio's subscription library was **not needed** and was not used — everything
above is on the open Xiph mirror. No paywall was encountered, so the $0 rule was never
tested.

**Surveying all 81 plate shots** (one frame each, `outputs/_scout/tos_plates_survey.jpg`)
found roughly 25 green-screen setups. Critically, **`08_3a` is the same actor as our
`hair` shot — long grey backlit hair — standing against a clean green screen**, 847
frames at 1920x1012. That is the hardest case in the project with keyable reference
alpha available.

Plates for our three graded shots also exist: `01_2a` is the bridge two-shot (`dialogue`),
`04_2d`/`04_3e` the warm interior (`hair`), `05_1c`/`05_3*` the canal walk (`walk`). Those
are ungraded source, not alpha, so they give resolution but not truth.

### Decision

Build **two truth tiers**, and never conflate them:

- **Tier A — synthetic, exact.** VideoMatte240K foreground+alpha clips composited over
  known backgrounds by a seeded script. The alpha is ground truth *by construction*:
  it is the input to the composite. Used to validate the metrics and to give an
  unarguable accuracy number.
- **Tier B — keyed reference, not gospel.** A scripted chroma key on ToS `08_3a`. This
  is real footage with real hair, at 1920 width, of the same actor as our hair shot — but
  the reference alpha is produced by a keyer, so it carries the keyer's own errors. Every
  table that uses it is labelled accordingly.

Tier B is the more *relevant* hair test; Tier A is the more *trustworthy* number. Reporting
both, separately, is the point.

---

## D3 — Hair experiments: what was tried and why (Task 4, 2026-08-23)

Task 2 left one headline failure: the matte's softness is a ~2 px anti-aliased outline,
not interior hair transparency. Five things were tried against measured truth. All five
share the same oracle prompt, the same clips and the same metrics, so they differ only
in what they compute.

| # | Method | Hypothesis being tested |
|---|---|---|
| a | `fullres_1920` | The detail is lost to **resolution**: MatAnyone downsamples internally, so running the whole pipeline at 1920 rather than 960 should recover strands for free. |
| a | `hairzoom_960` | Same hypothesis, cheaper: keep the pipeline at 960 but re-run the matting stage on a **2x magnified crop of the head only**, feathered back in. Pays for resolution only where it matters. |
| b | `vitmatte_960` | The detail is lost to the **model class**: a dedicated image matter given a trimap can solve strands that a segmentation-derived video matter cannot. SAM 2 mask -> trimap with a wide unknown band at the hair -> per-frame ViTMatte on the head patch -> temporal smoothing (ViTMatte has no memory at all, so without it the boil comes straight back) -> blended into the MatAnyone alpha inside the unknown band. |
| c | `guided_960` | The detail is already **in the plate**: a guided filter using the greyscale frame as guide should pull the alpha onto the image's own edges for almost no compute. The cheap-shot hypothesis, included precisely because it might have worked. |
| d | `matanyone2_960` | The detail is lost to the **model generation**: MatAnyone 2 (CVPR 2026 Highlight, same authors) is explicitly about "avoiding segmentation-like boundaries". Chosen over VideoMaMa on evidence: same mask-guided interface as v1, an identical `InferenceCore.step()` API so it is a genuine drop-in, first-class MPS support in-tree, and one 135 MB checkpoint — where VideoMaMa is a diffusion-prior model whose cost on an 18 GB Mac is unproven. One candidate, per the brief. |

Licence note: MatAnyone 2 is **S-Lab License 1.0, non-commercial**, exactly like v1, and
is handled the same way — cloned to gitignored `vendor/`, never bundled. ViTMatte
(`hustvl/vitmatte-small-composition-1k`) is **Apache-2.0**, which makes the trimap route
the commercially usable option if that ever matters; that was the fallback flagged in D1.

---

## D4 — Video inpainting for removal: ProPainter (Task 5, 2026-08-24)

**Decision.** Use **ProPainter** (ICCV 2023, `sczhou/ProPainter`) as the removal stage.

### Candidates

| | ProPainter | DiffuEraser | E2FGVI |
|---|---|---|---|
| Licence | **S-Lab 1.0, non-commercial** | **Apache-2.0** | NOASSERTION |
| Approach | flow-guided propagation + transformer | diffusion, refines a ProPainter-style prior | flow-guided |
| Memory, reported | 8 GB fp16 / 13 GB fp32 at 720x480x80f; 25 GB at 1280x720 | ~12 GB at 1280x720 | — |
| Chunking | **yes** — `--subvideo_length`, default 80 | via resolution only | limited |
| Apple Silicon | **yes, in-tree**: `model/misc.py:get_device()` returns `mps` and is *preferred over CUDA* | not stated | not stated |
| Extra weights | 3 files, ~150 MB total | Stable Diffusion stack on top | 1 file |
| Last pushed | 2025-02 | 2025-04 | 2023-04 |

### Why ProPainter, on evidence

1. **It is the only candidate with MPS support written into the repo.** `get_device()`
   checks `torch.backends.mps.is_available()` *before* CUDA. Neither alternative mentions
   Apple Silicon at all, and Task 4 showed that "it will probably work on MPS" is not a
   safe assumption on an 18 GB machine.
2. **It has the memory controls this project needs.** `--subvideo_length` chunks a long
   shot, `--resize_ratio` trades resolution for memory, `--fp16` halves it. Our target —
   96 frames at 960x400 — is 384k pixels/frame against the 345k of the reported
   720x480 case, so the reported 8–13 GB is the right ballpark, and chunking brings it
   under control. That maps directly onto the new memory-guard requirement.
3. **DiffuEraser is a refiner, not a replacement.** It produces better completeness in
   its own paper, but it works by improving a propagation-based prior, and it drags in a
   Stable Diffusion sampling loop. A diffusion denoiser over 96 frames on MPS is exactly
   the unproven, memory-hungry shape that cost forty minutes in Task 4. Its Apache-2.0
   licence is a genuine advantage and the reason it stays on the shelf as the
   commercially usable option — same role ViTMatte plays for matting.

### Cost accepted

**S-Lab License 1.0, non-commercial**, like MatAnyone. Handled identically: cloned into
gitignored `vendor/propainter/`, driven through a thin adapter, never bundled, recorded
in THIRD_PARTY.md. CleanPlate's MIT core does not depend on it — without ProPainter the
app simply has no Remove mode.

## D5 — Harmonize v0: what the review kept, and what it cannot do

**Reviewed by eye on the walk-over-Mars comp, Task 7, with before/after renders**
(`scripts/harmonize_review.py`, outputs in `outputs/_harmonize/`).

| Option | Measured | Verdict |
|---|---|---|
| Colour, **full match at strength 0.5** | gain 0.875, offset +23/+12/+7 (his mean [76, 92, 93] against the plate's [102, 92, 84]) | **Kept, enabled by default** for image backdrops |
| Colour, cast only at 0.4 | gain 1.10/0.98/0.94, no level change | Implemented, not the default (`--harmonize-mode cast`) |
| Grain match | plate 0.65 vs upscaled panorama 0.45 → **+0.00 added** | Kept, inert on this comp; it only fires when the background is grainier than the cut-out, which is the case for real plates |
| Background 2D track | plate pans **−1166 px in x, +91 in y** over 96 frames, measured from background pixels only | **Kept, opt-in** (`--track-bg`): used for the reel's Mars shot, off by default because it needs a background with the resolution to survive the pan |

Both are global and static for the whole shot, deliberately: a per-frame match on a
moving matte is a flicker generator.

**What harmonize v0 does not do**, and what will still look wrong in the reel: nothing
directional (a subject lit from the left dropped into a plate lit from the right stays
wrong), no contact shadow, no light wrap, no per-region control — a face and a dark
jacket get the same gain, which is why the strength is a half and not a one.

Two bugs found while building it, both worth recording because both produced
plausible-looking wrong answers:

- The grain estimator (MAD of image minus a median blur) returned **0.00 for every
  image**, because most of a frame is flat and a median filter changes nothing there.
  Replaced with Immerkaer's estimator, which measures 0.65 on the plate and 1.72 on a
  real background.
- The first review render used a background crop beyond the panorama's bounds, so half
  the "plate" was black — and the colour match dutifully learned that the background
  was nearly black and tried to darken the subject by 20 levels. The crop is now chosen
  by searching for the largest black-free band that needs the least upscaling.
