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
