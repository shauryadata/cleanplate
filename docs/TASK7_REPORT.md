# Task 7 — the launch build

Everything the project can do, packaged into the things a stranger actually meets: a
reel, a one-click notebook, and a repo that reads as a product rather than a lab
notebook.

Manual time asked of Shaurya: one shoot (~20 min), two review stops, one Colab run, one
image upload. Everything else is automated and re-runnable from the repo.

---

## Checkpoint 0 — 21 GB moved off the boot disk

`datasets/tos_pro` held 21 GB of raw EXR — a *rebuild cache*, not an input: the benchmark
only ever reads `truth/P*/`. It is now on the Archive SSD at
`/Volumes/ARCHIVE/cleanplate/tos_pro`.

Copied first, verified (2,969 files, 22,990,790,227 bytes, matching SHA-1 on a spot
sample), and only then deleted. Boot disk went from **39 GB free (91%) to 61 GB (86%)**.

`./scripts/download.sh pro` now links the archive back when the drive is attached
(`CLEANPLATE_ARCHIVE` overrides the path) and re-fetches when it is not. Proven
end to end: rebuilding from the archived EXR reproduces the clips **bit-identically**
(P12's alpha SHA is unchanged).

## Checkpoint 1 — harmonize v0, and what the review kept

Two things, each built, rendered before/after, and judged by eye rather than by me.

**Colour and exposure.** One gain and offset for the whole shot — static on purpose,
because a per-frame match on a moving matte is a flicker generator. The review chose the
**full match at strength 0.5** (gain 0.875, offset +23/+12/+7) over a gentler cast-only
variant; both ship, the chosen one by default for image backdrops.

**Background track.** The walk plate pans **−1166 px in x and +91 in y** over 96 frames,
measured from background pixels only (Lucas–Kanade features outside the matte, median
displacement). Panning the new background by that much makes the comp stop reading as a
sticker. Kept, but **opt-in**: it needs a background with the resolution to survive the
pan, and it only helps when the camera actually moves.

Two bugs found by looking at the render rather than the number, both of which produced
plausible wrong answers:

- The grain estimator returned **0.00 for every image**. It took the MAD of
  image-minus-median-blur over the whole frame, and most of a frame is flat, where a
  median filter changes nothing. Immerkær's estimator measures 0.65 on the plate and 1.72
  on a real background — the grain match had been silently inert.
- The first review render used a background crop **beyond the panorama's bounds**, so half
  the plate was black. The colour match dutifully learned that the background was nearly
  black and tried to darken the subject by 20 levels. The crop is now found by searching
  for the largest black-free band that needs the least upscaling, and the pan pads only in
  the direction of travel, which halves the upscale.

What v0 does not do, and still looks wrong: nothing directional, no contact shadow, no
light wrap, no per-region control. Recorded in [DECISIONS.md](DECISIONS.md) D5.

## Checkpoint 2 — the double-role shot

[SHOOT_CARD.md](SHOOT_CARD.md) is the phone-readable card: locked camera, 4K landscape,
AE/AF locked, stabilisation off, floor marks, each performance on its own half of frame,
ten seconds of empty frame as the clean plate, three takes a side, two recordings so the
camera is touched twice.

<!-- STATUS:double_role -->
**Pending his footage.** The reel carries a labelled placeholder in the cold-open slot
until it lands; the shot and its breakdown layers drop in and both cuts re-render.
<!-- /STATUS:double_role -->

## Checkpoint 3 — the reel

[docs/reel.json](reel.json) is the shot list; `scripts/build_reel.py` renders each
segment and concatenates. The reel re-renders from the repo rather than existing only as
an export someone made once.

**50 s horizontal (1080p) and a 46 s vertical cut (1080×1920).** The brief asked for
60–90 s; review stop 2 asked for it tighter, and his call wins — noted here because it is
a deviation from the spec, not an accident.

Order: cold open → title → click-to-matte → hair, default against high-quality → the
Changes tab catching a retroactive edit → three removals → the RotoBench card with the
frozen-control row → end card with credits.

Three things fixed by watching it back:

- the hair segment was comped over **a personal photo with no clear licence** — fine on a
  laptop, not in a public reel. It now uses P01, the professional-truth green-screen shot,
  despilled as the pipeline would.
- the marker removal was three pixels wide at reel scale; it is now zoomed to where the
  removal actually changed pixels, found by differencing before and after.
- the vertical cut laid out from a fixed top, leaving the bottom half of a 1080×1920 frame
  empty. Image and caption are centred as one block now, with a heading above.

Music: **"DreamScape" by HoliznaCC0, CC0 1.0**, verified from the archive.org item's
`licenseurl`; 70 s committed (1.6 MB) so the reel is self-contained. Font: **DejaVu Sans**
(Bitstream Vera licence), already present via matplotlib. Both in
[THIRD_PARTY.md](../THIRD_PARTY.md).

## Checkpoint 4 — the Colab notebook

[notebooks/cleanplate_colab.ipynb](../notebooks/cleanplate_colab.ipynb) is a thin wrapper
around `scripts/colab_demo.py`, so the thing Colab runs is the thing that can be tested on
a laptop. Both paths verified locally before it was ever opened in a browser: the bundled
demo shot (96 frames, 103 s on MPS) and an uploaded clip via `--video`.

One manual step, cued in the first cell: **Runtime → Change runtime type → T4 GPU**, then
Run all. It checks the GPU, clones and pins, downloads weights and footage, cuts the demo
shot, shows frame 0 with a coordinate grid, produces a matte and a Mars comp, then takes
an uploaded clip with click coordinates.

Three reproducibility holes closed while writing it:

- SAM 2, MatAnyone and ProPainter were cloned from **moving default branches**. They are
  now pinned to the commits every published number was produced with, with a graceful
  fallback if a pin is unreachable. (Two of the SHAs I first wrote were invented by padding
  short hashes; corrected from the actual checkouts.)
- the notebook composited onto a background that `.gitignore` excluded, so it would have
  failed on a bare clone. The two Mars images (NASA/JPL, public domain, 0.6 MB) are now
  committed *and* re-fetchable via `download.sh backgrounds`.
- the `--video` path was broken by a wrong `extract_shot` signature, and the demo start
  time in the first draft was invented. Both found by running it.

<!-- STATUS:colab -->
**Pending his browser run**, which is the verification step; the notebook gets fixed until
that run passes clean.
<!-- /STATUS:colab -->

## Checkpoint 5 — the repo as a landing page

README rewritten top-down for someone who has never seen it: pitch line, hero GIF, then
the one table that justifies the benchmark (a frozen matte scores 1.0000 on
self-consistency and is the worst matte in the set), then how to try it, the app, removal,
the local-by-design guarantees, and what is still wrong.

- `docs/img/reel.gif` — the click-to-matte stretch, held under 2 MB (720p/12 fps was
  6.7 MB).
- `docs/img/social_preview.png` — 1280×640 over a real Mars comp. Its first version cropped
  the tops off alpha mattes and looked like clipped garbage.
- Repo topics set via `gh` (12). Every internal link checked, and checked again for links
  to files that are gitignored and would 404 on GitHub: none.
