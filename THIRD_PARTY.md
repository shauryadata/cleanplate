# Third-Party Components

CleanPlate's own source code is MIT licensed (see [LICENSE](LICENSE)). It depends on,
and is demonstrated with, the third-party work listed below. Each item keeps its own
license; nothing here is relicensed by inclusion.

## Models and code

### SAM 2 (Segment Anything Model 2) — Meta AI / Facebook Research
- Source: https://github.com/facebookresearch/sam2
- **License: Apache License 2.0** — verified against `LICENSE` at the repository root
  (GitHub reports SPDX `Apache-2.0`). The repository additionally ships `LICENSE_cctorch`,
  a separate notice covering vendored third-party CUDA/torch extension code.
  The SAM 2.1 model checkpoints are distributed under the same Apache 2.0 terms.
- Used for: the video object segmentation / mask propagation core.
- Cloned into `vendor/sam2/` and weights into `checkpoints/` by `scripts/download.sh`.
  Neither is committed to this repository.
- Citation:
  > Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv:2408.00714, 2024.

### MatAnyone — Nanyang Technological University, S-Lab
- Source: https://github.com/pq-yang/MatAnyone
- **License: S-Lab License 1.0 — NON-COMMERCIAL USE ONLY.** Verified by reading
  `LICENSE` at the repository root: "Redistribution and use **for non-commercial
  purpose** in source and binary forms ... are permitted". Commercial use requires
  written permission from the authors. GitHub reports the licence as `NOASSERTION`
  because it is a custom licence, not an SPDX-recognised one.
- **This is why MatAnyone is not bundled.** CleanPlate's own code is MIT and must stay
  usable commercially. MatAnyone is therefore an *optional* stage: it is cloned into
  `vendor/matanyone/` (gitignored) by `scripts/download.sh`, driven through a thin
  adapter (`src/refine_matte.py`), and no MatAnyone code or weights enter this
  repository. CleanPlate runs without it and produces the binary SAM 2 matte.
  Anyone needing a commercially-usable soft matte should substitute a permissively
  licensed matting model — see `docs/DECISIONS.md` for the ViTMatte fallback.
- Used for: refining SAM 2's binary mask into a soft alpha with temporal propagation.
- Weights: `matanyone.pth` (135 MB), from the project's GitHub release v1.0.0, under the
  same S-Lab 1.0 terms. Not committed.
- Citation:
  > Yang et al., *MatAnyone: Stable Video Matting with Consistent Memory Propagation*,
  > CVPR 2025, arXiv:2501.14677.

### MatAnyone 2 — Nanyang Technological University, S-Lab
- Source: https://github.com/pq-yang/MatAnyone2
- Paper: *MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator*,
  CVPR 2026 Highlight, arXiv:2512.11782.
- **License: S-Lab License 1.0 — NON-COMMERCIAL USE ONLY**, verified from `LICENSE.txt`
  at the repository root ("Redistribution and use for non-commercial purpose..."),
  identical terms to MatAnyone v1. GitHub reports `NOASSERTION` because it is a custom
  licence.
- Handled exactly like v1: cloned into gitignored `vendor/matanyone2/`, driven through
  `cleanplate/refine.py`, never bundled, and CleanPlate runs without it.
- Weights: `matanyone2.pth` (135 MB), GitHub release v1.0.0, same terms. Not committed.
- Why it is here: it won the Task 4 whole-frame benchmark at essentially baseline cost.
  See docs/BENCH.md and docs/DECISIONS.md D3.

### ProPainter — Nanyang Technological University, S-Lab
- Source: https://github.com/sczhou/ProPainter
- Paper: Zhou et al., *ProPainter: Improving Propagation and Transformer for Video
  Inpainting*, ICCV 2023.
- **License: S-Lab License 1.0 — NON-COMMERCIAL USE ONLY**, verified from `LICENSE` at
  the repository root. Same family and same handling as MatAnyone: cloned into
  gitignored `vendor/propainter/`, driven as a subprocess by `cleanplate/remove.py`,
  never bundled. Without it CleanPlate simply has no Remove mode; the MIT core is
  unaffected.
- Weights: `ProPainter.pth` (150 MB), `recurrent_flow_completion.pth` (19 MB),
  `raft-things.pth` (20 MB), from GitHub release v0.1.0. Not committed.
- Chosen over DiffuEraser (Apache-2.0) on measured grounds — see docs/DECISIONS.md D4.
  DiffuEraser remains the commercially usable fallback if that is ever needed.

### ViTMatte — hustvl, via Hugging Face Transformers
- Model: `hustvl/vitmatte-small-composition-1k`
- **License: Apache-2.0** (per the Hugging Face model card metadata). This is the only
  matting model in the project that is *not* research-only, which is why the
  trimap→ViTMatte route stays documented as the commercially usable fallback.
- Paper: Yao et al., *ViTMatte: Boosting Image Matting with Pretrained Plain Vision
  Transformers*.
- Downloaded on demand by `transformers` into the Hugging Face cache; not committed.

### Hugging Face Transformers
- Source: https://github.com/huggingface/transformers
- License: Apache-2.0. Used only to run ViTMatte.

### torchvision ImageNet backbone weights (ResNet-50, ResNet-18)
- Downloaded automatically by MatAnyone on first run to `~/.cache/torch/hub/checkpoints/`
  (`resnet50-19c8e357.pth`, `resnet18-5c106cde.pth`; ~143 MB combined).
- Source: https://download.pytorch.org/models/ — part of torchvision.
- License: BSD-3-Clause (torchvision).
- Not committed; outside the repository tree entirely.

### PyTorch / torchvision
- Source: https://github.com/pytorch/pytorch
- License: BSD-3-Clause.

### OpenCV (opencv-python)
- Source: https://github.com/opencv/opencv-python
- License: Apache License 2.0 (OpenCV 4.5.0+).

### Pillow
- Source: https://github.com/python-pillow/Pillow
- License: MIT-CMU.

### NumPy
- Source: https://github.com/numpy/numpy
- License: BSD-3-Clause.

### Gradio
- Source: https://github.com/gradio-app/gradio
- License: Apache License 2.0.
- Used for the local web app (`app.py`). Pinned to `6.25.0` in `requirements.txt`:
  the API moves between majors (in 6.x `theme` and `css` moved from `Blocks()` to
  `launch()`, and `gr.Image` lost `show_download_button`).
- Note: Gradio collects analytics by default. CleanPlate disables it three ways -
  `analytics_enabled=False`, `GRADIO_ANALYTICS_ENABLED=False`, and never calling
  `share=True`. The Image component's default "share to Hugging Face Spaces
  Discussions" button is also removed.

### Playwright  (development only)
- Source: https://github.com/microsoft/playwright-python
- License: Apache License 2.0.
- Listed in `requirements-dev.txt`, not `requirements.txt`. Used only by
  `scripts/make_docs_assets.py` and `scripts/capture_retroactive_demo.py` to generate
  the README screenshots from the live app. Not needed to run CleanPlate.

### SciPy
- Source: https://github.com/scipy/scipy
- License: BSD-3-Clause.
- Used for connected-component analysis in `src/metrics.py`.

### Matplotlib
- Source: https://github.com/matplotlib/matplotlib
- License: PSF-based Matplotlib License (BSD-compatible).

### FFmpeg
- Source: https://ffmpeg.org
- License: LGPL-2.1+ / GPL-2.0+ depending on build configuration.
- Invoked as an external binary; not linked into or redistributed with CleanPlate.

## Datasets used for ground truth

### VideoMatte240K — University of Washington GRAIL (Background Matting V2)
- Source: https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets
- Paper: Lin et al., *Real-Time High-Resolution Background Matting*, CVPR 2021,
  arXiv:2012.07810. The BackgroundMattingV2 **code** is MIT.
- **Dataset terms, quoted from the project page: "All datasets are licensed for
  commercial and non-commercial purposes. We require you to cite our paper for
  acedemic works. For use in commercial products, please fill out this survey."**
  No form is required for non-commercial research use; no payment at any point.
- Contents used: the 5-clip **test** split of the HEVC package (foreground and alpha
  as paired mp4s, 3840x2160 / 4096x2304, 30fps). The 479 training clips are left
  inside the archive unused.
- Note on provenance: the alpha was extracted by the dataset authors from purchased
  green-screen stock with After Effects, and is redistributed as HEVC. CleanPlate uses
  it as *exact* truth regardless, because the decoded alpha is the number used to make
  the composite — see docs/DECISIONS.md D2.
- Not committed; `scripts/download.sh datasets` re-fetches it.

### Tears of Steel VFX plates (green screen)
- Source: https://media.xiph.org/tearsofsteel/tearsofsteel-footage-exr/
- **License: Creative Commons Attribution 3.0**, per the READMEs in the archive:
  "These are VFX plates from the mango open movie. (CC) Blender Foundation |
  mango.blender.org."
- Used: shot `08_3a`, 96 frames of `linear_hd` (1920x1012 half-float OpenEXR) — the
  same actor as our `hair` shot, on a green screen, for the Tier B keyed reference.
  From Task 6 also the plate windows of the twelve RotoBench Tier P clips (the frames
  each professional key was pulled from; see `truth/P*/recipe.json`).
- Not committed; `scripts/download.sh datasets` and `scripts/download.sh pro` re-fetch.

### Tears of Steel cleaned plates and mattes — RotoBench Tier P truth
- Source: https://media.xiph.org/tearsofsteel/tearsofsteel-cleaned-exr/
- **License: Creative Commons Attribution 3.0**, verified from that directory's own
  `README.txt` (2026-09-19): "All of the files may be reused and redistributed under
  the Creative Commons Attribution 3.0 license." Required attribution:
  **(CC) Blender Foundation | mango.blender.org**. Logos and trademarks are excluded
  from the licence; none are used.
- What it is, per the README: "cleaned plates and mattes used as input to the
  compositing pipeline". For character-key shots the frames are the despilled
  foreground premultiplied over black with the compositor's matte in a fourth channel;
  that channel is RotoBench's Tier P reference. A professional key, still a key.
- Used: the committed windows of 12 shots (08_3a, 04_1b, 04_5k, 09_1a, 08_4a, 04_3d,
  04_5n, 07_1c, 04_5g, 04_3b, 07_3c, 07_3b), 4K keys area-downscaled to 1920x1012.
- Not committed (about 15 GB of EXR); `scripts/download.sh pro` re-fetches and rebuilds.

## Footage

### Tears of Steel
Test footage used for development and the Task 1 demo:

> **(CC) Blender Foundation | mango.blender.org**

- Licensed under the Creative Commons Attribution 3.0 license (CC BY 3.0).
- Source used: https://media.xiph.org/tearsofsteel/tears_of_steel_1080p.webm
  (official Xiph.Org mirror of the Blender Foundation release; the Blender mirror at
  https://download.blender.org/demo/movies/ToS/ serves the same film as a `.zip`).
- No footage is committed to this repository. `scripts/download.sh` re-fetches it.

## Background plates

Any background images used in comp previews are recorded in
`outputs/<shot>/backgrounds/SOURCES.txt` with their origin and license at the time they
are fetched. Only public-domain or explicitly free-to-use images are used.

## Reel assets

### Music — "DreamScape" by HoliznaCC0
- Source: https://archive.org/details/holizna-cc-0-cosmic-waves
- **License: CC0 1.0 Universal** (public domain dedication), verified from the item's
  `licenseurl` metadata on archive.org: `creativecommons.org/publicdomain/zero/1.0/`.
  CC0 requires no attribution; the reel credits it anyway.
- Used: 70 seconds from 1:30, faded, at half volume, committed as
  `assets/reel_music.mp3` (1.6 MB) so the reel re-renders from the repo.

### Font — DejaVu Sans
- Ships with matplotlib (`mpl-data/fonts/ttf`), already a dependency; nothing extra
  downloaded.
- **License: Bitstream Vera Fonts Copyright** (free to use, redistribute and modify;
  DejaVu changes are public domain), verified from `LICENSE_DEJAVU` in that directory.
- Used for every title, caption and card in the reel.
