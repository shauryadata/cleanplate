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

### Matplotlib
- Source: https://github.com/matplotlib/matplotlib
- License: PSF-based Matplotlib License (BSD-compatible).

### FFmpeg
- Source: https://ffmpeg.org
- License: LGPL-2.1+ / GPL-2.0+ depending on build configuration.
- Invoked as an external binary; not linked into or redistributed with CleanPlate.

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
