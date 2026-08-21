#!/usr/bin/env bash
# Re-fetch everything that is deliberately NOT committed to this repository:
# the Tears of Steel footage, the SAM 2 source tree, and the SAM 2.1 checkpoints.
#
# Usage:  ./scripts/download.sh [footage|sam2|checkpoints|all]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FOOTAGE_URL="https://media.xiph.org/tearsofsteel/tears_of_steel_1080p.webm"
FOOTAGE_DST="shots/source/tears_of_steel_1080p.webm"
# Mirror, if Xiph is unreachable. Note: the Blender mirror serves a .zip.
FOOTAGE_MIRROR="https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip"

SAM2_REPO="https://github.com/facebookresearch/sam2.git"
SAM2_DIR="vendor/sam2"

# SAM 2.1 checkpoints (2024-09-28 release). hiera_small is what CleanPlate uses today.
CKPT_BASE="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
CKPT_NAME="sam2.1_hiera_small.pt"

# MatAnyone: optional soft-matte stage. S-Lab License 1.0, NON-COMMERCIAL only.
# Deliberately not vendored - see THIRD_PARTY.md and docs/DECISIONS.md.
MATANYONE_REPO="https://github.com/pq-yang/MatAnyone.git"
MATANYONE_DIR="vendor/matanyone"
MATANYONE_CKPT_URL="https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth"
MATANYONE_CKPT="checkpoints/matanyone.pth"

PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

fetch_footage() {
  mkdir -p shots/source
  if [ -s "$FOOTAGE_DST" ]; then
    log "footage already present: $FOOTAGE_DST ($(du -h "$FOOTAGE_DST" | cut -f1))"
    return
  fi
  log "downloading Tears of Steel 1080p (~545 MB) from Xiph"
  log "  (CC) Blender Foundation | mango.blender.org"
  curl -fL --retry 3 --progress-bar -o "$FOOTAGE_DST.part" "$FOOTAGE_URL"
  mv "$FOOTAGE_DST.part" "$FOOTAGE_DST"
  log "footage -> $FOOTAGE_DST"
  log "mirror, if ever needed: $FOOTAGE_MIRROR"
}

fetch_sam2() {
  mkdir -p vendor
  if [ -d "$SAM2_DIR/.git" ]; then
    log "SAM 2 already cloned at $SAM2_DIR"
  else
    log "cloning SAM 2"
    git clone --depth 1 "$SAM2_REPO" "$SAM2_DIR"
  fi
  # SAM 2's setup.py optionally builds a CUDA extension. Skip it on machines
  # without an NVIDIA toolchain (Apple Silicon, CPU-only) so the install succeeds.
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    export SAM2_BUILD_CUDA=0
    log "no nvidia-smi found -> SAM2_BUILD_CUDA=0"
  fi
  log "pip install -e $SAM2_DIR"
  "$PY" -m pip install -e "$SAM2_DIR"
}

fetch_checkpoints() {
  mkdir -p checkpoints
  if [ -s "checkpoints/$CKPT_NAME" ]; then
    log "checkpoint already present: checkpoints/$CKPT_NAME"
    return
  fi
  log "downloading $CKPT_NAME (~184 MB)"
  curl -fL --retry 3 --progress-bar -o "checkpoints/$CKPT_NAME.part" "$CKPT_BASE/$CKPT_NAME"
  mv "checkpoints/$CKPT_NAME.part" "checkpoints/$CKPT_NAME"
  log "checkpoint -> checkpoints/$CKPT_NAME"
}

fetch_matanyone() {
  mkdir -p vendor checkpoints
  log "MatAnyone is licensed S-Lab 1.0 - NON-COMMERCIAL USE ONLY. See THIRD_PARTY.md."
  if [ -d "$MATANYONE_DIR/.git" ]; then
    log "MatAnyone already cloned at $MATANYONE_DIR"
  else
    log "cloning MatAnyone"
    git clone --depth 1 "$MATANYONE_REPO" "$MATANYONE_DIR"
    # 70 MB of demo videos and stills we never use
    rm -rf "$MATANYONE_DIR/inputs" "$MATANYONE_DIR/assets"
  fi
  # Its pyproject demands PySide6, gradio, tensorboard, pycocotools, netifaces and
  # cchardet (abandoned; fails to build on py3.11+), none of which inference imports.
  log "installing MatAnyone without its dependency wall"
  "$PY" -m pip install --no-deps -e "$MATANYONE_DIR"
  "$PY" -m pip install einops safetensors huggingface_hub scipy imageio requests
  if [ -s "$MATANYONE_CKPT" ]; then
    log "MatAnyone checkpoint already present"
  else
    log "downloading matanyone.pth (~135 MB)"
    curl -fL --retry 3 --progress-bar -o "$MATANYONE_CKPT.part" "$MATANYONE_CKPT_URL"
    mv "$MATANYONE_CKPT.part" "$MATANYONE_CKPT"
  fi
  log "note: the first refine run also pulls ResNet-50/18 ImageNet weights (~143 MB)"
  log "      into ~/.cache/torch/hub/checkpoints/ (torchvision, BSD-3-Clause)."
}

case "${1:-all}" in
  footage)     fetch_footage ;;
  sam2)        fetch_sam2 ;;
  checkpoints) fetch_checkpoints ;;
  matanyone)   fetch_matanyone ;;
  all)         fetch_footage; fetch_sam2; fetch_checkpoints; fetch_matanyone ;;
  *)           echo "usage: $0 [footage|sam2|checkpoints|matanyone|all]" >&2; exit 2 ;;
esac

log "done."
