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

case "${1:-all}" in
  footage)     fetch_footage ;;
  sam2)        fetch_sam2 ;;
  checkpoints) fetch_checkpoints ;;
  all)         fetch_footage; fetch_sam2; fetch_checkpoints ;;
  *)           echo "usage: $0 [footage|sam2|checkpoints|all]" >&2; exit 2 ;;
esac

log "done."
