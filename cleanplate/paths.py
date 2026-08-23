"""Where things live. One place, so the CLI and the app never disagree."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SHOTS = ROOT / "shots"
OUTPUTS = ROOT / "outputs"
CHECKPOINTS = ROOT / "checkpoints"
VENDOR = ROOT / "vendor"

SOURCE_MOVIE = SHOTS / "source" / "tears_of_steel_1080p.webm"
SAM2_CKPT = CHECKPOINTS / "sam2.1_hiera_small.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
MATANYONE_CKPT = CHECKPOINTS / "matanyone.pth"


def shot_dir(shot: str) -> Path:
    return SHOTS / shot


def frames_dir(shot: str) -> Path:
    return SHOTS / shot / "frames"


def out_dir(name: str) -> Path:
    return OUTPUTS / name


def rel(p: Path | str) -> str:
    """Render a path relative to the repo root when possible, else absolute."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def list_shots() -> list[str]:
    if not SHOTS.is_dir():
        return []
    return sorted(d.name for d in SHOTS.iterdir()
                  if d.is_dir() and (d / "frames").is_dir()
                  and any((d / "frames").glob("*.jpg")))
