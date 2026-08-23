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


def is_built(shot: str) -> bool:
    d = SHOTS / shot / "frames"
    return d.is_dir() and any(d.glob("*.jpg"))


def list_shots(include_unbuilt: bool = True) -> list[str]:
    """Shots on disk.

    A fresh clone has no frames - they are gitignored - but it does have each shot's
    shot.json, which records the exact ffmpeg cut. Those count as shots the app can
    build on demand, so include them by default.
    """
    if not SHOTS.is_dir():
        return []
    out = []
    for d in sorted(SHOTS.iterdir()):
        if not d.is_dir() or d.name == "source":
            continue
        if is_built(d.name) or (include_unbuilt and (d / "shot.json").exists()):
            out.append(d.name)
    return out
