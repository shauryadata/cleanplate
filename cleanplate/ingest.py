"""Getting frames off disk, and cutting new shots out of a source movie."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import ROOT, SOURCE_MOVIE, frames_dir, shot_dir

FOOTAGE_CREDIT = "(CC) Blender Foundation | mango.blender.org"
FOOTAGE_URL = "https://media.xiph.org/tearsofsteel/tears_of_steel_1080p.webm"


def frame_paths(shot: str) -> list[Path]:
    """Frames in numeric order. SAM 2 needs integer stems, so we rely on that."""
    d = frames_dir(shot)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.jpg"), key=lambda p: int(p.stem))


def n_frames(shot: str) -> int:
    return len(frame_paths(shot))


def load_frame(shot: str, idx: int) -> np.ndarray:
    """One frame as RGB uint8."""
    ps = frame_paths(shot)
    if not ps:
        raise FileNotFoundError(f"no frames for shot {shot!r}")
    idx = max(0, min(idx, len(ps) - 1))
    return np.asarray(Image.open(ps[idx]).convert("RGB"))


def frame_size(shot: str) -> tuple[int, int]:
    ps = frame_paths(shot)
    if not ps:
        raise FileNotFoundError(f"no frames for shot {shot!r}")
    with Image.open(ps[0]) as im:
        return im.size


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}... failed:\n{p.stderr[-2000:]}")
    return p.stdout


def extract_shot(name: str, start: str | float, duration: float = 4.0,
                 source: Path = SOURCE_MOVIE, width: int = 960, fps: float = 24.0,
                 quality: int = 2, force: bool = False) -> dict:
    """Cut a shot out of the source movie into numbered JPEG frames.

    Writes shot.json alongside, recording the exact ffmpeg invocation, so the shot
    can be rebuilt from the source movie and nothing else.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(
            f"source movie not found: {source}\nRun ./scripts/download.sh footage")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH (brew install ffmpeg)")

    fdir = frames_dir(name)
    if fdir.exists() and any(fdir.iterdir()):
        if not force:
            raise FileExistsError(f"{fdir} already has frames; pass force=True")
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True, exist_ok=True)

    probe = json.loads(_run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "json", str(source)]))
    stream = probe["streams"][0]

    # -ss before -i seeks fast; no -copyts, so the output timeline restarts at 0
    # and frame 00000 really is the first frame of the shot.
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", str(start), "-i", str(source), "-t", str(duration),
           "-vf", f"fps={fps},scale={width}:-2:flags=lanczos",
           "-q:v", str(quality), "-start_number", "0", str(fdir / "%05d.jpg")]
    _run(cmd)

    frames = frame_paths(name)
    if not frames:
        raise RuntimeError("ffmpeg produced no frames - is start past the end?")
    with Image.open(frames[0]) as im:
        w, h = im.size

    meta = {
        "name": name,
        "source_movie": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_url": FOOTAGE_URL,
        "credit": FOOTAGE_CREDIT,
        "start": start,
        "duration_s": duration,
        "fps": fps,
        "frame_count": len(frames),
        "resolution": [w, h],
        "source_resolution": [stream.get("width"), stream.get("height")],
        "source_fps": stream.get("r_frame_rate"),
        "ffmpeg_cmd": " ".join(cmd),
    }
    (shot_dir(name) / "shot.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
