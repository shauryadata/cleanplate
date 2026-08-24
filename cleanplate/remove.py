"""Removal: take a tracked matte, turn it into a hole, and fill the hole.

Wraps ProPainter (S-Lab License 1.0, NON-COMMERCIAL — cloned into gitignored vendor/,
never bundled; see THIRD_PARTY.md). It is driven as a subprocess rather than imported,
for two reasons: its inference script already handles flow, chunking and I/O correctly,
and a child process is something the memory guard can actually kill. Importing it would
leave a thrashing run inside our own interpreter with no clean way out.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .memguard import MemoryAbort, MemoryGuard
from .paths import ROOT, peak_rss_mb, rel

PROPAINTER = ROOT / "vendor" / "propainter"
WORK = ROOT / "outputs" / "_remove_work"


def available() -> tuple[bool, str]:
    if not PROPAINTER.is_dir():
        return False, ("ProPainter not present. Run ./scripts/download.sh propainter")
    w = PROPAINTER / "weights"
    need = ["ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth"]
    missing = [n for n in need if not (w / n).exists()]
    if missing:
        return False, f"ProPainter weights missing: {missing}. ./scripts/download.sh propainter"
    return True, ""


def make_hole(alpha: np.ndarray, dilate: int = 12, feather: int = 0) -> np.ndarray:
    """Matte -> removal hole. Binary uint8, 255 where the inpainter should paint.

    The hole is deliberately grown past the matte: a matte that is a pixel tight leaves
    a rim of the subject's own colour behind, which the inpainter then happily
    propagates into the fill. Erring outward costs a little more area and looks far
    better.
    """
    import cv2
    out = np.zeros(alpha.shape, np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2) \
        if dilate > 0 else None
    for i, a in enumerate(alpha):
        m = ((a > 12) * 255).astype(np.uint8)      # anything non-transparent counts
        if k is not None:
            m = cv2.dilate(m, k, iterations=1)
        if feather > 0:
            m = cv2.GaussianBlur(m, (0, 0), feather)
            m = ((m > 40) * 255).astype(np.uint8)
        out[i] = m
    return out


@dataclass
class RemoveResult:
    frames: np.ndarray                 # cleaned plate, uint8 (N, H, W, 3)
    holes: np.ndarray                  # the hole masks actually used
    stats: dict


def remove(frames_dir: Path, alpha: np.ndarray, dilate: int = 12,
           subvideo_length: int = 8, resize_ratio: float = 1.0, fp16: bool = True,
           neighbor_length: int = 10, ref_stride: int = 10,
           guard_label: str = "inpaint", progress=None,
           timeout_s: int = 7200) -> RemoveResult:
    """Fill the hole left by removing the matted subject.

    Defaults measured on this machine, not guessed. ProPainter's own defaults
    (subvideo_length 80, fp32) grew swap by 6 GB on twenty-four 960x400 frames and the
    guard killed it. fp16 with chunks of 8 runs the same clip at 2.17 s/frame with
    **zero** swap growth and 2.3 GB of headroom, so those are the defaults. Halving
    resolution as well costs 0.92 s/frame if a machine needs the extra room.
    """
    ok, why = available()
    if not ok:
        raise FileNotFoundError(why)

    frames_dir = Path(frames_dir)
    fps = sorted((p for p in frames_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                 key=lambda q: int(q.stem))
    if len(fps) != len(alpha):
        raise ValueError(f"{len(fps)} frames but {len(alpha)} alpha frames")

    holes = make_hole(alpha, dilate=dilate)
    stamp = f"{frames_dir.parent.name}_{int(time.time())}"
    work = WORK / stamp
    vdir, mdir, odir = work / "video", work / "mask", work / "out"
    for d in (vdir, mdir, odir):
        d.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(fps):
        shutil.copy(p, vdir / f"{i:05d}{p.suffix}")
        Image.fromarray(holes[i], mode="L").save(mdir / f"{i:05d}.png")

    cmd = [sys.executable, "inference_propainter.py",
           "--video", str(vdir), "--mask", str(mdir), "--output", str(odir),
           "--save_frames", "--subvideo_length", str(subvideo_length),
           "--neighbor_length", str(neighbor_length), "--ref_stride", str(ref_stride)]
    if resize_ratio != 1.0:
        cmd += ["--resize_ratio", str(resize_ratio)]
    if fp16:
        cmd += ["--fp16"]

    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1", TQDM_DISABLE="1")
    if progress:
        progress(0.05, f"ProPainter: {len(fps)} frames, chunks of {subvideo_length}")

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(PROPAINTER), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    killed = None
    with MemoryGuard(guard_label) as g:
        while proc.poll() is None:
            time.sleep(2.0)
            if g.aborted:
                killed = g.aborted
                proc.send_signal(signal.SIGKILL)
                break
            if time.perf_counter() - t0 > timeout_s:
                killed = f"exceeded the {timeout_s}s timeout"
                proc.send_signal(signal.SIGKILL)
                break
            if progress:
                progress(min(0.9, 0.05 + (time.perf_counter() - t0) / 120.0),
                         f"ProPainter running ({time.perf_counter() - t0:.0f}s)")
    out_txt = (proc.stdout.read() if proc.stdout else "") or ""
    wall = time.perf_counter() - t0

    if killed:
        shutil.rmtree(work, ignore_errors=True)
        raise MemoryAbort(f"removal aborted — {killed}. {g.report()}")
    if proc.returncode != 0:
        tail = "\n".join(out_txt.strip().splitlines()[-12:])
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"ProPainter exited {proc.returncode}:\n{tail}")

    made = sorted(odir.rglob("frames/*.png"))
    if len(made) != len(fps):
        tail = "\n".join(out_txt.strip().splitlines()[-12:])
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"ProPainter wrote {len(made)} frames for {len(fps)} inputs:\n{tail}")
    cleaned = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in made])
    shutil.rmtree(work, ignore_errors=True)

    stats = {"stage": "remove (ProPainter)",
             "licence": "S-Lab License 1.0, NON-COMMERCIAL. Not bundled.",
             "frames": len(fps), "dilate_px": dilate,
             "subvideo_length": subvideo_length, "resize_ratio": resize_ratio,
             "fp16": fp16, "neighbor_length": neighbor_length, "ref_stride": ref_stride,
             "total_s": round(wall, 2),
             "seconds_per_frame": round(wall / len(fps), 4),
             "hole_area_fraction_mean": round(float((holes > 127).mean()), 5),
             "peak_rss_mb": round(peak_rss_mb(), 1),
             "memory": g.stats()}
    if progress:
        progress(1.0, f"removed in {wall:.0f}s ({stats['seconds_per_frame']:.2f} s/frame)")
    return RemoveResult(cleaned, holes, stats)
