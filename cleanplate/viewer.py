"""Rendering helpers for the app: markers, preview modes, the correction chart.

Kept out of app.py so the UI file stays about wiring, and so these can be unit-checked.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .compose import checkerboard, over

KEEP = (34, 197, 94)          # emerald - "this is the subject"
EXCLUDE = (239, 68, 68)       # red     - "this is not"
MATTE_TINT = (255, 0, 200)


def draw_points(rgb: np.ndarray, positive, negative, radius: int = 7) -> np.ndarray:
    """Draw click markers. Ringed in black so they read on any footage."""
    im = Image.fromarray(rgb.copy())
    d = ImageDraw.Draw(im)
    for pts, colour in ((positive, KEEP), (negative, EXCLUDE)):
        for x, y in pts:
            d.ellipse([x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2],
                      outline=(0, 0, 0), width=3)
            d.ellipse([x - radius, y - radius, x + radius, y + radius],
                      fill=colour, outline=(255, 255, 255), width=2)
            if colour == EXCLUDE:      # a minus bar, so it reads without colour
                d.line([x - radius + 3, y, x + radius - 3, y], fill=(255, 255, 255),
                       width=2)
            else:
                d.line([x - radius + 3, y, x + radius - 3, y], fill=(255, 255, 255),
                       width=2)
                d.line([x, y - radius + 3, x, y + radius - 3], fill=(255, 255, 255),
                       width=2)
    return np.asarray(im)


def overlay(rgb: np.ndarray, alpha: np.ndarray, opacity: float = 0.45) -> np.ndarray:
    a = (alpha.astype(np.float32) / 255.0 if alpha.dtype != bool
         else alpha.astype(np.float32))
    a = a[..., None] * opacity
    tint = np.array(MATTE_TINT, dtype=np.float32)
    return np.clip(rgb.astype(np.float32) * (1 - a) + tint * a, 0, 255).astype(np.uint8)


def matte_view(alpha: np.ndarray) -> np.ndarray:
    a = (alpha.astype(np.uint8) * 255) if alpha.dtype == bool else alpha
    return np.dstack([a, a, a])


def checker_view(rgb: np.ndarray, alpha: np.ndarray, cell: int = 16) -> np.ndarray:
    h, w = alpha.shape[:2]
    return over(rgb, alpha, checkerboard((w, h), cell))


def iou_chart(iou: np.ndarray, correction_frame: int | None,
              threshold: float = 0.99, width: int = 900, height: int = 320) -> np.ndarray:
    """Per-frame IoU against the previous run, with the retroactive zone called out."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#171a21")
    ax.set_facecolor("#0f1115")
    x = np.arange(len(iou))
    changed = iou < threshold

    if correction_frame is not None and correction_frame > 0:
        ax.axvspan(-0.5, correction_frame - 0.5, color="#d97706", alpha=0.12, lw=0,
                   label="before the corrective click")
        ax.axvline(correction_frame, color="#d97706", ls="--", lw=1.6)
        ax.annotate(f"click @ f{correction_frame}", (correction_frame, 1.002),
                    color="#f0b04a", fontsize=9, ha="left", va="bottom",
                    xytext=(4, 0), textcoords="offset points")

    ax.plot(x, iou, color="#6ee7a8", lw=1.4, zorder=3)
    ax.fill_between(x, iou, 1.0, where=changed, color="#ef4444", alpha=0.35,
                    step=None, zorder=2)
    ax.scatter(x[changed], iou[changed], s=9, color="#ef4444", zorder=4)

    ax.set_ylim(min(0.9, float(iou.min()) - 0.01), 1.005)
    ax.set_xlim(-0.5, len(iou) - 0.5)
    ax.set_xlabel("frame", color="#8b93a7", fontsize=9)
    ax.set_ylabel("IoU vs previous run", color="#8b93a7", fontsize=9)
    ax.tick_params(colors="#8b93a7", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#262b36")
    ax.grid(True, color="#1f2430", lw=0.6)
    if correction_frame is not None and correction_frame > 0:
        ax.legend(loc="lower left", fontsize=8, facecolor="#171a21",
                  edgecolor="#262b36", labelcolor="#8b93a7")
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf
