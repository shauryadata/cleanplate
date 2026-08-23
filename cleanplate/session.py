"""Reading and writing the two sidecars that make a run reproducible.

shot.json   how the frames were cut out of the source movie
point.json  the clicks, stored per frame

Both are small, both are committed. Everything else in a run is derived.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .paths import shot_dir


@dataclass
class FramePrompt:
    """The clicks on one frame."""
    frame: int
    positive: list[tuple[int, int]] = field(default_factory=list)
    negative: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_clicks(self) -> int:
        return len(self.positive) + len(self.negative)

    def to_dict(self) -> dict:
        return {"frame": int(self.frame),
                "positive": [[int(x), int(y)] for x, y in self.positive],
                "negative": [[int(x), int(y)] for x, y in self.negative]}


class Prompt:
    """All clicks for a shot, keyed by frame."""

    def __init__(self, frames: dict[int, FramePrompt] | None = None):
        self.frames: dict[int, FramePrompt] = frames or {}

    # -- editing -------------------------------------------------------------
    def add(self, frame: int, x: int, y: int, positive: bool = True) -> None:
        e = self.frames.setdefault(frame, FramePrompt(frame))
        (e.positive if positive else e.negative).append((int(x), int(y)))

    def undo(self, frame: int) -> tuple[int, int] | None:
        """Remove the most recently added click on this frame."""
        e = self.frames.get(frame)
        if not e:
            return None
        for bucket in ("negative", "positive"):
            lst = getattr(e, bucket)
            if lst:
                p = lst.pop()
                if e.n_clicks == 0:
                    del self.frames[frame]
                return p
        return None

    def nearest(self, frame: int, x: int, y: int) -> tuple[float, str] | None:
        """Distance to the closest existing click on this frame, and its kind.

        Two near-coincident positive points do NOT reinforce each other in SAM 2 -
        they change which mask hypothesis wins. Measured on the walk shot, a second
        keep point one pixel from the first shrank the matte by 34% on average
        (minimum frame area 8.87% -> 1.96%). Callers should warn.
        """
        e = self.frames.get(frame)
        if not e:
            return None
        best = None
        for pts, kind in ((e.positive, "keep"), (e.negative, "exclude")):
            for px, py in pts:
                d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
                if best is None or d < best[0]:
                    best = (d, kind)
        return best

    def clear_frame(self, frame: int) -> None:
        self.frames.pop(frame, None)

    def clear(self) -> None:
        self.frames.clear()

    # -- inspection ----------------------------------------------------------
    @property
    def total_clicks(self) -> int:
        return sum(e.n_clicks for e in self.frames.values())

    @property
    def prompt_frames(self) -> list[int]:
        return sorted(self.frames)

    def negative_only_frames(self) -> list[int]:
        """Frames SAM 2 would read as 'the object is absent here'.

        A conditioning frame carrying only negative points yields an empty mask.
        Callers must refuse to run rather than silently produce a hole.
        """
        return sorted(f for f, e in self.frames.items()
                      if e.negative and not e.positive)

    def for_sam2(self) -> dict[int, dict[str, list]]:
        return {f: {"positive": list(e.positive), "negative": list(e.negative)}
                for f, e in sorted(self.frames.items())}

    # -- persistence ---------------------------------------------------------
    def to_dict(self, shot: str, image_size: tuple[int, int] | None = None) -> dict:
        d: dict = {"shot": shot}
        if image_size:
            d["image_size"] = [int(image_size[0]), int(image_size[1])]
        d["prompts"] = [self.frames[f].to_dict() for f in sorted(self.frames)]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Prompt":
        p = cls()
        if "prompts" in d:
            for e in d["prompts"]:
                f = int(e["frame"])
                fp = FramePrompt(f,
                                 [tuple(map(int, xy)) for xy in e.get("positive", [])],
                                 [tuple(map(int, xy)) for xy in e.get("negative", [])])
                if fp.n_clicks:
                    p.frames[f] = fp
        else:                                        # Task 1 flat format
            f = int(d.get("prompt_frame", 0))
            fp = FramePrompt(f,
                             [tuple(map(int, xy)) for xy in d.get("positive", [])],
                             [tuple(map(int, xy)) for xy in d.get("negative", [])])
            if fp.n_clicks:
                p.frames[f] = fp
        return p

    @classmethod
    def load(cls, shot: str) -> "Prompt":
        path = shot_dir(shot) / "point.json"
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))

    def save(self, shot: str, image_size: tuple[int, int] | None = None,
             note: str | None = None) -> Path:
        path = shot_dir(shot) / "point.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict(shot, image_size)
        if note:
            d["_note"] = note
        path.write_text(json.dumps(d, indent=2) + "\n")
        return path


def load_shot_json(shot: str) -> dict:
    path = shot_dir(shot) / "shot.json"
    return json.loads(path.read_text()) if path.exists() else {}
