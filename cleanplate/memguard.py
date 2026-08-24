"""Abort a heavy stage before it thrashes the machine, and report pressure honestly.

Task 4 lost forty minutes to a run that drove swap to 13.8 GB of 14.3 GB on an 18 GB
Mac and made no progress. `peak_rss_mb` did not catch it and could not have: ru_maxrss
counts *resident* pages, so a process being paged out reports a small, reassuring
number while the system dies. This watches the two things that actually matter — free
physical memory and swap growth — and raises before the machine is unusable.

    with MemoryGuard("refine at 1920") as g:
        ...                       # g.check() inside the loop, or let the thread poll
    print(g.report())
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field


class MemoryAbort(RuntimeError):
    """Raised when a stage is stopped to protect the machine."""


def _vm() -> dict:
    """Available memory (MB) and swap used (MB), from the OS.

    'Pages free' is the wrong number on macOS: the kernel deliberately keeps almost
    nothing free and uses the rest as cache, so a healthy machine reports ~100 MB free
    and a naive guard aborts every run. What matters is memory that can be *reclaimed*
    (free + inactive + speculative + purgeable) and, above all, whether swap is
    growing — that is the signature of the Task 4 failure.
    """
    out = {"available_mb": None, "free_mb": None, "swap_used_mb": None,
           "swap_total_mb": None, "compressed_mb": None}
    try:
        vs = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        page = 4096
        vals = {}
        for line in vs.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split("bytes")[0].strip())
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip().rstrip(".")
                if v.isdigit():
                    vals[k.strip()] = int(v)
        mb = lambda k: vals.get(k, 0) * page / 1e6
        out["free_mb"] = mb("Pages free")
        out["available_mb"] = (mb("Pages free") + mb("Pages inactive")
                               + mb("Pages speculative") + mb("Pages purgeable"))
        out["compressed_mb"] = mb("Pages occupied by compressor")
    except Exception:
        pass
    try:
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                            text=True, timeout=5).stdout
        nums = [t for t in sw.replace("=", " ").split() if t.endswith("M")]
        if len(nums) >= 2:
            out["swap_total_mb"] = float(nums[0].rstrip("M"))
            out["swap_used_mb"] = float(nums[1].rstrip("M"))
    except Exception:
        pass
    return out


@dataclass
class MemoryGuard:
    """Poll memory during a stage; abort if it crosses the limits.

    Defaults tuned for an 18 GB machine, and set from observed behaviour rather than
    intuition. Two earlier versions of this guard were wrong in ways worth recording:

      1. Triggering on `Pages free` aborted a perfectly healthy run. macOS keeps almost
         nothing free by design and uses the rest as cache; 110 MB free is normal. The
         right quantity is *reclaimable* memory (free + inactive + speculative +
         purgeable), which read 4.7 GB at the same moment.
      2. Triggering on "swap is N% full" also aborted a healthy run, because macOS
         **grows the swap file on demand** - the total went 5120 -> 6144 MB mid-stage.
         A percentage of a moving denominator means nothing.

    What is left is what actually distinguished the Task 4 failure: swap growing by
    gigabytes *within one stage* while reclaimable memory collapses. A normal 960px
    stage on this machine grows swap by ~1.5 GB, so the limit sits well above that.
    """
    label: str = "stage"
    min_available_mb: float = 500.0
    max_swap_growth_mb: float = 6000.0
    poll_s: float = 2.0
    enabled: bool = True

    start: dict = field(default_factory=dict)
    worst_free_mb: float = field(default=float("inf"))
    peak_swap_mb: float = field(default=0.0)
    aborted: str | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> "MemoryGuard":
        self.start = _vm()
        self.worst_free_mb = self.start.get("available_mb") or float("inf")
        self.peak_swap_mb = self.start.get("swap_used_mb") or 0.0
        if self.enabled:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *a) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.poll_s + 1)

    def _poll(self) -> None:
        while not self._stop.wait(self.poll_s):
            self._sample()

    def _sample(self) -> None:
        v = _vm()
        free = v.get("available_mb")
        swap = v.get("swap_used_mb")
        tot = v.get("swap_total_mb")
        if free is not None:
            self.worst_free_mb = min(self.worst_free_mb, free)
        if swap is not None:
            self.peak_swap_mb = max(self.peak_swap_mb, swap)
            growth = swap - (self.start.get("swap_used_mb") or 0.0)
            if growth > self.max_swap_growth_mb:
                self.aborted = (f"swap grew {growth:.0f} MB during this stage "
                                f"(limit {self.max_swap_growth_mb:.0f} MB)")
        if free is not None and free < self.min_available_mb:
            self.aborted = (f"only {free:.0f} MB of reclaimable memory available "
                            f"(limit {self.min_available_mb:.0f} MB)")

    def check(self) -> None:
        """Call inside a loop. Raises MemoryAbort if a limit has been crossed."""
        if self.aborted:
            raise MemoryAbort(f"{self.label}: aborted to protect the machine — "
                              f"{self.aborted}. {self.report()}")

    def report(self) -> str:
        s0 = self.start.get("swap_used_mb")
        return (f"[{self.label}] lowest available memory "
                f"{self.worst_free_mb:.0f} MB; swap "
                f"{s0:.0f} -> {self.peak_swap_mb:.0f} MB"
                + (f" of {self.start['swap_total_mb']:.0f} MB"
                   if self.start.get("swap_total_mb") else "")
                + ". Note: peak RSS understates pressure, because a process being "
                  "paged out reports a small resident size.")

    def stats(self) -> dict:
        return {"label": self.label,
                "lowest_available_mb": round(self.worst_free_mb, 1),
                "swap_start_mb": self.start.get("swap_used_mb"),
                "swap_peak_mb": round(self.peak_swap_mb, 1),
                "swap_total_mb": self.start.get("swap_total_mb"),
                "aborted": self.aborted}
