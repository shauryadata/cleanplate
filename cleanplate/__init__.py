"""CleanPlate - open-source, fully local AI rotoscoping and cleanup for film.

The pipeline as importable functions. `src/*.py` are thin CLIs over this package,
and `app.py` is a local Gradio UI over the same functions, so the two can never
drift apart.
"""
__version__ = "0.3.0"

from . import compose, ingest, metrics, paths, refine, runs, session, track  # noqa: F401

__all__ = ["compose", "ingest", "metrics", "paths", "refine", "runs", "session",
           "track"]
