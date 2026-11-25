"""Render start target. `uvicorn main:app` loads the existing FastAPI app."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from accident_detection.mlops.api import app

__all__ = ["app"]
