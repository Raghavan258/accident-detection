"""Render start target. `uvicorn main:app` loads the FastAPI app.

Scoring checks the MLflow store first. When that database is missing, prediction
and the simulator load models/production_v1/ instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from accident_detection.mlops.api import app

__all__ = ["app"]
