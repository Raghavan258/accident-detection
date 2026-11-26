"""Load the git-tracked copy of Production Model v1."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

FROZEN_DIR = Path(__file__).resolve().parents[3] / "models" / "production_v1"


def load_frozen(root: Path | None = None) -> dict[str, object]:
    """Read the exported pipeline, feature list, and frozen threshold."""

    directory = FROZEN_DIR if root is None else root
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return {
        "model": joblib.load(directory / "model.joblib"),
        "version": str(metadata["version"]),
        "decision": {
            "threshold": float(metadata["threshold"]),
            "feature_names": [str(name) for name in metadata["feature_names"]],
        },
    }
