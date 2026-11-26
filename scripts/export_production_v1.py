"""Copy Production Model v1 out of the local MLflow store into models/production_v1/."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

from accident_detection.mlops import MODEL_NAME
from accident_detection.mlops.registry import _from_store


def main() -> None:
    """Export the current production alias. This does not refit the model."""

    uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlops/mlflow.db")
    bundle = _from_store(uri)
    root = Path(os.environ.get("IVERAS_EXPORT_DIR", "models/production_v1"))
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle["model"], root / "model.joblib")
    metadata = {
        "model_name": MODEL_NAME,
        "version": str(bundle["version"]),
        "threshold": float(bundle["decision"]["threshold"]),
        "feature_names": list(bundle["decision"]["feature_names"]),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(root), "version": metadata["version"], "threshold": metadata["threshold"]}))


if __name__ == "__main__":
    main()
