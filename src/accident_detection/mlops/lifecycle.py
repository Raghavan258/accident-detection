"""Run quality checks, training, registration, and drift monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from accident_detection.mlops.cloud import cloud_status
from accident_detection.mlops.drift import monitor_drift
from accident_detection.mlops.quality import validate_frame, write_validation
from accident_detection.mlops.registry import register_and_promote
from accident_detection.mlops.training import train_frame


def run_lifecycle(frame: pd.DataFrame, tracking_uri: str, report_dir: Path, run_name: str) -> dict[str, object]:
    """Validate, train, and promote only when the validation gate passes."""

    report_dir.mkdir(parents=True, exist_ok=True)
    quality = validate_frame(frame)
    write_validation(report_dir / "quality.json", quality)
    _write(report_dir / "dataset.json", {"rows": int(len(frame)), "columns": list(frame.columns)})
    if not quality["passed"]:
        return {"status": "blocked", "quality": quality}
    trained = train_frame(frame, tracking_uri, run_name)
    deployment = register_and_promote(
        tracking_uri,
        str(trained["model_uri"]),
        trained["run_id"],
        bool(trained["gate"]["passed"]),
        threshold=float(trained["threshold"]),
        feature_names=list(trained["feature_names"]),
    )
    drift = monitor_drift(frame, report_dir / "drift.html")
    _write(report_dir / "drift.json", {key: drift[key] for key in ("drift_detected", "drifted_features", "share_threshold", "evidently")})
    _write(report_dir / "deployment.json", deployment)
    _write(report_dir / "retraining.json", {"run_id": trained["run_id"], "gate": trained["gate"], "run_name": run_name})
    return {
        "status": "completed",
        "quality": quality,
        "gate": trained["gate"],
        "validation": trained["validation"],
        "test": trained["test"],
        "deployment": deployment,
        "drift": drift,
        "cloud": cloud_status(),
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
