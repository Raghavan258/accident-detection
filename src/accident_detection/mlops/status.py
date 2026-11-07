"""Assemble dashboard state from the registry and written reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from accident_detection.mlops import EXPERIMENT_NAME
from accident_detection.mlops.cloud import cloud_status
from accident_detection.mlops.registry import registry_history


def build_status(tracking_uri: str, report_dir: Path | None = None) -> dict[str, object]:
    """Read live MLflow state and any reports already written to disk."""

    mlflow.set_tracking_uri(tracking_uri)
    experiments = _experiments()
    latest = experiments[0] if experiments else {}
    status = {
        "registry": registry_history(tracking_uri),
        "experiments": experiments,
        "validation": latest.get("validation", {}),
        "test": latest.get("test", {}),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cloud": cloud_status(),
        "dataset": _read(report_dir, "dataset.json"),
        "quality": _read(report_dir, "quality.json"),
        "drift": _read(report_dir, "drift.json"),
        "deployment": _read(report_dir, "deployment.json"),
        "retraining": _read(report_dir, "retraining.json"),
        "investigation": _read(report_dir, "investigation.json"),
    }
    return status


def _experiments() -> list[dict[str, object]]:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []
    runs = client.search_runs([experiment.experiment_id], max_results=20, order_by=["start_time DESC"])
    return [_run_view(client, run) for run in runs]


def _run_view(client, run) -> dict[str, object]:
    metrics = dict(run.data.metrics)
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": metrics,
        "validation": _prefixed(metrics, "validation_"),
        "test": _prefixed(metrics, "test_"),
        "validation_confusion_matrix": _artifact(client, run.info.run_id, "validation_confusion_matrix.json"),
        "test_confusion_matrix": _artifact(client, run.info.run_id, "test_confusion_matrix.json"),
    }


def _prefixed(metrics: dict[str, object], prefix: str) -> dict[str, object]:
    return {key.removeprefix(prefix): value for key, value in metrics.items() if key.startswith(prefix)}


def _artifact(client, run_id: str, name: str):
    try:
        path = Path(client.download_artifacts(run_id, name))
    except Exception:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read(report_dir: Path | None, name: str) -> dict[str, object]:
    if report_dir is None:
        return {}
    path = Path(report_dir) / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
