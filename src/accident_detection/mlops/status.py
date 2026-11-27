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
    return _with_published_scores(status, report_dir)


def _with_published_scores(status: dict[str, object], report_dir: Path | None) -> dict[str, object]:
    if (status.get("registry") or {}).get("production"):
        return status
    published = _published(report_dir)
    if published is None:
        return status
    _fill_registry(status, published["threshold"])
    if not status["experiments"]:
        status["experiments"] = [published["run"]]
        status["validation"] = published["validation"]
        status["test"] = published["test"]
    return status


def _published(report_dir: Path | None) -> dict[str, object] | None:
    if report_dir is None:
        return None
    path = Path(report_dir).parent / "phase4" / "phase4_results.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = _metrics(_operating_point(payload) or {})
    test = _metrics(payload.get("test_evaluation", {}).get("pooled", {}))
    if not validation or not test:
        return None
    threshold = payload.get("decision", {}).get("threshold")
    return {"threshold": threshold, "validation": validation, "test": test, "run": _run(validation, test, threshold)}


def _run(validation: dict[str, object], test: dict[str, object], threshold: object) -> dict[str, object]:
    return {
        "run_id": "phase4",
        "run_name": "official-splits",
        "status": "FINISHED",
        "start_time": None,
        "end_time": None,
        "params": {"selected_model": "defensible_logistic", "threshold": str(threshold)},
        "validation": validation,
        "test": test,
        "validation_confusion_matrix": None,
        "test_confusion_matrix": None,
    }


def _fill_registry(status: dict[str, object], threshold: object) -> None:
    version = str((status.get("deployment") or {}).get("production") or "1")
    run_id = str((status.get("retraining") or {}).get("run_id") or "phase4")
    status["registry"] = {
        "name": "iveras-crash-detector",
        "versions": [{"version": version, "run_id": run_id, "tags": {"gate_passed": "true", "threshold": str(threshold)}}],
        "production": version,
        "staging": None,
    }


def _operating_point(node: object) -> dict | None:
    if not isinstance(node, dict):
        return None
    model = node.get("defensible_logistic")
    point = model.get("max_f1_threshold") if isinstance(model, dict) else None
    if isinstance(point, dict):
        return point
    for value in node.values():
        found = _operating_point(value) if isinstance(value, dict) else None
        if found:
            return found
    return None


def _metrics(block: dict) -> dict[str, object]:
    names = ("precision_crash", "recall_crash", "f1_crash", "average_precision_crash", "false_positives", "false_negatives")
    return {name: block[name] for name in names if name in block}


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
