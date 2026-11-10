"""Lifecycle, serving, drift, and rollback tests. Metrics come from the fitted model."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from accident_detection.features import DEFENSIBLE_FEATURE_NAMES, FEATURE_NAMES
from accident_detection.mlops import DRIFT_SHARE_THRESHOLD, MIN_VALIDATION_AP, MIN_VALIDATION_RECALL
from accident_detection.mlops.api import create_app
from accident_detection.mlops.cloud import cloud_status
from accident_detection.mlops.drift import monitor_drift
from accident_detection.mlops.lifecycle import run_lifecycle
from accident_detection.mlops.quality import validate_frame
from accident_detection.mlops.registry import registry_history, rollback_production

ROOT = Path(__file__).resolve().parents[1]


def test_declared_gates_match_params():
    import yaml

    declared = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    assert declared["min_validation_average_precision"] == MIN_VALIDATION_AP
    assert declared["min_validation_recall"] == MIN_VALIDATION_RECALL
    assert declared["drift_share_threshold"] == DRIFT_SHARE_THRESHOLD


def test_schema_and_missing_drift_report(tmp_path: Path):
    from fastapi.testclient import TestClient

    from accident_detection.mlops.api import create_app

    client = TestClient(create_app(f"sqlite:///{tmp_path / 'mlflow.db'}", tmp_path))
    schema = client.get("/mlops/schema")
    assert schema.status_code == 200
    assert len(schema.json()["features"]) == 12
    assert client.get("/mlops/reports/drift").status_code == 404


def test_invalid_labels_fail_validation():
    frame = _frame(np.array([0, 1]))
    frame.loc[0, "label"] = 9
    report = validate_frame(frame)
    assert report["passed"] is False


def test_lifecycle_promotes_only_a_validated_model(tmp_path: Path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    first = run_lifecycle(_separable(), uri, tmp_path / "reports", "valid")
    assert first["gate"]["passed"] is True
    assert first["gate"]["used_test_labels"] is False
    assert first["deployment"]["promoted"] is True
    production = first["deployment"]["production"]
    second = run_lifecycle(_unlearnable(), uri, tmp_path / "reports2", "rejected")
    assert second["gate"]["passed"] is False
    assert second["deployment"]["promoted"] is False
    assert second["deployment"]["production"] == production


def test_rollback_restores_the_previous_validated_version(tmp_path: Path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    reports = tmp_path / "reports"
    first = run_lifecycle(_separable(), uri, reports, "v1")
    second = run_lifecycle(_separable(), uri, reports, "v2")
    restored = rollback_production(uri, first["deployment"]["version"])
    assert restored["production"] == first["deployment"]["version"]
    assert second["deployment"]["version"] != restored["production"]
    blocked = run_lifecycle(_unlearnable(), uri, tmp_path / "blocked", "blocked")
    assert blocked["gate"]["passed"] is False
    with pytest.raises(ValueError):
        rollback_production(uri, blocked["deployment"]["version"])
    assert registry_history(uri)["production"] == first["deployment"]["version"]


def test_prediction_api_returns_probability_and_latency(tmp_path: Path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_lifecycle(_separable(), uri, tmp_path / "reports", "serve")
    client = TestClient(create_app(uri, tmp_path / "reports"))
    status = client.get("/mlops/status")
    assert status.status_code == 200
    assert status.json()["registry"]["production"]
    page = client.get("/mlops")
    assert "Production version" in page.text
    assert "0.8244" not in page.text
    body = {name: 0.0 for name in DEFENSIBLE_FEATURE_NAMES}
    body["peak_horizontal_g"] = 8.0
    response = client.post("/predict", json={"features": body})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in {"crash", "non_crash"}
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["latency_ms"] >= 0.0
    window = client.post("/predict/window", json={"gsensor": [[8.0, 0.0, 1.0]], "gps_speed": [30.0], "vehiclesize": "Light"})
    assert window.status_code == 200


def test_drift_report_flags_a_shifted_feature(tmp_path: Path):
    frame = _separable()
    frame.loc[frame["split"] == "validation", "peak_horizontal_g"] += 50.0
    report = monitor_drift(frame, tmp_path / "drift.html", share_threshold=0.3)
    assert report["drift_detected"] is True
    assert "peak_horizontal_g" in report["drifted_features"]
    assert (tmp_path / "drift.html").stat().st_size > 0


def test_cloud_status_stays_disabled_without_credentials(monkeypatch):
    for name in ("GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT", "GCS_BUCKET"):
        monkeypatch.delenv(name, raising=False)
    status = cloud_status()
    assert status["enabled"] is False
    assert status["apis_called"] is False


def test_dvc_versions_a_processed_table(tmp_path: Path):
    table = tmp_path / "features.parquet"
    _separable().to_parquet(table, index=False)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([*_dvc(), "init", "--no-scm"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run([*_dvc(), "add", table.name], cwd=tmp_path, check=True, capture_output=True)
    assert (tmp_path / "features.parquet.dvc").exists()


def test_dockerfile_targets_the_inference_app():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "accident_detection.mlops.api:app" in text
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker is not installed; the image was not built")
    result = subprocess.run([docker, "version"], capture_output=True)
    if result.returncode != 0:
        pytest.skip("Docker is installed but the daemon is not available; the image was not built")


def _unlearnable() -> pd.DataFrame:
    frame = _separable()
    for name in FEATURE_NAMES:
        frame[name] = 0.0
    return frame


def _separable() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for split, count in (("train", 80), ("validation", 40), ("test", 40)):
        for index in range(count):
            crash = index % 2 == 0
            values = {name: float(rng.normal(0, 0.01)) for name in FEATURE_NAMES}
            values["peak_horizontal_g"] = 6.0 if crash else 0.2
            values["peak_magnitude_g"] = values["peak_horizontal_g"]
            rows.append({"split": split, "label": 0 if crash else 2, "event_id": len(rows), **values})
    return pd.DataFrame(rows)


def _frame(labels: np.ndarray) -> pd.DataFrame:
    frame = _separable().iloc[: len(labels)].copy()
    frame["label"] = labels
    return frame


def _dvc() -> list[str]:
    return [sys.executable, "-m", "dvc"]
