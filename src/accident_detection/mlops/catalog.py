"""Read-only descriptions for the frontend. Training is not started here."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from accident_detection.features import DEFENSIBLE_FEATURE_NAMES
from accident_detection.mlops.cloud import cloud_status
from accident_detection.mlops.registry import registry_history

_CACHE: dict[str, tuple[float, dict[str, object]]] = {}
_CACHE_SECONDS = 30.0

FEATURE_FIELDS: tuple[dict[str, str], ...] = (
    {"name": "peak_horizontal_g", "label": "Peak horizontal acceleration", "unit": "g", "group": "acceleration"},
    {"name": "peak_magnitude_g", "label": "Peak acceleration magnitude", "unit": "g", "group": "acceleration"},
    {"name": "min_ax_g", "label": "Minimum longitudinal acceleration", "unit": "g", "group": "acceleration"},
    {"name": "max_ax_g", "label": "Maximum longitudinal acceleration", "unit": "g", "group": "acceleration"},
    {"name": "std_horizontal_g", "label": "Horizontal acceleration standard deviation", "unit": "g", "group": "acceleration"},
    {"name": "gps_start_kmh", "label": "GPS speed at window start", "unit": "km/h", "group": "gps"},
    {"name": "gps_end_kmh", "label": "GPS speed at window end", "unit": "km/h", "group": "gps"},
    {"name": "gps_min_kmh", "label": "Minimum GPS speed", "unit": "km/h", "group": "gps"},
    {"name": "gps_drop_kmh", "label": "GPS speed drop", "unit": "km/h", "group": "gps"},
    {"name": "size_light", "label": "Light vehicle", "unit": "0 or 1", "group": "size"},
    {"name": "size_medium", "label": "Medium vehicle", "unit": "0 or 1", "group": "size"},
    {"name": "size_heavy", "label": "Heavy vehicle", "unit": "0 or 1", "group": "size"},
)


def feature_schema() -> dict[str, object]:
    """Return the twelve features the production logistic model scores."""

    return {"features": list(FEATURE_FIELDS), "count": len(DEFENSIBLE_FEATURE_NAMES)}


def example_rows(path: Path) -> dict[str, object]:
    """Pick train rows nearest each class median peak. Missing files return no rows."""

    if not path.exists():
        return {"available": False, "path": str(path), "rows": []}
    frame = pd.read_parquet(path)
    rows = [_nearest(frame, 0, "Train crash nearest the crash median peak"), _nearest(frame, 2, "Train normal nearest the normal median peak")]
    return {"available": True, "path": str(path), "rows": [row for row in rows if row]}


def system_health(tracking_uri: str, report_dir: Path | None, db_path: Path) -> dict[str, object]:
    """Check services that this process can actually reach."""

    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "fastapi": {"ok": True},
        "mlflow": _mlflow(tracking_uri),
        "sqlite": _sqlite(db_path),
        "reports": _reports(report_dir),
        "model": _model(tracking_uri),
        "docker": _docker(),
        "cloud": cloud_status(),
        "predictions_recorded": False,
    }


def _nearest(frame: pd.DataFrame, label: int, title: str) -> dict[str, object] | None:
    subset = frame.loc[(frame["split"] == "train") & (frame["label"] == label)]
    if subset.empty:
        return None
    median = float(subset["peak_horizontal_g"].median())
    index = (subset["peak_horizontal_g"] - median).abs().idxmin()
    row = subset.loc[index]
    features = {name: None if pd.isna(row[name]) else float(row[name]) for name in DEFENSIBLE_FEATURE_NAMES}
    return {"title": title, "split": "train", "label": int(row["label"]), "event_id": int(row["event_id"]), "features": features}


def _mlflow(tracking_uri: str) -> dict[str, object]:
    try:
        history = registry_history(tracking_uri)
    except Exception as error:
        return {"ok": False, "detail": str(error)}
    return {"ok": True, "tracking_uri": tracking_uri, "production": history.get("production")}


def _sqlite(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"ok": False, "path": str(path), "detail": "database file is missing"}
    try:
        connection = sqlite3.connect(path)
        count = int(connection.execute("SELECT COUNT(*) FROM sensor_events").fetchone()[0])
        connection.close()
    except Exception as error:
        return {"ok": False, "path": str(path), "detail": str(error)}
    return {"ok": True, "path": str(path), "sensor_events": count}


def _reports(report_dir: Path | None) -> dict[str, object]:
    if report_dir is None:
        return {"ok": False, "detail": "IVERAS_REPORT_DIR is not set"}
    drift = report_dir / "drift.html"
    quality = report_dir / "quality.json"
    return {
        "ok": drift.exists() and quality.exists(),
        "directory": str(report_dir),
        "drift_html": drift.exists(),
        "drift_bytes": drift.stat().st_size if drift.exists() else None,
        "quality_json": quality.exists(),
    }


def _model(tracking_uri: str) -> dict[str, object]:
    return _cached(f"model:{tracking_uri}", lambda: _load_model(tracking_uri))


def _load_model(tracking_uri: str) -> dict[str, object]:
    try:
        from accident_detection.mlops.registry import load_production

        bundle = load_production(tracking_uri)
    except Exception as error:
        return {"loaded": False, "detail": str(error)}
    return {"loaded": True, "version": bundle["version"], "threshold": bundle["decision"]["threshold"]}


def _cached(key: str, build) -> dict[str, object]:
    cached = _CACHE.get(key)
    now = time.monotonic()
    if cached is not None and now - cached[0] < _CACHE_SECONDS:
        return cached[1]
    value = build()
    _CACHE[key] = (now, value)
    return value


def _docker() -> dict[str, object]:
    return _cached("docker", _docker_now)


def _docker_now() -> dict[str, object]:
    if shutil.which("docker") is None:
        return {"operational": False, "checked": True, "detail": "docker executable was not found"}
    try:
        result = subprocess.run(["docker", "version"], capture_output=True, timeout=5, check=False)
    except Exception as error:
        return {"operational": False, "checked": True, "detail": str(error)}
    detail = (result.stderr or result.stdout).decode(errors="replace")[:240]
    return {"operational": result.returncode == 0, "checked": True, "detail": detail}
