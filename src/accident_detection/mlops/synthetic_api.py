"""Serve the synthetic v2 map sample and score it with the production model."""

from __future__ import annotations

import json
from pathlib import Path

from accident_detection.mlops.service import predict_from_features

PUBLIC_FIELDS = (
    "synthetic_id",
    "synthetic",
    "scenario",
    "label",
    "latitude",
    "longitude",
    "locality",
    "occurred_at",
)


def load_catalog(path: Path) -> dict[str, object]:
    """Read the synthetic catalog from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def present_catalog(catalog: dict[str, object], tracking_uri: str) -> dict[str, object]:
    """Return map rows. Predictions come from the production model when it loads."""

    bundle = _bundle(tracking_uri)
    events = [_row(event, bundle) for event in catalog["events"]]
    return {
        "schema": catalog["schema"],
        "synthetic": True,
        "note": catalog["note"],
        "bounds": catalog["bounds"],
        "counts": catalog["counts"],
        "scored_with_production_model": bundle is not None,
        "events": events,
    }


def _bundle(tracking_uri: str):
    try:
        from accident_detection.mlops.registry import load_production

        return load_production(tracking_uri)
    except Exception:
        return None


def _row(event: dict[str, object], bundle) -> dict[str, object]:
    row = {field: event[field] for field in PUBLIC_FIELDS}
    features = event["features"]
    row["peak_horizontal_g"] = features["peak_horizontal_g"]
    row["gps_drop_kmh"] = features["gps_drop_kmh"]
    if bundle is None:
        row["prediction"] = None
        return row
    scored = predict_from_features(bundle, features)
    row["prediction"] = scored["prediction"]
    row["probability"] = scored["probability"]
    row["threshold"] = scored["threshold"]
    row["model_version"] = scored["model_version"]
    return row
