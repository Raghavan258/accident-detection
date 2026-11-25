"""Simulator session tests. They mock scoring so CI does not need the MLflow store."""

from __future__ import annotations

import asyncio

import pandas as pd
from fastapi.testclient import TestClient

from accident_detection.mlops.api import create_app
from accident_detection.synthetic_v2 import v2_routes
from accident_detection.synthetic_v2.dispatch import interpret
from accident_detection.synthetic_v2.generate import generate_events
from accident_detection.synthetic_v2.live import LiveSession
from accident_detection.synthetic_v2.telemetry import features_from_telemetry, simulation_label


def test_telemetry_uses_the_twelve_production_features():
    features = features_from_telemetry(3.8, 48, 0, "Medium")
    assert features["peak_horizontal_g"] == 3.8
    assert features["gps_drop_kmh"] == 48
    assert features["size_medium"] == 1.0
    assert features["size_light"] == 0.0
    assert simulation_label("non_crash", "near_miss") == "near_miss"
    assert simulation_label("crash", "near_miss") == "crash"


def test_broadcast_reaches_a_subscriber():
    session = LiveSession(pd.DataFrame())
    queue: asyncio.Queue[str] = asyncio.Queue()
    session.subscribers.add(queue)
    asyncio.run(session.broadcast({"event_id": "SIM-1001"}))
    assert "SIM-1001" in queue.get_nowait()


def test_density_question_names_the_busiest_zone():
    session = LiveSession(generate_events())
    catalog = session.catalog("all", "all", False)
    result = interpret(catalog["events"], catalog["localities"], "Which locality has the highest crash density?")
    assert "Indiranagar" in result["reply"]
    assert result["focus_target"]["event_id"]
    assert "112" not in result["reply"] or "not a live dispatch" in result["reply"]


def test_simulate_stores_a_live_crash_and_the_helper_can_filter_it(monkeypatch, tmp_path):
    frame = generate_events().head(20)
    path = tmp_path / "events.parquet"
    frame.to_parquet(path)
    monkeypatch.setenv("IVERAS_SYNTHETIC", str(path))
    monkeypatch.setattr(v2_routes, "load_production", lambda uri: {"bundle": True})
    monkeypatch.setattr(v2_routes, "predict_from_features", lambda bundle, features: _crash())
    client = TestClient(create_app("sqlite:///unused"))
    created = client.post("/api/v2/simulate", json=_body())
    assert created.status_code == 200
    payload = created.json()
    assert payload["event_id"] == "SIM-1001"
    assert payload["alert_triggered"] is True
    assert payload["threshold"] == 0.6152
    listed = client.get("/api/v2/events", params={"label": "crash", "locality": "Koramangala", "include_live": True})
    assert "SIM-1001" in {row["event_id"] for row in listed.json()["events"]}
    chat = client.post("/api/v2/helper-chat", json={"message": "Show live simulated crashes"})
    assert chat.json()["focus_target"]["event_id"] == "SIM-1001"
    severe = client.post("/api/v2/helper-chat", json={"message": "Severe crashes (>3.5g) in Koramangala"})
    assert all(row["locality"] == "Koramangala" for row in severe.json()["matched_events"])


def test_a_point_outside_bengaluru_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "events.parquet"
    generate_events().head(5).to_parquet(path)
    monkeypatch.setenv("IVERAS_SYNTHETIC", str(path))
    client = TestClient(create_app("sqlite:///unused"))
    body = _body()
    body["latitude"] = 1.0
    response = client.post("/api/v2/simulate", json=body)
    assert response.status_code == 422


def test_schema_lists_twelve_model_columns_and_excludes_gyro(monkeypatch, tmp_path):
    _point_synthetic(monkeypatch, tmp_path)
    payload = TestClient(create_app("sqlite:///unused")).get("/api/v2/schema-ranges").json()
    assert len(payload["feature_names"]) == 12
    assert "peak_abs_gyro_deg_s" not in payload["feature_names"]
    assert payload["gyro"]["used_by_model"] is False
    severe = next(item for item in payload["presets"] if item["id"] == "crash")
    assert severe["features"]["peak_horizontal_g"] == 3.8
    assert severe["features"]["gps_drop_kmh"] == 48


def test_small_accidents_and_navigation_use_live_rows():
    session = LiveSession(generate_events())
    catalog = session.catalog("all", "all", True)
    small = interpret(catalog["events"], catalog["localities"], "Filter small accidents (1-2g)")
    assert small["matched_events"]
    assert all(1 <= float(row["peak_horizontal_g"]) <= 2.2 for row in small["matched_events"])
    session.add(_body(), {"prediction": "crash", "probability": 0.99, "threshold": 0.6152, "model_version": "1"})
    catalog = session.catalog("all", "all", True)
    route = interpret(catalog["events"], catalog["localities"], "Navigate to latest accident")
    assert route["action"] == "navigate"
    assert str(route["focus_target"]["event_id"]) == "SIM-1001"


def test_dispatch_ack_reaches_the_user_status(monkeypatch, tmp_path):
    _point_synthetic(monkeypatch, tmp_path)
    monkeypatch.setattr(v2_routes, "load_production", lambda uri: {"bundle": True})
    monkeypatch.setattr(v2_routes, "predict_from_features", lambda bundle, features: _crash())
    client = TestClient(create_app("sqlite:///unused"))
    created = client.post("/api/v2/simulate", json=_body())
    ack = client.post("/api/v2/dispatch-ack", json={"event_id": created.json()["event_id"], "action": "eta", "eta_min": 6})
    assert ack.status_code == 200
    assert ack.json()["message"] == "Ambulance En Route (ETA: 6 mins)"
    status = client.get("/api/v2/dispatch-status", params={"event_id": "SIM-1001"})
    assert status.json()["message"] == "Ambulance En Route (ETA: 6 mins)"


def test_log_odds_shift_points_toward_crash_when_peak_rises():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from accident_detection.synthetic_v2.explain import column_contributions

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])
    pipe.fit([[0.2], [0.4], [3.5], [4.2]], [0, 0, 1, 1])
    bundle = {"model": pipe, "decision": {"feature_names": ["peak_horizontal_g"]}}
    high = column_contributions(bundle, {"peak_horizontal_g": 4.0})
    low = column_contributions(bundle, {"peak_horizontal_g": 0.2})
    assert high[0]["direction"] == "toward_crash"
    assert low[0]["direction"] == "toward_normal"


def _point_synthetic(monkeypatch, tmp_path):
    path = tmp_path / "events.parquet"
    generate_events().head(20).to_parquet(path)
    monkeypatch.setenv("IVERAS_SYNTHETIC", str(path))


def _crash() -> dict[str, object]:
    return {"prediction": "crash", "probability": 0.99, "threshold": 0.6152, "model_version": "1"}


def _body() -> dict[str, object]:
    return {
        "locality": "Koramangala",
        "latitude": 12.935,
        "longitude": 77.625,
        "peak_g": 3.8,
        "speed_before": 48,
        "speed_after": 0,
        "vehicle_size": "Medium",
        "scenario_preset": "crash",
    }
