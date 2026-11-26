"""The frozen bundle scores when no MLflow database is present."""

from fastapi.testclient import TestClient

from accident_detection.mlops.api import create_app
from accident_detection.mlops.registry import load_production
from accident_detection.synthetic_v2.telemetry import features_from_telemetry

MISSING = "sqlite:///missing/mlflow.db"
CRASH = {"peak_g": 3.8, "speed_before": 42.0, "speed_after": 0.0, "vehicle_size": "Medium"}


def test_missing_store_scores_the_frozen_crash() -> None:
    bundle = load_production(MISSING)
    features = features_from_telemetry(**CRASH)
    client = TestClient(create_app(MISSING))
    simulated = client.post("/api/v2/simulate", json={
        "locality": "Koramangala",
        "latitude": 12.935,
        "longitude": 77.625,
        "scenario_preset": "crash",
        **CRASH,
    })
    predicted = client.post("/predict", json={"features": features})

    assert bundle["version"] == "1"
    assert bundle["decision"]["threshold"] == 0.6152401422448208
    assert simulated.status_code == 200
    assert predicted.status_code == 200
    assert simulated.json()["probability"] == predicted.json()["probability"]
    assert abs(simulated.json()["probability"] - 0.9993410276278286) < 1e-9
    assert simulated.json()["prediction"] == "crash"
