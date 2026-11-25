"""In-memory simulated events and SSE subscriber queues."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pandas as pd

from accident_detection.synthetic_v2.hubs import nearest_hub
from accident_detection.synthetic_v2.query import crash_density
from accident_detection.synthetic_v2.telemetry import simulation_label

FIRST_SIM_ID = 1001


class LiveSession:
    """Baseline parquet plus events created during this process."""

    def __init__(self, baseline: pd.DataFrame) -> None:
        self.baseline = baseline
        self.live_events: list[dict[str, object]] = []
        self.subscribers: set[asyncio.Queue[str]] = set()
        self.updates: dict[str, dict[str, object]] = {}
        self._next_id = FIRST_SIM_ID

    def add(self, body: dict[str, object], scored: dict[str, object]) -> dict[str, object]:
        """Store one scored simulation and say whether helpers should hear it."""

        event = _event(self._next_id, body, scored)
        self._next_id += 1
        self.live_events.append(event)
        return event

    def acknowledge(self, event_id: str, action: str, eta_min: float | None) -> dict[str, object] | None:
        """Record a responder status for one live simulation."""

        if not any(row["event_id"] == event_id for row in self.live_events):
            return None
        update = {
            "kind": "dispatch",
            "event_id": event_id,
            "action": action,
            "message": _status_message(action, eta_min),
            "eta_min": eta_min,
            "synthetic": True,
        }
        self.updates[event_id] = update
        return update

    async def broadcast(self, event: dict[str, object]) -> None:
        """Push one JSON payload to every connected helper."""

        import json

        payload = json.dumps(event)
        for queue in list(self.subscribers):
            _offer(queue, payload)

    def catalog(self, label: str, locality: str, include_live: bool) -> dict[str, object]:
        """Return filtered events and per-locality crash density."""

        rows = _baseline_rows(self.baseline)
        if include_live:
            rows.extend(self.live_events)
        selected = [row for row in rows if _keep(row, label, locality)]
        return {"events": selected, "localities": _locality_stats(rows), "synthetic": True}


def _event(number: int, body: dict[str, object], scored: dict[str, object]) -> dict[str, object]:
    threshold = float(scored["threshold"])
    probability = float(scored["probability"])
    prediction = str(scored["prediction"])
    preset = str(body["scenario_preset"])
    label_name = simulation_label(prediction, preset)
    alert = probability >= threshold
    latitude = float(body["latitude"])
    longitude = float(body["longitude"])
    return {
        **_identity(number, body, latitude, longitude),
        **_motion(body),
        **_decision(prediction, probability, threshold, alert, label_name, scored),
        "observed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hub": nearest_hub(latitude, longitude),
        "synthetic": True,
    }


def _identity(number: int, body: dict[str, object], latitude: float, longitude: float) -> dict[str, object]:
    return {
        "event_id": f"SIM-{number}",
        "source": "live",
        "locality": body["locality"],
        "latitude": latitude,
        "longitude": longitude,
        "scenario_preset": body["scenario_preset"],
        "vehiclesize": body["vehicle_size"],
    }


def _motion(body: dict[str, object]) -> dict[str, object]:
    features = body.get("features") if isinstance(body.get("features"), dict) else {}
    start = float(features.get("gps_start_kmh", body["speed_before"]))
    end = float(features.get("gps_end_kmh", body["speed_after"]))
    return {
        "peak_horizontal_g": float(features.get("peak_horizontal_g", body["peak_g"])),
        "gps_start_kmh": start,
        "gps_end_kmh": end,
        "gps_drop_kmh": float(features.get("gps_drop_kmh", start - end)),
        "gyro_deg_s": body.get("gyro_deg_s"),
        "tier_labels": body.get("tier_labels") or {},
    }


def _decision(prediction: str, probability: float, threshold: float, alert: bool, label_name: str, scored: dict[str, object]) -> dict[str, object]:
    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "model_version": scored.get("model_version"),
        "label_name": label_name,
        "alert_triggered": alert,
        "broadcast": alert or label_name == "near_miss",
    }


def _status_message(action: str, eta_min: float | None) -> str:
    eta = f"{eta_min:.0f}" if eta_min is not None else None
    if action == "secured":
        return "Scene Secured"
    if action == "eta" and eta is not None:
        return f"Ambulance En Route (ETA: {eta} mins)"
    if eta is not None:
        return f"Ambulance En Route (108). ETA: {eta} mins"
    return "Ambulance En Route (108)"


def _offer(queue: asyncio.Queue[str], payload: str) -> None:
    if queue.full():
        queue.get_nowait()
    queue.put_nowait(payload)


def _plain(row: dict[str, object]) -> None:
    row["source"] = "synthetic_v2"
    row["event_id"] = str(int(row["event_id"]))
    row["observed_at"] = str(row["observed_at"])
    row["gps_drop_kmh"] = float(row["gps_start_kmh"]) - float(row["gps_end_kmh"])
    for key in ("latitude", "longitude", "peak_horizontal_g", "gps_start_kmh", "gps_end_kmh", "gps_drop_kmh"):
        row[key] = float(row[key])
    row["hub"] = nearest_hub(float(row["latitude"]), float(row["longitude"]))
    row["synthetic"] = True


def _baseline_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    columns = ["event_id", "label_name", "locality", "latitude", "longitude", "observed_at", "peak_horizontal_g", "gps_start_kmh", "gps_end_kmh", "vehiclesize"]
    rows = frame.loc[:, columns].to_dict(orient="records")
    for row in rows:
        _plain(row)
    return rows


def _keep(row: dict[str, object], label: str, locality: str) -> bool:
    if locality != "all" and str(row["locality"]).lower() != locality.lower():
        return False
    if label == "all":
        return True
    return str(row["label_name"]) == label


def _locality_stats(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["locality", "label_name"])
    density = {item["locality"]: item for item in crash_density(_density_frame(frame))}
    names = sorted(set(frame["locality"])) if not frame.empty else list(density)
    return [_stat(name, frame, density.get(name, {})) for name in names]


def _density_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["locality", "label"])
    labels = frame["label_name"].map({"crash": 0, "near_miss": 1, "normal_driving": 2}).fillna(2)
    return pd.DataFrame({"locality": frame["locality"], "label": labels.astype(int)})


def _stat(name: str, frame: pd.DataFrame, density: dict[str, object]) -> dict[str, object]:
    subset = frame.loc[frame["locality"] == name] if not frame.empty else frame
    total = int(len(subset))
    crashes = int((subset["label_name"] == "crash").sum()) if total else 0
    near = int((subset["label_name"] == "near_miss").sum()) if total else 0
    rate = (100.0 * crashes / total) if total else 0.0
    return {
        "locality": name,
        "total_events": total,
        "crash_count": crashes,
        "near_miss_count": near,
        "crash_rate_pct": round(rate, 2),
        "crashes_per_km2": density.get("crashes_per_km2", 0.0),
        "severity_tier": _tier(rate),
    }


def _tier(rate: float) -> str:
    if rate >= 25:
        return "high"
    if rate >= 15:
        return "elevated"
    return "low"
