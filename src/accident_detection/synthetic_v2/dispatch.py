"""Requirement parser for the helper dispatch chat. Rule based, not an LLM."""

from __future__ import annotations

import re

from accident_detection.synthetic_v2.hubs import nearest_hub
from accident_detection.synthetic_v2.localities import LOCALITIES

NAMES = tuple(str(box["name"]) for box in LOCALITIES)
MATCH_LIMIT = 40


def interpret(
    events: list[dict[str, object]],
    localities: list[dict[str, object]],
    message: str,
    current: dict[str, object] | None = None,
) -> dict[str, object]:
    """Turn one helper sentence into a reply, matching events, and a map target."""

    if _asks_comms(message):
        return _comms_reply(events, message)
    if _asks_navigation(message):
        return _navigation_reply(events)
    if _asks_latest(message):
        return _latest_reply(events)
    if _asks_density(message):
        return _density_reply(events, localities)
    filters = _apply_current(_filters(message), current)
    matched = [row for row in events if _matches(row, filters)][:MATCH_LIMIT]
    focus = _focus(matched)
    hub = nearest_hub(float(focus["lat"]), float(focus["lng"])) if focus else None
    return {
        "reply": _summary(matched, filters, localities, hub),
        "matched_events": matched,
        "focus_target": focus,
        "filters": filters,
        "hub": hub,
        "synthetic": True,
    }


def _apply_current(filters: dict[str, object], current: dict[str, object] | None) -> dict[str, object]:
    if not current:
        return filters
    for key, value in filters.items():
        if value in (None, False) and current.get(key) not in (None, False):
            filters[key] = current[key]
    return filters


def _asks_density(message: str) -> bool:
    text = message.lower()
    return "density" in text or "busiest" in text or "hotspot" in text


def _density_reply(events: list[dict[str, object]], localities: list[dict[str, object]]) -> dict[str, object]:
    top = max(localities, key=lambda item: float(item.get("crashes_per_km2", 0))) if localities else None
    name = str(top["locality"]) if top else ""
    matched = [row for row in events if row["locality"] == name and row["label_name"] == "crash"][:MATCH_LIMIT]
    focus = _focus(matched)
    hub = nearest_hub(float(focus["lat"]), float(focus["lng"])) if focus else None
    rate = top.get("crash_rate_pct", 0) if top else 0
    per_km = top.get("crashes_per_km2", 0) if top else 0
    reply = (
        f"{name} has the highest crash density in this synthetic table: "
        f"{per_km} crashes/km², crash rate {rate}%. "
        f"{_hub_sentence(hub)} Synthetic pins are not a live dispatch."
    )
    return {"reply": reply, "matched_events": matched, "focus_target": focus, "filters": {"locality": name, "label": "crash"}, "hub": hub, "synthetic": True}


def _filters(message: str) -> dict[str, object]:
    text = message.lower()
    return {
        "locality": next((name for name in NAMES if name.lower() in text), None),
        "event_id": _event_id(message),
        "label": _label(text),
        "severe": "3.5" in text or "severe" in text,
        "high_probability": "0.9" in text or "high probability" in text,
        "live_only": "live" in text,
        "high_speed": "high-speed" in text or "high speed" in text,
        "peak_min": 1.0 if _small_band(text) else None,
        "peak_max": 2.2 if _small_band(text) else None,
    }


def _label(text: str) -> str | None:
    if "small accident" in text or "1-2" in text or "1–2" in text:
        return None
    if "near-miss" in text or "near miss" in text or "near_miss" in text:
        return "near_miss"
    if "normal" in text:
        return "normal_driving"
    if "non-crash" in text or "non crash" in text:
        return "non_crash"
    if any(word in text for word in ("crash", "accident", "ambulance")):
        return "crash"
    return None


def _event_id(message: str) -> str | None:
    match = re.search(r"SIM-\d+", message, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def _matches(row: dict[str, object], filters: dict[str, object]) -> bool:
    if filters["event_id"] and str(row["event_id"]).upper() != filters["event_id"]:
        return False
    if filters["locality"] and str(row["locality"]).lower() != str(filters["locality"]).lower():
        return False
    if not _label_ok(row, filters["label"]):
        return False
    if filters["severe"] and float(row.get("peak_horizontal_g") or 0) <= 3.5:
        return False
    if filters["high_probability"] and not _high_probability(row):
        return False
    if filters["live_only"] and row.get("source") != "live":
        return False
    if filters["high_speed"] and float(row.get("gps_drop_kmh") or 0) < 30:
        return False
    return _peak_ok(row, filters)


def _label_ok(row: dict[str, object], label: object) -> bool:
    if label is None:
        return True
    if label == "non_crash":
        return row["label_name"] != "crash"
    return row["label_name"] == label


def _high_probability(row: dict[str, object]) -> bool:
    probability = row.get("probability")
    return probability is not None and float(probability) > 0.9


def _focus(rows: list[dict[str, object]]) -> dict[str, object] | None:
    if not rows:
        return None
    row = max(rows, key=lambda item: float(item.get("probability") or 0))
    return {"lat": row["latitude"], "lng": row["longitude"], "zoom": 16, "event_id": row["event_id"]}


def _summary(matched: list[dict[str, object]], filters: dict[str, object], localities: list[dict[str, object]], hub: dict[str, object] | None) -> str:
    place = filters["locality"] or "all 8 Bengaluru zones"
    kind = filters["label"] or "event"
    rate = _rate(localities, filters["locality"])
    return (
        f"{len(matched)} {kind} records in {place}. Crash rate {rate}%. "
        f"{_hub_sentence(hub)} Public numbers 112, 100, and 108. Synthetic pins are not a live dispatch."
    )


def _rate(localities: list[dict[str, object]], name: object) -> object:
    if name is None:
        crashes = sum(int(item["crash_count"]) for item in localities)
        total = sum(int(item["total_events"]) for item in localities)
        return round(100.0 * crashes / total, 2) if total else 0.0
    row = next((item for item in localities if item["locality"] == name), None)
    return row["crash_rate_pct"] if row else 0.0


def _hub_sentence(hub: dict[str, object] | None) -> str:
    if hub is None:
        return "No matching pin to route."
    return f"Nearest simulation hub: {hub['name']} ({hub['zone']}). Traffic contact: {hub['traffic_contact']}."


def _small_band(text: str) -> bool:
    return "small accident" in text or "1-2" in text or "1–2" in text


def _peak_ok(row: dict[str, object], filters: dict[str, object]) -> bool:
    peak = float(row.get("peak_horizontal_g") or 0)
    low, high = filters.get("peak_min"), filters.get("peak_max")
    if low is not None and peak < float(low):
        return False
    if high is not None and peak > float(high):
        return False
    return True


def _asks_comms(message: str) -> bool:
    text = message.lower()
    return any(phrase in text for phrase in ("dispatch ambulance", "notify user", "eta update", "scene secured"))


def _asks_navigation(message: str) -> bool:
    return "navigate" in message.lower()


def _asks_latest(message: str) -> bool:
    text = message.lower()
    return "latest" in text and ("live" in text or "alert" in text)


def _comms_reply(events: list[dict[str, object]], message: str) -> dict[str, object]:
    action = _comms_action(message.lower())
    latest = _latest_live(events)
    focus = _focus([latest] if latest else [])
    reply = {
        "ambulance": "Dispatching ambulance 108 and notifying the simulated driver.",
        "eta": "Sending the current responder ETA to the simulated driver.",
        "secured": "Marking the simulated scene secured.",
    }[action]
    return {"reply": reply, "matched_events": [latest] if latest else [], "focus_target": focus, "filters": {}, "hub": latest.get("hub") if latest else None, "action": action, "synthetic": True}


def _comms_action(text: str) -> str:
    if "scene secured" in text:
        return "secured"
    if "eta" in text:
        return "eta"
    return "ambulance"


def _navigation_reply(events: list[dict[str, object]]) -> dict[str, object]:
    latest = _latest_live(events)
    focus = _focus([latest] if latest else [])
    reply = "No live user alert to navigate to." if latest is None else f"Routing to {latest['event_id']} in {latest['locality']}."
    return {"reply": reply, "matched_events": [latest] if latest else [], "focus_target": focus, "filters": {}, "hub": latest.get("hub") if latest else None, "action": "navigate", "synthetic": True}


def _latest_reply(events: list[dict[str, object]]) -> dict[str, object]:
    latest = _latest_live(events)
    focus = _focus([latest] if latest else [])
    reply = "No live user alert yet." if latest is None else f"Latest live alert is {latest['event_id']} in {latest['locality']}."
    return {"reply": reply, "matched_events": [latest] if latest else [], "focus_target": focus, "filters": {"live_only": True}, "hub": latest.get("hub") if latest else None, "action": "filter", "synthetic": True}


def _latest_live(events: list[dict[str, object]]) -> dict[str, object] | None:
    live = [row for row in events if row.get("source") == "live"]
    if not live:
        return None
    return sorted(live, key=lambda row: str(row.get("observed_at") or ""))[-1]

