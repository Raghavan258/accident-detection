"""Rule-based answers over the synthetic v2 table."""

from __future__ import annotations

import pandas as pd

from accident_detection.synthetic_v2.contacts import EMERGENCY_CONTACTS
from accident_detection.synthetic_v2.localities import LOCALITIES
from accident_detection.synthetic_v2.query import crash_density, filter_events, preview_rows

_NON_CRASH = ("non-crash", "non crash", "not crash", "not a crash")
_DENSITY = ("density", "busiest", "hotspot", "per km")


def answer(frame: pd.DataFrame, message: str) -> dict[str, object]:
    """Filter synthetic events and attach the public emergency numbers."""

    text = " ".join(message.lower().split())
    locality = _locality(text)
    mode = _label_mode(text)
    selected = filter_events(frame, locality, mode)
    density = crash_density(frame)
    return {
        "synthetic": True,
        "reply": _reply(selected, locality, mode, _wants(text, _DENSITY), density),
        "locality": locality,
        "label_filter": mode,
        "matched_rows": int(len(selected)),
        "rows": preview_rows(selected),
        "density": density,
        "emergency": [dict(item) for item in EMERGENCY_CONTACTS],
    }


def _locality(text: str) -> str | None:
    for box in LOCALITIES:
        if box["name"].lower() in text:
            return box["name"]
    return None


def _label_mode(text: str) -> str | None:
    if any(phrase in text for phrase in _NON_CRASH):
        return "non_crash"
    if "near miss" in text or "near-miss" in text:
        return "near_miss"
    if "normal" in text:
        return "normal_driving"
    if "crash" in text or "accident" in text:
        return "crash"
    return None


def _wants(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _reply(selected: pd.DataFrame, locality: str | None, mode: str | None, density: bool, ranked: list[dict[str, object]]) -> str:
    place = locality or "all eight sampling areas"
    kind = mode or "events"
    top = ranked[0]
    head = f"Synthetic v2 only, not recorded crashes. {len(selected)} {kind} in {place}."
    if density and ranked:
        head += f" Highest crash density is {top['locality']}: {top['crashes']} crashes, {top['crashes_per_km2']} per km²."
    return head + " Police 100, ambulance 108, and national emergency 112 are public numbers. They are not dispatched to these pins."
