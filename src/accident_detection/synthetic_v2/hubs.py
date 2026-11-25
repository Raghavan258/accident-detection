"""Simulation routing hubs. Public campus points, not a live dispatch feed."""

from __future__ import annotations

import math

HUBS: tuple[dict[str, object], ...] = (
    {"name": "St. John's Hospital", "zone": "Koramangala", "lat": 12.9274, "lng": 77.6205},
    {"name": "Manipal Hospital Whitefield", "zone": "Whitefield", "lat": 12.9880, "lng": 77.7276},
    {"name": "Baptist Hospital", "zone": "Hebbal", "lat": 13.0352, "lng": 77.5895},
    {"name": "Victoria Hospital", "zone": "City Market", "lat": 12.9634, "lng": 77.5736},
)
TRAFFIC_CONTACT = "Bengaluru Traffic Police — 100"


def nearest_hub(latitude: float, longitude: float) -> dict[str, object]:
    """Return the closest listed hospital campus and the public traffic number."""

    hub = min(HUBS, key=lambda item: _km(latitude, longitude, float(item["lat"]), float(item["lng"])))
    return {**hub, "traffic_contact": TRAFFIC_CONTACT, "public_numbers": ["112", "100", "108"]}


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))
