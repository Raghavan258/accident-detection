"""Sampling rectangles for synthetic coordinates. These are not crash sites."""

from __future__ import annotations

from typing import TypedDict


class LocalityBox(TypedDict):
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


# Approximate rectangles inside Bengaluru for synthetic sampling only.
# They are not administrative boundaries and not a record of real crashes.
LOCALITIES: tuple[LocalityBox, ...] = (
    {"name": "Koramangala", "lat_min": 12.925, "lat_max": 12.945, "lon_min": 77.610, "lon_max": 77.640},
    {"name": "Indiranagar", "lat_min": 12.970, "lat_max": 12.985, "lon_min": 77.635, "lon_max": 77.655},
    {"name": "Whitefield", "lat_min": 12.960, "lat_max": 12.990, "lon_min": 77.720, "lon_max": 77.760},
    {"name": "Jayanagar", "lat_min": 12.920, "lat_max": 12.940, "lon_min": 77.580, "lon_max": 77.600},
    {"name": "Malleswaram", "lat_min": 13.000, "lat_max": 13.020, "lon_min": 77.560, "lon_max": 77.580},
    {"name": "Hebbal", "lat_min": 13.030, "lat_max": 13.050, "lon_min": 77.590, "lon_max": 77.610},
    {"name": "Electronic City", "lat_min": 12.830, "lat_max": 12.855, "lon_min": 77.660, "lon_max": 77.690},
    {"name": "Yelahanka", "lat_min": 13.090, "lat_max": 13.115, "lon_min": 77.580, "lon_max": 77.610},
)
