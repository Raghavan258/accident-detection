"""Published VZCrash sampling spec and column names.

Nominal lengths come from the dataset definition (16-second windows), not from
the most common length in any shard.
"""

from __future__ import annotations

IMU_SAMPLE_RATE_HZ = 100
GPS_SAMPLE_RATE_HZ = 1
WINDOW_SECONDS = 16
IMU_AXES = 3
NOMINAL_IMU_SAMPLES = IMU_SAMPLE_RATE_HZ * WINDOW_SECONDS
NOMINAL_GPS_SAMPLES = GPS_SAMPLE_RATE_HZ * WINDOW_SECONDS

LABEL_NAMES: dict[int, str] = {
    0: "crash",
    1: "near_miss",
    2: "normal_driving",
}
LABEL_ORDER: tuple[int, ...] = (0, 1, 2)
ALLOWED_VEHICLE_SIZES: frozenset[str] = frozenset(
    {"Light", "Medium", "Heavy", "Unknown"}
)
REQUIRED_COLUMNS: tuple[str, ...] = (
    "event_id",
    "label",
    "vehiclesize",
    "gyro_is_hd",
    "gsensor",
    "gyro",
    "gps_speed",
)
SIGNAL_COLUMNS: tuple[str, ...] = ("gsensor", "gyro", "gps_speed")
IMU_COLUMNS: tuple[str, ...] = ("gsensor", "gyro")
KNOWN_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})
UNKNOWN_SPLIT = "unknown"
OFFICIAL_SHARD_COUNTS: dict[str, int] = {
    "train": 11,
    "validation": 3,
    "test": 2,
}
