"""Data objects shared by the audit and the row preprocessor."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from accident_detection.constants import (
    ALLOWED_VEHICLE_SIZES,
    NOMINAL_GPS_SAMPLES,
    NOMINAL_IMU_SAMPLES,
    GPS_SAMPLE_RATE_HZ,
    IMU_SAMPLE_RATE_HZ,
)


@dataclass(frozen=True)
class AuditConfig:
    """Sampling spec used to flag records. It does not drop them."""

    nominal_imu_samples: int = NOMINAL_IMU_SAMPLES
    nominal_gps_samples: int = NOMINAL_GPS_SAMPLES
    imu_hz: int = IMU_SAMPLE_RATE_HZ
    gps_hz: int = GPS_SAMPLE_RATE_HZ
    allowed_vehicle_sizes: frozenset[str] = field(
        default_factory=lambda: ALLOWED_VEHICLE_SIZES
    )


DEFAULT_CONFIG = AuditConfig()


@dataclass(frozen=True)
class Flag:
    """One explicit data-quality finding for a single event."""

    code: str
    detail: str


@dataclass(frozen=True)
class ChannelStats:
    """Quality counts for one signal, without the signal values."""

    is_null: bool
    length: int
    nonfinite_count: int
    null_element_count: int
    malformed_detail: str | None = None


@dataclass(frozen=True)
class PreparedEvent:
    """Float64 copies of one event. Flags describe problems; values stay raw."""

    accelerometer_g: np.ndarray
    gyroscope_deg_s: np.ndarray
    gps_speed_kmh: np.ndarray
    flags: tuple[Flag, ...]


@dataclass(frozen=True)
class PlotExample:
    """One training event chosen for a class illustration."""

    label: int
    event_id: int
    source_file: str
    row_index: int
    accelerometer_g: np.ndarray
    nominal_timing: bool


@dataclass(frozen=True)
class DuplicateId:
    """An event_id that occurs more than once in the audited files."""

    event_id: int
    locations: tuple[str, ...]


@dataclass
class FileScan:
    """Audit facts for one Parquet file. Anomalies are records, not deletions."""

    path: str
    file_name: str
    split: str
    num_rows: int
    num_row_groups: int
    file_size_bytes: int
    schema_problems: tuple[str, ...]
    read_error: str | None
    signals_scanned: bool
    label_counts: dict[int, int]
    null_counts: dict[str, int]
    null_element_totals: dict[str, int]
    nonfinite_totals: dict[str, int]
    length_counts: dict[str, dict[int, int]]
    extrema: dict[str, dict[str, float | int | None]]
    vehiclesize_counts: dict[str, int]
    gyro_hd_counts: dict[str, int]
    anomalies: list[dict[str, object]]
    event_ids: np.ndarray
    event_id_null: np.ndarray
    labels: np.ndarray
    label_null: np.ndarray
    peak_acceleration_g: np.ndarray
    peak_horizontal_g: np.ndarray
    examples: dict[int, PlotExample]


@dataclass(frozen=True)
class AuditReport:
    """Directory-level audit. Raw files are never rewritten."""

    data_dir: str
    files: tuple[FileScan, ...]
    coverage_notes: tuple[str, ...]
    duplicates: tuple[DuplicateId, ...]
    examples: dict[int, PlotExample]
    config: AuditConfig
    descriptive_stats: dict[str, object]
