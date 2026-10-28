"""Parse one event and flag quality problems without changing raw numbers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from accident_detection.constants import IMU_AXES, LABEL_NAMES, SIGNAL_COLUMNS
from accident_detection.models import (
    DEFAULT_CONFIG,
    AuditConfig,
    ChannelStats,
    Flag,
    PreparedEvent,
)


def prepare_event(
    gsensor,
    gyro,
    gps_speed,
    *,
    label: int | None,
    vehiclesize: str | None,
    event_id: int | None = None,
    config: AuditConfig | None = None,
) -> PreparedEvent:
    """Copy one event into float64 arrays and attach quality flags.

    Null and malformed channels become empty arrays and a flag. Parsed numbers
    are not clipped, resampled, or imputed.
    """

    active = DEFAULT_CONFIG if config is None else config
    accel, accel_stats = _prepare_xyz(gsensor)
    gyro_array, gyro_stats = _prepare_xyz(gyro)
    gps_array, gps_stats = _prepare_gps(gps_speed)
    channels = {"gsensor": accel_stats, "gyro": gyro_stats, "gps_speed": gps_stats}
    flags = assess_record_stats(
        label=label,
        vehiclesize=vehiclesize,
        event_id=event_id,
        channels=channels,
        config=active,
    )
    return PreparedEvent(accel, gyro_array, gps_array, flags)


def assess_record_stats(
    *,
    label: int | None,
    vehiclesize: str | None,
    channels: Mapping[str, ChannelStats],
    event_id: int | None = None,
    config: AuditConfig | None = None,
) -> tuple[Flag, ...]:
    """Build stable quality flags from already computed channel counts."""

    active = DEFAULT_CONFIG if config is None else config
    flags: list[Flag] = []
    flags.extend(_identity_flags(event_id, label, vehiclesize, active))
    for name in SIGNAL_COLUMNS:
        nominal = _nominal_length(name, active)
        flags.extend(_channel_flags(name, channels[name], nominal))
    flags.extend(_mismatch_flags(channels["gsensor"], channels["gyro"]))
    return tuple(flags)


def acceleration_magnitude(accel_xyz: np.ndarray) -> np.ndarray:
    """Return the 3-axis acceleration magnitude in g for each sample."""

    return np.linalg.norm(np.asarray(accel_xyz, dtype=np.float64), axis=1)


def as_xyz(samples) -> tuple[np.ndarray, int]:
    """Return shape (n, 3) and the number of missing scalar axes."""

    if samples is None:
        raise ValueError("signal is null")
    if isinstance(samples, np.ndarray):
        return _xyz_from_array(samples)
    return _xyz_from_sequence(samples)


def as_gps(samples) -> tuple[np.ndarray, int]:
    """Return a 1-D speed array and the number of missing samples."""

    if samples is None:
        raise ValueError("signal is null")
    values = np.empty(len(samples), dtype=np.float64)
    nulls = 0
    for index, sample in enumerate(samples):
        if sample is None:
            values[index] = np.nan
            nulls += 1
            continue
        if isinstance(sample, (list, tuple, np.ndarray)):
            raise ValueError(f"sample {index} is not a scalar")
        values[index] = float(sample)
    return values, nulls


def _prepare_xyz(samples) -> tuple[np.ndarray, ChannelStats]:
    empty = np.empty((0, IMU_AXES), dtype=np.float64)
    if samples is None:
        return empty, ChannelStats(True, 0, 0, 0, None)
    try:
        matrix, nulls = as_xyz(samples)
    except (TypeError, ValueError) as exc:
        return empty, ChannelStats(False, 0, 0, 0, str(exc))
    nonfinite = int((~np.isfinite(matrix)).sum() - nulls)
    return matrix, ChannelStats(False, int(matrix.shape[0]), nonfinite, nulls, None)


def _prepare_gps(samples) -> tuple[np.ndarray, ChannelStats]:
    if samples is None:
        return np.empty(0, dtype=np.float64), ChannelStats(True, 0, 0, 0, None)
    try:
        values, nulls = as_gps(samples)
    except (TypeError, ValueError) as exc:
        return np.empty(0, dtype=np.float64), ChannelStats(False, 0, 0, 0, str(exc))
    nonfinite = int((~np.isfinite(values)).sum() - nulls)
    return values, ChannelStats(False, int(values.size), nonfinite, nulls, None)


def _xyz_from_array(samples: np.ndarray) -> tuple[np.ndarray, int]:
    if samples.ndim != 2 or samples.shape[1] != IMU_AXES:
        raise ValueError(f"expected shape (n, {IMU_AXES})")
    return samples.astype(np.float64, copy=True), 0


def _xyz_from_sequence(samples) -> tuple[np.ndarray, int]:
    matrix = np.empty((len(samples), IMU_AXES), dtype=np.float64)
    nulls = 0
    for index, sample in enumerate(samples):
        if sample is None:
            matrix[index] = np.nan
            nulls += IMU_AXES
            continue
        if len(sample) != IMU_AXES:
            raise ValueError(f"sample {index} does not have {IMU_AXES} axes")
        matrix[index] = (float(sample[0]), float(sample[1]), float(sample[2]))
    return matrix, nulls


def _identity_flags(event_id, label, vehiclesize, config: AuditConfig) -> list[Flag]:
    flags: list[Flag] = []
    if event_id is None:
        flags.append(Flag("null_field:event_id", "value is null"))
    if label is None:
        flags.append(Flag("null_field:label", "value is null"))
    elif int(label) not in LABEL_NAMES:
        flags.append(Flag("unknown_label", f"label={label}"))
    if vehiclesize is None:
        flags.append(Flag("null_field:vehiclesize", "value is null"))
    elif vehiclesize not in config.allowed_vehicle_sizes:
        flags.append(Flag("unknown_vehiclesize", f"vehiclesize={vehiclesize}"))
    return flags


def _channel_flags(name: str, stats: ChannelStats, nominal: int) -> list[Flag]:
    if stats.malformed_detail is not None:
        return [Flag(f"malformed_signal:{name}", stats.malformed_detail)]
    if stats.is_null:
        return [Flag(f"null_signal:{name}", "value is null")]
    flags: list[Flag] = []
    if stats.length == 0:
        flags.append(Flag(f"empty_signal:{name}", "length=0"))
    elif stats.length != nominal:
        detail = f"length={stats.length}; nominal={nominal}"
        flags.append(Flag(f"length_not_nominal:{name}", detail))
    if stats.null_element_count:
        flags.append(Flag(f"null_elements:{name}", f"count={stats.null_element_count}"))
    if stats.nonfinite_count:
        flags.append(Flag(f"nonfinite_signal:{name}", f"count={stats.nonfinite_count}"))
    return flags


def _mismatch_flags(gsensor: ChannelStats, gyro: ChannelStats) -> list[Flag]:
    blocked = (gsensor.malformed_detail, gyro.malformed_detail)
    if any(blocked) or gsensor.is_null or gyro.is_null:
        return []
    if gsensor.length == gyro.length:
        return []
    detail = f"gsensor_length={gsensor.length}; gyro_length={gyro.length}"
    return [Flag("imu_length_mismatch", detail)]


def _nominal_length(name: str, config: AuditConfig) -> int:
    if name == "gps_speed":
        return config.nominal_gps_samples
    return config.nominal_imu_samples
