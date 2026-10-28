"""Feature extraction keeps unusual lengths and malformed channels."""

import math

import numpy as np

from accident_detection.features import row_features
from accident_detection.models import AuditConfig
from accident_detection.signals import prepare_event

SHORT = AuditConfig(nominal_imu_samples=2, nominal_gps_samples=2)


def test_horizontal_peak_uses_x_and_y_only() -> None:
    features = row_features(
        [[3.0, 4.0, 100.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [10.0, 4.0],
        gyro_is_hd=False,
        vehiclesize="Heavy",
    )
    assert features["peak_horizontal_g"] == 5.0
    assert features["peak_magnitude_g"] == math.sqrt(3**2 + 4**2 + 100**2)
    assert features["min_ax_g"] == 0.0
    assert features["max_ax_g"] == 3.0
    assert features["size_heavy"] == 1.0
    assert features["size_light"] == 0.0


def test_gps_length_17_is_not_truncated() -> None:
    gps = [float(value) for value in range(17)]
    features = row_features(
        [[0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0]],
        gps,
        gyro_is_hd=True,
        vehiclesize="Unknown",
    )
    assert features["gps_samples"] == 17
    assert features["gps_end_kmh"] == 16.0
    assert features["gps_start_kmh"] == 0.0
    assert features["gyro_is_hd"] == 1.0
    assert features["size_light"] == 0.0


def test_imu_length_mismatch_is_preserved() -> None:
    features = row_features(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
        [[9.0, 0.0, 0.0]],
        [3.0, 1.0],
        gyro_is_hd=False,
        vehiclesize=None,
    )
    assert features["imu_samples"] == 2
    assert features["gyro_samples"] == 1
    assert features["peak_abs_gyro_deg_s"] == 9.0
    assert features["gps_drop_kmh"] == 2.0


def test_malformed_accelerometer_becomes_nan_and_is_kept() -> None:
    prepared = prepare_event(
        [[1.0, 2.0]],
        [[0.0, 0.0, 1.0]],
        [5.0],
        label=1,
        vehiclesize="Light",
        event_id=4,
        config=SHORT,
    )
    assert any(flag.code == "malformed_signal:gsensor" for flag in prepared.flags)
    features = row_features(
        prepared.accelerometer_g,
        prepared.gyroscope_deg_s,
        prepared.gps_speed_kmh,
        gyro_is_hd=False,
        vehiclesize="Light",
    )
    assert math.isnan(features["peak_horizontal_g"])
    assert features["gyro_samples"] == 1.0
    assert features["gps_samples"] == 1.0


def test_large_gyroscope_value_is_not_clipped() -> None:
    features = row_features(
        [[0.0, 0.0, 1.0]],
        [[32762.0, 0.0, 0.0]],
        [0.0],
        gyro_is_hd=False,
        vehiclesize="Light",
    )
    assert features["peak_abs_gyro_deg_s"] == 32762.0
