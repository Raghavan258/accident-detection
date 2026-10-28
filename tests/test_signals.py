"""Row parsing keeps raw values and records quality flags."""

import math

import numpy as np

from accident_detection.models import AuditConfig
from accident_detection.signals import acceleration_magnitude, prepare_event

SHORT = AuditConfig(nominal_imu_samples=2, nominal_gps_samples=2)


def _codes(prepared) -> set[str]:
    return {flag.code for flag in prepared.flags}


def test_nominal_event_has_no_flags_and_keeps_values() -> None:
    prepared = prepare_event(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]],
        [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
        [10.0, 11.0],
        label=0,
        vehiclesize="Heavy",
        event_id=7,
        config=SHORT,
    )
    assert _codes(prepared) == set()
    assert prepared.accelerometer_g.shape == (2, 3)
    assert prepared.accelerometer_g[0, 0] == 1.0
    assert prepared.gps_speed_kmh.tolist() == [10.0, 11.0]
    assert acceleration_magnitude(prepared.accelerometer_g)[0] == math.sqrt(1 + 4 + 9)


def test_non_nominal_gps_is_flagged_and_not_truncated() -> None:
    prepared = prepare_event(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [1.0, 2.0, 3.0],
        label=2,
        vehiclesize="Light",
        event_id=1,
        config=SHORT,
    )
    assert "length_not_nominal:gps_speed" in _codes(prepared)
    assert prepared.gps_speed_kmh.tolist() == [1.0, 2.0, 3.0]


def test_nan_is_preserved_and_flagged() -> None:
    prepared = prepare_event(
        [[math.nan, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [1.0, 2.0],
        label=1,
        vehiclesize="Medium",
        event_id=3,
        config=SHORT,
    )
    assert "nonfinite_signal:gsensor" in _codes(prepared)
    assert np.isnan(prepared.accelerometer_g[0, 0])


def test_null_empty_and_malformed_signals_are_flagged() -> None:
    prepared = prepare_event(
        [[1.0, 2.0]],
        None,
        [],
        label=99,
        vehiclesize="Bus",
        event_id=None,
        config=SHORT,
    )
    assert "malformed_signal:gsensor" in _codes(prepared)
    assert "null_signal:gyro" in _codes(prepared)
    assert "empty_signal:gps_speed" in _codes(prepared)
    assert "unknown_label" in _codes(prepared)
    assert "unknown_vehiclesize" in _codes(prepared)
    assert "null_field:event_id" in _codes(prepared)
    assert prepared.gyroscope_deg_s.shape == (0, 3)


def test_null_sample_is_not_counted_as_a_stored_nan() -> None:
    prepared = prepare_event(
        [None, [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [None, 4.0],
        label=2,
        vehiclesize="Unknown",
        event_id=5,
        config=SHORT,
    )
    assert "null_elements:gsensor" in _codes(prepared)
    assert "null_elements:gps_speed" in _codes(prepared)
    assert "nonfinite_signal:gsensor" not in _codes(prepared)
    assert prepared.accelerometer_g.shape == (2, 3)
    assert np.isnan(prepared.accelerometer_g[0]).all()


def test_imu_length_mismatch_is_explicit() -> None:
    prepared = prepare_event(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0]],
        [1.0, 2.0],
        label=0,
        vehiclesize="Light",
        event_id=8,
        config=SHORT,
    )
    assert "imu_length_mismatch" in _codes(prepared)
    mismatch = next(flag for flag in prepared.flags if flag.code == "imu_length_mismatch")
    assert mismatch.detail == "gsensor_length=2; gyro_length=1"
