"""Checks for the synthetic v2 table. It must stay separate from VZCrash."""

from __future__ import annotations

import numpy as np
import pandas as pd

from accident_detection.synthetic_v2.generate import COUNTS, SEED, generate_events, locality_boxes


def test_generation_is_repeatable_and_changes_with_the_seed():
    first = generate_events(SEED)
    second = generate_events(SEED)
    other = generate_events(SEED + 1)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_crash_count_is_inside_the_requested_range_and_ids_are_unique():
    frame = generate_events()
    crashes = int((frame["label"] == 0).sum())
    assert crashes == COUNTS["crash"]
    assert 100 <= crashes <= 1000
    assert frame["event_id"].is_unique
    assert set(frame["label_name"]) == {"crash", "near_miss", "normal_driving"}
    assert (frame["source"] == "synthetic_v2").all()


def test_coordinates_fall_inside_the_named_rectangle():
    frame = generate_events()
    boxes = {box["name"]: box for box in locality_boxes()}
    for row in frame.itertuples(index=False):
        box = boxes[row.locality]
        assert box["lat_min"] <= row.latitude <= box["lat_max"]
        assert box["lon_min"] <= row.longitude <= box["lon_max"]
        assert row.crs == "EPSG:4326"


def test_splits_do_not_share_events_and_keep_every_label():
    frame = generate_events()
    assert set(frame["split"]) == {"train", "validation", "test"}
    for label in (0, 1, 2):
        present = set(frame.loc[frame["label"] == label, "split"])
        assert present == {"train", "validation", "test"}
    assert frame["event_id"].is_unique


def test_speed_and_size_follow_the_stated_constraints():
    frame = generate_events()
    assert np.all(frame["gps_min_kmh"] <= frame[["gps_start_kmh", "gps_end_kmh"]].min(axis=1) + 1e-9)
    drop = frame["gps_start_kmh"] - frame["gps_end_kmh"]
    assert np.allclose(frame["gps_drop_kmh"], drop)
    assert np.all(frame["peak_magnitude_g"] >= frame["peak_horizontal_g"])
    sizes = frame[["size_light", "size_medium", "size_heavy"]].sum(axis=1)
    assert np.allclose(sizes, 1.0)


def test_chat_filters_crash_locality_and_lists_public_emergency_numbers():
    from accident_detection.synthetic_v2.chat import answer

    frame = generate_events()
    found = answer(frame, "crashes in Koramangala")
    assert found["locality"] == "Koramangala"
    assert found["label_filter"] == "crash"
    assert found["matched_rows"] == int(((frame["locality"] == "Koramangala") & (frame["label"] == 0)).sum())
    assert all(row["label_name"] == "crash" and row["locality"] == "Koramangala" for row in found["rows"])
    numbers = {item["number"] for item in found["emergency"]}
    assert numbers == {"112", "100", "108"}


def test_chat_non_crash_and_density_use_only_existing_rows():
    from accident_detection.synthetic_v2.chat import answer

    frame = generate_events()
    found = answer(frame, "non-crash in Whitefield")
    assert found["label_filter"] == "non_crash"
    assert all(row["label_name"] != "crash" and row["locality"] == "Whitefield" for row in found["rows"])
    density = answer(frame, "accident density")
    assert density["label_filter"] == "crash"
    assert density["density"][0]["crashes_per_km2"] >= density["density"][-1]["crashes_per_km2"]
    assert sum(item["crashes"] for item in density["density"]) == int((frame["label"] == 0).sum())

