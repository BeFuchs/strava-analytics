"""Tests for power-based interval detection and set grouping."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.intervals import detect_intervals, intervals_frame

CONFIG = AthleteConfig(ftp_watts=250, threshold_hr=160, weight_kg=70.0, max_hr=190)
START = datetime(2026, 5, 1, 8, 0)
FTP110 = 275  # 110 % of 250 W FTP
REST_W = 120  # below the 220 W (88 % FTP) threshold


def ride_from_segments(segments, hr=145, cadence=90):
    """Build a 1 Hz ride from (duration_s, power_w) segments."""
    power = []
    for duration, watts in segments:
        power.extend([watts] * duration)
    ts = pd.date_range(START, periods=len(power), freq="s")
    return pd.DataFrame({"timestamp": ts, "power": power, "heart_rate": hr, "cadence": cadence})


def four_by_five():
    segments = [(300, 150)]  # warm-up
    for _ in range(4):
        segments += [(300, FTP110), (180, REST_W)]
    segments += [(300, 150)]  # cool-down
    return segments


def test_four_by_five_min_is_one_set_of_four():
    sets = detect_intervals(ride_from_segments(four_by_five()), CONFIG)
    assert len(sets) == 1
    assert sets[0].reps == 4
    for interval in sets[0].intervals:
        assert interval.duration_s == pytest.approx(300, abs=12)
    assert sets[0].avg_rest_duration_s == pytest.approx(180, abs=15)


def test_five_second_dip_does_not_split_an_interval():
    segments = [(300, 150)]
    for i in range(4):
        if i == 1:
            # Second interval carries a 5 s dip below threshold in its middle.
            segments += [(147, FTP110), (5, 90), (148, FTP110), (180, REST_W)]
        else:
            segments += [(300, FTP110), (180, REST_W)]
    segments += [(300, 150)]

    sets = detect_intervals(ride_from_segments(segments), CONFIG)
    total_intervals = sum(s.reps for s in sets)
    assert total_intervals == 4  # the dip is bridged, not counted as a split


def test_short_surge_is_not_an_interval():
    sets = detect_intervals(ride_from_segments([(300, 150), (20, 320), (300, 150)]), CONFIG)
    assert sets == []


def test_ride_below_threshold_has_no_intervals():
    sets = detect_intervals(ride_from_segments([(3600, 150)]), CONFIG)
    assert sets == []


def test_ride_without_power_returns_empty():
    ts = pd.date_range(START, periods=600, freq="s")
    df = pd.DataFrame({"timestamp": ts, "heart_rate": 150})
    assert detect_intervals(df, CONFIG) == []


def test_distinct_blocks_form_two_sets():
    # 3 x 1 min, then 3 x 4 min — different durations -> two sets.
    segments = [(300, 150)]
    for _ in range(3):
        segments += [(60, FTP110), (60, REST_W)]
    segments += [(300, 130)]
    for _ in range(3):
        segments += [(240, FTP110), (180, REST_W)]
    sets = detect_intervals(ride_from_segments(segments), CONFIG)
    assert len(sets) == 2
    assert sorted(s.reps for s in sets) == [3, 3]


def test_intervals_frame_is_flat_with_set_membership():
    sets = detect_intervals(ride_from_segments(four_by_five()), CONFIG)
    frame = intervals_frame(sets)
    assert len(frame) == 4
    assert list(frame["set_index"].unique()) == [0]
    assert list(frame["rep_index"]) == [0, 1, 2, 3]
    assert (frame["avg_power_watts"] > 260).all()
    # kJ before each rep rises across the set.
    assert frame["kj_before"].is_monotonic_increasing
