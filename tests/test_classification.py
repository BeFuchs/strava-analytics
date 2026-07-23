"""Tests for rule-based ride classification."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.classification import (
    BASE,
    COMMUTE,
    INTERVALS,
    LONG_BASE,
    OTHER,
    RACE,
    RECOVERY,
    THRESHOLD,
    classify_ride,
)

CONFIG = AthleteConfig(ftp_watts=250, threshold_hr=160, weight_kg=70.0, max_hr=190)
START = datetime(2026, 5, 1, 8, 0)


def ride(segments, hr=150, cadence=90, distance_km=None):
    """Build a 1 Hz ride from (duration_s, power_w) segments; power None -> no column."""
    power = []
    for duration, watts in segments:
        power.extend([watts] * duration)
    n = len(power)
    ts = pd.date_range(START, periods=n, freq="s")
    data = {"timestamp": ts, "heart_rate": np.full(n, hr), "cadence": np.full(n, cadence)}
    if segments[0][1] is not None:
        data["power"] = power
    if distance_km is not None:
        data["distance"] = np.linspace(0, distance_km * 1000, n)
    return pd.DataFrame(data)


def blocks(low, high, seconds_each, reps):
    return [(seconds_each, low if i % 2 == 0 else high) for i in range(reps)]


def test_race_ride():
    df = ride(blocks(100, 360, 120, 36))  # 72 min, high & surgy: IF ≫ 0.85, VI ≫ 1.15
    result = classify_ride(df, CONFIG)
    assert result.ride_type == RACE
    assert result.confidence == "high"


def test_interval_ride():
    segments = [(300, 150)]
    for _ in range(4):
        segments += [(300, 300), (180, 120)]  # 4 × 5 min at 300 W
    segments += [(300, 150)]
    result = classify_ride(ride(segments), CONFIG)
    assert result.ride_type == INTERVALS
    assert result.confidence == "high"


def test_threshold_ride():
    df = ride([(300, 150), (1980, 235), (300, 150)])  # 33 min sweetspot block, no set
    result = classify_ride(df, CONFIG)
    assert result.ride_type == THRESHOLD
    assert result.confidence == "high"


def test_long_base_ride():
    df = ride([(9600, 140)], hr=120)  # 160 min steady endurance
    result = classify_ride(df, CONFIG)
    assert result.ride_type == LONG_BASE
    assert result.confidence == "high"


def test_base_ride():
    df = ride([(5400, 140)], hr=120)  # 90 min steady, under the long-base duration
    result = classify_ride(df, CONFIG)
    assert result.ride_type == BASE
    assert result.confidence == "high"


def test_commute_ride():
    df = ride(blocks(40, 210, 120, 22), distance_km=14)  # 44 min stop-and-go, short
    result = classify_ride(df, CONFIG)
    assert result.ride_type == COMMUTE
    assert result.confidence == "high"


def test_recovery_ride():
    df = ride(blocks(40, 160, 120, 34))  # 68 min, very easy but variable (fails base on VI)
    result = classify_ride(df, CONFIG)
    assert result.ride_type == RECOVERY
    assert result.confidence == "high"


def test_ambiguous_ride_is_other_with_low_confidence():
    df = ride([(2400, 205)])  # 40 min steady tempo — matches no rule cleanly
    result = classify_ride(df, CONFIG)
    assert result.ride_type == OTHER
    assert result.confidence == "low"


def test_ride_without_power_uses_hr_and_caps_confidence():
    ts = pd.date_range(START, periods=5400, freq="s")
    df = pd.DataFrame({"timestamp": ts, "heart_rate": np.full(5400, 128)})  # 90 min mid-Z2 HR
    result = classify_ride(df, CONFIG)
    assert result.ride_type in (BASE, LONG_BASE)
    assert result.confidence != "high"
    assert "HF" in result.matched_rules
