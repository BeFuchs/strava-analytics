"""Tests for aerobic decoupling, efficiency factor and cardiac drift."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.decoupling import (
    cardiac_drift_bpm_per_h,
    efficiency_factor,
    ride_decoupling,
)

CONFIG = AthleteConfig(ftp_watts=250, threshold_hr=160, weight_kg=70.0, max_hr=190)
START = datetime(2026, 5, 1, 8, 0)


def ride_df(seconds, power, hr, start=START):
    ts = pd.date_range(start, periods=seconds, freq="s")

    def col(value):
        return value if hasattr(value, "__len__") else np.full(seconds, float(value))

    data = {"timestamp": ts}
    if power is not None:
        data["power"] = col(power)
    if hr is not None:
        data["heart_rate"] = col(hr)
    return pd.DataFrame(data)


def test_constant_ride_has_zero_decoupling():
    df = ride_df(7200, 200.0, 140.0)  # 2 h, flat power and HR
    result = ride_decoupling(df, CONFIG)
    assert result.valid
    assert result.ef == pytest.approx(200 / 140)
    assert result.decoupling_pct == pytest.approx(0.0, abs=1e-6)


def test_higher_hr_in_second_half_gives_expected_decoupling():
    hr = np.concatenate([np.full(3600, 140.0), np.full(3600, 154.0)])  # +10 % HR
    df = ride_df(7200, 200.0, hr)
    result = ride_decoupling(df, CONFIG)
    # EF1 = 200/140, EF2 = 200/154 -> (1 - 140/154) * 100 ≈ 9.09 %
    assert result.decoupling_pct == pytest.approx(100 * (1 - 140 / 154), abs=0.05)
    assert result.valid


def test_ride_under_60_min_is_invalid():
    df = ride_df(1800, 200.0, 140.0)  # 30 min
    result = ride_decoupling(df, CONFIG)
    assert not result.valid
    assert result.decoupling_pct is None
    assert "60" in result.reason


def test_variable_power_is_invalid():
    block = np.concatenate([np.full(300, 350.0), np.full(300, 50.0)])  # 5 min high / low
    power = np.tile(block, 7)  # 70 min, VI well above 1.15
    df = ride_df(len(power), power, 150.0)
    result = ride_decoupling(df, CONFIG)
    assert not result.valid
    assert "variabel" in result.reason.lower()


def test_interval_type_marked_not_comparable():
    df = ride_df(7200, 200.0, 140.0)
    result = ride_decoupling(df, CONFIG, ride_type="intervals")
    assert not result.valid
    assert "vergleichbar" in result.reason.lower()
    # EF is still computed even when decoupling isn't comparable.
    assert result.ef == pytest.approx(200 / 140)


def test_missing_hr_yields_no_ef():
    df = ride_df(7200, 200.0, None)
    result = ride_decoupling(df, CONFIG)
    assert result.ef is None
    assert not result.valid
    assert efficiency_factor(df) is None


def test_cardiac_drift_matches_linear_rise():
    hr = np.linspace(130.0, 150.0, 3600)  # +20 bpm over ~1 h at flat power
    df = ride_df(3600, 200.0, hr)
    assert cardiac_drift_bpm_per_h(df) == pytest.approx(20.0, abs=1.0)


def test_low_coverage_is_invalid():
    hr = np.full(7200, 140.0)
    hr[: int(7200 * 0.2)] = np.nan  # 20 % of HR samples missing
    df = ride_df(7200, 200.0, hr)
    result = ride_decoupling(df, CONFIG)
    assert not result.valid
    assert "unvollständig" in result.reason.lower()
