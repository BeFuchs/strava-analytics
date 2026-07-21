from datetime import datetime, timedelta

import pandas as pd
import pytest

from ride_analytics.metrics.power_curve import (
    STANDARD_WINDOWS_S,
    aggregate_power_curve,
    estimate_ftp,
    ride_power_curve,
)

START = datetime(2024, 5, 1, 8, 0, 0)


def df_from(offsets, power):
    return pd.DataFrame(
        {"timestamp": [START + timedelta(seconds=o) for o in offsets], "power": power}
    )


def test_constant_hour_fills_all_windows():
    df = df_from(range(3600), [250] * 3600)

    curve = ride_power_curve(df)

    assert set(curve) == set(STANDARD_WINDOWS_S)
    for watts in curve.values():
        assert watts == pytest.approx(250.0)


def test_short_ride_omits_long_windows():
    df = df_from(range(100), [200] * 100)

    curve = ride_power_curve(df)

    assert set(curve) == {5, 15, 30, 60}


def test_burst_is_picked_up_per_window():
    # 45 s at 100 W, 10 s at 400 W, 45 s at 100 W
    power = [100] * 45 + [400] * 10 + [100] * 45
    df = df_from(range(100), power)

    curve = ride_power_curve(df)

    assert curve[5] == pytest.approx(400.0)
    assert curve[15] == pytest.approx((10 * 400 + 5 * 100) / 15)
    assert curve[30] == pytest.approx((10 * 400 + 20 * 100) / 30)


def test_gap_counts_as_zero_power():
    offsets = list(range(30)) + [60 + i for i in range(30)]
    df = df_from(offsets, [300] * 60)

    curve = ride_power_curve(df)

    assert curve[30] == pytest.approx(300.0)
    assert curve[60] == pytest.approx(150.0)  # 30 s at 300 W + 30 s gap at 0 W


def test_no_power_yields_empty_curve():
    df = pd.DataFrame({"timestamp": [START, START + timedelta(seconds=1)]})

    assert ride_power_curve(df) == {}


def test_aggregate_takes_elementwise_max():
    curves = [{5: 300.0, 60: 200.0}, {5: 250.0, 60: 280.0, 300: 220.0}]

    assert aggregate_power_curve(curves) == {5: 300.0, 60: 280.0, 300: 220.0}


def test_estimate_ftp_from_20min_best():
    assert estimate_ftp({1200: 200.0}) == pytest.approx(190.0)
    assert estimate_ftp({60: 350.0}) is None
    assert estimate_ftp({}) is None
