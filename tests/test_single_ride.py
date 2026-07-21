from datetime import datetime, timedelta

import pandas as pd
import pytest

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.single_ride import (
    compute_ride_metrics,
    elapsed_time_s,
    moving_time_s,
    normalized_power,
    work_kj,
)

START = datetime(2024, 5, 1, 8, 0, 0)
CONFIG = AthleteConfig(ftp_watts=200, threshold_hr=170, weight_kg=75.0, max_hr=190)


def df_from(offsets, **columns):
    data = {"timestamp": [START + timedelta(seconds=o) for o in offsets]}
    data.update(columns)
    return pd.DataFrame(data)


def test_constant_hour_at_ftp():
    df = df_from(range(3600), power=[200] * 3600)

    metrics = compute_ride_metrics(df, CONFIG)

    assert metrics.np_watts == pytest.approx(200.0)
    assert metrics.intensity_factor == pytest.approx(1.0)
    assert metrics.tss == pytest.approx(100.0)
    assert metrics.variability_index == pytest.approx(1.0)
    assert metrics.work_kj == pytest.approx(720.0)
    assert metrics.tss_estimated is False


def test_np_uses_fourth_power_weighting():
    # Samples 30 s apart: each rolling window holds exactly one sample, so the
    # formula reduces to the plain fourth-power mean of [100, 200].
    df = df_from([0, 30], power=[100, 200])

    expected = ((100**4 + 200**4) / 2) ** 0.25
    assert normalized_power(df) == pytest.approx(expected)


def test_np_exceeds_avg_for_variable_power():
    df = df_from(range(600), power=[100] * 300 + [300] * 300)

    metrics = compute_ride_metrics(df, CONFIG)

    assert metrics.avg_power == pytest.approx(200.0)
    assert metrics.np_watts > 200.0
    assert metrics.variability_index > 1.0


def test_moving_vs_elapsed_with_pause():
    offsets = list(range(600)) + [900 + i for i in range(600)]
    df = df_from(offsets, power=[150] * 1200)

    assert moving_time_s(df) == pytest.approx(1200.0)
    assert elapsed_time_s(df) == pytest.approx(1499.0)


def test_work_kj_integrates_power():
    df = df_from(range(100), power=[250] * 100)

    assert work_kj(df) == pytest.approx(25.0)


def test_hr_fallback_when_power_missing():
    df = df_from(range(3600), heart_rate=[170] * 3600)

    metrics = compute_ride_metrics(df, CONFIG)

    assert metrics.np_watts is None
    assert metrics.intensity_factor is None
    assert metrics.tss == pytest.approx(100.0)  # 1 h at threshold HR
    assert metrics.tss_estimated is True


def test_no_power_no_hr_yields_no_tss():
    df = df_from(range(60))

    metrics = compute_ride_metrics(df, CONFIG)

    assert metrics.tss is None
    assert metrics.tss_estimated is False
    assert metrics.avg_power is None
    assert metrics.avg_hr is None


def test_distance_and_avg_speed():
    df = df_from(range(3600), power=[200] * 3600, distance=[i * 10.0 for i in range(3600)])

    metrics = compute_ride_metrics(df, CONFIG)

    assert metrics.distance_km == pytest.approx(35.99)
    assert metrics.avg_speed_kmh == pytest.approx(35.99)


def test_max_values_and_cadence():
    df = df_from(
        range(4),
        power=[100, 400, 200, 100],
        heart_rate=[120, 160, 150, 130],
        cadence=[80, 90, 100, 90],
    )

    metrics = compute_ride_metrics(df, CONFIG)

    assert metrics.max_power == 400
    assert metrics.max_hr == 160
    assert metrics.avg_cadence == pytest.approx(90.0)
