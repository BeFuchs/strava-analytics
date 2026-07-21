from datetime import datetime, timedelta

import pandas as pd
import pytest

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.zones import (
    aggregate_zone_distributions,
    hr_zone_distribution,
    power_zone_distribution,
)

START = datetime(2024, 5, 1, 8, 0, 0)
CONFIG = AthleteConfig(ftp_watts=250, threshold_hr=170, weight_kg=75.0, max_hr=190)


def df_from(offsets, **columns):
    data = {"timestamp": [START + timedelta(seconds=o) for o in offsets]}
    data.update(columns)
    return pd.DataFrame(data)


def test_power_at_ftp_is_threshold_zone():
    df = df_from(range(600), power=[250] * 600)

    dist = power_zone_distribution(df, CONFIG)

    assert dist.labels[3] == "Z4 Threshold"
    assert dist.seconds[3] == pytest.approx(600.0)
    assert dist.percent[3] == pytest.approx(100.0)


def test_power_zones_split_by_intensity():
    # 100 s at 40% FTP (Z1), 200 s at 80% (Z3), 100 s at 120% (Z5)
    power = [100] * 100 + [200] * 200 + [300] * 100
    df = df_from(range(400), power=power)

    dist = power_zone_distribution(df, CONFIG)

    assert dist.seconds == (100.0, 0.0, 200.0, 0.0, 100.0, 0.0, 0.0)
    assert dist.percent[0] == pytest.approx(25.0)
    assert dist.percent[2] == pytest.approx(50.0)
    assert dist.percent[4] == pytest.approx(25.0)


def test_zone_upper_bound_is_inclusive():
    df = df_from(range(10), power=[int(0.55 * 250)] * 10)  # exactly Z1/Z2 boundary

    dist = power_zone_distribution(df, CONFIG)

    assert dist.seconds[0] == pytest.approx(10.0)
    assert dist.seconds[1] == 0.0


def test_hr_at_threshold_is_zone_four():
    df = df_from(range(600), heart_rate=[170] * 600)

    dist = hr_zone_distribution(df, CONFIG)

    assert dist.labels[3] == "Z4 Threshold"
    assert dist.percent[3] == pytest.approx(100.0)


def test_pause_adds_no_zone_time():
    offsets = list(range(10)) + [300 + i for i in range(10)]
    df = df_from(offsets, power=[200] * 20)

    dist = power_zone_distribution(df, CONFIG)

    assert sum(dist.seconds) == pytest.approx(20.0)


def test_missing_sensor_returns_none():
    df = df_from(range(10), heart_rate=[150] * 10)

    assert power_zone_distribution(df, CONFIG) is None
    assert hr_zone_distribution(df_from(range(10), power=[200] * 10), CONFIG) is None


def test_aggregate_sums_seconds():
    a = power_zone_distribution(df_from(range(100), power=[100] * 100), CONFIG)
    b = power_zone_distribution(df_from(range(300), power=[100] * 300), CONFIG)

    total = aggregate_zone_distributions([a, None, b])

    assert total.seconds[0] == pytest.approx(400.0)
    assert total.percent[0] == pytest.approx(100.0)


def test_aggregate_of_nothing_is_none():
    assert aggregate_zone_distributions([None, None]) is None
