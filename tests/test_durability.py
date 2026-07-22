from datetime import datetime, timedelta

import pandas as pd
import pytest

from ride_analytics.metrics.durability import (
    bucket_labels,
    compute_durability,
    ride_bucket_curves,
)

START = datetime(2024, 5, 1, 8, 0, 0)


def df_from(power):
    return pd.DataFrame(
        {
            "timestamp": [START + timedelta(seconds=i) for i in range(len(power))],
            "power": power,
        }
    )


def row(result, bucket, window_s):
    match = result[(result["bucket"] == bucket) & (result["window_s"] == window_s)]
    assert len(match) == 1
    return match.iloc[0]


def test_bucket_labels():
    assert bucket_labels((1000, 2000, 3000)) == (
        "0-1000 kJ",
        "1000-2000 kJ",
        "2000-3000 kJ",
        "3000+ kJ",
    )
    assert bucket_labels((100,)) == ("0-100 kJ", "100+ kJ")


def test_constant_power_gives_identical_buckets_and_index_one():
    # 200 W for 2 h = 1440 kJ: buckets 0-1000 and 1000-2000 filled, both at 200 W.
    result = compute_durability([df_from([200] * 7200)])

    for bucket in ("0-1000 kJ", "1000-2000 kJ"):
        for window in (5, 60, 300, 1200):
            r = row(result, bucket, window)
            assert r["mmp_watts"] == pytest.approx(200.0)
            assert r["durability_index"] == pytest.approx(1.0)


def test_power_drop_shows_up_as_durability_index():
    # 200 W until 1500 kJ (7500 s), then 170 W: the 2000-3000 kJ bucket contains
    # only fatigued riding, so its index must be exactly 170/200 = 0.85.
    result = compute_durability([df_from([200] * 7500 + [170] * 9000)])

    for window in (5, 60, 300, 1200):
        r = row(result, "2000-3000 kJ", window)
        assert r["mmp_watts"] == pytest.approx(170.0)
        assert r["durability_index"] == pytest.approx(0.85)


def test_short_bucket_segment_omits_long_windows():
    # Same ride: past 3000 kJ only ~177 s remain — short windows fill, long stay missing.
    result = compute_durability([df_from([200] * 7500 + [170] * 9000)])

    assert row(result, "3000+ kJ", 5)["mmp_watts"] == pytest.approx(170.0)
    assert pd.isna(row(result, "3000+ kJ", 300)["mmp_watts"])
    assert pd.isna(row(result, "3000+ kJ", 1200)["mmp_watts"])


def test_empty_buckets_stay_missing_not_zero():
    # A short ride never reaches 1000 kJ; deeper buckets must be NaN, not 0.
    result = compute_durability([df_from([200] * 600)])

    assert row(result, "0-1000 kJ", 5)["mmp_watts"] == pytest.approx(200.0)
    for bucket in ("1000-2000 kJ", "2000-3000 kJ", "3000+ kJ"):
        r = row(result, bucket, 5)
        assert pd.isna(r["mmp_watts"])
        assert pd.isna(r["durability_index"])
        assert r["n_rides"] == 0


def test_aggregates_elementwise_max_and_counts_rides():
    short_hard = df_from([300] * 100)  # only 5 s and 60 s windows
    long_steady = df_from([200] * 3600)

    result = compute_durability([short_hard, long_steady])

    assert row(result, "0-1000 kJ", 5)["mmp_watts"] == pytest.approx(300.0)
    assert row(result, "0-1000 kJ", 5)["n_rides"] == 2
    assert row(result, "0-1000 kJ", 300)["mmp_watts"] == pytest.approx(200.0)
    assert row(result, "0-1000 kJ", 300)["n_rides"] == 1


def test_rides_without_power_are_skipped():
    no_power = pd.DataFrame(
        {
            "timestamp": [START + timedelta(seconds=i) for i in range(100)],
            "heart_rate": [150] * 100,
        }
    )

    assert ride_bucket_curves(no_power) is None

    result = compute_durability([no_power])
    assert result["mmp_watts"].isna().all()
    assert (result["n_rides"] == 0).all()


def test_custom_bucket_edges():
    # 200 W for 1000 s = 200 kJ; with a single 100 kJ edge both buckets fill.
    result = compute_durability([df_from([200] * 1000)], edges_kj=(100,), windows=(5, 60))

    assert set(result["bucket"]) == {"0-100 kJ", "100+ kJ"}
    assert row(result, "100+ kJ", 60)["durability_index"] == pytest.approx(1.0)
