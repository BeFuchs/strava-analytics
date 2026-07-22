from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from ride_analytics.config import AthleteConfig
from ride_analytics.ingest import Ride, RideMeta
from ride_analytics.metrics.comparison import compare_periods

CONFIG = AthleteConfig(ftp_watts=200, threshold_hr=160, weight_kg=70.0, max_hr=190)

JAN_A = (date(2025, 1, 1), date(2025, 1, 7))
JAN_B = (date(2025, 2, 1), date(2025, 2, 7))
JAN_B_TWO_WEEKS = (date(2025, 2, 1), date(2025, 2, 14))


def make_ride(day: date, hours: float = 1.0, power: int = 200) -> Ride:
    n = int(hours * 3600)
    start = datetime(day.year, day.month, day.day, 9, 0, 0)
    df = pd.DataFrame(
        {
            "timestamp": [start + timedelta(seconds=i) for i in range(n)],
            "power": [power] * n,
            "distance": [i * 8.0 for i in range(n)],  # 8 m/s
        }
    )
    meta = RideMeta(source=f"{day}.fit", start_time=start, duration_s=float(n), sport="cycling")
    return Ride(metadata=meta, df=df)


def row(result, metric):
    match = result.table[result.table["metric"] == metric]
    assert len(match) == 1
    return match.iloc[0]


def test_known_sums_and_deltas():
    # A: two 1-h rides at FTP (TSS 100 each); B: one -> known totals and deltas.
    rides = [
        make_ride(date(2025, 1, 2)),
        make_ride(date(2025, 1, 5)),
        make_ride(date(2025, 2, 3)),
    ]

    result = compare_periods(rides, CONFIG, JAN_A, JAN_B)

    assert result.equal_length
    n = row(result, "n_rides")
    assert (n["period_a"], n["period_b"], n["delta_abs"]) == (2, 1, -1)

    tss = row(result, "total_tss")
    assert tss["period_a"] == pytest.approx(200.0, abs=0.5)
    assert tss["period_b"] == pytest.approx(100.0, abs=0.5)
    assert tss["delta_pct"] == pytest.approx(-50.0, abs=0.5)

    distance = row(result, "total_distance_km")
    assert distance["period_a"] == pytest.approx(2 * 28.8, abs=0.1)

    # Equal-length periods need no per-week normalization rows.
    assert "distance_per_week_km" not in set(result.table["metric"])


def test_unequal_periods_add_per_week_rows():
    # Same weekly training in a 1-week vs. 2-week window.
    rides = [
        make_ride(date(2025, 1, 3)),
        make_ride(date(2025, 2, 4)),
        make_ride(date(2025, 2, 11)),
    ]

    result = compare_periods(rides, CONFIG, JAN_A, JAN_B_TWO_WEEKS)

    assert not result.equal_length
    assert row(result, "total_tss")["delta_pct"] == pytest.approx(100.0, abs=1.0)
    # Normalized per week the two periods are identical.
    assert row(result, "avg_tss_per_week")["delta_pct"] == pytest.approx(0.0, abs=1.0)
    assert row(result, "distance_per_week_km")["delta_pct"] == pytest.approx(0.0, abs=1.0)
    assert row(result, "rides_per_week")["delta_pct"] == pytest.approx(0.0, abs=1.0)


def test_rides_outside_periods_are_ignored():
    rides = [make_ride(date(2025, 1, 2)), make_ride(date(2025, 6, 15))]

    result = compare_periods(rides, CONFIG, JAN_A, JAN_B)

    assert row(result, "n_rides")["period_a"] == 1
    assert row(result, "n_rides")["period_b"] == 0


def test_empty_period_does_not_crash():
    rides = [make_ride(date(2025, 1, 2))]

    result = compare_periods(rides, CONFIG, JAN_A, JAN_B)

    assert row(result, "total_tss")["period_b"] == 0
    assert row(result, "total_tss")["delta_pct"] == pytest.approx(-100.0)
    assert pd.isna(row(result, "avg_ctl")["period_b"])
    assert result.period_b.ftp_estimate is None
    assert result.period_b.power_zones is None
    assert result.period_b.power_curve == {}

    # Empty baseline period: percent delta stays blank — no division by zero.
    reversed_result = compare_periods(rides, CONFIG, JAN_B, JAN_A)
    assert pd.isna(row(reversed_result, "total_tss")["delta_pct"])
    assert row(reversed_result, "total_tss")["delta_abs"] == pytest.approx(100.0, abs=0.5)


def test_summary_carries_curves_and_zones():
    rides = [make_ride(date(2025, 1, 2)), make_ride(date(2025, 2, 3))]

    result = compare_periods(rides, CONFIG, JAN_A, JAN_B)

    assert result.period_a.power_curve[1200] == pytest.approx(200.0)
    assert result.period_a.ftp_estimate == pytest.approx(190.0)
    assert result.period_a.power_zones is not None
    assert row(result, "ftp_estimate_watts")["delta_abs"] == pytest.approx(0.0)
