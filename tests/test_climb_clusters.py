"""Tests for stable climb clustering."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import pandas as pd

from ride_analytics.clustering.climb_clusters import (
    ClimbEffort,
    climb_avg_hr,
    cluster_climbs,
)
from ride_analytics.metrics.climbs import EARTH_RADIUS_M, Climb

LAT0, LON0 = 49.4094, 8.6946
T0 = datetime(2026, 3, 1, 10, 0)


def lat_offset(meters: float) -> float:
    """Degrees of latitude corresponding to ``meters`` of northward travel."""
    return math.degrees(meters / EARTH_RADIUS_M)


def make_climb(
    start_time: datetime = T0,
    lat: float | None = LAT0,
    lon: float | None = LON0,
    length_m: float = 2000.0,
    gain_m: float = 150.0,
    duration_s: float = 480.0,
    avg_power: float | None = None,
) -> Climb:
    return Climb(
        start_time=start_time,
        start_offset_s=0.0,
        length_m=length_m,
        elevation_gain_m=gain_m,
        avg_gradient_pct=gain_m / length_m * 100,
        max_gradient_pct=gain_m / length_m * 100,
        duration_s=duration_s,
        avg_speed_kmh=length_m / duration_s * 3.6,
        vam_m_per_h=gain_m / (duration_s / 3600),
        avg_power_watts=avg_power,
        np_watts=None,
        watts_per_kg=avg_power / 80 if avg_power else None,
        quarter_avg_power_watts=None,
        kj_before_climb=None,
        start_lat=lat,
        start_lon=lon,
    )


def efforts(*climbs: Climb) -> list[ClimbEffort]:
    return [ClimbEffort(climb=c) for c in climbs]


def test_repeated_hill_and_distant_hill_form_separate_clusters():
    home = [
        make_climb(T0, lat=LAT0 + lat_offset(jitter), length_m=length)
        for jitter, length in ((0, 2000.0), (30, 2050.0), (-25, 1980.0))
    ]
    far = make_climb(T0 + timedelta(days=1), lat=LAT0 + lat_offset(5000))
    clusters = cluster_climbs(efforts(*home, far))
    assert sorted(c.ascent_count for c in clusters) == [1, 3]


def test_start_distance_exactly_at_threshold_matches():
    a = make_climb(T0)
    b = make_climb(T0 + timedelta(days=1), lat=LAT0 + lat_offset(200.0))
    assert len(cluster_climbs(efforts(a, b))) == 1


def test_start_distance_beyond_threshold_splits():
    a = make_climb(T0)
    b = make_climb(T0 + timedelta(days=1), lat=LAT0 + lat_offset(210.0))
    assert len(cluster_climbs(efforts(a, b))) == 2


def test_length_exactly_at_tolerance_matches():
    # 2000 - 1700 == 0.15 * 2000 — exactly at the ±15 % limit.
    a = make_climb(T0, length_m=1700.0)
    b = make_climb(T0 + timedelta(days=1), length_m=2000.0)
    assert len(cluster_climbs(efforts(a, b))) == 1


def test_length_beyond_tolerance_splits():
    a = make_climb(T0, length_m=1690.0)
    b = make_climb(T0 + timedelta(days=1), length_m=2000.0)
    assert len(cluster_climbs(efforts(a, b))) == 2


def test_nearest_cluster_wins_on_multiple_matches():
    a = make_climb(T0)
    b = make_climb(T0 + timedelta(days=1), lat=LAT0 + lat_offset(390.0))
    # 200 m from a, 190 m from b — both match, b is closer.
    c = make_climb(T0 + timedelta(days=2), lat=LAT0 + lat_offset(200.0))
    clusters = cluster_climbs(efforts(a, b, c))
    counts = {cluster.ascent_count for cluster in clusters}
    assert counts == {1, 2}
    joined = next(cl for cl in clusters if cl.ascent_count == 2)
    assert {ascent.date for ascent in joined.ascents} == {b.start_time, c.start_time}


def test_cluster_ids_are_deterministic():
    climbs = [
        make_climb(T0 + timedelta(days=i), lat=LAT0 + lat_offset(random.Random(i).uniform(0, 40)))
        for i in range(6)
    ]
    ids_a = sorted(c.cluster_id for c in cluster_climbs(efforts(*climbs)))
    shuffled = list(climbs)
    random.Random(42).shuffle(shuffled)
    ids_b = sorted(c.cluster_id for c in cluster_climbs(efforts(*shuffled)))
    assert ids_a == ids_b


def test_climbs_without_gps_are_excluded():
    clusters = cluster_climbs(efforts(make_climb(lat=None, lon=None), make_climb()))
    assert len(clusters) == 1
    assert clusters[0].ascent_count == 1


def test_cluster_metadata():
    fast = make_climb(T0, duration_s=400.0, avg_power=250.0)
    slow = make_climb(T0 + timedelta(days=5), duration_s=500.0)
    (cluster,) = cluster_climbs([ClimbEffort(fast, avg_hr=155.0), ClimbEffort(slow)])

    assert cluster.best_time_s == 400.0
    assert cluster.last_ridden_date == slow.start_time.date()
    assert cluster.length_km == 2.0
    assert cluster.location_label == "49.409 N, 8.695 E"
    # Newest first; ride context travels with each ascent.
    assert [a.date for a in cluster.ascents] == [slow.start_time, fast.start_time]
    assert cluster.ascents[1].avg_hr == 155.0
    assert cluster.ascents[1].avg_power_watts == 250.0


def test_climb_avg_hr_uses_climb_window():
    timestamps = pd.date_range("2026-03-01 10:00", periods=10, freq="s")
    df = pd.DataFrame({"timestamp": timestamps, "heart_rate": range(100, 110)})
    climb = make_climb(start_time=timestamps[2].to_pydatetime(), duration_s=4.0)
    climb = Climb(**{**climb.__dict__, "start_offset_s": 2.0, "duration_s": 4.0})
    assert climb_avg_hr(df, climb) == 104.0  # mean of samples at offsets 2..6

    no_hr = pd.DataFrame({"timestamp": timestamps})
    assert climb_avg_hr(no_hr, climb) is None
