from datetime import datetime, timedelta

import pandas as pd
import pytest

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.climbs import (
    detect_climbs,
    haversine_m,
    match_climbs,
    ride_elevation_gain_m,
)

CONFIG = AthleteConfig(ftp_watts=250, threshold_hr=170, weight_kg=70.0, max_hr=190)
START = datetime(2024, 5, 1, 8, 0, 0)
SPEED = 5.0  # m/s, constant so distance maps 1:1 to time

# Degrees of latitude per meter (spherical earth, matches haversine_m).
LAT_DEG_PER_M = 1 / 111_195.0


def track_df(segments, lat0=None, start=START):
    """1 Hz ride from (length_m, gradient, power) segments; power None = no sensor."""
    rows = []
    dist, alt, t = 0.0, 100.0, 0
    for length, gradient, power in segments:
        for _ in range(round(length / SPEED)):
            dist += SPEED
            alt += SPEED * gradient
            row = {"timestamp": start + timedelta(seconds=t), "distance": dist, "altitude": alt}
            if power is not None:
                row["power"] = power
            if lat0 is not None:
                row["position_lat"] = lat0 + dist * LAT_DEG_PER_M
                row["position_long"] = 8.5
            rows.append(row)
            t += 1
    return pd.DataFrame(rows)


def test_single_climb_detected_with_exact_metrics():
    # Briefing case: 5 km flat, 3 km at 6 %, 4 km flat -> exactly one climb.
    df = track_df([(5000, 0.0, None), (3000, 0.06, None), (4000, 0.0, None)])

    climbs = detect_climbs(df, CONFIG)

    assert len(climbs) == 1
    climb = climbs[0]
    assert climb.elevation_gain_m == pytest.approx(180.0, abs=3.0)
    assert climb.avg_gradient_pct == pytest.approx(6.0, abs=0.15)
    assert climb.max_gradient_pct == pytest.approx(6.0, abs=0.3)  # uniform slope
    assert climb.length_m == pytest.approx(3000.0, rel=0.03)
    assert climb.duration_s == pytest.approx(600.0, rel=0.03)
    assert climb.avg_speed_kmh == pytest.approx(18.0, rel=0.03)
    assert climb.vam_m_per_h == pytest.approx(1080.0, rel=0.05)
    assert climb.start_offset_s == pytest.approx(1000.0, abs=20.0)


def test_short_flat_gap_merges_into_one_climb():
    # Briefing case: a short flat stretch mid-climb must not split the climb.
    df = track_df(
        [
            (1000, 0.0, None),
            (2000, 0.06, None),
            (150, 0.0, None),
            (2000, 0.06, None),
            (1000, 0.0, None),
        ]
    )

    climbs = detect_climbs(df, CONFIG)

    assert len(climbs) == 1
    assert climbs[0].elevation_gain_m == pytest.approx(240.0, abs=4.0)
    assert climbs[0].length_m == pytest.approx(4150.0, rel=0.03)


def test_long_valley_separates_two_climbs():
    df = track_df(
        [
            (1000, 0.0, None),
            (1000, 0.06, None),
            (2000, 0.0, None),
            (1000, 0.06, None),
            (1000, 0.0, None),
        ]
    )

    climbs = detect_climbs(df, CONFIG)

    assert len(climbs) == 2
    for climb in climbs:
        assert climb.elevation_gain_m == pytest.approx(60.0, abs=3.0)


def test_gentle_and_short_sections_are_not_climbs():
    gentle = track_df([(1000, 0.0, None), (3000, 0.02, None), (1000, 0.0, None)])
    assert detect_climbs(gentle, CONFIG) == []

    short_steep = track_df([(1000, 0.0, None), (300, 0.08, None), (1000, 0.0, None)])
    assert detect_climbs(short_steep, CONFIG) == []


def test_missing_altitude_or_distance_gives_no_climbs():
    no_altitude = pd.DataFrame(
        {
            "timestamp": [START + timedelta(seconds=i) for i in range(100)],
            "distance": [i * 5.0 for i in range(100)],
        }
    )
    assert detect_climbs(no_altitude, CONFIG) == []

    no_distance = pd.DataFrame(
        {
            "timestamp": [START + timedelta(seconds=i) for i in range(100)],
            "altitude": [100.0 + i for i in range(100)],
        }
    )
    assert detect_climbs(no_distance, CONFIG) == []


def test_power_metrics_and_pacing_quarters():
    # 2.5 km approach at 100 W (~50 kJ), then a 3 km climb ridden 300 W -> 200 W.
    df = track_df(
        [
            (2500, 0.0, 100),
            (1500, 0.06, 300),
            (1500, 0.06, 200),
            (1000, 0.0, 100),
        ]
    )

    climbs = detect_climbs(df, CONFIG)

    assert len(climbs) == 1
    climb = climbs[0]
    assert climb.avg_power_watts == pytest.approx(250.0, abs=5.0)
    assert climb.watts_per_kg == pytest.approx(climb.avg_power_watts / 70.0, abs=0.01)
    assert climb.np_watts is not None
    assert climb.kj_before_climb == pytest.approx(50.0, abs=2.0)

    quarters = climb.quarter_avg_power_watts
    assert len(quarters) == 4
    assert quarters[0] == pytest.approx(300.0, abs=10.0)
    assert quarters[1] == pytest.approx(300.0, abs=10.0)
    assert quarters[2] == pytest.approx(200.0, abs=10.0)
    assert quarters[3] == pytest.approx(200.0, abs=10.0)


def test_climb_without_power_has_no_power_metrics():
    df = track_df([(1000, 0.0, None), (3000, 0.06, None), (1000, 0.0, None)])

    climb = detect_climbs(df, CONFIG)[0]

    assert climb.avg_power_watts is None
    assert climb.np_watts is None
    assert climb.watts_per_kg is None
    assert climb.quarter_avg_power_watts is None
    assert climb.kj_before_climb is None


def test_repeated_climbs_are_matched_across_rides():
    segments = [(1000, 0.0, None), (2000, 0.06, None), (1000, 0.0, None)]
    ride_a = track_df(segments, lat0=49.0)
    ride_b = track_df(segments, lat0=49.0, start=START + timedelta(days=7))
    elsewhere = track_df(segments, lat0=49.5, start=START + timedelta(days=14))

    climbs = (
        detect_climbs(ride_a, CONFIG)
        + detect_climbs(ride_b, CONFIG)
        + detect_climbs(elsewhere, CONFIG)
    )
    assert len(climbs) == 3

    groups = match_climbs(climbs)

    assert sorted(len(g) for g in groups) == [1, 2]
    repeated = next(g for g in groups if len(g) == 2)
    assert {c.start_time for c in repeated} == {
        climbs[0].start_time,
        climbs[1].start_time,
    }


def test_same_start_different_climb_is_not_matched():
    # Same foot of the hill, but one effort goes three times higher.
    short = track_df([(1000, 0.0, None), (1000, 0.06, None), (1000, 0.0, None)], lat0=49.0)
    long = track_df([(1000, 0.0, None), (3000, 0.06, None), (1000, 0.0, None)], lat0=49.0)

    climbs = detect_climbs(short, CONFIG) + detect_climbs(long, CONFIG)
    assert len(climbs) == 2

    assert all(len(g) == 1 for g in match_climbs(climbs))


def test_climbs_without_coordinates_are_never_matched():
    segments = [(1000, 0.0, None), (2000, 0.06, None), (1000, 0.0, None)]
    climbs = detect_climbs(track_df(segments), CONFIG) + detect_climbs(track_df(segments), CONFIG)

    assert all(len(g) == 1 for g in match_climbs(climbs))


def test_haversine_known_distance():
    # One degree of latitude is ~111.2 km.
    assert haversine_m(49.0, 8.5, 50.0, 8.5) == pytest.approx(111_195, rel=0.01)


def test_ride_elevation_gain_sums_ascents_only():
    df = track_df([(1000, 0.05, None), (1000, -0.05, None), (1000, 0.05, None)])

    # Median smoothing clips the summit/valley corners (~2 m each) and the
    # ascent deadband swallows up to ASCENT_HYSTERESIS_M per climb end.
    assert ride_elevation_gain_m(df) == pytest.approx(100.0, abs=9.0)
    assert ride_elevation_gain_m(pd.DataFrame({"timestamp": [START]})) is None
