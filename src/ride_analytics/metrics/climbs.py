"""Automatic climb detection from elevation data plus per-climb metrics.

Detection works on the smoothed barometric altitude — smoothing must happen
before the gradient is derived, otherwise sensor noise turns into absurd
per-sample gradients. Candidate stretches above the gradient threshold are
merged across short flat/descent gaps (a hairpin road with flat corners is one
climb, not twenty) and then filtered against minimum gradient, elevation gain
and length.

Repeated efforts up the same climb are matched by start proximity (haversine)
and similar length/gain, which enables personal bests over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.single_ride import normalized_power, sample_durations_s

# Altitude smoothing window (centered rolling median).
SMOOTH_WINDOW = "15s"

# Deadband for ascent totals: rises only count once they exceed this from the
# running reference, the way head units do it — summing every positive delta
# accumulates barometric noise into 5-10 % excess elevation gain.
ASCENT_HYSTERESIS_M = 2.0

# A climb must have at least this average gradient, elevation gain and length.
MIN_AVG_GRADIENT = 0.03
MIN_ELEVATION_GAIN_M = 30.0
MIN_LENGTH_M = 500.0

# Flat/descent stretches inside a climb merge when shorter than either limit.
MAX_GAP_LENGTH_M = 200.0
MAX_GAP_DURATION_S = 30.0

# Samples moving less than this are standstill — no gradient is derived there.
MIN_SAMPLE_DISTANCE_M = 0.5

# Distance base for the max-gradient metric. Sample-to-sample gradients explode
# at crawling speed (1 m altitude step over 1.5 m of travel = 67 %); the
# steepest 50 m stretch is what a rider would call the maximum gradient.
MAX_GRADIENT_BASE_M = 50.0

# Two efforts count as the same climb within these tolerances.
MATCH_START_RADIUS_M = 200.0
MATCH_LENGTH_TOLERANCE = 0.15
MATCH_GAIN_TOLERANCE = 0.15

N_PACING_QUARTERS = 4

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class Climb:
    start_time: datetime
    start_offset_s: float
    length_m: float
    elevation_gain_m: float
    avg_gradient_pct: float
    max_gradient_pct: float
    duration_s: float
    avg_speed_kmh: float
    vam_m_per_h: float
    avg_power_watts: float | None
    np_watts: float | None
    watts_per_kg: float | None
    quarter_avg_power_watts: tuple[float, ...] | None
    kj_before_climb: float | None
    start_lat: float | None
    start_lon: float | None


def smooth_altitude(df: pd.DataFrame) -> pd.Series | None:
    """Altitude with sensor noise removed; ``None`` when the ride has none."""
    if "altitude" not in df.columns or df["altitude"].notna().sum() == 0:
        return None
    smoothed = (
        df.set_index("timestamp")["altitude"]
        .rolling(SMOOTH_WINDOW, center=True, min_periods=1)
        .median()
    )
    return pd.Series(smoothed.to_numpy(), index=df.index)


def ride_elevation_gain_m(df: pd.DataFrame) -> float | None:
    """Total ascent of a ride (smoothed, with deadband); ``None`` without altitude."""
    altitude = smooth_altitude(df)
    if altitude is None:
        return None
    return _total_ascent(altitude.to_numpy())


def _total_ascent(altitude: np.ndarray, threshold: float = ASCENT_HYSTERESIS_M) -> float:
    """Sum of ascents, ignoring oscillations smaller than ``threshold``."""
    gain = 0.0
    reference = None
    for value in altitude:
        if np.isnan(value):
            continue
        if reference is None:
            reference = value
        elif value >= reference + threshold:
            gain += value - reference
            reference = value
        elif value <= reference - threshold:
            reference = value
    return gain


def detect_climbs(df: pd.DataFrame, config: AthleteConfig) -> list[Climb]:
    """Detect climbs in one ride; needs altitude and distance data, else ``[]``."""
    track = _prepare_track(df)
    if track is None:
        return []
    runs = _merge_gaps(track, _candidate_runs(track))
    return [
        _build_climb(df, track, start, end, config)
        for start, end in runs
        if _passes_minimums(track, start, end)
    ]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def match_climbs(climbs: list[Climb]) -> list[list[Climb]]:
    """Group efforts that are the same climb; unmatched climbs form 1-element groups.

    Climbs without GPS coordinates are never matched to anything.
    """
    groups: list[list[Climb]] = []
    for climb in climbs:
        for group in groups:
            if _same_climb(group[0], climb):
                group.append(climb)
                break
        else:
            groups.append([climb])
    return groups


def _same_climb(a: Climb, b: Climb) -> bool:
    if a.start_lat is None or b.start_lat is None:
        return False
    if haversine_m(a.start_lat, a.start_lon, b.start_lat, b.start_lon) > MATCH_START_RADIUS_M:
        return False
    if abs(a.length_m - b.length_m) > MATCH_LENGTH_TOLERANCE * max(a.length_m, b.length_m):
        return False
    gain_limit = MATCH_GAIN_TOLERANCE * max(a.elevation_gain_m, b.elevation_gain_m)
    return abs(a.elevation_gain_m - b.elevation_gain_m) <= gain_limit


def _prepare_track(df: pd.DataFrame) -> pd.DataFrame | None:
    altitude = smooth_altitude(df)
    if altitude is None or "distance" not in df.columns or df["distance"].notna().sum() == 0:
        return None

    track = pd.DataFrame(
        {
            "time_s": (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds(),
            "dist_m": df["distance"],
            "alt_m": altitude,
        }
    )
    d_dist = track["dist_m"].diff()
    gradient = track["alt_m"].diff() / d_dist
    gradient[d_dist < MIN_SAMPLE_DISTANCE_M] = np.nan
    track["gradient"] = gradient
    return track


def _candidate_runs(track: pd.DataFrame) -> list[tuple[int, int]]:
    """Contiguous stretches whose smoothed gradient clears the threshold.

    A run ``(start, end)`` spans sample indices; ``start`` is the base of the
    first climbing interval, so climb length/gain measure from its foot.
    """
    climbing = (track["gradient"] >= MIN_AVG_GRADIENT).to_numpy()
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(climbing):
        if climbing[i]:
            start = i
            while i + 1 < len(climbing) and climbing[i + 1]:
                i += 1
            runs.append((max(start - 1, 0), i))
        i += 1
    return runs


def _merge_gaps(track: pd.DataFrame, runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not runs:
        return []
    dist = track["dist_m"].to_numpy()
    time = track["time_s"].to_numpy()

    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        gap_length = dist[start] - dist[prev_end]
        gap_duration = time[start] - time[prev_end]
        if gap_length < MAX_GAP_LENGTH_M or gap_duration < MAX_GAP_DURATION_S:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _passes_minimums(track: pd.DataFrame, start: int, end: int) -> bool:
    length = track["dist_m"].iloc[end] - track["dist_m"].iloc[start]
    if length < MIN_LENGTH_M:
        return False
    if _elevation_gain(track, start, end) < MIN_ELEVATION_GAIN_M:
        return False
    avg_gradient = (track["alt_m"].iloc[end] - track["alt_m"].iloc[start]) / length
    return avg_gradient >= MIN_AVG_GRADIENT


def _elevation_gain(track: pd.DataFrame, start: int, end: int) -> float:
    return _total_ascent(track["alt_m"].iloc[start : end + 1].to_numpy())


def _build_climb(
    df: pd.DataFrame, track: pd.DataFrame, start: int, end: int, config: AthleteConfig
) -> Climb:
    segment = df.iloc[start : end + 1]
    length_m = float(track["dist_m"].iloc[end] - track["dist_m"].iloc[start])
    gain_m = _elevation_gain(track, start, end)
    duration_s = float(track["time_s"].iloc[end] - track["time_s"].iloc[start])
    net_climb = float(track["alt_m"].iloc[end] - track["alt_m"].iloc[start])

    has_power = "power" in segment.columns and segment["power"].notna().any()
    avg_power = float(segment["power"].mean()) if has_power else None

    quarters = _quarter_powers(segment, track, start, end) if has_power else None

    kj_before = None
    if "power" in df.columns and df["power"].notna().any():
        work_ws = (df["power"].fillna(0) * sample_durations_s(df)).iloc[:start].sum()
        kj_before = float(work_ws / 1000)

    lat = _value_at(df, "position_lat", start)
    lon = _value_at(df, "position_long", start)

    return Climb(
        start_time=df["timestamp"].iloc[start].to_pydatetime(),
        start_offset_s=float(track["time_s"].iloc[start]),
        length_m=length_m,
        elevation_gain_m=gain_m,
        avg_gradient_pct=net_climb / length_m * 100,
        max_gradient_pct=_max_gradient_pct(track, start, end),
        duration_s=duration_s,
        avg_speed_kmh=length_m / duration_s * 3.6 if duration_s > 0 else 0.0,
        vam_m_per_h=gain_m / (duration_s / 3600) if duration_s > 0 else 0.0,
        avg_power_watts=avg_power,
        np_watts=normalized_power(segment) if has_power else None,
        watts_per_kg=avg_power / config.weight_kg if avg_power is not None else None,
        quarter_avg_power_watts=quarters,
        kj_before_climb=kj_before,
        start_lat=lat,
        start_lon=lon,
    )


def _max_gradient_pct(track: pd.DataFrame, start: int, end: int) -> float:
    """Steepest ``MAX_GRADIENT_BASE_M`` stretch within the climb, in percent."""
    dist = track["dist_m"].to_numpy()[start : end + 1]
    alt = track["alt_m"].to_numpy()[start : end + 1]

    ahead = np.searchsorted(dist, dist + MAX_GRADIENT_BASE_M)
    origins = np.nonzero(ahead < len(dist))[0]
    if len(origins) == 0:
        return float((alt[-1] - alt[0]) / (dist[-1] - dist[0]) * 100)
    targets = ahead[origins]
    gradients = (alt[targets] - alt[origins]) / (dist[targets] - dist[origins])
    return float(np.nanmax(gradients) * 100)


def _quarter_powers(
    segment: pd.DataFrame, track: pd.DataFrame, start: int, end: int
) -> tuple[float, ...]:
    """Average power per equal-length quarter of the climb (pacing profile)."""
    dist = track["dist_m"].iloc[start : end + 1].to_numpy()
    power = segment["power"].to_numpy(dtype=float)
    bounds = np.linspace(dist[0], dist[-1], N_PACING_QUARTERS + 1)

    quarters = []
    for i in range(N_PACING_QUARTERS):
        if i == N_PACING_QUARTERS - 1:
            in_quarter = (dist >= bounds[i]) & (dist <= bounds[i + 1])
        else:
            in_quarter = (dist >= bounds[i]) & (dist < bounds[i + 1])
        values = power[in_quarter]
        values = values[~np.isnan(values)]
        quarters.append(float(values.mean()) if len(values) else float("nan"))
    return tuple(quarters)


def _value_at(df: pd.DataFrame, column: str, index: int) -> float | None:
    if column not in df.columns:
        return None
    value = df[column].iloc[index]
    return float(value) if pd.notna(value) else None
