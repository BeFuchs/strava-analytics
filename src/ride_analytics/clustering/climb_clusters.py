"""Group repeated efforts up the same hill into stable climb clusters.

Builds on the pairwise matching from ``metrics.climbs`` (same start radius,
length/gain tolerances widened by absolute noise floors) but produces
persistent cluster objects: efforts are
walked chronologically and matched against a running cluster representative —
the member median of start coordinate, length and gain. Median instead of mean
so a single GPS outlier cannot drag the representative away from the hill.

Cluster IDs are derived from the rounded representative (coordinate + length),
so the same data produces the same IDs on every upload.

Climbs without GPS coordinates cannot be located and are left out entirely.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from statistics import median

import pandas as pd

from ride_analytics.metrics.climbs import (
    MATCH_GAIN_TOLERANCE,
    MATCH_LENGTH_TOLERANCE,
    MATCH_START_RADIUS_M,
    Climb,
    haversine_m,
)

# Rounding for the stable cluster ID: ~100 m in coordinates and length, coarse
# enough that re-parsing the same rides cannot flip the ID through jitter.
ID_COORD_DECIMALS = 3
ID_LENGTH_STEP_M = 100

# Absolute floors for the ±15 % tolerances. On small climbs the relative
# tolerance drops below the measurement noise and split a verified real-world
# climb into two clusters: gain noise is barometric (~tens of meters missing),
# length varies with where detection cuts the climb — gaps up to 200 m are
# merged (``MAX_GAP_LENGTH_M``), so boundaries legitimately differ by that much
# between rides of the same hill.
GAIN_TOLERANCE_FLOOR_M = 20.0
LENGTH_TOLERANCE_FLOOR_M = 250.0


@dataclass(frozen=True)
class ClimbEffort:
    """One detected climb plus ride context the ``Climb`` itself doesn't carry."""

    climb: Climb
    avg_hr: float | None = None


@dataclass(frozen=True)
class ClusterAscent:
    """A single ride up a clustered climb."""

    date: datetime
    duration_s: float
    vam_m_per_h: float
    avg_power_watts: float | None
    watts_per_kg: float | None
    avg_hr: float | None


@dataclass(frozen=True)
class ClimbCluster:
    cluster_id: str
    location_label: str
    length_km: float
    avg_gradient_pct: float
    elevation_gain_m: float
    ascent_count: int
    best_time_s: float
    last_ridden_date: date
    ascents: tuple[ClusterAscent, ...]


def climb_avg_hr(df: pd.DataFrame, climb: Climb) -> float | None:
    """Average heart rate over the climb's time window in its ride DataFrame."""
    if "heart_rate" not in df.columns:
        return None
    offsets = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    window = (offsets >= climb.start_offset_s) & (
        offsets <= climb.start_offset_s + climb.duration_s
    )
    values = df.loc[window, "heart_rate"].dropna()
    return float(values.mean()) if len(values) else None


def cluster_climbs(efforts: list[ClimbEffort]) -> list[ClimbCluster]:
    """Cluster efforts agglomeratively; clusters ordered by first occurrence."""
    states: list[_ClusterState] = []
    with_gps = [e for e in efforts if e.climb.start_lat is not None]
    for effort in sorted(with_gps, key=lambda e: e.climb.start_time):
        best = _best_match(states, effort.climb)
        if best is None:
            states.append(_ClusterState(effort))
        else:
            best.add(effort)
    return [state.finalize() for state in states]


def _best_match(states: list[_ClusterState], climb: Climb) -> _ClusterState | None:
    """The matching cluster with the smallest start distance, if any."""
    best, best_distance = None, float("inf")
    for state in states:
        distance = haversine_m(climb.start_lat, climb.start_lon, state.rep_lat, state.rep_lon)
        if distance > MATCH_START_RADIUS_M or distance >= best_distance:
            continue
        if not _within(
            climb.length_m, state.rep_length_m, MATCH_LENGTH_TOLERANCE, LENGTH_TOLERANCE_FLOOR_M
        ):
            continue
        if not _within(
            climb.elevation_gain_m, state.rep_gain_m, MATCH_GAIN_TOLERANCE, GAIN_TOLERANCE_FLOOR_M
        ):
            continue
        best, best_distance = state, distance
    return best


def _within(a: float, b: float, tolerance: float, floor: float) -> bool:
    return abs(a - b) <= max(tolerance * max(a, b), floor)


class _ClusterState:
    """Mutable cluster under construction with a running median representative."""

    def __init__(self, effort: ClimbEffort) -> None:
        self.efforts: list[ClimbEffort] = []
        self.add(effort)

    def add(self, effort: ClimbEffort) -> None:
        self.efforts.append(effort)
        climbs = [e.climb for e in self.efforts]
        self.rep_lat = median(c.start_lat for c in climbs)
        self.rep_lon = median(c.start_lon for c in climbs)
        self.rep_length_m = median(c.length_m for c in climbs)
        self.rep_gain_m = median(c.elevation_gain_m for c in climbs)

    def finalize(self) -> ClimbCluster:
        climbs = [e.climb for e in self.efforts]
        ascents = tuple(
            ClusterAscent(
                date=e.climb.start_time,
                duration_s=e.climb.duration_s,
                vam_m_per_h=e.climb.vam_m_per_h,
                avg_power_watts=e.climb.avg_power_watts,
                watts_per_kg=e.climb.watts_per_kg,
                avg_hr=e.avg_hr,
            )
            for e in sorted(self.efforts, key=lambda e: e.climb.start_time, reverse=True)
        )
        return ClimbCluster(
            cluster_id=_cluster_id(self.rep_lat, self.rep_lon, self.rep_length_m),
            location_label=_location_label(self.rep_lat, self.rep_lon),
            length_km=self.rep_length_m / 1000,
            avg_gradient_pct=median(c.avg_gradient_pct for c in climbs),
            elevation_gain_m=self.rep_gain_m,
            ascent_count=len(self.efforts),
            best_time_s=min(c.duration_s for c in climbs),
            last_ridden_date=max(c.start_time for c in climbs).date(),
            ascents=ascents,
        )


def _cluster_id(lat: float, lon: float, length_m: float) -> str:
    key = (
        f"{round(lat, ID_COORD_DECIMALS):.3f}|"
        f"{round(lon, ID_COORD_DECIMALS):.3f}|"
        f"{round(length_m / ID_LENGTH_STEP_M) * ID_LENGTH_STEP_M:d}"
    )
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _location_label(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.3f} {ns}, {abs(lon):.3f} {ew}"
