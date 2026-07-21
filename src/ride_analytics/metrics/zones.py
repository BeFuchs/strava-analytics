"""Time-in-zone distributions: Coggan power zones (7) and HR zones (5).

Zone boundaries are fractions of FTP resp. threshold HR; a sample sits in a
zone when its value is at or below the zone's upper bound. Sample time uses
the same gap-aware durations as the single-ride metrics, so pauses add no
zone time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.single_ride import sample_durations_s

# Upper bounds of Z1..Z6 as fraction of FTP; Z7 is open-ended.
POWER_ZONE_BOUNDS = (0.55, 0.75, 0.90, 1.05, 1.20, 1.50)
POWER_ZONE_LABELS = (
    "Z1 Active Recovery",
    "Z2 Endurance",
    "Z3 Tempo",
    "Z4 Threshold",
    "Z5 VO2max",
    "Z6 Anaerobic",
    "Z7 Neuromuscular",
)

# Upper bounds of Z1..Z4 as fraction of threshold HR; Z5 is open-ended.
HR_ZONE_BOUNDS = (0.68, 0.83, 0.94, 1.05)
HR_ZONE_LABELS = (
    "Z1 Recovery",
    "Z2 Aerobic",
    "Z3 Tempo",
    "Z4 Threshold",
    "Z5 VO2max",
)


@dataclass(frozen=True)
class ZoneDistribution:
    labels: tuple[str, ...]
    seconds: tuple[float, ...]

    @property
    def percent(self) -> tuple[float, ...]:
        total = sum(self.seconds)
        if total == 0:
            return tuple(0.0 for _ in self.seconds)
        return tuple(s / total * 100 for s in self.seconds)


def power_zone_distribution(df: pd.DataFrame, config: AthleteConfig) -> ZoneDistribution | None:
    """Time per Coggan power zone for one ride; ``None`` without power data."""
    return _distribution(df, "power", config.ftp_watts, POWER_ZONE_BOUNDS, POWER_ZONE_LABELS)


def hr_zone_distribution(df: pd.DataFrame, config: AthleteConfig) -> ZoneDistribution | None:
    """Time per HR zone for one ride; ``None`` without heart-rate data."""
    return _distribution(df, "heart_rate", config.threshold_hr, HR_ZONE_BOUNDS, HR_ZONE_LABELS)


def aggregate_zone_distributions(
    distributions: list[ZoneDistribution | None],
) -> ZoneDistribution | None:
    """Element-wise sum of per-ride zone times; ``None`` entries are skipped."""
    present = [dist for dist in distributions if dist is not None]
    if not present:
        return None
    labels = present[0].labels
    seconds = [0.0] * len(labels)
    for dist in present:
        for i, value in enumerate(dist.seconds):
            seconds[i] += value
    return ZoneDistribution(labels=labels, seconds=tuple(seconds))


def _distribution(
    df: pd.DataFrame,
    column: str,
    reference: int,
    bounds: tuple[float, ...],
    labels: tuple[str, ...],
) -> ZoneDistribution | None:
    if column not in df.columns or df[column].notna().sum() == 0:
        return None

    edges = [-math.inf] + [b * reference for b in bounds] + [math.inf]
    zone_index = pd.cut(df[column], bins=edges, labels=False, right=True)
    dt = sample_durations_s(df)

    seconds = tuple(float(dt[zone_index == i].sum()) for i in range(len(labels)))
    return ZoneDistribution(labels=labels, seconds=seconds)
