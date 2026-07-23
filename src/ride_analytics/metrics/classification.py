"""Assign each ride a training type from transparent, hand-set rules.

The rules are checked in a fixed order, first match wins. Every threshold is a
named constant with its rationale — they are deliberately set, not empirically
optimized, and there is no machine learning here on purpose: the classification
stays readable and explainable. A confidence level (high/medium/low) rides
along so the report can flag borderline calls instead of feigning certainty.

Pure functions over the normalized record DataFrame from ``ingest``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.intervals import IntervalSet, detect_intervals
from ride_analytics.metrics.power_curve import power_grid_1hz
from ride_analytics.metrics.single_ride import RideMetrics, compute_ride_metrics
from ride_analytics.metrics.zones import hr_zone_distribution, power_zone_distribution

# Type keys (stable identifiers used across the API, CSV and decoupling).
RACE = "race"
INTERVALS = "intervals"
THRESHOLD = "threshold"
LONG_BASE = "long_base"
BASE = "base"
COMMUTE = "commute"
RECOVERY = "recovery"
OTHER = "other"

# German display labels for the UI.
RIDE_TYPE_LABELS = {
    RACE: "Rennen / Wettkampf",
    INTERVALS: "Intervalle",
    THRESHOLD: "Schwelle / Sweetspot",
    LONG_BASE: "Grundlage (lang)",
    BASE: "Grundlage",
    COMMUTE: "Pendeln",
    RECOVERY: "Erholung",
    OTHER: "Sonstige",
}

# Race: sustained hard and surgy — high intensity over a real duration with the
# stop-and-go variability of a bunch race.
RACE_MIN_IF = 0.85
RACE_MIN_DURATION_S = 3600.0
RACE_MIN_VI = 1.15

# Intervals: at least one repeated set plus enough total time above the interval
# threshold that it reads as a workout, not a couple of climbs.
INTERVAL_MIN_TIME_OVER_THRESHOLD_S = 480.0  # 8 min

# Threshold / sweetspot: a long continuous block in the 84–105 % FTP band with
# no repeated-set structure.
THRESHOLD_BAND = (0.84, 1.05)
THRESHOLD_MIN_CONTIG_S = 1200.0  # 20 min
# Coasting into a corner shouldn't break the block — smooth before banding.
THRESHOLD_SMOOTH_S = 30

# Endurance: most of the ride easy, steady power.
BASE_MIN_ZONE1_2_PCT = 70.0
LONG_BASE_MIN_DURATION_S = 9000.0  # 150 min
LONG_BASE_MAX_VI = 1.10
BASE_MAX_VI = 1.15

# Commute: short, stop-and-go, low distance.
COMMUTE_MAX_DURATION_S = 3600.0
COMMUTE_MAX_DISTANCE_KM = 25.0
COMMUTE_MIN_VI = 1.20

# Recovery: short and easy.
RECOVERY_MAX_DURATION_S = 5400.0  # 90 min
RECOVERY_MAX_IF = 0.60


@dataclass(frozen=True)
class RideClassification:
    ride_type: str
    confidence: str  # "high" | "medium" | "low"
    matched_rules: str

    @property
    def label(self) -> str:
        return RIDE_TYPE_LABELS[self.ride_type]


def classify_ride(
    df: pd.DataFrame,
    config: AthleteConfig,
    *,
    interval_sets: list[IntervalSet] | None = None,
) -> RideClassification:
    """Classify one ride. ``interval_sets`` can be passed to avoid recomputing."""
    metrics = compute_ride_metrics(df, config)
    has_power = metrics.np_watts is not None
    if interval_sets is None:
        interval_sets = detect_intervals(df, config)

    if not has_power:
        return _classify_by_hr(df, config, metrics)
    return _classify_by_power(df, config, metrics, interval_sets)


def _classify_by_power(
    df: pd.DataFrame,
    config: AthleteConfig,
    metrics: RideMetrics,
    interval_sets: list[IntervalSet],
) -> RideClassification:
    duration = metrics.moving_time_s
    vi = metrics.variability_index or 0.0
    if_factor = metrics.intensity_factor or 0.0
    zone1_2 = _zone1_2_pct(power_zone_distribution(df, config))
    has_set = any(s.reps >= 2 for s in interval_sets)
    time_over = _time_over_fraction_s(df, config, 0.88)

    if if_factor >= RACE_MIN_IF and duration >= RACE_MIN_DURATION_S and vi >= RACE_MIN_VI:
        conf = "high" if if_factor >= 0.90 and vi >= 1.20 else "medium"
        return RideClassification(RACE, conf, f"IF {if_factor:.2f}, VI {vi:.2f}, ≥60 min")

    if has_set and time_over >= INTERVAL_MIN_TIME_OVER_THRESHOLD_S:
        max_reps = max(s.reps for s in interval_sets)
        conf = "high" if max_reps >= 3 and time_over >= 720 else "medium"
        return RideClassification(
            INTERVALS, conf, f"{max_reps}er-Set, {time_over / 60:.0f} min über Schwelle"
        )

    contig = _longest_band_run_s(df, config, *THRESHOLD_BAND)
    if contig >= THRESHOLD_MIN_CONTIG_S and not has_set:
        conf = "high" if contig >= 1800 else "medium"
        return RideClassification(THRESHOLD, conf, f"{contig / 60:.0f} min am Stück 84–105 % FTP")

    if (
        duration >= LONG_BASE_MIN_DURATION_S
        and zone1_2 >= BASE_MIN_ZONE1_2_PCT
        and vi <= LONG_BASE_MAX_VI
    ):
        conf = "high" if zone1_2 >= 80 and vi <= 1.05 else "medium"
        return RideClassification(LONG_BASE, conf, f"≥150 min, {zone1_2:.0f} % Z1–2, VI {vi:.2f}")

    if zone1_2 >= BASE_MIN_ZONE1_2_PCT and vi <= BASE_MAX_VI:
        conf = "high" if zone1_2 >= 80 else "medium"
        return RideClassification(BASE, conf, f"{zone1_2:.0f} % Z1–2, VI {vi:.2f}")

    distance = metrics.distance_km or 0.0
    if (
        duration <= COMMUTE_MAX_DURATION_S
        and distance <= COMMUTE_MAX_DISTANCE_KM
        and vi >= COMMUTE_MIN_VI
    ):
        conf = "high" if vi >= 1.30 else "medium"
        return RideClassification(COMMUTE, conf, f"≤60 min, ≤25 km, VI {vi:.2f}")

    if duration <= RECOVERY_MAX_DURATION_S and if_factor <= RECOVERY_MAX_IF:
        conf = "high" if if_factor <= 0.55 else "medium"
        return RideClassification(RECOVERY, conf, f"≤90 min, IF {if_factor:.2f}")

    return RideClassification(OTHER, "low", "keine Regel eindeutig erfüllt")


def _classify_by_hr(
    df: pd.DataFrame, config: AthleteConfig, metrics: RideMetrics
) -> RideClassification:
    """HR-based fallback without power — zones stand in for IF/VI, confidence ≤ medium."""
    duration = metrics.moving_time_s
    zone1_2 = _zone1_2_pct(hr_zone_distribution(df, config))
    avg_hr = metrics.avg_hr or 0.0
    recovery_hr = 0.75 * config.threshold_hr  # easy-spin ceiling without IF

    if duration >= LONG_BASE_MIN_DURATION_S and zone1_2 >= BASE_MIN_ZONE1_2_PCT:
        return RideClassification(
            LONG_BASE, "medium", f"≥150 min, {zone1_2:.0f} % HF-Z1–2 (HF-basiert)"
        )
    if duration <= RECOVERY_MAX_DURATION_S and 0 < avg_hr <= recovery_hr:
        return RideClassification(RECOVERY, "medium", "≤90 min, Ø HF niedrig (HF-basiert)")
    if zone1_2 >= BASE_MIN_ZONE1_2_PCT:
        return RideClassification(BASE, "medium", f"{zone1_2:.0f} % HF-Z1–2 (HF-basiert)")
    return RideClassification(OTHER, "low", "keine Regel eindeutig erfüllt (HF-basiert)")


def _zone1_2_pct(distribution) -> float:
    """Share of time in the two easiest zones (recovery + endurance/aerobic)."""
    if distribution is None:
        return 0.0
    percent = distribution.percent
    return percent[0] + percent[1] if len(percent) >= 2 else 0.0


def _time_over_fraction_s(df: pd.DataFrame, config: AthleteConfig, fraction: float) -> float:
    """Seconds of riding time spent at or above ``fraction`` of FTP."""
    from ride_analytics.metrics.single_ride import sample_durations_s

    if "power" not in df.columns or df["power"].notna().sum() == 0:
        return 0.0
    dt = sample_durations_s(df)
    return float(dt[df["power"] >= fraction * config.ftp_watts].sum())


def _longest_band_run_s(
    df: pd.DataFrame, config: AthleteConfig, low_frac: float, high_frac: float
) -> float:
    """Longest continuous stretch (seconds) with smoothed power inside the band."""
    grid = power_grid_1hz(df)
    if grid is None:
        return 0.0
    smoothed = grid.rolling(THRESHOLD_SMOOTH_S, center=True, min_periods=1).mean()
    low, high = low_frac * config.ftp_watts, high_frac * config.ftp_watts
    in_band = ((smoothed >= low) & (smoothed <= high)).to_numpy()

    best = current = 0
    for on in in_band:
        current = current + 1 if on else 0
        best = max(best, current)
    return float(best)


def classification_frame(records: list[tuple[str, object, RideClassification]]) -> pd.DataFrame:
    """History table ``ride_id, date, ride_type, confidence, matched_rules``.

    ``records`` is a list of ``(ride_id, date, RideClassification)``.
    """
    return pd.DataFrame(
        [
            {
                "ride_id": ride_id,
                "date": date,
                "ride_type": c.ride_type,
                "confidence": c.confidence,
                "matched_rules": c.matched_rules,
            }
            for ride_id, date, c in records
        ],
        columns=["ride_id", "date", "ride_type", "confidence", "matched_rules"],
    )


def type_distribution(
    classifications: list[RideClassification], durations_s: list[float]
) -> pd.DataFrame:
    """Ride count and total time per type, for the training-distribution card."""
    rows: dict[str, dict[str, float]] = {}
    for c, seconds in zip(classifications, durations_s, strict=True):
        entry = rows.setdefault(c.ride_type, {"rides": 0, "seconds": 0.0})
        entry["rides"] += 1
        entry["seconds"] += seconds
    return pd.DataFrame(
        [
            {
                "ride_type": key,
                "label": RIDE_TYPE_LABELS[key],
                "rides": int(v["rides"]),
                "seconds": v["seconds"],
            }
            for key, v in rows.items()
        ],
        columns=["ride_type", "label", "rides", "seconds"],
    )
