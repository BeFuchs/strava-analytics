"""Aerobic decoupling, efficiency factor and cardiac drift.

Where durability asks *whether* power fades late in a ride, this asks *why*:
if heart rate drifts up while power stays flat, the aerobic base isn't deep
enough for that duration. The efficiency factor (NP per heartbeat) tracks that
efficiency over time; decoupling measures its drift within a single ride.

Pure functions over the normalized record DataFrame from ``ingest`` — no HTTP,
no HTML, no file system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from ride_analytics.config import AthleteConfig
from ride_analytics.metrics.single_ride import (
    normalized_power,
    sample_durations_s,
)

# Below this, the aerobic base counts as sufficient for the ride's duration.
DECOUPLING_THRESHOLD_PCT = 5.0

# Validity gates — outside these, decoupling is noise, not signal.
MIN_MOVING_TIME_S = 3600.0  # 60 min; shorter rides don't decouple meaningfully
MIN_SENSOR_COVERAGE = 0.90  # power and HR must each be present in ≥ 90 % of samples
MAX_VI_FOR_DECOUPLING = 1.15  # surging power (intervals, city traffic) breaks the metric

# Ride types whose decoupling is not comparable to steady rides (see classification).
NON_COMPARABLE_TYPES = frozenset({"race", "intervals"})

# Cardiac drift: HR slope over a narrow power band around the ride average.
DRIFT_POWER_BAND = 0.10  # ± 10 % of average power
MIN_DRIFT_SAMPLES = 120  # need a couple of minutes in-band for a stable slope

# Four-week moving average for the EF trend line.
EF_TREND_WINDOW_DAYS = 28


@dataclass(frozen=True)
class RideDecoupling:
    date: datetime
    ef: float | None
    decoupling_pct: float | None
    cardiac_drift_bpm_per_h: float | None
    valid: bool
    reason: str | None


def efficiency_factor(df: pd.DataFrame) -> float | None:
    """EF = Normalized Power / average heart rate; ``None`` without both sensors."""
    if not _has(df, "power") or not _has(df, "heart_rate"):
        return None
    np_watts = normalized_power(df)
    avg_hr = float(df["heart_rate"].mean())
    if np_watts is None or avg_hr <= 0:
        return None
    return np_watts / avg_hr


def ride_decoupling(
    df: pd.DataFrame, config: AthleteConfig, *, ride_type: str | None = None
) -> RideDecoupling:
    """Efficiency factor, aerobic decoupling and cardiac drift for one ride.

    ``ride_type`` (from classification) lets the caller mark interval/race rides
    as not comparable; it is optional so this module stays independent.
    """
    date = df["timestamp"].iloc[0].to_pydatetime()
    ef = efficiency_factor(df)
    drift = cardiac_drift_bpm_per_h(df)

    reason = _invalid_reason(df, config, ride_type)
    if reason is not None:
        return RideDecoupling(date, ef, None, drift, valid=False, reason=reason)

    decoupling = _decoupling_pct(df)
    if decoupling is None:
        return RideDecoupling(date, ef, None, drift, valid=False, reason="Hälften nicht auswertbar")
    return RideDecoupling(date, ef, decoupling, drift, valid=True, reason=None)


def _invalid_reason(df: pd.DataFrame, config: AthleteConfig, ride_type: str | None) -> str | None:
    """The first violated validity gate, or ``None`` when the ride qualifies."""
    if ride_type in NON_COMPARABLE_TYPES:
        return "Fahrtentyp nicht vergleichbar (Intervalle/Rennen)"
    if not _has(df, "power") or not _has(df, "heart_rate"):
        return "Leistung oder Herzfrequenz fehlt"
    if (
        _coverage(df, "power") < MIN_SENSOR_COVERAGE
        or _coverage(df, "heart_rate") < MIN_SENSOR_COVERAGE
    ):
        return "Leistung oder Herzfrequenz unvollständig (< 90 %)"
    if float(sample_durations_s(df).sum()) < MIN_MOVING_TIME_S:
        return "Fahrt unter 60 Minuten"
    vi = _variability_index(df)
    if vi is None or vi > MAX_VI_FOR_DECOUPLING:
        return "Leistung zu variabel (VI > 1,15)"
    return None


def _decoupling_pct(df: pd.DataFrame) -> float | None:
    """Percentage EF drop from the first to the second moving-time half."""
    first, second = _split_by_moving_time(df)
    ef_first = efficiency_factor(first)
    ef_second = efficiency_factor(second)
    if ef_first is None or ef_second is None or ef_first <= 0:
        return None
    return (ef_first - ef_second) / ef_first * 100


def _split_by_moving_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split at the sample where cumulative moving time crosses the halfway mark.

    Split on moving time, not elapsed time, so long mid-ride pauses don't push
    the boundary into the wrong half.
    """
    cumulative = sample_durations_s(df).cumsum()
    half = cumulative.iloc[-1] / 2
    split = int((cumulative <= half).sum())
    split = min(max(split, 1), len(df) - 1)
    return df.iloc[:split], df.iloc[split:]


def cardiac_drift_bpm_per_h(df: pd.DataFrame) -> float | None:
    """HR slope (bpm per hour) over samples within ±10 % of average power.

    Restricting to a narrow power band isolates the drift from intensity
    changes, so it survives on rides where plain decoupling would not.
    """
    if not _has(df, "power") or not _has(df, "heart_rate"):
        return None
    avg_power = float(df["power"].mean())
    if avg_power <= 0:
        return None
    low, high = avg_power * (1 - DRIFT_POWER_BAND), avg_power * (1 + DRIFT_POWER_BAND)
    in_band = df[(df["power"] >= low) & (df["power"] <= high) & df["heart_rate"].notna()]
    if len(in_band) < MIN_DRIFT_SAMPLES:
        return None
    elapsed_h = (in_band["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 3600
    if elapsed_h.max() - elapsed_h.min() <= 0:
        return None
    slope = np.polyfit(elapsed_h.to_numpy(), in_band["heart_rate"].to_numpy(), 1)[0]
    return float(slope)


def decoupling_frame(results: list[RideDecoupling]) -> pd.DataFrame:
    """History table ``date, ef, decoupling_pct, cardiac_drift_bpm_per_h, valid, reason``."""
    return pd.DataFrame(
        {
            "date": [r.date for r in results],
            "ef": [r.ef for r in results],
            "decoupling_pct": [r.decoupling_pct for r in results],
            "cardiac_drift_bpm_per_h": [r.cardiac_drift_bpm_per_h for r in results],
            "valid": [r.valid for r in results],
            "reason": [r.reason for r in results],
        }
    )


def ef_trend_series(frame: pd.DataFrame, window_days: int = EF_TREND_WINDOW_DAYS) -> pd.Series:
    """Time-based rolling mean of EF over valid rides, indexed by date."""
    valid = frame[frame["valid"] & frame["ef"].notna()]
    if valid.empty:
        return pd.Series(dtype=float)
    series = valid.set_index(pd.to_datetime(valid["date"]))["ef"].sort_index()
    return series.rolling(f"{window_days}D").mean()


def _has(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()


def _coverage(df: pd.DataFrame, column: str) -> float:
    return float(df[column].notna().mean()) if column in df.columns else 0.0


def _variability_index(df: pd.DataFrame) -> float | None:
    np_watts = normalized_power(df)
    avg_power = float(df["power"].mean())
    if np_watts is None or avg_power <= 0:
        return None
    return np_watts / avg_power
