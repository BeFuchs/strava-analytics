"""Single-ride metrics: NP, IF, TSS, VI, work, moving/elapsed time, averages.

All functions are pure and operate on the normalized record DataFrame from
``ingest`` (columns: timestamp plus whatever sensors the ride carried).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ride_analytics.config import AthleteConfig

# Timestamp gaps above this are treated as pauses (auto-pause / GPS dropout),
# not riding time. Below it, the gap counts as recording interval.
MAX_SAMPLE_GAP_S = 10.0

# Fallback duration assigned to the first sample and to samples right after a
# pause, matching the typical 1 Hz recording interval.
DEFAULT_SAMPLE_DT_S = 1.0

NP_WINDOW = "30s"


@dataclass(frozen=True)
class RideMetrics:
    np_watts: float | None
    intensity_factor: float | None
    tss: float | None
    tss_estimated: bool  # True when TSS is the HR-based estimate (no power data)
    variability_index: float | None
    work_kj: float | None
    avg_power: float | None
    max_power: float | None
    avg_hr: float | None
    max_hr: float | None
    avg_cadence: float | None
    moving_time_s: float
    elapsed_time_s: float
    distance_km: float | None
    avg_speed_kmh: float | None


def sample_durations_s(df: pd.DataFrame) -> pd.Series:
    """Riding seconds represented by each sample.

    The duration of a sample is the gap to its predecessor; the first sample
    and samples following a pause (> ``MAX_SAMPLE_GAP_S``) count as one
    recording interval instead, so pauses add no riding time.
    """
    dt = df["timestamp"].diff().dt.total_seconds()
    dt.iloc[0] = DEFAULT_SAMPLE_DT_S
    dt[dt > MAX_SAMPLE_GAP_S] = DEFAULT_SAMPLE_DT_S
    return dt


def moving_time_s(df: pd.DataFrame) -> float:
    """Total riding time in seconds, pauses excluded."""
    if df.empty:
        return 0.0
    return float(sample_durations_s(df).sum())


def elapsed_time_s(df: pd.DataFrame) -> float:
    """Wall-clock seconds from first to last sample, pauses included."""
    if df.empty:
        return 0.0
    return float((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds())


def normalized_power(df: pd.DataFrame) -> float | None:
    """Normalized Power: 30-s rolling mean of power -> ^4 -> mean -> ^(1/4).

    The rolling window is time-based, so irregular sampling and gaps are
    handled; partial windows at the ride start are included.
    """
    if "power" not in df.columns or df["power"].notna().sum() == 0:
        return None
    rolling = df.set_index("timestamp")["power"].rolling(NP_WINDOW).mean()
    return float((rolling.pow(4).mean()) ** 0.25)


def intensity_factor(np_watts: float, ftp_watts: int) -> float:
    """Intensity Factor: NP relative to FTP."""
    return np_watts / ftp_watts


def training_stress_score(duration_s: float, np_watts: float, ftp_watts: int) -> float:
    """TSS = (duration_s * NP * IF) / (FTP * 3600) * 100; one hour at FTP = 100."""
    if_factor = intensity_factor(np_watts, ftp_watts)
    return (duration_s * np_watts * if_factor) / (ftp_watts * 3600) * 100


def variability_index(np_watts: float, avg_power: float) -> float | None:
    """Variability Index: NP / average power (1.0 = perfectly steady)."""
    if avg_power <= 0:
        return None
    return np_watts / avg_power


def work_kj(df: pd.DataFrame) -> float | None:
    """Mechanical work in kJ: power integrated over riding time."""
    if "power" not in df.columns or df["power"].notna().sum() == 0:
        return None
    dt = sample_durations_s(df)
    return float((df["power"].fillna(0) * dt).sum() / 1000)


def hr_tss(df: pd.DataFrame, threshold_hr: int) -> float | None:
    """HR-based TSS estimate for rides without power.

    Per-sample analogue of ``duration_h * (HR / threshold_HR)^2 * 100`` —
    the HR ratio stands in for IF. An approximation, always marked estimated.
    """
    if "heart_rate" not in df.columns or df["heart_rate"].notna().sum() == 0:
        return None
    dt = sample_durations_s(df)
    ratio_sq = (df["heart_rate"] / threshold_hr) ** 2
    return float((ratio_sq.fillna(0) * dt).sum() / 3600 * 100)


def compute_ride_metrics(df: pd.DataFrame, config: AthleteConfig) -> RideMetrics:
    """Compute all single-ride metrics; missing sensors yield ``None`` fields."""
    has_power = "power" in df.columns and df["power"].notna().any()
    has_hr = "heart_rate" in df.columns and df["heart_rate"].notna().any()

    moving_s = moving_time_s(df)
    elapsed_s = elapsed_time_s(df)

    np_watts = normalized_power(df) if has_power else None
    avg_power = float(df["power"].mean()) if has_power else None

    if_factor = None
    tss = None
    tss_estimated = False
    vi = None
    if np_watts is not None:
        if_factor = intensity_factor(np_watts, config.ftp_watts)
        tss = training_stress_score(moving_s, np_watts, config.ftp_watts)
        if avg_power:
            vi = variability_index(np_watts, avg_power)
    elif has_hr:
        tss = hr_tss(df, config.threshold_hr)
        tss_estimated = tss is not None

    distance_km = None
    if "distance" in df.columns and df["distance"].notna().any():
        distance_km = float((df["distance"].max() - df["distance"].min()) / 1000)

    avg_speed_kmh = None
    if distance_km is not None and moving_s > 0:
        avg_speed_kmh = distance_km / (moving_s / 3600)
    elif "speed" in df.columns and df["speed"].notna().any():
        avg_speed_kmh = float(df["speed"].mean() * 3.6)

    return RideMetrics(
        np_watts=np_watts,
        intensity_factor=if_factor,
        tss=tss,
        tss_estimated=tss_estimated,
        variability_index=vi,
        work_kj=work_kj(df) if has_power else None,
        avg_power=avg_power,
        max_power=float(df["power"].max()) if has_power else None,
        avg_hr=float(df["heart_rate"].mean()) if has_hr else None,
        max_hr=float(df["heart_rate"].max()) if has_hr else None,
        avg_cadence=(
            float(df["cadence"].mean())
            if "cadence" in df.columns and df["cadence"].notna().any()
            else None
        ),
        moving_time_s=moving_s,
        elapsed_time_s=elapsed_s,
        distance_km=distance_km,
        avg_speed_kmh=avg_speed_kmh,
    )
