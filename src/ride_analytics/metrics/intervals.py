"""Detect structured interval blocks from power and evaluate them.

Pipeline: smooth power, mark stretches above a Zone-4 threshold as candidates,
clean them up (drop too-short ones, bridge brief dips, merge near-adjacent
ones), then group repetitions with similar duration and power into sets.

Power-based only — heart rate reacts far too slowly to delimit intervals, so
rides without power are skipped. Everything runs on a 1 Hz grid so that all the
"seconds" thresholds map directly to samples; pauses resample to 0 W, which
correctly ends an interval.

Pure functions over the normalized record DataFrame from ``ingest``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ride_analytics.config import AthleteConfig

# Candidate detection.
SMOOTH_WINDOW_S = 10  # rolling mean to strip cadence noise and single-sample spikes
INTERVAL_FTP_FRACTION = 0.88  # lower edge of Zone 4 / top of sweetspot
NP_WINDOW_S = 30  # normalized-power rolling window

# Cleanup. The spec's two rules — bridge dips < 10 s, merge candidates < 20 s
# apart — both join runs across a short sub-threshold gap, so they collapse to a
# single 20 s join threshold (the shorter dip case is a subset of it).
MIN_INTERVAL_S = 30  # shorter efforts are sprints/ramps, not intervals
MERGE_GAP_S = 20

# Set grouping: consecutive efforts of similar shape are repetitions of one set.
SET_DURATION_TOLERANCE = 0.20  # ± 20 %
SET_POWER_TOLERANCE = 0.10  # ± 10 %
MIN_SET_REPS = 2


@dataclass(frozen=True)
class Interval:
    start_offset_s: float
    duration_s: float
    avg_power: float
    np_watts: float
    max_power: float
    watts_per_kg: float
    avg_hr: float | None
    max_hr: float | None
    hr_end: float | None
    avg_cadence: float | None
    kj_before: float


@dataclass(frozen=True)
class IntervalSet:
    reps: int
    avg_duration_s: float
    avg_power: float
    power_fade_pct: float  # (first rep - last rep) / first rep, in %; positive = faded
    power_std: float  # SD of per-rep average power; low = even pacing
    avg_rest_duration_s: float | None
    avg_rest_power: float | None
    avg_rest_hr_drop: float | None  # mean HR recovered during the rests, bpm
    intervals: tuple[Interval, ...]


def detect_intervals(df: pd.DataFrame, config: AthleteConfig) -> list[IntervalSet]:
    """Detect interval sets in one ride; ``[]`` without power or without intervals."""
    if "power" not in df.columns or df["power"].notna().sum() == 0:
        return []
    grid = _one_hz_grid(df)
    threshold = INTERVAL_FTP_FRACTION * config.ftp_watts

    runs = _candidate_runs(grid["power"], threshold)
    runs = _merge_close_runs(runs, MERGE_GAP_S)
    runs = [(s, e) for s, e in runs if e - s + 1 >= MIN_INTERVAL_S]
    if not runs:
        return []

    return [_build_set(grid, group, config) for group in _group_into_sets(grid, runs)]


def _one_hz_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Resample the ride to a gap-free 1 Hz grid; power gaps become 0 W."""
    indexed = df.set_index("timestamp")
    grid = pd.DataFrame({"power": indexed["power"].resample("1s").mean().fillna(0.0)})
    for col in ("heart_rate", "cadence"):
        if col in indexed.columns:
            grid[col] = indexed[col].resample("1s").mean().interpolate(limit=30)
    return grid.reset_index(drop=True)


def _candidate_runs(power: pd.Series, threshold: float) -> list[tuple[int, int]]:
    """Contiguous stretches whose smoothed power clears the threshold (inclusive ends)."""
    smoothed = power.rolling(SMOOTH_WINDOW_S, center=True, min_periods=1).mean()
    return _runs((smoothed >= threshold).to_numpy())


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge_close_runs(runs: list[tuple[int, int]], max_gap_s: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 < max_gap_s:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _group_into_sets(
    grid: pd.DataFrame, runs: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    """Group consecutive runs of similar duration and power into sets.

    A lone run that matches neither neighbour becomes its own single-rep set.
    """
    power = grid["power"].to_numpy()
    groups: list[list[tuple[int, int]]] = [[runs[0]]]
    for run in runs[1:]:
        group = groups[-1]
        mean_dur = np.mean([e - s + 1 for s, e in group])
        mean_pow = np.mean([power[s : e + 1].mean() for s, e in group])
        dur = run[1] - run[0] + 1
        pw = power[run[0] : run[1] + 1].mean()
        if _within(dur, mean_dur, SET_DURATION_TOLERANCE) and _within(
            pw, mean_pow, SET_POWER_TOLERANCE
        ):
            group.append(run)
        else:
            groups.append([run])
    return groups


def _within(value: float, reference: float, tolerance: float) -> bool:
    return reference > 0 and abs(value - reference) <= tolerance * reference


def _build_interval(grid: pd.DataFrame, start: int, end: int, config: AthleteConfig) -> Interval:
    segment = grid.iloc[start : end + 1]
    power = segment["power"]
    avg_power = float(power.mean())
    np_watts = float((power.rolling(NP_WINDOW_S, min_periods=1).mean().pow(4).mean()) ** 0.25)
    kj_before = float(grid["power"].iloc[:start].sum() / 1000)
    return Interval(
        start_offset_s=float(start),
        duration_s=float(end - start + 1),
        avg_power=avg_power,
        np_watts=np_watts,
        max_power=float(power.max()),
        watts_per_kg=avg_power / config.weight_kg,
        avg_hr=_mean(segment, "heart_rate"),
        max_hr=_max(segment, "heart_rate"),
        hr_end=_last(segment, "heart_rate"),
        avg_cadence=_mean(segment, "cadence"),
        kj_before=kj_before,
    )


def _build_set(
    grid: pd.DataFrame, runs: list[tuple[int, int]], config: AthleteConfig
) -> IntervalSet:
    intervals = [_build_interval(grid, s, e, config) for s, e in runs]
    powers = [iv.avg_power for iv in intervals]
    fade = (powers[0] - powers[-1]) / powers[0] * 100 if powers[0] > 0 else 0.0

    rests = _rest_metrics(grid, runs)
    return IntervalSet(
        reps=len(intervals),
        avg_duration_s=float(np.mean([iv.duration_s for iv in intervals])),
        avg_power=float(np.mean(powers)),
        power_fade_pct=float(fade),
        power_std=float(np.std(powers)),
        avg_rest_duration_s=rests["duration"],
        avg_rest_power=rests["power"],
        avg_rest_hr_drop=rests["hr_drop"],
        intervals=tuple(intervals),
    )


def _rest_metrics(grid: pd.DataFrame, runs: list[tuple[int, int]]) -> dict[str, float | None]:
    """Average duration, power and HR recovery of the gaps between reps."""
    if len(runs) < 2:
        return {"duration": None, "power": None, "hr_drop": None}
    durations, powers, hr_drops = [], [], []
    hr = grid["heart_rate"] if "heart_rate" in grid.columns else None
    for (_, end), (nxt_start, _) in zip(runs, runs[1:], strict=False):
        rest = grid["power"].iloc[end + 1 : nxt_start]
        if rest.empty:
            continue
        durations.append(nxt_start - end - 1)
        powers.append(float(rest.mean()))
        if hr is not None:
            start_hr, end_hr = hr.iloc[end], hr.iloc[nxt_start]
            if pd.notna(start_hr) and pd.notna(end_hr):
                hr_drops.append(float(start_hr - end_hr))
    return {
        "duration": float(np.mean(durations)) if durations else None,
        "power": float(np.mean(powers)) if powers else None,
        "hr_drop": float(np.mean(hr_drops)) if hr_drops else None,
    }


def intervals_frame(sets: list[IntervalSet]) -> pd.DataFrame:
    """Flat one-row-per-interval table with set membership, for CSV export."""
    rows = []
    for set_index, interval_set in enumerate(sets):
        for rep_index, iv in enumerate(interval_set.intervals):
            rows.append(
                {
                    "set_index": set_index,
                    "rep_index": rep_index,
                    "set_reps": interval_set.reps,
                    "start_offset_s": round(iv.start_offset_s),
                    "duration_s": round(iv.duration_s),
                    "avg_power_watts": round(iv.avg_power, 1),
                    "np_watts": round(iv.np_watts, 1),
                    "max_power_watts": round(iv.max_power),
                    "watts_per_kg": round(iv.watts_per_kg, 2),
                    "avg_hr_bpm": _round(iv.avg_hr, 1),
                    "max_hr_bpm": _round(iv.max_hr, 0),
                    "hr_end_bpm": _round(iv.hr_end, 0),
                    "avg_cadence_rpm": _round(iv.avg_cadence, 1),
                    "kj_before": round(iv.kj_before, 1),
                }
            )
    return pd.DataFrame(rows)


def _mean(segment: pd.DataFrame, column: str) -> float | None:
    if column not in segment.columns or segment[column].notna().sum() == 0:
        return None
    return float(segment[column].mean())


def _max(segment: pd.DataFrame, column: str) -> float | None:
    if column not in segment.columns or segment[column].notna().sum() == 0:
        return None
    return float(segment[column].max())


def _last(segment: pd.DataFrame, column: str) -> float | None:
    if column not in segment.columns:
        return None
    valid = segment[column].dropna()
    return float(valid.iloc[-1]) if len(valid) else None


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)
