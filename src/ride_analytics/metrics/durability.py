"""Durability / fatigue resistance: mean-maximal power after cumulative work.

Instead of asking "how strong are you", this asks "how strong are you *still*,
after having already done x kJ of work": each ride is segmented by cumulative
work into kJ buckets, and the mean-maximal power per window is computed from
the samples inside each bucket only. Cumulative work is monotonic, so every
bucket covers one contiguous stretch of the ride.

Durability is power-based; rides without power are skipped (no HR fallback).
"""

from __future__ import annotations

import pandas as pd

from ride_analytics.metrics.power_curve import mean_max_power, power_grid_1hz

# Upper bucket edges in kJ; the last bucket is open-ended (3000+ kJ).
DEFAULT_BUCKET_EDGES_KJ = (1000, 2000, 3000)

DURABILITY_WINDOWS_S = (5, 60, 300, 1200)


def bucket_labels(edges_kj: tuple[int, ...] = DEFAULT_BUCKET_EDGES_KJ) -> tuple[str, ...]:
    """Human-readable labels for the kJ buckets, e.g. ``0-1000 kJ`` … ``3000+ kJ``."""
    labels = []
    lower = 0
    for edge in edges_kj:
        labels.append(f"{lower}-{edge} kJ")
        lower = edge
    labels.append(f"{lower}+ kJ")
    return tuple(labels)


def ride_bucket_curves(
    df: pd.DataFrame,
    edges_kj: tuple[int, ...] = DEFAULT_BUCKET_EDGES_KJ,
    windows: tuple[int, ...] = DURABILITY_WINDOWS_S,
) -> dict[int, dict[int, float]] | None:
    """MMP per kJ bucket for one ride: ``{bucket_index: {window_s: watts}}``.

    A window is only computed when the bucket's stretch of the ride is at
    least that long. Returns ``None`` for rides without power data.
    """
    grid = power_grid_1hz(df)
    if grid is None:
        return None

    cumulative_kj = grid.cumsum() / 1000.0
    edges = [0.0, *(float(e) for e in edges_kj), float("inf")]
    curves: dict[int, dict[int, float]] = {}
    for i in range(len(edges) - 1):
        segment = grid[(cumulative_kj >= edges[i]) & (cumulative_kj < edges[i + 1])]
        if segment.empty:
            continue
        curve = mean_max_power(segment, windows)
        if curve:
            curves[i] = curve
    return curves


def compute_durability(
    dfs: list[pd.DataFrame],
    edges_kj: tuple[int, ...] = DEFAULT_BUCKET_EDGES_KJ,
    windows: tuple[int, ...] = DURABILITY_WINDOWS_S,
) -> pd.DataFrame:
    """Aggregate durability over the ride history.

    Element-wise maximum per (bucket × window) across all rides with power.
    Returns one row per bucket × window: ``bucket, window_s, mmp_watts,
    durability_index, n_rides``. Buckets without enough data keep ``mmp_watts``
    missing (NaN) — never 0, which would fake a collapse where data is absent.
    The durability index is the drop relative to the fresh bucket:
    ``mmp(bucket) / mmp(bucket_0)`` for the same window.
    """
    labels = bucket_labels(edges_kj)
    best: dict[tuple[int, int], float] = {}
    n_rides: dict[tuple[int, int], int] = {}

    for df in dfs:
        curves = ride_bucket_curves(df, edges_kj, windows)
        if curves is None:
            continue
        for bucket, curve in curves.items():
            for window, watts in curve.items():
                key = (bucket, window)
                n_rides[key] = n_rides.get(key, 0) + 1
                if watts > best.get(key, float("-inf")):
                    best[key] = watts

    rows = []
    for bucket, label in enumerate(labels):
        for window in windows:
            mmp = best.get((bucket, window))
            fresh = best.get((0, window))
            index = mmp / fresh if mmp is not None and fresh else None
            rows.append(
                {
                    "bucket": label,
                    "window_s": window,
                    "mmp_watts": mmp,
                    "durability_index": index,
                    "n_rides": n_rides.get((bucket, window), 0),
                }
            )
    return pd.DataFrame(rows)
