"""Mean-maximal power curve and FTP estimation from the best 20-min effort."""

from __future__ import annotations

import pandas as pd

STANDARD_WINDOWS_S = (5, 15, 30, 60, 300, 480, 1200, 3600)

FTP_WINDOW_S = 1200
FTP_FACTOR = 0.95


def power_grid_1hz(df: pd.DataFrame) -> pd.Series | None:
    """Power on a 1 Hz grid over the ride's elapsed span; ``None`` without power.

    Seconds without a sample (pauses, dropouts) count as 0 W — you are not
    pedaling while paused, so long-window bests stay honest.
    """
    if "power" not in df.columns or df["power"].notna().sum() == 0:
        return None
    return df.set_index("timestamp")["power"].resample("1s").mean().fillna(0.0)


def mean_max_power(grid: pd.Series, windows: tuple[int, ...]) -> dict[int, float]:
    """Best rolling-mean power (W) per window over a contiguous 1 Hz power series.

    Windows longer than the series are omitted.
    """
    curve: dict[int, float] = {}
    for window in windows:
        if len(grid) < window:
            continue
        curve[window] = float(grid.rolling(window, min_periods=window).mean().max())
    return curve


def ride_power_curve(
    df: pd.DataFrame, windows: tuple[int, ...] = STANDARD_WINDOWS_S
) -> dict[int, float]:
    """Best average power (W) per window; windows longer than the ride are omitted."""
    grid = power_grid_1hz(df)
    if grid is None:
        return {}
    return mean_max_power(grid, windows)


def aggregate_power_curve(curves: list[dict[int, float]]) -> dict[int, float]:
    """Element-wise maximum over per-ride curves (all-time bests per window)."""
    best: dict[int, float] = {}
    for curve in curves:
        for window, watts in curve.items():
            if watts > best.get(window, float("-inf")):
                best[window] = watts
    return dict(sorted(best.items()))


def estimate_ftp(curve: dict[int, float]) -> float | None:
    """FTP estimate: best 20-min average power x 0.95.

    An estimate only — the configured FTP stays authoritative for TSS/IF.
    Returns ``None`` when no 20-min best exists.
    """
    best_20min = curve.get(FTP_WINDOW_S)
    if best_20min is None:
        return None
    return best_20min * FTP_FACTOR
