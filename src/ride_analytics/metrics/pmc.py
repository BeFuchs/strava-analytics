"""Performance Management Chart: CTL / ATL / TSB from the daily TSS series.

Classic impulse-response model: an exponentially weighted moving average of
daily TSS with a 42-day time constant (CTL, "fitness") and a 7-day one
(ATL, "fatigue"). TSB ("form") is yesterday's CTL minus yesterday's ATL.
"""

from __future__ import annotations

import pandas as pd

CTL_DAYS = 42
ATL_DAYS = 7


def compute_pmc(
    rides_tss: pd.DataFrame,
    ctl_days: int = CTL_DAYS,
    atl_days: int = ATL_DAYS,
) -> pd.DataFrame:
    """Build the day-continuous PMC DataFrame ``date, tss, ctl, atl, tsb``.

    ``rides_tss`` needs columns ``date`` and ``tss`` (one row per ride; rows
    with missing TSS are ignored). Multiple rides on one day are summed, days
    without a ride enter as TSS 0 so the model keeps decaying.
    """
    if rides_tss.empty:
        return pd.DataFrame(columns=["date", "tss", "ctl", "atl", "tsb"])

    rides = rides_tss.dropna(subset=["tss"])
    if rides.empty:
        return pd.DataFrame(columns=["date", "tss", "ctl", "atl", "tsb"])

    days = pd.to_datetime(rides["date"]).dt.normalize()
    per_day = rides.groupby(days)["tss"].sum()

    idx = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D")
    tss = per_day.reindex(idx, fill_value=0.0).astype(float)

    ctl = _ewma(tss, ctl_days)
    atl = _ewma(tss, atl_days)
    tsb = ctl.shift(1, fill_value=0.0) - atl.shift(1, fill_value=0.0)

    return pd.DataFrame(
        {
            "date": idx,
            "tss": tss.to_numpy(),
            "ctl": ctl.to_numpy(),
            "atl": atl.to_numpy(),
            "tsb": tsb.to_numpy(),
        }
    )


def _ewma(values: pd.Series, time_constant_days: int) -> pd.Series:
    """EWMA with ``alpha = 1/time_constant``, seeded at 0 (no prior training).

    Equivalent to the recursion ``load_today = load_yesterday +
    (tss_today - load_yesterday) / time_constant``.
    """
    seeded = pd.concat([pd.Series([0.0]), values.reset_index(drop=True)], ignore_index=True)
    smoothed = seeded.ewm(alpha=1 / time_constant_days, adjust=False).mean()
    return smoothed.iloc[1:].reset_index(drop=True)
