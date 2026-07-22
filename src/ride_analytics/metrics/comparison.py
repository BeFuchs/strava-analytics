"""Compare two date periods of the ride history ("this season vs. last season").

Each period is summarized independently (totals, training load, power curve,
zones), then laid side by side in a delta table. When the periods differ in
length, absolute totals alone would mislead, so per-week normalized rows are
added and the result is flagged for the report to label.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from ride_analytics.config import AthleteConfig
from ride_analytics.ingest import Ride
from ride_analytics.metrics.climbs import ride_elevation_gain_m
from ride_analytics.metrics.pmc import compute_pmc
from ride_analytics.metrics.power_curve import (
    aggregate_power_curve,
    estimate_ftp,
    ride_power_curve,
)
from ride_analytics.metrics.single_ride import compute_ride_metrics
from ride_analytics.metrics.zones import (
    ZoneDistribution,
    aggregate_zone_distributions,
    hr_zone_distribution,
    power_zone_distribution,
)

TABLE_COLUMNS = ("metric", "period_a", "period_b", "delta_abs", "delta_pct")


@dataclass(frozen=True)
class PeriodSummary:
    label: str
    start: date
    end: date
    days: int
    n_rides: int
    total_distance_km: float
    total_elevation_gain_m: float
    total_moving_time_h: float
    total_tss: float
    avg_tss_per_week: float
    avg_ctl: float | None
    peak_ctl: float | None
    power_curve: dict[int, float]
    ftp_estimate: float | None
    power_zones: ZoneDistribution | None
    hr_zones: ZoneDistribution | None


@dataclass(frozen=True)
class ComparisonResult:
    period_a: PeriodSummary
    period_b: PeriodSummary
    table: pd.DataFrame
    equal_length: bool


def summarize_period(
    rides: list[Ride], config: AthleteConfig, label: str, start: date, end: date
) -> PeriodSummary:
    """Aggregate all rides whose start date falls into ``start..end`` (inclusive)."""
    selected = [r for r in rides if start <= r.metadata.start_time.date() <= end]
    days = (end - start).days + 1
    weeks = days / 7

    metrics = [compute_ride_metrics(r.df, config) for r in selected]
    gains = [ride_elevation_gain_m(r.df) for r in selected]
    tss_values = [m.tss for m in metrics if m.tss is not None]
    total_tss = sum(tss_values)

    pmc = compute_pmc(
        pd.DataFrame(
            {
                "date": [r.metadata.start_time for r in selected],
                "tss": [m.tss for m in metrics],
            }
        )
    )
    curve = aggregate_power_curve([ride_power_curve(r.df) for r in selected])

    return PeriodSummary(
        label=label,
        start=start,
        end=end,
        days=days,
        n_rides=len(selected),
        total_distance_km=sum(m.distance_km or 0.0 for m in metrics),
        total_elevation_gain_m=sum(g or 0.0 for g in gains),
        total_moving_time_h=sum(m.moving_time_s for m in metrics) / 3600,
        total_tss=total_tss,
        avg_tss_per_week=total_tss / weeks,
        avg_ctl=float(pmc["ctl"].mean()) if not pmc.empty else None,
        peak_ctl=float(pmc["ctl"].max()) if not pmc.empty else None,
        power_curve=curve,
        ftp_estimate=estimate_ftp(curve),
        power_zones=aggregate_zone_distributions(
            [power_zone_distribution(r.df, config) for r in selected]
        ),
        hr_zones=aggregate_zone_distributions(
            [hr_zone_distribution(r.df, config) for r in selected]
        ),
    )


def compare_periods(
    rides: list[Ride],
    config: AthleteConfig,
    period_a: tuple[date, date],
    period_b: tuple[date, date],
) -> ComparisonResult:
    """Summarize both periods and build the delta table.

    Empty periods yield zero/None values and blank deltas — never a crash.
    """
    a = summarize_period(rides, config, "A", *period_a)
    b = summarize_period(rides, config, "B", *period_b)
    equal_length = a.days == b.days

    rows = [
        _row("n_rides", a.n_rides, b.n_rides),
        _row("total_distance_km", a.total_distance_km, b.total_distance_km),
        _row("total_elevation_gain_m", a.total_elevation_gain_m, b.total_elevation_gain_m),
        _row("total_moving_time_h", a.total_moving_time_h, b.total_moving_time_h),
        _row("total_tss", a.total_tss, b.total_tss),
        _row("avg_tss_per_week", a.avg_tss_per_week, b.avg_tss_per_week),
        _row("avg_ctl", a.avg_ctl, b.avg_ctl),
        _row("peak_ctl", a.peak_ctl, b.peak_ctl),
        _row("ftp_estimate_watts", a.ftp_estimate, b.ftp_estimate),
    ]
    if not equal_length:
        # Unequal periods: absolute totals mislead, add per-week rows.
        for name, value_a, value_b in (
            ("rides_per_week", a.n_rides, b.n_rides),
            ("distance_per_week_km", a.total_distance_km, b.total_distance_km),
            ("elevation_gain_per_week_m", a.total_elevation_gain_m, b.total_elevation_gain_m),
            ("moving_time_per_week_h", a.total_moving_time_h, b.total_moving_time_h),
        ):
            rows.append(_row(name, value_a / (a.days / 7), value_b / (b.days / 7)))

    return ComparisonResult(
        period_a=a,
        period_b=b,
        table=pd.DataFrame(rows, columns=list(TABLE_COLUMNS)),
        equal_length=equal_length,
    )


def _row(metric: str, value_a: float | None, value_b: float | None) -> dict:
    delta_abs = None
    delta_pct = None
    if value_a is not None and value_b is not None:
        delta_abs = value_b - value_a
        if value_a != 0:
            delta_pct = delta_abs / value_a * 100
    return {
        "metric": metric,
        "period_a": value_a,
        "period_b": value_b,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
    }
