"""Assemble the report data model and render the self-contained HTML report.

This module orchestrates the metric functions and formats their results; the
metric math itself lives in ``metrics/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.offline import get_plotlyjs

from ride_analytics.config import AthleteConfig
from ride_analytics.ingest import Ride
from ride_analytics.metrics.climbs import (
    Climb,
    detect_climbs,
    match_climbs,
    ride_elevation_gain_m,
    smooth_altitude,
)
from ride_analytics.metrics.comparison import ComparisonResult, PeriodSummary
from ride_analytics.metrics.durability import compute_durability
from ride_analytics.metrics.pmc import compute_pmc
from ride_analytics.metrics.power_curve import (
    aggregate_power_curve,
    estimate_ftp,
    ride_power_curve,
)
from ride_analytics.metrics.single_ride import RideMetrics, compute_ride_metrics
from ride_analytics.metrics.zones import (
    ZoneDistribution,
    aggregate_zone_distributions,
    hr_zone_distribution,
    power_zone_distribution,
)

# Reference dataviz palette (light mode), unchanged.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_CTL_BLUE = "#2a78d6"
_ATL_ORANGE = "#eb6834"
_TSB_AQUA = "#1baf7a"
_TSS_BAR = "#e1e0d9"
# Ordinal blue ramps (light -> dark = easy -> hard zone).
_POWER_RAMP = ("#86b6ef", "#5598e7", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b")
_HR_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
# Ramp steps light enough to need ink (not white) labels inside the segment.
_LIGHT_RAMP_STEPS = 2

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_WINDOW_LABELS = {
    5: "5s",
    15: "15s",
    30: "30s",
    60: "1m",
    300: "5m",
    480: "8m",
    1200: "20m",
    3600: "60m",
}

# One line color per kJ bucket, light -> dark = fresh -> fatigued.
_DURABILITY_RAMP = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")

# Climb highlight on the elevation profile.
_CLIMB_FILL = "rgba(235, 104, 52, 0.35)"
_PROFILE_FILL = "rgba(42, 120, 214, 0.15)"

_TOP_CLIMBS_N = 10
_EVOLUTION_MIN_EFFORTS = 3
_EVOLUTION_MAX_GROUPS = 3


@dataclass(frozen=True)
class AnalyzedRide:
    ride: Ride
    metrics: RideMetrics
    power_curve: dict[int, float]
    power_zones: ZoneDistribution | None
    hr_zones: ZoneDistribution | None
    climbs: list[Climb]
    elevation_gain_m: float | None


@dataclass(frozen=True)
class ReportData:
    rides: list[AnalyzedRide]
    pmc: pd.DataFrame
    power_curve: dict[int, float]
    ftp_estimate: float | None
    power_zones: ZoneDistribution | None
    hr_zones: ZoneDistribution | None
    durability: pd.DataFrame
    climb_groups: list[list[Climb]]


def build_report_data(rides: list[Ride], config: AthleteConfig) -> ReportData:
    """Run all metrics over the loaded rides and collect the report data model."""
    analyzed = [
        AnalyzedRide(
            ride=ride,
            metrics=compute_ride_metrics(ride.df, config),
            power_curve=ride_power_curve(ride.df),
            power_zones=power_zone_distribution(ride.df, config),
            hr_zones=hr_zone_distribution(ride.df, config),
            climbs=detect_climbs(ride.df, config),
            elevation_gain_m=ride_elevation_gain_m(ride.df),
        )
        for ride in rides
    ]

    pmc = compute_pmc(
        pd.DataFrame(
            {
                "date": [a.ride.metadata.start_time for a in analyzed],
                "tss": [a.metrics.tss for a in analyzed],
            }
        )
    )
    curve = aggregate_power_curve([a.power_curve for a in analyzed])

    return ReportData(
        rides=analyzed,
        pmc=pmc,
        power_curve=curve,
        ftp_estimate=estimate_ftp(curve),
        power_zones=aggregate_zone_distributions([a.power_zones for a in analyzed]),
        hr_zones=aggregate_zone_distributions([a.hr_zones for a in analyzed]),
        durability=compute_durability([ride.df for ride in rides]),
        climb_groups=match_climbs([climb for a in analyzed for climb in a.climbs]),
    )


def render_report(data: ReportData, config: AthleteConfig, out_path: str | Path) -> Path:
    """Render the self-contained HTML report (Plotly inline, no CDN) to ``out_path``."""
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("report.html.j2")

    has_durability = data.durability["mmp_watts"].notna().any()
    profile_ride = next((a for a in reversed(data.rides) if a.climbs), None)
    evolution = _evolution_groups(data)

    html = template.render(
        comparison=None,
        generated=f"{datetime.now():%d %b %Y %H:%M}",
        totals=totals(data),
        rows=ride_rows(data),
        config=config,
        ftp_wkg=f"{config.ftp_watts / config.weight_kg:.1f}",
        ftp_estimate=_fmt_ftp_estimate(data.ftp_estimate, config),
        any_estimated_tss=any(a.metrics.tss_estimated for a in data.rides),
        pmc_div=_to_div(_pmc_figure(data.pmc)) if not data.pmc.empty else None,
        power_curve_div=_to_div(_power_curve_figure(data.power_curve))
        if data.power_curve
        else None,
        power_zones_div=(
            _to_div(_zones_figure(data.power_zones, _POWER_RAMP)) if data.power_zones else None
        ),
        hr_zones_div=_to_div(_zones_figure(data.hr_zones, _HR_RAMP)) if data.hr_zones else None,
        durability_div=_to_div(_durability_figure(data.durability)) if has_durability else None,
        durability_index=durability_index_table(data) if has_durability else None,
        climb_rows=climb_rows(data),
        n_climbs=sum(len(a.climbs) for a in data.rides),
        profile_div=_to_div(_elevation_profile_figure(profile_ride)) if profile_ride else None,
        profile_caption=(
            f"{profile_ride.ride.metadata.start_time:%d %b %Y} · "
            f"{profile_ride.ride.metadata.source}"
            if profile_ride
            else None
        ),
        evolution_div=_to_div(_climb_evolution_figure(evolution)) if evolution else None,
        plotlyjs=get_plotlyjs(),
    )

    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_comparison_report(
    result: ComparisonResult, config: AthleteConfig, out_path: str | Path
) -> Path:
    """Render the period-comparison variant of the report."""
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("report.html.j2")

    a, b = result.period_a, result.period_b
    html = template.render(
        comparison=result,
        generated=f"{datetime.now():%d %b %Y %H:%M}",
        config=config,
        ftp_wkg=f"{config.ftp_watts / config.weight_kg:.1f}",
        period_a_label=f"{a.start:%d %b %Y} – {a.end:%d %b %Y}",
        period_b_label=f"{b.start:%d %b %Y} – {b.end:%d %b %Y}",
        comparison_rows=comparison_rows(result),
        comparison_curves_div=(
            _to_div(_comparison_curves_figure(a, b)) if a.power_curve or b.power_curve else None
        ),
        comparison_power_zones_div=(
            _to_div(_comparison_zones_figure(a.power_zones, b.power_zones))
            if a.power_zones and b.power_zones
            else None
        ),
        comparison_hr_zones_div=(
            _to_div(_comparison_zones_figure(a.hr_zones, b.hr_zones))
            if a.hr_zones and b.hr_zones
            else None
        ),
        plotlyjs=get_plotlyjs(),
    )

    out_path = Path(out_path)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def totals(data: ReportData) -> dict[str, str]:
    """Display-ready headline figures shared by report header and CLI summary."""
    rides = data.rides
    distance = sum(a.metrics.distance_km or 0.0 for a in rides)
    moving = sum(a.metrics.moving_time_s for a in rides)
    tss = sum(a.metrics.tss or 0.0 for a in rides)
    if rides:
        start = min(a.ride.metadata.start_time for a in rides)
        end = max(a.ride.metadata.start_time for a in rides)
        period = f"{start:%d %b %Y} – {end:%d %b %Y}"
    else:
        period = "–"
    return {
        "rides": str(len(rides)),
        "period": period,
        "distance": f"{distance:,.0f} km",
        "moving_time": _fmt_duration(moving),
        "tss": f"{tss:,.0f}",
    }


def ride_rows(data: ReportData) -> list[dict[str, str]]:
    """Display-ready table rows shared by the HTML report and the CLI summary."""
    rows = []
    for a in data.rides:
        m = a.metrics
        rows.append(
            {
                "date": f"{a.ride.metadata.start_time:%Y-%m-%d}",
                "source": a.ride.metadata.source,
                "distance": _fmt(m.distance_km, "{:.1f}"),
                "duration": _fmt_duration(m.moving_time_s),
                "np": _fmt(m.np_watts, "{:.0f}"),
                "if": _fmt(m.intensity_factor, "{:.2f}"),
                "tss": _fmt(m.tss, "{:.0f}") + ("*" if m.tss_estimated else ""),
                "avg_hr": _fmt(m.avg_hr, "{:.0f}"),
                "max_hr": _fmt(m.max_hr, "{:.0f}"),
            }
        )
    return rows


def _fmt(value, spec: str) -> str:
    return spec.format(value) if value is not None else "–"


def _fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _fmt_ftp_estimate(ftp_estimate: float | None, config: AthleteConfig) -> str | None:
    if ftp_estimate is None:
        return None
    wkg = ftp_estimate / config.weight_kg
    return f"{ftp_estimate:.0f} W ({wkg:.1f} W/kg)"


def _to_div(fig: go.Figure) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"displayModeBar": False, "responsive": True},
    )


def _base_layout(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        height=height,
        paper_bgcolor=_SURFACE,
        plot_bgcolor=_SURFACE,
        font=dict(family=_FONT, size=13, color=_INK_SECONDARY),
        margin=dict(l=56, r=24, t=16, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=_INK)),
        barcornerradius=4,
    )
    fig.update_xaxes(gridcolor=_GRID, linecolor=_AXIS, tickfont=dict(color=_MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=_GRID, linecolor=_AXIS, tickfont=dict(color=_MUTED), zeroline=False)


def _pmc_figure(pmc: pd.DataFrame) -> go.Figure:
    """CTL/ATL/TSB lines over daily TSS bars — all in TSS points, one axis."""
    fig = go.Figure()
    fig.add_bar(
        x=pmc["date"],
        y=pmc["tss"],
        name="Daily TSS",
        marker=dict(color=_TSS_BAR),
        hovertemplate="%{y:.0f}<extra>TSS</extra>",
    )
    for column, name, color in (
        ("ctl", "CTL · fitness", _CTL_BLUE),
        ("atl", "ATL · fatigue", _ATL_ORANGE),
        ("tsb", "TSB · form", _TSB_AQUA),
    ):
        fig.add_scatter(
            x=pmc["date"],
            y=pmc[column],
            name=name,
            mode="lines",
            line=dict(color=color, width=2, shape="spline", smoothing=0.6),
            hovertemplate="%{y:.1f}<extra>" + name.split(" ")[0] + "</extra>",
        )
    _base_layout(fig, height=380)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(
        title_text="TSS / training load",
        title_font=dict(color=_MUTED),
        zeroline=True,
        zerolinecolor=_AXIS,
        zerolinewidth=1,
    )
    return fig


def _power_curve_figure(curve: dict[int, float]) -> go.Figure:
    windows = sorted(curve)
    fig = go.Figure(
        go.Scatter(
            x=windows,
            y=[curve[w] for w in windows],
            mode="lines+markers",
            line=dict(color=_CTL_BLUE, width=2),
            marker=dict(size=8, color=_CTL_BLUE, line=dict(color=_SURFACE, width=2)),
            hovertemplate="best %{customdata}: %{y:.0f} W<extra></extra>",
            customdata=[_WINDOW_LABELS.get(w, f"{w}s") for w in windows],
        )
    )
    _base_layout(fig, height=360)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(
        type="log",
        tickvals=windows,
        ticktext=[_WINDOW_LABELS.get(w, f"{w}s") for w in windows],
    )
    fig.update_yaxes(title_text="Best avg power (W)", title_font=dict(color=_MUTED))
    return fig


def _zones_figure(dist: ZoneDistribution, ramp: tuple[str, ...]) -> go.Figure:
    """One horizontal stacked bar; segment = share of time in that zone."""
    fig = go.Figure()
    for i, (label, seconds, pct) in enumerate(
        zip(dist.labels, dist.seconds, dist.percent, strict=True)
    ):
        fig.add_bar(
            y=[""],
            x=[pct],
            orientation="h",
            name=label,
            marker=dict(color=ramp[i], line=dict(color=_SURFACE, width=1)),
            text=f"{pct:.0f}%" if pct >= 6 else "",
            textposition="inside",
            insidetextanchor="middle",
            insidetextfont=dict(color=_INK if i < _LIGHT_RAMP_STEPS else "#ffffff", size=12),
            customdata=[[label, _fmt_duration(seconds)]],
            hovertemplate="%{customdata[0]}: %{customdata[1]} (%{x:.1f} %)<extra></extra>",
        )
    _base_layout(fig, height=180)
    fig.update_layout(
        barmode="stack",
        margin=dict(l=16, r=16, t=8, b=36),
        legend=dict(traceorder="normal"),  # plotly reverses stacked-bar legends by default
    )
    fig.update_xaxes(range=[0, 100], ticksuffix=" %", showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return fig


def durability_index_table(data: ReportData) -> dict:
    """Durability index per window x bucket for the report's mini table."""
    buckets = list(dict.fromkeys(data.durability["bucket"]))
    windows = sorted(data.durability["window_s"].unique())
    by_key = {
        (row["bucket"], row["window_s"]): row["durability_index"]
        for _, row in data.durability.iterrows()
    }
    rows = []
    for window in windows:
        cells = []
        for bucket in buckets:
            index = by_key.get((bucket, window))
            cells.append("–" if index is None or pd.isna(index) else f"{index:.2f}")
        rows.append({"window": _WINDOW_LABELS.get(window, f"{window}s"), "cells": cells})
    return {"buckets": buckets, "rows": rows}


def climb_rows(data: ReportData, top_n: int = _TOP_CLIMBS_N) -> list[dict[str, str]]:
    """Display rows for the biggest climbs (by elevation gain) across the history."""
    efforts = {id(climb): len(group) for group in data.climb_groups for climb in group}
    climbs = [climb for a in data.rides for climb in a.climbs]
    climbs.sort(key=lambda c: c.elevation_gain_m, reverse=True)

    rows = []
    for climb in climbs[:top_n]:
        pacing = "–"
        if climb.quarter_avg_power_watts is not None:
            pacing = "/".join(
                "–" if pd.isna(q) else f"{q:.0f}" for q in climb.quarter_avg_power_watts
            )
        rows.append(
            {
                "date": f"{climb.start_time:%Y-%m-%d}",
                "length": f"{climb.length_m / 1000:.1f}",
                "gain": f"{climb.elevation_gain_m:.0f}",
                "avg_grad": f"{climb.avg_gradient_pct:.1f}",
                "max_grad": f"{climb.max_gradient_pct:.1f}",
                "time": _fmt_duration(climb.duration_s),
                "vam": f"{climb.vam_m_per_h:.0f}",
                "avg_power": _fmt(climb.avg_power_watts, "{:.0f}"),
                "wkg": _fmt(climb.watts_per_kg, "{:.1f}"),
                "pacing": pacing,
                "kj_before": _fmt(climb.kj_before_climb, "{:.0f}"),
                "efforts": f"{efforts.get(id(climb), 1)}×",
            }
        )
    return rows


def comparison_rows(result: ComparisonResult) -> list[dict[str, str]]:
    """Display rows for the comparison table, formatted per metric."""
    formats = {
        "n_rides": ("Rides", "{:.0f}"),
        "total_distance_km": ("Distance (km)", "{:,.0f}"),
        "total_elevation_gain_m": ("Elevation gain (m)", "{:,.0f}"),
        "total_moving_time_h": ("Moving time (h)", "{:.1f}"),
        "total_tss": ("Total TSS", "{:,.0f}"),
        "avg_tss_per_week": ("Ø TSS / week", "{:.0f}"),
        "avg_ctl": ("Ø CTL", "{:.1f}"),
        "peak_ctl": ("Peak CTL", "{:.1f}"),
        "ftp_estimate_watts": ("FTP estimate (W)", "{:.0f}"),
        "rides_per_week": ("Rides / week *", "{:.1f}"),
        "distance_per_week_km": ("Distance / week (km) *", "{:.0f}"),
        "elevation_gain_per_week_m": ("Elevation / week (m) *", "{:,.0f}"),
        "moving_time_per_week_h": ("Moving time / week (h) *", "{:.1f}"),
    }
    rows = []
    for _, row in result.table.iterrows():
        label, spec = formats.get(row["metric"], (row["metric"], "{:.1f}"))
        delta_abs = row["delta_abs"]
        delta_pct = row["delta_pct"]
        if delta_abs is None or pd.isna(delta_abs):
            delta, trend = "–", ""
        else:
            delta = f"{delta_abs:+,.1f}".rstrip("0").rstrip(".")
            if delta_pct is not None and not pd.isna(delta_pct):
                delta += f" ({delta_pct:+.0f} %)"
            trend = "pos" if delta_abs > 0 else "neg" if delta_abs < 0 else ""
        rows.append(
            {
                "metric": label,
                "a": _fmt(None if pd.isna(row["period_a"]) else row["period_a"], spec),
                "b": _fmt(None if pd.isna(row["period_b"]) else row["period_b"], spec),
                "delta": delta,
                "trend": trend,
            }
        )
    return rows


def _evolution_groups(data: ReportData) -> list[list[Climb]]:
    """Most-ridden climb groups, big enough for a best-time-over-date chart."""
    groups = [g for g in data.climb_groups if len(g) >= _EVOLUTION_MIN_EFFORTS]
    groups.sort(key=len, reverse=True)
    return groups[:_EVOLUTION_MAX_GROUPS]


def _durability_figure(durability: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    buckets = list(dict.fromkeys(durability["bucket"]))
    for i, bucket in enumerate(buckets):
        sub = durability[
            (durability["bucket"] == bucket) & durability["mmp_watts"].notna()
        ].sort_values("window_s")
        if sub.empty:
            continue
        windows = sub["window_s"].tolist()
        fig.add_scatter(
            x=windows,
            y=sub["mmp_watts"],
            name=bucket,
            mode="lines+markers",
            line=dict(color=_DURABILITY_RAMP[i % len(_DURABILITY_RAMP)], width=2),
            marker=dict(size=7),
            hovertemplate="%{customdata}: %{y:.0f} W<extra>" + bucket + "</extra>",
            customdata=[_WINDOW_LABELS.get(w, f"{w}s") for w in windows],
        )
    _base_layout(fig, height=360)
    all_windows = sorted(durability["window_s"].unique())
    fig.update_xaxes(
        type="log",
        tickvals=all_windows,
        ticktext=[_WINDOW_LABELS.get(w, f"{w}s") for w in all_windows],
    )
    fig.update_yaxes(title_text="Best avg power (W)", title_font=dict(color=_MUTED))
    return fig


def _elevation_profile_figure(analyzed: AnalyzedRide) -> go.Figure:
    df = analyzed.ride.df
    altitude = smooth_altitude(df)
    distance_km = df["distance"] / 1000

    fig = go.Figure(
        go.Scatter(
            x=distance_km,
            y=altitude,
            mode="lines",
            line=dict(color=_CTL_BLUE, width=1.5),
            fill="tozeroy",
            fillcolor=_PROFILE_FILL,
            name="Elevation",
            hovertemplate="%{x:.1f} km · %{y:.0f} m<extra></extra>",
        )
    )
    for climb in analyzed.climbs:
        start = climb.start_time
        mask = (df["timestamp"] >= pd.Timestamp(start)) & (
            df["timestamp"] <= pd.Timestamp(start) + pd.Timedelta(seconds=climb.duration_s)
        )
        fig.add_scatter(
            x=distance_km[mask],
            y=altitude[mask],
            mode="lines",
            line=dict(color=_ATL_ORANGE, width=2),
            fill="tozeroy",
            fillcolor=_CLIMB_FILL,
            showlegend=False,
            hovertemplate=(
                f"climb: {climb.length_m / 1000:.1f} km · +{climb.elevation_gain_m:.0f} m · "
                f"{climb.avg_gradient_pct:.1f} %<extra></extra>"
            ),
        )
    _base_layout(fig, height=300)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="km", title_font=dict(color=_MUTED))
    fig.update_yaxes(title_text="m", title_font=dict(color=_MUTED))
    return fig


def _climb_evolution_figure(groups: list[list[Climb]]) -> go.Figure:
    fig = go.Figure()
    for i, group in enumerate(groups):
        efforts = sorted(group, key=lambda c: c.start_time)
        label = f"{group[0].length_m / 1000:.1f} km · +{group[0].elevation_gain_m:.0f} m"
        fig.add_scatter(
            x=[c.start_time for c in efforts],
            y=[c.duration_s / 60 for c in efforts],
            name=label,
            mode="lines+markers",
            line=dict(color=_DURABILITY_RAMP[i % len(_DURABILITY_RAMP)], width=2),
            marker=dict(size=7),
            hovertemplate="%{x|%d %b %Y}: %{y:.1f} min<extra>" + label + "</extra>",
        )
    _base_layout(fig, height=320)
    fig.update_yaxes(title_text="Time (min)", title_font=dict(color=_MUTED))
    return fig


def _comparison_curves_figure(a: PeriodSummary, b: PeriodSummary) -> go.Figure:
    fig = go.Figure()
    for period, color in ((a, _CTL_BLUE), (b, _ATL_ORANGE)):
        if not period.power_curve:
            continue
        windows = sorted(period.power_curve)
        label = f"{period.start:%Y-%m-%d} – {period.end:%Y-%m-%d}"
        fig.add_scatter(
            x=windows,
            y=[period.power_curve[w] for w in windows],
            name=label,
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=7),
            hovertemplate="best %{customdata}: %{y:.0f} W<extra></extra>",
            customdata=[_WINDOW_LABELS.get(w, f"{w}s") for w in windows],
        )
    _base_layout(fig, height=360)
    windows = sorted(set(a.power_curve) | set(b.power_curve))
    fig.update_xaxes(
        type="log",
        tickvals=windows,
        ticktext=[_WINDOW_LABELS.get(w, f"{w}s") for w in windows],
    )
    fig.update_yaxes(title_text="Best avg power (W)", title_font=dict(color=_MUTED))
    return fig


def _comparison_zones_figure(zones_a: ZoneDistribution, zones_b: ZoneDistribution) -> go.Figure:
    fig = go.Figure()
    for dist, name, color in ((zones_a, "Period A", _CTL_BLUE), (zones_b, "Period B", _ATL_ORANGE)):
        fig.add_bar(
            x=list(dist.labels),
            y=list(dist.percent),
            name=name,
            marker=dict(color=color),
            hovertemplate="%{x}: %{y:.1f} %<extra>" + name + "</extra>",
        )
    _base_layout(fig, height=320)
    fig.update_layout(barmode="group")
    fig.update_yaxes(ticksuffix=" %", title_text="Share of time", title_font=dict(color=_MUTED))
    return fig
