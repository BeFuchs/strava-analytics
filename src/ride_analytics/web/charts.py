"""Plotly figure dicts for the dashboard — JSON only, rendered client-side.

Styling follows the dashboard's emerald/champagne scheme: main series in
Emerald Ink, secondary series in derived tones, transparent background so the
card surface shows through, recessive grid. One y-axis per chart, always.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go

from ride_analytics.metrics.zones import ZoneDistribution

EMERALD = "#064E3B"
EMERALD_LIGHT = "#0A6B52"
EMERALD_DARK = "#043528"
AMBER = "#B45309"
SAGE = "#6F8478"
TSS_BAR = "#E8D0A8"

TEXT_SECONDARY = "#5A5A5A"
GRID = "#EFE5D0"
AXIS = "#D9C9A6"

# Sequential ramps, champagne-light -> emerald-dark (easy -> hard zone).
POWER_RAMP = ("#F3E2C0", "#DCCFA0", "#B4BC93", "#8CA680", "#5C8F6D", "#2E7A58", "#064E3B")
HR_RAMP = ("#F3E2C0", "#CBC498", "#94A984", "#4F8666", "#064E3B")
# Ramp steps light enough to need dark (not white) labels inside the segment.
LIGHT_RAMP_STEPS = {7: 3, 5: 2}

# Fresh -> fatigued kJ buckets, light -> dark.
DURABILITY_RAMP = ("#A9BC94", "#6FA07E", "#2E7A58", "#043528")

FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'

WINDOW_LABELS = {
    5: "5s",
    15: "15s",
    30: "30s",
    60: "1m",
    300: "5m",
    480: "8m",
    1200: "20m",
    3600: "60m",
}


def _fig_dict(fig: go.Figure) -> dict:
    # Round-trip through plotly's encoder: numpy arrays and timestamps become
    # plain JSON types, NaN becomes null.
    return json.loads(fig.to_json())


def _base_layout(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=TEXT_SECONDARY),
        margin=dict(l=56, r=24, t=16, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        barcornerradius=4,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)


def pmc_figure(pmc: pd.DataFrame) -> dict:
    """CTL/ATL/TSB lines over daily TSS bars — all in TSS points, one axis."""
    fig = go.Figure()
    fig.add_bar(
        x=pmc["date"],
        y=pmc["tss"],
        name="Daily TSS",
        marker=dict(color=TSS_BAR),
        hovertemplate="%{y:.0f}<extra>TSS</extra>",
    )
    for column, name, color in (
        ("ctl", "CTL · Fitness", EMERALD),
        ("atl", "ATL · Ermüdung", AMBER),
        ("tsb", "TSB · Form", SAGE),
    ):
        fig.add_scatter(
            x=pmc["date"],
            y=pmc[column],
            name=name,
            mode="lines",
            line=dict(color=color, width=2, shape="spline", smoothing=0.6),
            hovertemplate="%{y:.1f}<extra>" + name.split(" ")[0] + "</extra>",
        )
    _base_layout(fig, height=360)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="TSS / Trainingslast", zeroline=True, zerolinecolor=AXIS)
    return _fig_dict(fig)


def power_curve_figure(curve: dict[int, float]) -> dict:
    windows = sorted(curve)
    fig = go.Figure(
        go.Scatter(
            x=windows,
            y=[curve[w] for w in windows],
            mode="lines+markers",
            line=dict(color=EMERALD, width=2),
            marker=dict(size=8, color=EMERALD, line=dict(color="#FFFFFF", width=2)),
            hovertemplate="Best %{customdata}: %{y:.0f} W<extra></extra>",
            customdata=[WINDOW_LABELS.get(w, f"{w}s") for w in windows],
        )
    )
    _base_layout(fig, height=340)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(
        type="log",
        tickvals=windows,
        ticktext=[WINDOW_LABELS.get(w, f"{w}s") for w in windows],
    )
    fig.update_yaxes(title_text="Beste Ø-Leistung (W)")
    return _fig_dict(fig)


def zones_figure(dist: ZoneDistribution, ramp: tuple[str, ...]) -> dict:
    """One horizontal stacked bar; segment = share of time in that zone."""
    fig = go.Figure()
    light_steps = LIGHT_RAMP_STEPS.get(len(ramp), 2)
    for i, (label, seconds, pct) in enumerate(
        zip(dist.labels, dist.seconds, dist.percent, strict=True)
    ):
        fig.add_bar(
            y=[""],
            x=[pct],
            orientation="h",
            name=label,
            marker=dict(color=ramp[i], line=dict(color="#FFFFFF", width=1)),
            text=f"{pct:.0f}%" if pct >= 6 else "",
            textposition="inside",
            insidetextanchor="middle",
            insidetextfont=dict(color="#1A1A1A" if i < light_steps else "#FFFFFF", size=12),
            customdata=[[label, _fmt_duration(seconds)]],
            hovertemplate="%{customdata[0]}: %{customdata[1]} (%{x:.1f} %)<extra></extra>",
        )
    _base_layout(fig, height=170)
    fig.update_layout(
        barmode="stack",
        margin=dict(l=16, r=16, t=8, b=36),
        legend=dict(traceorder="normal"),  # plotly reverses stacked-bar legends by default
    )
    fig.update_xaxes(range=[0, 100], ticksuffix=" %", showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return _fig_dict(fig)


def durability_figure(durability: pd.DataFrame) -> dict:
    """Mean-max power per kJ bucket — the curve family, fresh to fatigued."""
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
            line=dict(color=DURABILITY_RAMP[i % len(DURABILITY_RAMP)], width=2),
            marker=dict(size=7),
            hovertemplate="%{customdata}: %{y:.0f} W<extra>" + bucket + "</extra>",
            customdata=[WINDOW_LABELS.get(w, f"{w}s") for w in windows],
        )
    _base_layout(fig, height=340)
    all_windows = sorted(durability["window_s"].unique())
    fig.update_xaxes(
        type="log",
        tickvals=all_windows,
        ticktext=[WINDOW_LABELS.get(w, f"{w}s") for w in all_windows],
    )
    fig.update_yaxes(title_text="Beste Ø-Leistung (W)")
    return _fig_dict(fig)


def _fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"
