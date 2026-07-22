"""Write the computed metrics as CSV files for analysis in Excel/Sheets.

Conventions: UTF-8, comma separator, dot as decimal separator, ISO-8601 dates
(YYYY-MM-DD), snake_case headers with the unit in the name. Missing values are
written as empty fields — never 0 or a "NaN" string.

This layer only formats and writes; every number comes from ``metrics/`` via
the assembled ``ReportData``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ride_analytics.report.builder import ReportData


def export_csv(data: ReportData, weight_kg: float, out_dir: str | Path) -> list[Path]:
    """Write all metric CSVs into ``out_dir`` (created if needed); returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    durability = data.durability.round({"mmp_watts": 1, "durability_index": 3})
    written = [
        _write(_rides_frame(data), out_dir / "rides.csv"),
        _write(_pmc_frame(data), out_dir / "pmc.csv"),
        _write(_power_curve_frame(data, weight_kg), out_dir / "power_curve.csv"),
        _write(_zones_frame(data), out_dir / "zones.csv"),
        _write(durability, out_dir / "durability.csv"),
    ]
    return written


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, encoding="utf-8", na_rep="")
    return path


def _rides_frame(data: ReportData) -> pd.DataFrame:
    rows = []
    for a in data.rides:
        m = a.metrics
        rows.append(
            {
                "date": f"{a.ride.metadata.start_time:%Y-%m-%d}",
                "source_file": a.ride.metadata.source,
                "distance_km": _round(m.distance_km, 2),
                "moving_time_s": round(m.moving_time_s),
                "elapsed_time_s": round(m.elapsed_time_s),
                "np_watts": _round(m.np_watts, 1),
                "intensity_factor": _round(m.intensity_factor, 3),
                "tss": _round(m.tss, 1),
                "tss_estimated": "true" if m.tss_estimated else "false",
                "variability_index": _round(m.variability_index, 3),
                "work_kj": _round(m.work_kj, 1),
                "avg_power_watts": _round(m.avg_power, 1),
                "max_power_watts": _round(m.max_power, 0),
                "avg_hr_bpm": _round(m.avg_hr, 1),
                "max_hr_bpm": _round(m.max_hr, 0),
                "avg_cadence_rpm": _round(m.avg_cadence, 1),
                "avg_speed_kmh": _round(m.avg_speed_kmh, 1),
            }
        )
    return pd.DataFrame(rows)


def _pmc_frame(data: ReportData) -> pd.DataFrame:
    pmc = data.pmc.copy()
    if not pmc.empty:
        pmc["date"] = pmc["date"].dt.strftime("%Y-%m-%d")
        pmc = pmc.round({"tss": 1, "ctl": 1, "atl": 1, "tsb": 1})
    return pmc


def _power_curve_frame(data: ReportData, weight_kg: float) -> pd.DataFrame:
    windows = sorted(data.power_curve)
    return pd.DataFrame(
        {
            "window_s": windows,
            "watts": [round(data.power_curve[w], 1) for w in windows],
            "watts_per_kg": [round(data.power_curve[w] / weight_kg, 2) for w in windows],
        }
    )


def _zones_frame(data: ReportData) -> pd.DataFrame:
    rows = []
    for zone_type, dist in (("power", data.power_zones), ("hr", data.hr_zones)):
        if dist is None:
            continue
        for label, seconds, percent in zip(dist.labels, dist.seconds, dist.percent, strict=True):
            rows.append(
                {
                    "zone": label,
                    "type": zone_type,
                    "seconds": round(seconds),
                    "percent": round(percent, 1),
                }
            )
    return pd.DataFrame(rows, columns=["zone", "type", "seconds", "percent"])


def _round(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(value, digits)
