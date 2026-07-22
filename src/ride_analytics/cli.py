"""Click entry point — wiring only: config, ingest, metrics via report builder."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import click

from ride_analytics.config import AthleteConfig, ConfigError, load_config
from ride_analytics.export.csv_export import export_comparison_csv, export_csv
from ride_analytics.ingest import IngestError, load_rides
from ride_analytics.metrics.comparison import ComparisonResult, compare_periods
from ride_analytics.report.builder import (
    ReportData,
    build_report_data,
    comparison_rows,
    render_comparison_report,
    render_report,
    ride_rows,
    totals,
)

DEFAULT_CONFIGS = ("config.yaml", "config.example.yaml")

PRESET_LAST_TWO_SEASONS = "last-two-seasons"


@click.group()
def main() -> None:
    """Analyze Strava/Garmin FIT exports locally — no API, no cloud."""


@main.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("report.html"),
    show_default=True,
    help="Output path for the HTML report.",
)
@click.option("--summary", is_flag=True, help="Also print a compact summary table.")
@click.option(
    "--export-csv",
    "csv_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Also write all metrics as CSV files into this directory.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Athlete config YAML (default: config.yaml, else config.example.yaml).",
)
def analyze(
    path: Path,
    report_path: Path,
    summary: bool,
    csv_dir: Path | None,
    config_path: Path | None,
) -> None:
    """Analyze PATH — a FIT file or a folder of FIT files — into an HTML report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = _resolve_config(config_path)
    rides = _load_rides_checked(path)

    data = build_report_data(rides, config)
    out = render_report(data, config, report_path)
    click.echo(f"{len(rides)} ride(s) analyzed → {out}")

    if csv_dir is not None:
        written = export_csv(data, config.weight_kg, csv_dir)
        click.echo(f"{len(written)} CSV file(s) → {csv_dir}")

    if summary:
        _print_summary(data)


@main.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.option("--period-a", default=None, help="First period as YYYY-MM-DD:YYYY-MM-DD.")
@click.option("--period-b", default=None, help="Second period as YYYY-MM-DD:YYYY-MM-DD.")
@click.option(
    "--preset",
    type=click.Choice([PRESET_LAST_TWO_SEASONS]),
    default=None,
    help="Shortcut: previous calendar year vs. current calendar year.",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Also write an HTML comparison report to this path.",
)
@click.option(
    "--export-csv",
    "csv_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Also write the comparison table as comparison.csv into this directory.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Athlete config YAML (default: config.yaml, else config.example.yaml).",
)
def compare(
    path: Path,
    period_a: str | None,
    period_b: str | None,
    preset: str | None,
    report_path: Path | None,
    csv_dir: Path | None,
    config_path: Path | None,
) -> None:
    """Compare two periods of the ride history in PATH."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if preset == PRESET_LAST_TWO_SEASONS:
        year = date.today().year
        range_a = (date(year - 1, 1, 1), date(year - 1, 12, 31))
        range_b = (date(year, 1, 1), date(year, 12, 31))
    elif period_a and period_b:
        range_a = _parse_period(period_a)
        range_b = _parse_period(period_b)
    else:
        raise click.UsageError(
            f"pass --period-a and --period-b, or --preset {PRESET_LAST_TWO_SEASONS}"
        )

    config = _resolve_config(config_path)
    rides = _load_rides_checked(path)

    result = compare_periods(rides, config, range_a, range_b)
    for summary in (result.period_a, result.period_b):
        if summary.n_rides == 0:
            click.echo(
                f"note: period {summary.label} ({summary.start} – {summary.end}) contains no rides"
            )

    _print_comparison(result)

    if report_path is not None:
        out = render_comparison_report(result, config, report_path)
        click.echo(f"\ncomparison report → {out}")
    if csv_dir is not None:
        written = export_comparison_csv(result, csv_dir)
        click.echo(f"comparison table → {written}")


def _parse_period(value: str) -> tuple[date, date]:
    try:
        start_raw, end_raw = value.split(":")
        start, end = date.fromisoformat(start_raw), date.fromisoformat(end_raw)
    except ValueError as exc:
        raise click.BadParameter(f"{value!r} — expected YYYY-MM-DD:YYYY-MM-DD") from exc
    if end < start:
        raise click.BadParameter(f"{value!r} — period ends before it starts")
    return start, end


def _load_rides_checked(path: Path) -> list:
    try:
        rides = load_rides(path)
    except IngestError as exc:
        raise click.ClickException(str(exc)) from exc
    if not rides:
        raise click.ClickException(f"no cycling activities found in {path}")
    return rides


def _resolve_config(config_path: Path | None) -> AthleteConfig:
    candidates = [config_path] if config_path else [Path(name) for name in DEFAULT_CONFIGS]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return load_config(candidate)
            except ConfigError as exc:
                raise click.ClickException(str(exc)) from exc
    raise click.ClickException(
        "no config found — pass --config or create config.yaml (see config.example.yaml)"
    )


def _print_summary(data: ReportData) -> None:
    headers = {
        "date": "Date",
        "source": "File",
        "distance": "km",
        "duration": "Moving",
        "np": "NP",
        "if": "IF",
        "tss": "TSS",
        "avg_hr": "Ø HR",
    }
    rows = ride_rows(data)
    widths = {
        key: max(len(label), *(len(row[key]) for row in rows)) for key, label in headers.items()
    }

    def line(values: dict[str, str]) -> str:
        left = ("date", "source")
        return "  ".join(
            values[key].ljust(widths[key]) if key in left else values[key].rjust(widths[key])
            for key in headers
        )

    click.echo()
    click.echo(line(headers))
    click.echo("  ".join("─" * widths[key] for key in headers))
    for row in rows:
        click.echo(line(row))

    total = totals(data)
    click.echo(
        f"\nTotal: {total['rides']} rides · {total['distance']} · "
        f"{total['moving_time']} moving · TSS {total['tss']}"
    )
    if any(a.metrics.tss_estimated for a in data.rides):
        click.echo("* TSS estimated from heart rate (no power data)")


def _print_comparison(result: ComparisonResult) -> None:
    a, b = result.period_a, result.period_b
    click.echo(f"\nPeriod A: {a.start} – {a.end} ({a.n_rides} rides)")
    click.echo(f"Period B: {b.start} – {b.end} ({b.n_rides} rides)")
    click.echo()

    rows = comparison_rows(result)
    headers = {"metric": "Metric", "a": "Period A", "b": "Period B", "delta": "Δ"}
    widths = {
        key: max(len(label), *(len(row[key]) for row in rows)) for key, label in headers.items()
    }

    def line(values: dict[str, str]) -> str:
        return "  ".join(
            values[key].ljust(widths[key]) if key == "metric" else values[key].rjust(widths[key])
            for key in headers
        )

    click.echo(line(headers))
    click.echo("  ".join("─" * widths[key] for key in headers))
    for row in rows:
        click.echo(line(row))

    if not result.equal_length:
        click.echo("\n* periods differ in length — per-week rows are the fair comparison")


if __name__ == "__main__":
    main()
