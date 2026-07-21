"""Click entry point — wiring only: config, ingest, metrics via report builder."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from ride_analytics.config import AthleteConfig, ConfigError, load_config
from ride_analytics.ingest import IngestError, load_rides
from ride_analytics.report.builder import (
    ReportData,
    build_report_data,
    render_report,
    ride_rows,
    totals,
)

DEFAULT_CONFIGS = ("config.yaml", "config.example.yaml")


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
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Athlete config YAML (default: config.yaml, else config.example.yaml).",
)
def analyze(path: Path, report_path: Path, summary: bool, config_path: Path | None) -> None:
    """Analyze PATH — a FIT file or a folder of FIT files — into an HTML report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = _resolve_config(config_path)
    try:
        rides = load_rides(path)
    except IngestError as exc:
        raise click.ClickException(str(exc)) from exc
    if not rides:
        raise click.ClickException(f"no cycling activities found in {path}")

    data = build_report_data(rides, config)
    out = render_report(data, config, report_path)
    click.echo(f"{len(rides)} ride(s) analyzed → {out}")

    if summary:
        _print_summary(data)


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


if __name__ == "__main__":
    main()
