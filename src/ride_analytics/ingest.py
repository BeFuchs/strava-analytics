"""Read Garmin/Strava FIT files into normalized per-ride DataFrames.

This module only parses and normalizes; it computes no metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from fitparse import FitFile
from fitparse.utils import FitParseError

logger = logging.getLogger(__name__)

RECORD_FIELDS = ("timestamp", "power", "heart_rate", "cadence", "speed", "altitude", "distance")


class IngestError(Exception):
    """Raised when FIT input cannot be read."""


@dataclass(frozen=True)
class RideMeta:
    source: str
    start_time: datetime
    duration_s: float
    sport: str


@dataclass(frozen=True)
class Ride:
    metadata: RideMeta
    df: pd.DataFrame


def load_fit(path: str | Path) -> Ride | None:
    """Parse a single FIT file.

    Returns ``None`` for non-cycling activities (logged, not an error).
    Raises ``IngestError`` for missing/unreadable files or files without records.
    """
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"FIT file not found: {path}")

    try:
        fit = FitFile(str(path))
        records = [msg.get_values() for msg in fit.get_messages("record")]
        sessions = [msg.get_values() for msg in fit.get_messages("session")]
    except FitParseError as exc:
        raise IngestError(f"{path.name}: cannot parse FIT file ({exc})") from exc

    sport = _extract_sport(sessions)
    if sport != "cycling":
        logger.info("skipping %s: sport is %r, not cycling", path.name, sport)
        return None
    if not records:
        raise IngestError(f"{path.name}: no record messages found")

    df = _normalize_records(records)
    if df.empty:
        raise IngestError(f"{path.name}: no records with valid timestamps")

    first_ts = df["timestamp"].iloc[0].to_pydatetime()
    last_ts = df["timestamp"].iloc[-1].to_pydatetime()
    start_time = _session_value(sessions, "start_time") or first_ts
    duration_s = _session_value(sessions, "total_elapsed_time")
    if duration_s is None:
        duration_s = (last_ts - first_ts).total_seconds()

    meta = RideMeta(
        source=path.name,
        start_time=start_time,
        duration_s=float(duration_s),
        sport=sport,
    )
    return Ride(metadata=meta, df=df)


def load_rides(path: str | Path) -> list[Ride]:
    """Load one FIT file or every ``*.fit`` in a directory, sorted by start time.

    Non-cycling activities are skipped. In directory mode, unreadable files are
    logged and skipped; for a single explicit file the error is raised.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() == ".fit")
        if not files:
            raise IngestError(f"no .fit files found in {path}")
    elif path.is_file():
        files = [path]
    else:
        raise IngestError(f"path not found: {path}")

    rides: list[Ride] = []
    for file in files:
        try:
            ride = load_fit(file)
        except IngestError as exc:
            if len(files) == 1:
                raise
            logger.warning("skipping file: %s", exc)
            continue
        if ride is not None:
            rides.append(ride)

    rides.sort(key=lambda ride: ride.metadata.start_time)
    return rides


def _extract_sport(sessions: list[dict]) -> str:
    for session in sessions:
        sport = session.get("sport")
        if sport is not None:
            return str(sport)
    return "unknown"


def _session_value(sessions: list[dict], field: str):
    for session in sessions:
        value = session.get(field)
        if value is not None:
            return value
    return None


def _normalize_records(records: list[dict]) -> pd.DataFrame:
    """Build a clean DataFrame: known columns only, sorted, unique timestamps.

    Gaps (auto-pause, GPS dropout) are kept as-is — no interpolation; the
    metrics layer derives moving vs. elapsed time from the raw timestamps.
    """
    df = pd.DataFrame.from_records(records)
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp"])

    present = [field for field in RECORD_FIELDS if field in df.columns]
    df = df[present]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    for col in df.columns.drop("timestamp"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="first")

    # Drop sensor columns that never delivered a valid value.
    data_cols = df.columns.drop("timestamp")
    all_nan = [col for col in data_cols if df[col].isna().all()]
    df = df.drop(columns=all_nan)

    return df.reset_index(drop=True)
