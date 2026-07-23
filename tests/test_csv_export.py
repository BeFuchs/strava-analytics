import csv
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from ride_analytics.config import AthleteConfig
from ride_analytics.export.csv_export import CLIMB_COLUMNS, export_comparison_csv, export_csv
from ride_analytics.ingest import Ride, RideMeta
from ride_analytics.metrics.comparison import compare_periods
from ride_analytics.report.builder import build_report_data
from test_climbs import track_df

CONFIG = AthleteConfig(ftp_watts=200, threshold_hr=160, weight_kg=70.0, max_hr=190)
START = datetime(2024, 5, 1, 8, 0, 0)

EXPECTED_FILES = (
    "rides.csv",
    "pmc.csv",
    "power_curve.csv",
    "zones.csv",
    "durability.csv",
    "climbs.csv",
)

RIDES_COLUMNS = [
    "date",
    "source_file",
    "distance_km",
    "elevation_gain_m",
    "moving_time_s",
    "elapsed_time_s",
    "np_watts",
    "intensity_factor",
    "tss",
    "tss_estimated",
    "variability_index",
    "work_kj",
    "avg_power_watts",
    "max_power_watts",
    "avg_hr_bpm",
    "max_hr_bpm",
    "avg_cadence_rpm",
    "avg_speed_kmh",
]


def make_ride(name: str, start: datetime, columns: dict) -> Ride:
    n = len(next(iter(columns.values())))
    df = pd.DataFrame({"timestamp": [start + timedelta(seconds=i) for i in range(n)], **columns})
    meta = RideMeta(source=name, start_time=start, duration_s=float(n), sport="cycling")
    return Ride(metadata=meta, df=df)


@pytest.fixture
def report_data():
    power_ride = make_ride(
        "with_power.fit",
        START,
        {
            "power": [200] * 3600,
            "heart_rate": [140] * 3600,
            "distance": [i * 8.0 for i in range(3600)],
        },
    )
    hr_only = make_ride("hr_only.fit", START + timedelta(days=1), {"heart_rate": [150] * 1800})
    return build_report_data([power_ride, hr_only], CONFIG)


def read_rows(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_writes_all_files(tmp_path, report_data):
    written = export_csv(report_data, CONFIG.weight_kg, tmp_path / "out")

    assert [p.name for p in written] == list(EXPECTED_FILES)
    for path in written:
        assert path.is_file()


def test_rides_csv_columns_and_rows(tmp_path, report_data):
    export_csv(report_data, CONFIG.weight_kg, tmp_path)
    rows = read_rows(tmp_path / "rides.csv")

    assert list(rows[0].keys()) == RIDES_COLUMNS
    assert len(rows) == 2
    assert rows[0]["date"] == "2024-05-01"
    assert rows[0]["source_file"] == "with_power.fit"
    assert float(rows[0]["np_watts"]) == pytest.approx(200.0)
    assert rows[0]["tss_estimated"] == "false"
    assert rows[1]["tss_estimated"] == "true"


def test_missing_values_are_empty_fields(tmp_path, report_data):
    export_csv(report_data, CONFIG.weight_kg, tmp_path)
    rows = read_rows(tmp_path / "rides.csv")

    # HR-only ride has no power metrics — empty fields, not 0 or NaN.
    assert rows[1]["np_watts"] == ""
    assert rows[1]["work_kj"] == ""

    for name in EXPECTED_FILES:
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "NaN" not in text
        assert "None" not in text


def test_pmc_csv_daily_series(tmp_path, report_data):
    export_csv(report_data, CONFIG.weight_kg, tmp_path)
    rows = read_rows(tmp_path / "pmc.csv")

    assert list(rows[0].keys()) == ["date", "tss", "ctl", "atl", "tsb"]
    assert rows[0]["date"] == "2024-05-01"
    assert len(rows) == 2  # two consecutive days
    assert float(rows[0]["tss"]) == pytest.approx(100.0)  # 1 h at FTP


def test_power_curve_csv_includes_watts_per_kg(tmp_path, report_data):
    export_csv(report_data, CONFIG.weight_kg, tmp_path)
    rows = read_rows(tmp_path / "power_curve.csv")

    assert list(rows[0].keys()) == ["window_s", "watts", "watts_per_kg"]
    for row in rows:
        assert float(row["watts_per_kg"]) == pytest.approx(
            float(row["watts"]) / CONFIG.weight_kg, abs=0.01
        )


def test_zones_csv_covers_power_and_hr(tmp_path, report_data):
    export_csv(report_data, CONFIG.weight_kg, tmp_path)
    rows = read_rows(tmp_path / "zones.csv")

    assert list(rows[0].keys()) == ["zone", "type", "seconds", "percent"]
    types = {row["type"] for row in rows}
    assert types == {"power", "hr"}
    power_pct = sum(float(r["percent"]) for r in rows if r["type"] == "power")
    assert power_pct == pytest.approx(100.0, abs=0.5)


def test_climbs_csv_empty_without_altitude_data(tmp_path, report_data):
    export_csv(report_data, CONFIG.weight_kg, tmp_path)

    rows = read_rows(tmp_path / "climbs.csv")
    assert rows == []
    header = (tmp_path / "climbs.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(CLIMB_COLUMNS)


def test_climbs_csv_lists_detected_climbs(tmp_path):
    df = track_df([(1000, 0.0, 150), (3000, 0.06, 250), (1000, 0.0, 150)])
    ride = Ride(
        metadata=RideMeta(source="climb.fit", start_time=START, duration_s=1000.0, sport="cycling"),
        df=df,
    )
    data = build_report_data([ride], CONFIG)

    export_csv(data, CONFIG.weight_kg, tmp_path)

    rows = read_rows(tmp_path / "climbs.csv")
    assert len(rows) == 1
    climb = rows[0]
    assert list(climb.keys()) == CLIMB_COLUMNS
    assert climb["source_file"] == "climb.fit"
    assert float(climb["elevation_gain_m"]) == pytest.approx(180.0, abs=3.0)
    assert float(climb["avg_gradient_pct"]) == pytest.approx(6.0, abs=0.2)
    assert climb["matched_climb_id"] == "0"
    assert climb["start_lat"] == ""  # synthetic track has no GPS

    rides = read_rows(tmp_path / "rides.csv")
    assert float(rides[0]["elevation_gain_m"]) == pytest.approx(180.0, abs=5.0)


def test_comparison_csv_round_trip(tmp_path):
    may = make_ride("a.fit", datetime(2025, 5, 2, 9, 0, 0), {"power": [200] * 3600})
    june = make_ride("b.fit", datetime(2025, 6, 3, 9, 0, 0), {"power": [200] * 1800})
    result = compare_periods(
        [may, june],
        CONFIG,
        (date(2025, 5, 1), date(2025, 5, 31)),
        (date(2025, 6, 1), date(2025, 6, 30)),
    )

    path = export_comparison_csv(result, tmp_path)

    assert path.name == "comparison.csv"
    rows = read_rows(path)
    assert list(rows[0].keys()) == ["metric", "period_a", "period_b", "delta_abs", "delta_pct"]
    by_metric = {row["metric"]: row for row in rows}
    assert float(by_metric["n_rides"]["period_a"]) == 1
    assert float(by_metric["total_tss"]["delta_pct"]) == pytest.approx(-50.0, abs=1.0)
    assert "distance_per_week_km" in by_metric  # May vs June differ in length


def test_durability_csv_columns(tmp_path, report_data):
    export_csv(report_data, CONFIG.weight_kg, tmp_path)
    rows = read_rows(tmp_path / "durability.csv")

    assert list(rows[0].keys()) == [
        "bucket",
        "window_s",
        "mmp_watts",
        "durability_index",
        "n_rides",
    ]
    fresh = [r for r in rows if r["bucket"] == "0-1000 kJ" and r["window_s"] == "1200"]
    assert float(fresh[0]["mmp_watts"]) == pytest.approx(200.0)


def test_export_training_csv_writes_three_files(tmp_path):
    from ride_analytics.export.csv_export import export_training_csv

    decoupling = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-01", "2026-05-03"]),
            "ef": [1.6123, None],
            "decoupling_pct": [3.456, None],
            "cardiac_drift_bpm_per_h": [1.23, None],
            "valid": [True, False],
            "reason": [None, "Fahrt unter 60 Minuten"],
        }
    )
    intervals = pd.DataFrame(
        {
            "ride_id": ["r1", "r1"],
            "date": ["2026-05-01", "2026-05-01"],
            "set_index": [0, 0],
            "rep_index": [0, 1],
            "avg_power_watts": [275.0, 273.0],
        }
    )
    classification = pd.DataFrame(
        {
            "ride_id": ["r1"],
            "date": ["2026-05-01"],
            "ride_type": ["intervals"],
            "confidence": ["high"],
            "matched_rules": ["4er-Set, 20 min über Schwelle"],
        }
    )

    written = export_training_csv(decoupling, intervals, classification, tmp_path)
    assert {p.name for p in written} == {"decoupling.csv", "intervals.csv", "classification.csv"}

    dec_rows = read_rows(tmp_path / "decoupling.csv")
    assert dec_rows[0]["date"] == "2026-05-01"  # datetime -> ISO
    assert dec_rows[0]["ef"] == "1.612"  # rounded to 3 dp
    assert dec_rows[1]["ef"] == ""  # missing stays empty, not 0

    cls_rows = read_rows(tmp_path / "classification.csv")
    assert cls_rows[0]["ride_type"] == "intervals"
    assert cls_rows[0]["confidence"] == "high"
